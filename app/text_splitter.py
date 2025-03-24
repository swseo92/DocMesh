from typing import List, Dict
from langchain.text_splitter import TokenTextSplitter


def default_length_function(text: str) -> int:
    # 간단하게 공백을 기준으로 단어 수를 셉니다.
    return len(text.split())


def default_tokenizer(text: str) -> List[str]:
    return text.split()


class DocumentChunkPipeline:
    """
    DocumentChunkPipeline은 크롤러에서 나온 청크(딕셔너리 리스트)를 대상으로
    1. 합병 → 2. 분할 → 3. overlap 적용의 순서로 처리합니다.

    - 합병: 같은 source 내에서 토큰 수가 min_tokens(500) 미만인 청크들을 합쳐 최소 크기를 보장합니다.
    - 분할: 합병된 청크가 max_tokens(500)를 초과하면, TokenTextSplitter를 이용해 추가 분할합니다.
    - overlap 적용: 동일 source의 인접 청크에 대해 앞 청크의 마지막 desired_overlap(100) 토큰을 현재 청크 앞에 추가
    """

    def __init__(
        self,
        max_tokens: int = 1000,
        min_tokens: int = 500,
        desired_overlap: int = 100,
        length_function=None,
        tokenizer=None,
    ):
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.desired_overlap = desired_overlap
        self.length_function = length_function or default_length_function
        self.tokenizer = tokenizer or default_tokenizer

        # 분할 단계에서는 overlap 없이 단순 분할
        self.token_text_splitter = TokenTextSplitter(
            chunk_size=self.max_tokens,
            chunk_overlap=0,
            length_function=self.length_function,
        )

    def merge_chunks(self, docs: List[Dict]) -> List[Dict]:
        """
        같은 source 내에서, 각 청크의 토큰 수가 min_tokens 미만인 경우 인접 청크들을 합병합니다.
        """
        merged_docs = []
        i = 0
        while i < len(docs):
            current_doc = docs[i]
            source = current_doc.get("source", "")
            merged_text = current_doc.get("text", "").strip()
            # 이미 충분히 큰 청크라면 그대로 사용
            if self.length_function(merged_text) >= self.min_tokens:
                merged_docs.append(
                    {
                        "source": source,
                        "text": merged_text,
                        "metadata": current_doc.get("metadata", {}),
                    }
                )
                i += 1
            else:
                # 현재 청크가 작다면, 같은 source의 다음 청크들을 합하여 최소 토큰 수에 도달할 때까지 병합
                j = i + 1
                while (
                    self.length_function(merged_text) < self.min_tokens
                    and j < len(docs)
                    and docs[j].get("source", "") == source
                ):
                    merged_text += " " + docs[j].get("text", "").strip()
                    j += 1
                merged_docs.append(
                    {
                        "source": source,
                        "text": merged_text,
                        "metadata": current_doc.get("metadata", {}),
                    }
                )
                i = j
        return merged_docs

    def split_merged_chunks(self, docs: List[Dict]) -> List[Dict]:
        """
        합병된 청크가 max_tokens를 초과하면 TokenTextSplitter를 사용해 추가 분할합니다.
        """
        split_docs = []
        for doc in docs:
            text = doc.get("text", "").strip()
            source = doc.get("source", "")
            if self.length_function(text) > self.max_tokens:
                split_texts = self.token_text_splitter.split_text(text)
                for idx, chunk in enumerate(split_texts):
                    split_docs.append(
                        {
                            "source": source,
                            "text": chunk,
                            "metadata": {**doc.get("metadata", {}), "split_index": idx},
                        }
                    )
            else:
                split_docs.append(doc)
        return split_docs

    def apply_overlap(self, docs: List[Dict]) -> List[Dict]:
        """
        동일 source 내에서 인접 청크에 대해, 앞 청크의 마지막 desired_overlap 토큰을
        현재 청크의 앞부분에 추가하여 overlap을 생성합니다.
        """
        final_docs = []
        for idx, doc in enumerate(docs):
            source = doc.get("source", "")
            text = doc.get("text", "").strip()
            if idx > 0 and docs[idx - 1].get("source", "") == source:
                prev_text = docs[idx - 1].get("text", "").strip()
                prev_tokens = self.tokenizer(prev_text)
                # 이전 청크의 마지막 desired_overlap 토큰 추출
                if len(prev_tokens) >= self.desired_overlap:
                    overlap_tokens = prev_tokens[-self.desired_overlap :]
                else:
                    overlap_tokens = prev_tokens
                overlap_text = " ".join(overlap_tokens)
                text = overlap_text + " " + text
            final_docs.append(
                {"source": source, "text": text, "metadata": doc.get("metadata", {})}
            )
        return final_docs

    def process(self, docs: List[Dict]) -> List[Dict]:
        # 1. 합병
        merged = self.merge_chunks(docs)
        # 2. 분할
        split_docs = self.split_merged_chunks(merged)
        # 3. overlap 적용
        final_docs = self.apply_overlap(split_docs)
        return final_docs


# ----------------------- 테스트 코드 -----------------------
if __name__ == "__main__":
    # 모의 크롤러 결과: 이미 HTML의 문단을 고려해 분할된 상태
    crawler_results = [
        {
            "source": "https://example.com",
            "text": "첫 번째 문단 " * 20,  # 약 20*3 = 60 토큰 정도 (min_tokens 미만)
            "metadata": {"source": "https://example.com"},
        },
        {
            "source": "https://example.com",
            "text": "두 번째 문단 " * 30,  # 약 90 토큰 정도 (여전히 작음)
            "metadata": {"source": "https://example.com"},
        },
        {
            "source": "https://example.com",
            "text": "세 번째 문단 "
            * 40,  # 약 120 토큰 정도 (합병하면 60+90+120 ≒ 270 토큰, 여전히 부족할 수 있음)
            "metadata": {"source": "https://example.com"},
        },
        {
            "source": "https://example.org",
            "text": "다른 사이트 문단 " * 200,  # 약 200*3 = 600 토큰 (이미 min_tokens 이상)
            "metadata": {"source": "https://example.org"},
        },
    ]

    # 파이프라인 파라미터: max_tokens=500, min_tokens=500, desired_overlap=100
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    final_chunks = pipeline.process(crawler_results)

    # 결과 검증: 각 최종 청크의 토큰 수와 overlap 적용 여부 출력
    for idx, doc in enumerate(final_chunks, start=1):
        tokens = default_tokenizer(doc["text"])
        print(f"[Chunk {idx}] Source: {doc['source']} - Token count: {len(tokens)}")
        # 두 번째 청크 이상에서 overlap 적용 여부 확인
        if idx > 1 and final_chunks[idx - 2]["source"] == doc["source"]:
            prev_tokens = default_tokenizer(final_chunks[idx - 2]["text"])
            expected_overlap = (
                " ".join(prev_tokens[-100:])
                if len(prev_tokens) >= 100
                else " ".join(prev_tokens)
            )
            if doc["text"].startswith(expected_overlap):
                print("  → Overlap 적용됨.")
            else:
                print("  → Overlap 미적용!")
        print("Excerpt:", doc["text"][:200], "\n")
