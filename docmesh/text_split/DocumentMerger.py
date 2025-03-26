from typing import List
from langchain.schema import Document
from docmesh.text_split.tools import default_length_function
from docmesh.config import Config


class DocumentMerger:
    """
    LangChain Document의 리스트를 받아, 인접 문서들이 같은 source일 때
    'min_tokens' 미만인 문서를 합쳐 최소 길이를 보장한다.

    - source 판단은 `doc.metadata["source"]`를 사용한다고 가정.
    - '합칠 수 없음' 상황(예: 다음 문서가 다른 source)이면, 그대로 단독으로 남긴다.
    """

    def __init__(
        self, min_tokens: int = 200, encoding_name: str = Config.TIKTOKEN_ENCODING
    ):
        """
        :param min_tokens: 병합 후 문서의 최소 길이(토큰 수)
        :param length_function: 텍스트 길이를 측정하는 함수
        """
        self.min_tokens = min_tokens
        self.encoding_name = encoding_name

    def length_function(self, text: str) -> int:
        length = default_length_function(text, encoding_name=self.encoding_name)
        return length

    def merge_documents(self, docs: List[Document]) -> List[Document]:
        merged_docs: List[Document] = []
        i = 0

        while i < len(docs):
            current_doc = docs[i]
            source = current_doc.metadata.get("source", None)
            text = current_doc.page_content.strip()

            # 길이 측정
            current_length = self.length_function(text)

            if current_length >= self.min_tokens:
                # 이미 충분히 길면 그대로 추가
                merged_docs.append(current_doc)
                i += 1
            else:
                # min_tokens 미만이므로, 같은 source의 다음 문서와 병합 시도
                j = i + 1
                while (
                    current_length < self.min_tokens
                    and j < len(docs)
                    and docs[j].metadata.get("source", None) == source
                ):
                    # 다음 문서를 이어 붙이고 길이 갱신
                    text += " " + docs[j].page_content.strip()
                    current_length = self.length_function(text)
                    j += 1

                # 병합 결과로 새 Document 생성
                new_doc = Document(
                    page_content=text, metadata=current_doc.metadata.copy()
                )
                merged_docs.append(new_doc)
                i = j

        return merged_docs
