import pytest
from app.text_splitter import DocumentChunkPipeline
from app.embedding import Document, EmbeddingStoreManager, LangchainFAISSVectorStore


# --- FakeEmbeddingModel 정의 ---
class FakeEmbeddingModel:
    """
    FakeEmbeddingModel은 실제 OpenAI API 호출 대신,
    텍스트 길이에 따라 고정 차원(1536)의 임베딩 벡터를 반환합니다.
    FAISS 벡터스토어가 요구하는 callable 인터페이스, vector_dim, embed_query 등을 구현합니다.
    """

    def __init__(self, dim: int = 1536):
        self.dim = dim
        self.vector_dim = dim
        self.embeddings = self  # FAISSVectorStore 초기화 시 embedding_function으로 사용

    def __call__(self, text: str) -> list:
        return self.get_embedding(text)

    def embed_query(self, text: str) -> list:
        return self.get_embedding(text)

    def get_embedding(self, text: str) -> list:
        # 테스트용: 텍스트 길이를 100으로 나눈 값을 모든 요소에 채워 반환
        value = len(text) / 100.0
        return [value] * self.dim


# --- 통합 테스트 ---
def test_integration_text_splitter_embedding():
    # 1. 샘플 텍스트 준비: 여러 문단으로 구성된 긴 텍스트
    sample_text = (
        "This is the first paragraph. " * 50
        + "\n\n"
        + "This is the second paragraph" * 70
        + "\n\n"
        + "This is the third paragraph" * 30
    )

    # 2. 크롤러 결과를 모의: text_splitter는 이미 HTML 문단 단위로 분할된 결과(딕셔너리 리스트)를 받는다고 가정
    crawler_results = [
        {
            "source": "https://integration-test.com",
            "text": sample_text,
            "metadata": {"source": "https://integration-test.com"},
        }
    ]

    # 3. DocumentChunkPipeline을 사용해 텍스트 분할, 합병, 분할 후 overlap 적용
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    final_chunks = pipeline.process(crawler_results)
    assert len(final_chunks) > 0, "최종 청크가 하나 이상 생성되어야 합니다."

    # 4. 각 최종 청크를 Document 객체로 변환 (embedding.py 의 Document)
    documents = [
        Document(chunk["text"], {"source": chunk["source"]}) for chunk in final_chunks
    ]

    # 5. FakeEmbeddingModel과 LangchainFAISSVectorStore를 이용해 EmbeddingStoreManager 초기화
    fake_model = FakeEmbeddingModel(dim=1536)
    vector_store = LangchainFAISSVectorStore(embedding_model=fake_model)
    manager = EmbeddingStoreManager(fake_model, vector_store)

    # 6. Document 객체들을 임베딩 및 저장
    manager.embed_and_store(documents)

    # 7. 간단한 쿼리 실행: "first paragraph"라는 쿼리로 검색
    query = "first paragraph"
    results = manager.search_chunks(query, k=3)
    assert len(results) > 0, "검색 결과가 하나 이상 반환되어야 합니다."

    # 8. 반환된 결과의 메타데이터에 올바른 출처가 포함되어 있는지 검증
    for res in results:
        assert "source" in res.metadata, "검색 결과의 metadata에 source 정보가 포함되어야 합니다."
        assert (
            res.metadata["source"] == "https://integration-test.com"
        ), "출처 정보가 일치해야 합니다."


if __name__ == "__main__":
    pytest.main()
