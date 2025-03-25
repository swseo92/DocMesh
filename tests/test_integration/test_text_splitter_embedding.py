import pytest
from docmesh.format import Document
from docmesh.text_splitter import DocumentChunkPipeline
from docmesh.embedding import EmbeddingModelFactory, BaseEmbeddingModel
from docmesh.vector_store import (
    VectorStoreFactory,
    EmbeddingStoreManager,
    BaseVectorStore,
)


# --- FakeEmbeddingModel (dummy embedding) ---
class FakeEmbeddingModel(BaseEmbeddingModel):
    """
    FakeEmbeddingModel은 실제 API 호출 없이, 입력 텍스트 길이에 따라 고정된 임베딩 벡터를 반환합니다.
    여기서는 1536 차원 벡터를 반환하며, 모든 요소는 텍스트 길이를 100으로 나눈 값으로 채웁니다.
    """

    def __init__(self, dim: int = 1536):
        self._dim = dim

    def get_embedding(self, text: str) -> list:
        value = len(text) / 100.0
        return [value] * self._dim

    @property
    def vector_dim(self) -> int:
        return self._dim


# --- DummyVectorStore (dummy vector store) ---
class DummyVectorStore(BaseVectorStore):
    """
    DummyVectorStore는 단순히 문서를 리스트에 저장하고,
    검색 시 저장된 문서 중 앞에서 k개를 반환합니다.
    """

    def __init__(self, embedding_model: BaseEmbeddingModel):
        self.embedding_model = embedding_model
        self.documents = []

    def add_documents(self, documents: list) -> None:
        self.documents.extend(documents)

    def search(self, query: str, k: int = 3) -> list:
        return self.documents[:k]


# --- 통합 테스트 헬퍼 함수 ---
def run_integration_test(
    embedding_model: BaseEmbeddingModel, vector_store: BaseVectorStore
):
    # 1. 샘플 텍스트 준비 (여러 문단으로 구성된 긴 텍스트)
    sample_text = (
        "This is the first paragraph. " * 50
        + "\n\n"
        + "This is the second paragraph. " * 70
        + "\n\n"
        + "This is the third paragraph. " * 30
    )
    # 2. 모의 크롤러 결과 생성 (단일 문서)
    crawler_results = [
        {
            "source": "https://integration-test.com",
            "text": sample_text,
            "metadata": {"source": "https://integration-test.com"},
        }
    ]
    # 3. DocumentChunkPipeline으로 분할 및 합병 처리
    pipeline = DocumentChunkPipeline(
        max_tokens=500, min_tokens=500, desired_overlap=100
    )
    final_chunks = pipeline.process(crawler_results)
    assert len(final_chunks) > 0, "최종 청크가 하나 이상 생성되어야 합니다."
    # 4. 각 청크를 Document 객체로 변환
    documents = [
        Document(chunk.page_content, {"source": chunk.metadata.get("source")})
        for chunk in final_chunks
    ]
    # 5. EmbeddingStoreManager 생성 및 문서 임베딩/저장
    manager = EmbeddingStoreManager(embedding_model, vector_store)
    manager.embed_and_store(documents)
    # 6. 간단한 쿼리 실행: "first paragraph"라는 쿼리로 검색
    results = manager.search_chunks("first paragraph", k=3)
    assert len(results) > 0, "검색 결과가 하나 이상 반환되어야 합니다."
    # 7. 반환된 결과의 metadata에 올바른 출처가 포함되어 있는지 검증
    for res in results:
        assert "source" in res.metadata, "검색 결과의 metadata에 source 정보가 포함되어야 합니다."
        assert (
            res.metadata["source"] == "https://integration-test.com"
        ), "출처 정보가 일치해야 합니다."


# --- 파라미터화된 통합 테스트 ---
@pytest.mark.parametrize(
    "embedding_provider, vector_store_provider, extra_params",
    [
        # 실제 embedding (openai)와 FAISS 조합
        ("openai", "faiss", {}),
        # 실제 embedding (openai)와 Annoy 조합
        ("openai", "annoy", {"n_trees": 5}),
        # dummy embedding와 dummy vector store 조합
        ("dummy", "dummy", {}),
    ],
)
def test_embedding_vector_store_integration(
    embedding_provider, vector_store_provider, extra_params
):
    # 임베딩 모델 생성
    if embedding_provider == "dummy":
        embedding_model = FakeEmbeddingModel(dim=1536)
    elif embedding_provider == "openai":
        embedding_model = EmbeddingModelFactory.create_embedding_model(
            provider="openai", model_name="text-embedding-ada-002"
        )
    else:
        pytest.skip("Unsupported embedding provider")

    # 벡터 스토어 생성
    if vector_store_provider == "dummy":
        vector_store = DummyVectorStore(embedding_model)
    elif vector_store_provider in ["faiss", "annoy"]:
        vector_store = VectorStoreFactory.create_vector_store(
            provider=vector_store_provider,
            embedding_model=embedding_model,
            **extra_params
        )
    else:
        pytest.skip("Unsupported vector store provider")

    run_integration_test(embedding_model, vector_store)
