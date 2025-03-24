import pytest
from docmesh.embedding import (
    LangchainOpenAIEmbeddingModel,
    EmbeddingModelFactory,
    VectorStoreFactory,
    EmbeddingStoreManager,
    Document,
    main,
)


def test_main():
    main()


# 더미 embed_query 함수: 입력 텍스트와 상관없이 고정된 벡터를 반환
def dummy_embed_query(text):
    # 예: 768차원 벡터 (실제 차원은 상관없으므로 고정값 사용)
    return [0.1] * 768


@pytest.fixture(autouse=True)
def override_embed_query(monkeypatch):
    def dummy_init(self, model_name: str = "text-embedding-ada-002"):
        self._model_name = model_name
        # 더미 embed_query 함수를 직접 할당합니다.
        self.embeddings = type(
            "DummyEmbeddings", (), {"embed_query": staticmethod(dummy_embed_query)}
        )
        self._vector_dim = len(dummy_embed_query("hello world"))

    monkeypatch.setattr(LangchainOpenAIEmbeddingModel, "__init__", dummy_init)


def test_embedding_model_factory():
    # EmbeddingModelFactory를 이용해 임베딩 모델 생성
    embedding_model = EmbeddingModelFactory.create_embedding_model(
        provider="openai", model_name="text-embedding-ada-002"
    )
    assert isinstance(embedding_model, LangchainOpenAIEmbeddingModel)
    # dummy_embed_query를 사용하므로 vector_dim은 768로 예상됩니다.
    assert embedding_model.vector_dim == 768
    # get_embedding이 dummy 값을 반환하는지 확인
    vec = embedding_model.get_embedding("Test text")
    assert isinstance(vec, list)
    assert len(vec) == 768
    assert all(v == 0.1 for v in vec)


def test_vector_store_factory():
    # 임베딩 모델 생성
    embedding_model = EmbeddingModelFactory.create_embedding_model(
        provider="openai", model_name="text-embedding-ada-002"
    )
    # VectorStoreFactory를 이용해 벡터 스토어 생성
    vector_store = VectorStoreFactory.create_vector_store(
        provider="faiss", embedding_model=embedding_model
    )
    from docmesh.embedding import LangchainFAISSVectorStore

    assert isinstance(vector_store, LangchainFAISSVectorStore)


def test_embedding_store_manager_embed_and_search():
    # 임베딩 모델 생성 (더미 embed_query 사용)
    embedding_model = EmbeddingModelFactory.create_embedding_model(
        provider="openai", model_name="text-embedding-ada-002"
    )
    # VectorStoreFactory를 이용해 벡터 스토어 생성
    vector_store = VectorStoreFactory.create_vector_store(
        provider="faiss", embedding_model=embedding_model
    )
    manager = EmbeddingStoreManager(embedding_model, vector_store)

    # 테스트용 Document 객체 생성
    doc1 = Document(
        "This is the content of the first document.", {"source": "https://example.com"}
    )
    doc2 = Document(
        "The second document contains more detailed testing info.",
        {"source": "https://example.org"},
    )
    documents = [doc1, doc2]

    # 문서 임베딩 및 저장
    manager.embed_and_store(documents)

    # 어떤 쿼리든 저장한 문서가 검색 결과로 반환되는지 확인
    results = manager.search_chunks("first document", k=2)
    assert isinstance(results, list)
    assert len(results) > 0
    # 각 결과는 LC_Document 형태로 metadata에 source가 포함되어야 합니다.
    for res in results:
        assert hasattr(res, "page_content")
        assert hasattr(res, "metadata")
        assert "source" in res.metadata


if __name__ == "__main__":
    pytest.main()
