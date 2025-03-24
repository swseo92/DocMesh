from langchain.schema import Document as LC_Document

from app.embedding import (
    Document,
    LangchainFAISSVectorStore,
    EmbeddingStoreManager,
)


# --- FakeEmbeddingModel 수정 ---
class FakeEmbeddingModel:
    """
    FakeEmbeddingModel은 실제 OpenAI API 호출 대신,
    텍스트 길이에 따라 고정 차원(1536)의 임베딩 벡터를 반환합니다.
    이 클래스는 LangchainOpenAIEmbeddingModel과 동일한 인터페이스
    (get_embedding, embed_query, vector_dim, embeddings)를 구현하며,
    __call__ 메서드를 추가하여 인스턴스 자체가 callable하도록 합니다.
    """

    def __init__(self, dim: int = 1536):
        self.dim = dim
        self.vector_dim = dim  # FAISSVectorStore가 요구하는 속성
        self.embeddings = self  # FAISSVectorStore 생성 시 사용

    def __call__(self, text: str) -> list:
        return self.get_embedding(text)

    def embed_query(self, text: str) -> list:
        # 텍스트 길이를 100으로 나눈 값을 모든 요소로 채워 반환합니다.
        value = len(text) / 100.0
        return [value] * self.dim

    def get_embedding(self, text: str) -> list:
        return self.embed_query(text)


# --- Document 클래스 테스트 ---
def test_document_to_langchain_document():
    doc = Document("Test content", {"source": "https://example.com"})
    lc_doc = doc.to_langchain_document()
    assert isinstance(lc_doc, LC_Document)
    assert lc_doc.page_content == "Test content"
    assert lc_doc.metadata == {"source": "https://example.com"}


# --- FakeEmbeddingModel 테스트 ---
def test_fake_embedding_model():
    fake_model = FakeEmbeddingModel(dim=1536)
    text = "Hello world!"
    embedding = fake_model.get_embedding(text)
    assert len(embedding) == 1536
    expected_value = len(text) / 100.0
    for val in embedding:
        assert abs(val - expected_value) < 1e-6


# --- FAISSVectorStore 테스트 ---
def test_faiss_vector_store():
    fake_model = FakeEmbeddingModel(dim=1536)
    vector_store = LangchainFAISSVectorStore(embedding_model=fake_model)
    # 생성할 Document 객체
    doc = Document("Test content for FAISS", {"source": "https://example.com"})
    vector_store.add_documents([doc])
    # 검색: 같은 텍스트로 검색하면 저장한 문서가 반환되어야 함
    results = vector_store.search("Test content for FAISS", k=1)
    assert len(results) == 1
    res = results[0]
    assert hasattr(res, "metadata")
    assert hasattr(res, "page_content")
    assert res.metadata.get("source") == "https://example.com"


# --- EmbeddingStoreManager 테스트 ---
def test_embedding_store_manager():
    fake_model = FakeEmbeddingModel(dim=1536)
    vector_store = LangchainFAISSVectorStore(embedding_model=fake_model)
    manager = EmbeddingStoreManager(fake_model, vector_store)

    docs = [
        Document("Content of document one", {"source": "https://example.com"}),
        Document("Content of document two", {"source": "https://example.org"}),
        Document(
            "Another document from example.com", {"source": "https://example.com"}
        ),
    ]
    manager.embed_and_store(docs)
    results = manager.search_chunks("document", k=2)
    assert len(results) == 2
    for res in results:
        assert hasattr(res, "metadata")
        assert hasattr(res, "page_content")


# --- 전체 파이프라인 통합 테스트 ---
def test_embedding_pipeline_integration():
    fake_model = FakeEmbeddingModel(dim=1536)
    vector_store = LangchainFAISSVectorStore(embedding_model=fake_model)
    manager = EmbeddingStoreManager(fake_model, vector_store)

    docs = [
        Document("Document one content. " * 10, {"source": "https://example.com"}),
        Document("Document two content. " * 15, {"source": "https://example.org"}),
        Document("Document three content. " * 8, {"source": "https://example.com"}),
    ]
    manager.embed_and_store(docs)
    results = manager.search_chunks("content", k=3)
    assert len(results) == 3
    for res in results:
        assert hasattr(res, "metadata")
        assert hasattr(res, "page_content")
