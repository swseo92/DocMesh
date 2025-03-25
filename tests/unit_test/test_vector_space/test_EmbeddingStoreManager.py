from docmesh.vector_store.BaseVectorStore import EmbeddingStoreManager
from docmesh.vector_store.FAISSVectorStore import LangchainFAISSVectorStore
from langchain.schema import Document as LC_Document
from tests.utils.dummy_embedding import DummyEmbeddingModel


def create_test_documents():
    doc1 = LC_Document(page_content="Test document one", metadata={"source": "doc1"})
    doc2 = LC_Document(
        page_content="Another test document", metadata={"source": "doc2"}
    )
    doc3 = LC_Document(
        page_content="Yet another document for testing", metadata={"source": "doc3"}
    )
    return [doc1, doc2, doc3]


def test_embedding_store_manager():
    embedding_model = DummyEmbeddingModel()
    store = LangchainFAISSVectorStore(embedding_model)
    manager = EmbeddingStoreManager(embedding_model, store)
    docs = create_test_documents()
    manager.embed_and_store(docs)

    results = manager.search_chunks("Test document", k=2)
    assert len(results) > 0, "EmbeddingStoreManager 검색 결과가 없습니다."
