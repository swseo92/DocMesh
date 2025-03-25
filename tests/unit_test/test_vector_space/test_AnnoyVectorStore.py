from langchain.schema import Document as LC_Document
from docmesh.vector_store.AnnoyVectorStore import AnnoyVectorStore
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


def test_annoy_vector_store_search():
    embedding_model = DummyEmbeddingModel()
    # index_path가 None이면 메모리 내에서 동작
    store = AnnoyVectorStore(embedding_model, n_trees=5, index_path=None)
    docs = create_test_documents()
    store.add_documents(docs)

    results = store.search("Test document", k=2)
    assert len(results) > 0, "AnnoyVectorStore 검색 결과가 없습니다."
    for doc in results:
        assert "source" in doc.metadata, "검색된 문서에 'source' 메타데이터가 없습니다."
