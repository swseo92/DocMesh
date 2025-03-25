from langchain.schema import Document as LC_Document
from docmesh.vector_store.FAISSVectorStore import (
    LangchainFAISSVectorStore,
)
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


def test_langchain_faiss_vector_store():
    embedding_model = DummyEmbeddingModel()
    store = LangchainFAISSVectorStore(embedding_model)
    docs = create_test_documents()
    store.add_documents(docs)

    results = store.search("Test document", k=2)
    assert len(results) > 0, "LangchainFAISSVectorStore 검색 결과가 없습니다."
    for doc in results:
        assert "source" in doc.metadata, "검색된 문서에 'source' 메타데이터가 없습니다."


# Todo 아직 완성안됨
# def test_disk_based_faiss_vector_store():
#     embedding_model = DummyEmbeddingModel()
#     with tempfile.TemporaryDirectory() as tmpdir:
#         index_file = os.path.join(tmpdir, "faiss_index.index")
#         docstore_file = os.path.join(tmpdir, "docstore.json")
#
#         store = DiskBasedLangchainFAISSVectorStore(embedding_model, index_file, docstore_file)
#         docs = create_test_documents()
#         store.add_documents(docs)
#
#         # FAISS 인덱스 파일과 docstore 파일 생성 확인
#         assert os.path.exists(index_file), "FAISS 인덱스 파일이 생성되지 않았습니다."
#         assert os.path.exists(docstore_file), "docstore 파일이 생성되지 않았습니다."
#
#         results = store.search("Test document", k=2)
#         assert len(results) > 0, "DiskBasedLangchainFAISSVectorStore 검색 결과가 없습니다."
#
#         # 재시작(재로드) 테스트
#         new_store = DiskBasedLangchainFAISSVectorStore(embedding_model, index_file, docstore_file)
#         new_results = new_store.search("Test document", k=2)
#         assert len(new_results) > 0, "재로드 후 검색 결과가 없습니다."
