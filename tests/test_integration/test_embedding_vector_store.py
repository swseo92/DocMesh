import pytest
from docmesh.embedding import EmbeddingModelFactory, Document
from docmesh.vector_store import VectorStoreFactory, EmbeddingStoreManager


# 공통 테스트 함수: 주어진 embedding 모델과 vector store 조합에 대해 문서 추가 및 검색 검증
def run_integration_test(embedding_model, vector_store):
    # EmbeddingStoreManager 생성
    manager = EmbeddingStoreManager(embedding_model, vector_store)
    # 테스트용 Document 객체 생성
    doc1 = Document(
        "This is the content of document one.", {"source": "https://doc1.com"}
    )
    doc2 = Document(
        "Content of document two for testing.", {"source": "https://doc2.com"}
    )
    documents = [doc1, doc2]
    # 문서 임베딩 및 저장
    manager.embed_and_store(documents)
    # 검색 실행 (dummy 임베딩이 동일하므로, 쿼리 내용에 관계없이 저장한 문서들이 반환됩니다.)
    results = manager.search_chunks("any query", k=2)
    # 결과 검증: 반환된 객체는 LC_Document 형식이어야 하며, metadata에 source가 포함되어야 합니다.
    assert isinstance(results, list)
    assert len(results) >= 1
    for res in results:
        # LangChain Document 객체는 page_content와 metadata 속성을 가짐
        assert hasattr(res, "page_content")
        assert hasattr(res, "metadata")
        assert "source" in res.metadata


# 파라미터: embedding_provider, model_name, vector_store_provider, vector_store 추가 옵션
@pytest.mark.parametrize(
    "embedding_provider, model_name, vector_store_provider, extra_params",
    [
        ("openai", "text-embedding-ada-002", "faiss", {}),
        ("openai", "text-embedding-ada-002", "annoy", {"n_trees": 5}),
    ],
)
def test_embedding_vector_store_integration(
    embedding_provider, model_name, vector_store_provider, extra_params
):
    # 임베딩 모델 생성
    embedding_model = EmbeddingModelFactory.create_embedding_model(
        provider=embedding_provider, model_name=model_name
    )
    # 벡터 스토어 생성 (추가 옵션 포함)
    vector_store = VectorStoreFactory.create_vector_store(
        provider=vector_store_provider, embedding_model=embedding_model, **extra_params
    )
    # 공통 테스트 실행
    run_integration_test(embedding_model, vector_store)
