import os
import pytest
from dotenv import load_dotenv

# embedding.py 모듈에서 필요한 클래스들을 임포트합니다.
from docmesh.embedding import (
    Document,
    LangchainOpenAIEmbeddingModel,
    LangchainFAISSVectorStore,
    EmbeddingStoreManager,
)

# .env 파일에서 환경변수 로드
load_dotenv()

# OpenAI API 키가 없는 경우, 전체 모듈 테스트를 스킵합니다.
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    pytest.skip(
        "OPENAI_API_KEY가 설정되어 있지 않아 OpenAI 통합 테스트를 건너뜁니다.", allow_module_level=True
    )


def test_openai_embedding_model():
    """
    LangchainOpenAIEmbeddingModel이 올바른 임베딩 벡터를 반환하는지 검증합니다.
    """
    model = LangchainOpenAIEmbeddingModel("text-embedding-3-small")
    embedding = model.get_embedding("hello world")
    # 반환된 임베딩이 리스트이며, 모델의 vector_dim과 동일한 길이를 가져야 합니다.
    assert isinstance(embedding, list)
    assert len(embedding) == model.vector_dim


def test_openai_pipeline_integration():
    """
    LangChain 기반의 OpenAI 임베딩 모델과 FAISS 벡터스토어를 결합한 전체 파이프라인이
    문서를 임베딩 및 저장하고, 검색 기능을 통해 유사 문서를 올바르게 반환하는지 검증합니다.
    """
    # OpenAI 임베딩 모델과 FAISS 벡터스토어, 그리고 EmbeddingStoreManager 초기화
    model = LangchainOpenAIEmbeddingModel()
    vector_store = LangchainFAISSVectorStore(model)
    manager = EmbeddingStoreManager(model, vector_store)

    # 테스트용 Document 객체들 생성
    docs = [
        Document(
            """This is a test document.
            It contains information about testing OpenAI embeddings.""",
            {"source": "https://example.com"},
        ),
        Document(
            """Another test document with more details
            and different content for integration testing.""",
            {"source": "https://example.org"},
        ),
    ]

    # 문서 임베딩 및 저장
    manager.embed_and_store(docs)

    # 간단한 쿼리 실행: 결과가 하나 이상 반환되는지 검증합니다.
    query = "test document"
    results = manager.search_chunks(query, k=2)

    assert len(results) > 0, "검색 결과가 없습니다."
    for res in results:
        # 반환된 결과는 LangChain Document 객체와 유사하게 page_content와 metadata 속성을 포함해야 합니다.
        assert hasattr(res, "page_content"), "결과 객체에 page_content 속성이 없습니다."
        assert hasattr(res, "metadata"), "결과 객체에 metadata 속성이 없습니다."
        # 예시로 source 메타데이터가 존재하는지 확인합니다.
        assert "source" in res.metadata, "메타데이터에 source 정보가 없습니다."
