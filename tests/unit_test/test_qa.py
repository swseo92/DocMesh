import pytest
from docmesh.format import Document
from docmesh.qa_bot import QAService
from docmesh.llm import LLMFactory
from docmesh.embedding import LangchainOpenAIEmbeddingModel
from docmesh.vector_store import LangchainFAISSVectorStore, EmbeddingStoreManager


@pytest.fixture
def qa_service():
    # 임베딩 모델, 벡터 스토어, EmbeddingStoreManager 초기화
    embedding_model = LangchainOpenAIEmbeddingModel()
    vector_store = LangchainFAISSVectorStore(embedding_model)
    embedding_manager = EmbeddingStoreManager(embedding_model, vector_store)

    # FAISS 벡터 스토어의 index_to_docstore_id를 통해 문서가 이미 저장되어 있는지 확인합니다.
    if not vector_store.vectorstore.index_to_docstore_id:
        docs = [
            Document(
                "This is the content of the first document.",
                {"source": "https://example.com"},
            ),
            Document(
                "The second document contains more detailed information for testing.",
                {"source": "https://example.org"},
            ),
        ]
        embedding_manager.embed_and_store(docs)

    # 팩토리를 이용해 LangchainLLMService를 생성합니다.
    llm_service = LLMFactory.create_llm_service(
        provider="langchain", model="gpt-3.5-turbo", temperature=0.0
    )

    return QAService(llm_service, embedding_manager)


def test_answer_question_english(qa_service):
    question = "What is the content of the first document?"
    answer = qa_service.answer_question(question)

    # 답변이 비어있지 않고 문자열임을 확인합니다.
    assert isinstance(answer, str)
    assert len(answer) > 0
    # LLM이 프롬프트에 따라 "Sources:"를 포함하도록 요청했으므로, 결과에 해당 키워드가 포함되어 있는지 확인합니다.
    assert "Sources:" in answer


def test_answer_question_korean(qa_service):
    question = "첫 번째 문서의 내용은 무엇인가요?"
    answer = qa_service.answer_question(question)

    assert isinstance(answer, str)
    assert len(answer) > 0
    # 동일하게 참조 URL이 포함되었는지 확인합니다.
    assert "Sources:" in answer


if __name__ == "__main__":
    pytest.main()
