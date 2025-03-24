from fastapi import FastAPI
from models import CrawlRequest, QuestionRequest
from qa_bot import LLMServiceFactory, QAService
from embedding import (
    LangchainOpenAIEmbeddingModel,
    LangchainFAISSVectorStore,
    Document,
    EmbeddingStoreManager,
)
from config import Config

app = FastAPI()

# 시스템 초기화: 임베딩 모델, 벡터 스토어, EmbeddingStoreManager 및 QAService 생성
embedding_model = LangchainOpenAIEmbeddingModel()

# VECTOR_STORE_TYPE 설정에 따라 벡터 스토어 인스턴스를 생성 (현재는 FAISS만 구현)
if Config.VECTOR_STORE_TYPE == "faiss":
    vector_store = LangchainFAISSVectorStore(embedding_model)
else:
    raise ValueError(f"Unsupported VECTOR_STORE_TYPE: {Config.VECTOR_STORE_TYPE}")

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

# LLM 서비스 생성 (config.py의 설정값을 사용)
llm_service = LLMServiceFactory.create_llm_service(
    provider=Config.LLM_PROVIDER,
    model=Config.LLM_MODEL,
    temperature=Config.LLM_TEMPERATURE,
)
qa_service = QAService(llm_service, embedding_manager)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/crawl")
def crawl_endpoint(payload: CrawlRequest):
    # TODO: 크롤러 함수를 호출하여 문서를 수집, 임베딩 후 결과를 반환하도록 구현합니다.
    return {"status": "crawl completed"}


@app.post("/ask")
def ask_endpoint(question: QuestionRequest):
    # 사용자 질문을 받아 QAService를 통해 답변을 생성하고 반환합니다.
    answer = qa_service.answer_question(question.question)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
