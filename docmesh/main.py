from fastapi import FastAPI
from docmesh.crawler import WebCrawler
from docmesh.models import CrawlRequest, QuestionRequest
from docmesh.qa_bot import QAService
from docmesh.llm import LLMFactory
from docmesh.embedding import EmbeddingModelFactory
from docmesh.vector_store import VectorStoreFactory, EmbeddingStoreManager
from docmesh.config import Config

app = FastAPI()

# 1. 임베딩 모델 생성 (팩토리 사용)
embedding_model = EmbeddingModelFactory.create_embedding_model(
    provider=Config.EMBEDDING_MODEL_PROVIDER,
    model_name=Config.EMBEDDING_MODEL_NAME,
)

# 2. 벡터 스토어 생성 (팩토리 사용)
if Config.VECTOR_STORE_TYPE == "faiss":
    vector_store = VectorStoreFactory.create_vector_store(
        provider=Config.VECTOR_STORE_TYPE,
        embedding_model=embedding_model,
    )
else:
    raise ValueError(f"Unsupported VECTOR_STORE_TYPE: {Config.VECTOR_STORE_TYPE}")

# 3. EmbeddingStoreManager 생성
embedding_manager = EmbeddingStoreManager(embedding_model, vector_store)

# 4. LLM 서비스 생성 (팩토리 사용)
llm_service = LLMFactory.create_llm_service(
    provider=Config.LLM_PROVIDER,
    model=Config.LLM_MODEL,
    temperature=Config.LLM_TEMPERATURE,
)

# 5. QAService 생성
qa_service = QAService(llm_service, embedding_manager)


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/crawl")
def crawl_endpoint(payload: CrawlRequest):
    crawler = WebCrawler(payload.url, use_max_pages=True, max_pages=5)
    results = crawler.crawl()
    return {"status": "crawl completed", "results": results}


@app.post("/ask")
def ask_endpoint(question: QuestionRequest):
    answer = qa_service.answer_question(question.question)
    return {"answer": answer}


if __name__ == "__main__":
    import os
    import uvicorn
    from dotenv import load_dotenv

    load_dotenv()

    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", 8000))

    uvicorn.run(app, host=host, port=port)
