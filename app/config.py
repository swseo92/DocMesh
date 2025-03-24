# config.py


class Config:
    # LLM 관련 설정
    LLM_PROVIDER = "langchain"  # 추후 다른 LLM provider를 추가할 수 있습니다.
    LLM_MODEL = "gpt-3.5-turbo"  # 사용할 LLM 모델 이름
    LLM_TEMPERATURE = 0.0  # LLM 생성 온도 (0.0 ~ 1.0)

    # 벡터 스토어 관련 설정
    VECTOR_STORE_TYPE = "faiss"  # 현재 지원: "faiss", 추후 "pinecone", "milvus" 등 확장 가능

    # 임베딩 모델 관련 설정
    # 현재는 LangchainOpenAIEmbeddingModel을 사용하지만, 추후 다른 임베딩 모델로 확장할 수 있습니다.
    EMBEDDING_MODEL = "langchain_openai"
