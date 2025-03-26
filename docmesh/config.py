class Config:
    # tiktoken 관련 설정
    TIKTOKEN_ENCODING = "gpt2"

    # LLM 관련 설정
    # 사용할 LLM provider 및 모델 설정 (예: "langchain" 또는 추후 다른 provider)
    LLM_PROVIDER = "langchain"
    LLM_MODEL = "gpt-3.5-turbo"
    LLM_TEMPERATURE = 0.0

    # 벡터 스토어 관련 설정
    # 현재 "faiss"를 지원하며, 추후 "pinecone", "milvus" 등으로 확장 가능
    VECTOR_STORE_TYPE = "faiss"

    # 임베딩 모델 관련 설정
    # 사용할 임베딩 모델 provider 및 모델 이름 (예: "openai" 및 "text-embedding-ada-002")
    EMBEDDING_MODEL_PROVIDER = "openai"
    EMBEDDING_MODEL_NAME = "text-embedding-ada-002"
