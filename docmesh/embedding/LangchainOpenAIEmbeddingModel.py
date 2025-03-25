from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from docmesh.embedding.BaseEmbeddingModel import BaseEmbeddingModel


class LangchainOpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    LangchainOpenAIEmbeddingModel은 LangChain의 OpenAIEmbeddings를 사용하여 텍스트를 임베딩합니다.
    model_name 인자를 통해 다양한 OpenAI 임베딩 모델(e.g., text-embedding-ada-002, text-embedding-babbage-001 등)을 선택할 수 있습니다.
    """

    def __init__(self, model_name: str = "text-embedding-ada-002"):
        load_dotenv()

        self._model_name = model_name
        self.embeddings = OpenAIEmbeddings(model=model_name)
        # "hello world" 임베딩을 통해 임베딩 차원을 결정합니다.
        self._vector_dim = len(self.embeddings.embed_query("hello world"))

    def get_embedding(self, text: str) -> list:
        return self.embeddings.embed_query(text)

    @property
    def vector_dim(self) -> int:
        return self._vector_dim
