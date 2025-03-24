import os
import openai
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.schema import Document as LC_Document
from dotenv import load_dotenv
import abc

# .env 파일에서 OPENAI_API_KEY를 로드합니다.
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class Document:
    """
    Document 클래스는 임베딩 및 검색을 위한 텍스트 청크와 메타데이터를 캡슐화합니다.
    """

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

    def to_langchain_document(self) -> LC_Document:
        return LC_Document(page_content=self.page_content, metadata=self.metadata)


# --- Embedding Model 추상 클래스 및 구현 ---
class BaseEmbeddingModel(abc.ABC):
    @abc.abstractmethod
    def get_embedding(self, text: str) -> list:
        """주어진 텍스트의 임베딩 벡터를 반환합니다."""
        pass

    @property
    @abc.abstractmethod
    def vector_dim(self) -> int:
        """임베딩 벡터의 차원을 반환합니다."""
        pass


class LangchainOpenAIEmbeddingModel(BaseEmbeddingModel):
    """
    LangchainOpenAIEmbeddingModel은 LangChain의 OpenAIEmbeddings를 사용하여 텍스트를 임베딩합니다.
    model_name 인자를 통해 다양한 OpenAI 임베딩 모델(e.g., text-embedding-ada-002, text-embedding-babbage-001 등)을 선택할 수 있습니다.
    """

    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self._model_name = model_name
        self.embeddings = OpenAIEmbeddings(model=model_name)
        # "hello world" 임베딩을 통해 임베딩 차원을 결정합니다.
        self._vector_dim = len(self.embeddings.embed_query("hello world"))

    def get_embedding(self, text: str) -> list:
        return self.embeddings.embed_query(text)

    @property
    def vector_dim(self) -> int:
        return self._vector_dim


# --- Embedding Model Factory ---
class EmbeddingModelFactory:
    @staticmethod
    def create_embedding_model(
        provider: str = "openai", model_name: str = "text-embedding-ada-002"
    ) -> BaseEmbeddingModel:
        """
        provider와 model_name에 따라 적절한 임베딩 모델 인스턴스를 생성합니다.
        현재는 provider가 "openai"일 경우 LangchainOpenAIEmbeddingModel을 반환하며,
        향후 다른 공급자를 위한 구현체를 추가할 수 있습니다.
        """
        if provider == "openai":
            return LangchainOpenAIEmbeddingModel(model_name=model_name)
        else:
            raise ValueError(f"Unsupported embedding model provider: {provider}")


# --- Vector Store 구현 ---
class LangchainFAISSVectorStore:
    """
    LangchainFAISSVectorStore는 FAISS 인덱스, InMemoryDocstore, 빈 index_to_docstore_id 딕셔너리를 이용해
    벡터 스토어를 초기화하고, 임베딩 모델의 get_embedding()을 사용해 문서 임베딩 및 유사도 검색을 수행합니다.
    """

    def __init__(self, embedding_model: BaseEmbeddingModel):
        self.embedding_model = embedding_model
        # 임베딩 차원에 맞게 FAISS 인덱스 생성
        index = faiss.IndexFlatL2(self.embedding_model.vector_dim)
        self.vectorstore = FAISS(
            embedding_function=lambda text: self.embedding_model.get_embedding(text),
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def add_documents(self, documents: list) -> None:
        lc_documents = [doc.to_langchain_document() for doc in documents]
        self.vectorstore.add_documents(lc_documents)

    def search(self, query: str, k: int = 3) -> list:
        return self.vectorstore.similarity_search(query, k=k)


# --- Vector Store Factory ---
class VectorStoreFactory:
    @staticmethod
    def create_vector_store(
        provider: str = "faiss", embedding_model: BaseEmbeddingModel = None
    ):
        """
        provider와 embedding_model에 따라 적절한 벡터 스토어 인스턴스를 생성합니다.
        현재는 provider가 "faiss"일 경우 LangchainFAISSVectorStore를 반환하며,
        추후 다른 벡터 스토어 구현체(Pinecone, Milvus 등)를 추가할 수 있습니다.
        """
        if provider == "faiss":
            if embedding_model is None:
                raise ValueError(
                    "embedding_model must be provided for FAISS vector store."
                )
            return LangchainFAISSVectorStore(embedding_model)
        else:
            raise ValueError(f"Unsupported vector store provider: {provider}")


# --- EmbeddingStoreManager: 두 객체를 결합하는 Manager ---
class EmbeddingStoreManager:
    """
    EmbeddingStoreManager는 임베딩 모델과 벡터 스토어를 조합하여,
    Document 객체들의 임베딩 생성, 저장 및 유사도 검색 기능을 제공합니다.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: LangchainFAISSVectorStore,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def embed_and_store(self, documents: list) -> None:
        self.vector_store.add_documents(documents)

    def search_chunks(self, query: str, k: int = 3) -> list:
        return self.vector_store.search(query, k=k)


def main():
    # 테스트용 Document 객체들 생성
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

    # EmbeddingModelFactory를 통해 임베딩 모델 생성
    embedding_model = EmbeddingModelFactory.create_embedding_model(
        provider="openai", model_name="text-embedding-ada-002"
    )
    # VectorStoreFactory를 통해 벡터 스토어 생성
    vector_store = VectorStoreFactory.create_vector_store(
        provider="faiss", embedding_model=embedding_model
    )
    manager = EmbeddingStoreManager(embedding_model, vector_store)

    # Document 임베딩 및 저장
    manager.embed_and_store(docs)

    # 사용자 질문에 대해 유사도 검색 실행
    query = "first document content"
    results = manager.search_chunks(query, k=2)

    print("Search results:")
    for res in results:
        print(f"Source: {res.metadata.get('source', '')}")
        print(f"Content: {res.page_content}\n")


if __name__ == "__main__":
    main()
