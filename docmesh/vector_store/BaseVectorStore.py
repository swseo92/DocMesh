import abc

from docmesh.embedding import BaseEmbeddingModel


# --- Vector Store 상위 인터페이스 ---
class BaseVectorStore(abc.ABC):
    @abc.abstractmethod
    def add_documents(self, documents: list) -> None:
        """문서들을 벡터 스토어에 추가합니다."""
        pass

    @abc.abstractmethod
    def search(self, query: str, k: int = 3) -> list:
        """쿼리 텍스트에 대해 상위 k개의 유사 문서를 검색하여 반환합니다."""
        pass


class EmbeddingStoreManager:
    """
    EmbeddingStoreManager는 임베딩 모델과 벡터 스토어를 조합하여,
    Document 객체들의 임베딩 생성, 저장 및 유사도 검색 기능을 제공합니다.
    """

    def __init__(
        self, embedding_model: BaseEmbeddingModel, vector_store: BaseVectorStore
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def embed_and_store(self, documents: list) -> None:
        self.vector_store.add_documents(documents)

    def search_chunks(self, query: str, k: int = 3) -> list:
        return self.vector_store.search(query, k=k)
