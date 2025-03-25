import abc


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
