from langchain.schema import Document as LC_Document
import abc


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
