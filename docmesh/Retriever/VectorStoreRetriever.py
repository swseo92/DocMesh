# vector_store_retriever.py

from langchain.schema import Document, BaseRetriever
from typing import List
from langchain.vectorstores.base import VectorStore  # 또는 직접 만든 BaseVectorStore


class VectorStoreRetriever(BaseRetriever):
    """
    - VectorStore 인스턴스를 받아,
      query 시 vectorstore.similarity_search(query, k)로 문서를 가져옴
    """

    def __init__(self, vector_store: VectorStore, k: int = 3):
        self.vector_store = vector_store
        self.k = k

    def get_relevant_documents(self, query: str) -> List[Document]:
        """동기식 문서 검색."""
        return self.vector_store.similarity_search(query, k=self.k)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """비동기 버전 (필요할 경우 구현)."""
        raise NotImplementedError("Async retrieval not implemented.")
