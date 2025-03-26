from docmesh.vector_store.FAISSVectorStore import LangchainFAISSVectorStore
from docmesh.vector_store.BaseVectorStore import BaseVectorStore

from docmesh.embedding import BaseEmbeddingModel


class VectorStoreFactory:
    @staticmethod
    def create_vector_store(
        provider: str = "faiss", embedding_model: BaseEmbeddingModel = None, **kwargs
    ) -> BaseVectorStore:
        """
        provider와 embedding_model에 따라 적절한 벡터 스토어 인스턴스를 생성합니다.
        현재는 provider가 "faiss"일 경우 LangchainFAISSVectorStore,
        "annoy"일 경우 AnnoyVectorStore를 반환합니다.
        """
        if provider == "faiss":
            if embedding_model is None:
                raise ValueError(
                    "embedding_model must be provided for FAISS vector store."
                )
            return LangchainFAISSVectorStore(embedding_model, **kwargs)
        else:
            raise ValueError(f"Unsupported vector store provider: {provider}")
