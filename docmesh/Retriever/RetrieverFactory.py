from docmesh.vector_store.VectorStoreFactory import VectorStoreFactory
from docmesh.Retriever.VectorStoreRetriever import VectorStoreRetriever
from docmesh.embedding.BaseEmbeddingModel import BaseEmbeddingModel


class RetrieverFactory:
    @staticmethod
    def create_retriever(
        provider: str = "faiss",
        embedding_model: BaseEmbeddingModel = None,
        k: int = 3,
        **kwargs
    ) -> VectorStoreRetriever:
        """
        1) VectorStoreFactory를 통해 해당 provider에 맞는 벡터스토어를 생성
        2) 만들어진 벡터스토어를 VectorStoreRetriever에 넣어 반환
        :param provider: "faiss" / "annoy" 등
        :param embedding_model: 임베딩 모델 (faiss/annoy 등에서 사용)
        :param k: 검색 결과 상위 개수
        :param kwargs: 벡터스토어 생성시 필요한 추가 파라미터 (path, index_path 등)
        """
        vector_store = VectorStoreFactory.create_vector_store(
            provider=provider, embedding_model=embedding_model, **kwargs
        )
        return VectorStoreRetriever(vector_store, k=k)
