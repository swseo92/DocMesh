import pytest
from docmesh.embedding.BaseEmbeddingModel import BaseEmbeddingModel
from docmesh.vector_store.VectorStoreFactory import VectorStoreFactory
from docmesh.vector_store.FAISSVectorStore import LangchainFAISSVectorStore
from docmesh.vector_store.AnnoyVectorStore import AnnoyVectorStore


class DummyEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        self._vector_dim = 4

    def get_embedding(self, text: str) -> list:
        length = float(len(text))
        return [length, length + 1, length + 2, length + 3]

    @property
    def vector_dim(self) -> int:
        return self._vector_dim


def test_vector_store_factory_faiss():
    embedding_model = DummyEmbeddingModel()
    store = VectorStoreFactory.create_vector_store(
        provider="faiss", embedding_model=embedding_model
    )
    assert isinstance(
        store, LangchainFAISSVectorStore
    ), "생성된 벡터 스토어가 LangchainFAISSVectorStore가 아닙니다."


def test_vector_store_factory_annoy():
    embedding_model = DummyEmbeddingModel()
    store = VectorStoreFactory.create_vector_store(
        provider="annoy", embedding_model=embedding_model, n_trees=5, index_path=None
    )

    assert isinstance(store, AnnoyVectorStore), "생성된 벡터 스토어가 AnnoyVectorStore가 아닙니다."


def test_vector_store_factory_invalid():
    embedding_model = DummyEmbeddingModel()
    with pytest.raises(ValueError):
        VectorStoreFactory.create_vector_store(
            provider="unsupported", embedding_model=embedding_model
        )
