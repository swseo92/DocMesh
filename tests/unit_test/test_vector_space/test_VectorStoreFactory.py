import pytest
from docmesh.vector_store.VectorStoreFactory import VectorStoreFactory
from docmesh.vector_store.FAISSVectorStore import LangchainFAISSVectorStore
from tests.mocks.dummy_embedding import DummyEmbeddingModel


def test_vector_store_factory_faiss():
    embedding_model = DummyEmbeddingModel()
    store = VectorStoreFactory.create_vector_store(
        provider="faiss", embedding_model=embedding_model
    )
    assert isinstance(
        store, LangchainFAISSVectorStore
    ), "생성된 벡터 스토어가 LangchainFAISSVectorStore가 아닙니다."


def test_vector_store_factory_invalid():
    embedding_model = DummyEmbeddingModel()
    with pytest.raises(ValueError):
        VectorStoreFactory.create_vector_store(
            provider="unsupported", embedding_model=embedding_model
        )
