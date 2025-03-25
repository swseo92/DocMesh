import pytest
from docmesh.embedding.EmbeddingModelFactory import EmbeddingModelFactory
from docmesh.embedding.LangchainOpenAIEmbeddingModel import (
    LangchainOpenAIEmbeddingModel,
)


def test_create_openai_embedding_model():
    model = EmbeddingModelFactory.create_embedding_model(
        provider="openai", model_name="text-embedding-ada-002"
    )
    assert isinstance(model, LangchainOpenAIEmbeddingModel)


def test_create_invalid_embedding_model():
    with pytest.raises(ValueError):
        _ = EmbeddingModelFactory.create_embedding_model(provider="invalid")
