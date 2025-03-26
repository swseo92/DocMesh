import pytest
from docmesh.embedding.EmbeddingModelFactory import EmbeddingModelFactory
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv


def test_create_openai_embedding_model():
    load_dotenv()

    model = EmbeddingModelFactory.create_embedding_model(
        provider="openai", model_name="text-embedding-ada-002"
    )
    assert isinstance(model, OpenAIEmbeddings)


def test_create_invalid_embedding_model():
    with pytest.raises(ValueError):
        _ = EmbeddingModelFactory.create_embedding_model(provider="invalid")
