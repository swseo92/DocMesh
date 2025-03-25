import os
import pytest
from dotenv import load_dotenv
from docmesh.embedding.LangchainOpenAIEmbeddingModel import (
    LangchainOpenAIEmbeddingModel,
)


load_dotenv()


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY is not set"
)
def test_langchain_openai_embedding_model():
    model = LangchainOpenAIEmbeddingModel(model_name="text-embedding-ada-002")
    vec_dim = model.vector_dim
    assert isinstance(vec_dim, int)
    assert vec_dim > 0
    embedding = model.get_embedding("test")
    assert isinstance(embedding, list)
    assert len(embedding) == vec_dim
