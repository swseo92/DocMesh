import pytest
from docmesh.llm.LLMFactory import LLMFactory
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv


def test_llm_factory_langchain():
    load_dotenv()

    llm = LLMFactory.create_llm_service(
        provider="langchain", model="gpt-3.5-turbo", temperature=0.0
    )
    assert isinstance(llm, ChatOpenAI)


def test_llm_factory_invalid_provider():
    with pytest.raises(ValueError):
        _ = LLMFactory.create_llm_service(
            provider="invalid", model="gpt-3.5-turbo", temperature=0.0
        )
