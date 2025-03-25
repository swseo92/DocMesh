import pytest
from docmesh.llm.LLMFactory import LLMFactory
from docmesh.llm.LangchainOpenAILLM import LangchainOpenAILLM


def test_llm_factory_langchain():
    llm = LLMFactory.create_llm_service(
        provider="langchain", model="gpt-3.5-turbo", temperature=0.0
    )
    assert isinstance(llm, LangchainOpenAILLM)


def test_llm_factory_invalid_provider():
    with pytest.raises(ValueError):
        _ = LLMFactory.create_llm_service(
            provider="invalid", model="gpt-3.5-turbo", temperature=0.0
        )


def test_llm_factory_other_provider():
    with pytest.raises(NotImplementedError):
        _ = LLMFactory.create_llm_service(
            provider="other", model="gpt-3.5-turbo", temperature=0.0
        )
