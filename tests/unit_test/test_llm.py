import pytest
from docmesh.llm import LangchainLLMService, LLMServiceFactory


# Dummy 함수: LLM 호출 시 고정된 응답 객체를 반환합니다.
def dummy_llm_call(messages):
    class DummyResponse:
        def __init__(self, content):
            self.content = content

    # 예시로 "dummy answer"를 반환합니다.
    return DummyResponse("dummy answer")


def test_langchain_llm_service(monkeypatch):
    """
    LangchainLLMService.generate_answer()가 올바른 값을 반환하는지 테스트합니다.
    실제 API 호출을 피하기 위해 monkeypatch로 llm 속성을 대체합니다.
    """
    # LangchainLLMService 인스턴스 생성
    service = LangchainLLMService(model="gpt-3.5-turbo", temperature=0.0)
    # 서비스의 llm 속성을 dummy_llm_call로 교체합니다.
    monkeypatch.setattr(service, "llm", dummy_llm_call)

    prompt = "Test prompt"
    answer = service.generate_answer(prompt)

    # dummy_llm_call에 의해 "dummy answer"가 반환되어야 합니다.
    assert answer == "dummy answer"


def test_llm_service_factory_langchain():
    """
    LLMServiceFactory가 provider가 'langchain'일 때 올바른 LangchainLLMService 인스턴스를 생성하는지 테스트합니다.
    """
    service = LLMServiceFactory.create_llm_service(
        provider="langchain", model="gpt-3.5-turbo", temperature=0.0
    )
    assert isinstance(service, LangchainLLMService)


def test_llm_service_factory_other():
    """
    provider가 'other'인 경우 NotImplementedError를 발생하는지 테스트합니다.
    """
    with pytest.raises(NotImplementedError):
        LLMServiceFactory.create_llm_service(
            provider="other", model="gpt-3.5-turbo", temperature=0.0
        )


def test_llm_service_factory_unsupported():
    """
    지원되지 않는 provider를 전달할 경우 ValueError가 발생하는지 테스트합니다.
    """
    with pytest.raises(ValueError):
        LLMServiceFactory.create_llm_service(
            provider="unsupported", model="gpt-3.5-turbo", temperature=0.0
        )


if __name__ == "__main__":
    pytest.main()
