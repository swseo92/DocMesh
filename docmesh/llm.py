import abc
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv


# LLM 서비스 추상 클래스: 모든 LLM 모델은 이 인터페이스를 구현해야 합니다.
class BaseLLMService(abc.ABC):
    @abc.abstractmethod
    def generate_answer(self, prompt: str) -> str:
        """프롬프트를 받아 LLM을 통해 답변을 생성합니다."""
        pass


# LangChain 기반 LLM 서비스 구현
class LangchainLLMService(BaseLLMService):
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.0):
        load_dotenv()

        self.llm = ChatOpenAI(model_name=model, temperature=temperature)

    def generate_answer(self, prompt: str) -> str:
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=prompt),
        ]
        response = self.llm(messages)
        return response.content.strip()


# 팩토리 클래스: 설정에 따라 적절한 LLM 서비스를 생성할 수 있습니다.
class LLMServiceFactory:
    @staticmethod
    def create_llm_service(provider: str = "langchain", **kwargs) -> BaseLLMService:
        """
        provider 인자에 따라 적절한 LLM 서비스를 생성합니다.
        현재는 'langchain'이 기본 옵션이며, 향후 다른 모델을 추가할 수 있습니다.
        """
        if provider == "langchain":
            return LangchainLLMService(**kwargs)
        elif provider == "other":
            # 예시: 향후 다른 LLM 모델을 위한 구현체를 추가합니다.
            raise NotImplementedError(
                "Other LLM service provider is not implemented yet."
            )
        else:
            raise ValueError(f"Unsupported LLM service provider: {provider}")
