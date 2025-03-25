import abc


# LLM 서비스 추상 클래스: 모든 LLM 모델은 이 인터페이스를 구현해야 합니다.
class BaseLLM(abc.ABC):
    @abc.abstractmethod
    def generate_answer(self, prompt: str) -> str:
        """프롬프트를 받아 LLM을 통해 답변을 생성합니다."""
        pass
