import os
import pytest
from docmesh.llm import LangchainLLMService
from dotenv import load_dotenv


load_dotenv()


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is not set, skipping OpenAI integration test.",
)
def test_openai():
    """
    LangchainLLMService를 이용해 실제 OpenAI API와 연동되는지 확인합니다.
    테스트 프롬프트를 통해 LLM 호출 후, 비어있지 않은 문자열 답변이 반환되는지 검증합니다.
    """
    service = LangchainLLMService(model="gpt-3.5-turbo", temperature=0.0)
    prompt = "Please provide a one-sentence summary of the current date in English."
    answer = service.generate_answer(prompt)

    # 답변이 문자열이며, 공백을 제거한 길이가 0보다 큰지 확인합니다.
    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
    print("Generated answer:", answer)


if __name__ == "__main__":
    pytest.main()
