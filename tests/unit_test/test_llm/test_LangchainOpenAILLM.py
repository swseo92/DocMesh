import os
import pytest
from docmesh.llm.LangchainOpenAILLM import LangchainOpenAILLM
from dotenv import load_dotenv


load_dotenv()


@pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None, reason="OPENAI_API_KEY is not set"
)
def test_langchain_openai_llm_generate_answer():
    # gpt-3.5-turbo 모델을 사용하여 LLM 인스턴스 생성
    llm = LangchainOpenAILLM(model="gpt-3.5-turbo", temperature=0.0)
    prompt = "What is the capital of France?"
    answer = llm.generate_answer(prompt)

    # 반환값이 문자열이며, 비어있지 않은지 검증합니다.
    assert isinstance(answer, str)
    assert len(answer) > 0
