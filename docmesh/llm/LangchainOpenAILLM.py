from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from dotenv import load_dotenv
from docmesh.llm.BaseLLM import BaseLLM


class LangchainOpenAILLM(BaseLLM):
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
