# qa_bot.py
import abc
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


# LLM 서비스 추상 클래스: 모든 LLM 모델은 이 인터페이스를 구현해야 합니다.
class BaseLLMService(abc.ABC):
    @abc.abstractmethod
    def generate_answer(self, prompt: str) -> str:
        """프롬프트를 받아 LLM을 통해 답변을 생성합니다."""
        pass


# LangChain 기반 LLM 서비스 구현
class LangchainLLMService(BaseLLMService):
    def __init__(self, model: str = "gpt-3.5-turbo", temperature: float = 0.0):
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


# -------------------------------
# QAService: 질문에 대해 문서 검색 및 답변 생성 (참조 URL 포함)
# -------------------------------
class QAService:
    def __init__(
        self, llm_service: BaseLLMService, embedding_manager, default_k: int = 3
    ):
        self.llm_service = llm_service
        self.embedding_manager = embedding_manager
        self.default_k = default_k

    def answer_question(self, question: str, k: int = None) -> str:
        """
        사용자의 질문을 받아 관련 문서를 검색한 후, LLM을 통해 답변과 참조 URL을 포함한 결과를 생성합니다.
        프롬프트에 검색된 문서 내용과 함께 해당 문서들의 URL 목록을 전달하여,
        LLM이 답변 마지막에 "Sources:" 형태로 참조 URL을 표시하도록 요청합니다.
        또한, 질문과 동일한 언어로 답변하도록 요청합니다.
        """
        if k is None:
            k = self.default_k

        # FAISS 등 임베딩 기반 벡터 스토어에서 유사 문서 검색
        search_results = self.embedding_manager.search_chunks(question, k=k)
        context = "\n".join([doc.page_content for doc in search_results])

        # 검색 결과에서 중복 제거한 URL 목록 추출
        sources = list(
            {
                doc.metadata.get("source", "")
                for doc in search_results
                if doc.metadata.get("source")
            }
        )
        sources_str = ", ".join(sources)

        prompt = (
            f"Context: {context}\n\n"
            f"Question: {question}\n\n"
            f"""Please answer the question in the same language as the question, based on the above context. """
            f"After your answer, on a new line, list the source URLs used in your answer in the following format:\n"
            f"Sources: {sources_str}\n\n"
            f"Answer:"
        )
        answer = self.llm_service.generate_answer(prompt)
        return answer


# -------------------------------
# main() 함수: 전체 QA 흐름 테스트
# -------------------------------
def main():
    # embedding.py 모듈에서 임베딩 모델, FAISSVectorStore, Document, EmbeddingStoreManager를 가져옵니다.
    from embedding import (
        LangchainOpenAIEmbeddingModel,
        LangchainFAISSVectorStore,
        Document,
        EmbeddingStoreManager,
    )

    embedding_model = LangchainOpenAIEmbeddingModel()
    vector_store = LangchainFAISSVectorStore(embedding_model)
    embedding_manager = EmbeddingStoreManager(embedding_model, vector_store)

    # FAISS 벡터 스토어의 index_to_docstore_id를 사용해 문서가 이미 저장되어 있는지 확인합니다.
    if not vector_store.vectorstore.index_to_docstore_id:
        docs = [
            Document(
                "This is the content of the first document.",
                {"source": "https://example.com"},
            ),
            Document(
                "The second document contains more detailed information for testing.",
                {"source": "https://example.org"},
            ),
        ]
        embedding_manager.embed_and_store(docs)

    # 팩토리를 이용해 LLM 서비스를 생성합니다.
    llm_service = LLMServiceFactory.create_llm_service(
        provider="langchain", model="gpt-3.5-turbo", temperature=0.0
    )
    qa_service = QAService(llm_service, embedding_manager)

    # 테스트 질문
    question = "What is the content of the first document?"
    answer = qa_service.answer_question(question)
    print("Question:", question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
