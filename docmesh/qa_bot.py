from docmesh.llm import BaseLLM


# -------------------------------
# QAService: 질문에 대해 문서 검색 및 답변 생성 (참조 URL 포함)
# -------------------------------
class QAService:
    def __init__(self, llm_service: BaseLLM, embedding_manager, default_k: int = 3):
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
