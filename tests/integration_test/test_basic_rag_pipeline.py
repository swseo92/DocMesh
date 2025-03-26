from docmesh.embedding import EmbeddingModelFactory
from docmesh.vector_store import VectorStoreFactory
from docmesh.llm import LLMFactory
from docmesh.config import Config
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from docmesh.tools.load_config import load_config


from dotenv import load_dotenv


def test_basic_rag_pipeline():
    load_dotenv()
    config = load_config("../test_config.yaml")

    path_save_faiss = "./faiss_index"
    # EmbeddingModelFactory를 통해 임베딩 모델 생성
    embedding_model = EmbeddingModelFactory().create_embedding_model(
        **config["embedding_model"]
    )
    # VectorStoreFactory를 통해 벡터 스토어 생성
    vector_store = VectorStoreFactory().create_vector_store(
        **config["vector_store"], embedding_model=embedding_model, path=path_save_faiss
    )
    docs = vector_store.similarity_search("qux")
    # 로드가 정상적으로 되었는지 확인
    assert len(docs) > 0

    # 벡터 스토어를 리트리버로 사용
    retriever = vector_store.as_retriever()

    # 4. LLM 서비스 생성 (팩토리 사용)
    llm = LLMFactory().create_llm(
        provider=Config.LLM_PROVIDER,
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
    )

    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Answer in Korean.

        #Question:
        {question}
        #Context:
        {context}        
        #Answer:"""
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    chain.invoke("hi")
