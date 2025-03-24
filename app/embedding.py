import os
import openai
import faiss
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.schema import Document as LC_Document
from dotenv import load_dotenv

# .env 파일에서 OPENAI_API_KEY를 로드합니다.
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


class Document:
    """
    Document 클래스는 임베딩 및 검색을 위한 텍스트 청크와 메타데이터를 캡슐화합니다.
    """
    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

    def to_langchain_document(self) -> LC_Document:
        return LC_Document(page_content=self.page_content, metadata=self.metadata)


# --- Embedding Model 구현 ---
class LangchainOpenAIEmbeddingModel:
    """
    LangchainOpenAIEmbeddingModel은 LangChain의 OpenAIEmbeddings를 사용하여 텍스트를 임베딩합니다.
    """
    def __init__(self, model_name: str = "text-embedding-ada-002"):
        self.embeddings = OpenAIEmbeddings(model=model_name)
        # "hello world"에 대한 임베딩 결과를 통해 임베딩 차원을 동적으로 결정합니다.
        self.vector_dim = len(self.embeddings.embed_query("hello world"))

    def get_embedding(self, text: str) -> list:
        return self.embeddings.embed_query(text)


# --- Vector Store 구현 ---
class LangchainFAISSVectorStore:
    """
    LangchainFAISSVectorStore는 공식 문서 예제와 같이,
    FAISS 인덱스, InMemoryDocstore, 그리고 빈 index_to_docstore_id 딕셔너리를 이용해 벡터 스토어를 초기화합니다.
    """
    def __init__(self, embedding_model: LangchainOpenAIEmbeddingModel):
        self.embedding_model = embedding_model
        # 임베딩 차원에 맞게 FAISS 인덱스 생성
        index = faiss.IndexFlatL2(self.embedding_model.vector_dim)
        self.vectorstore = FAISS(
            embedding_function=self.embedding_model.embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

    def add_documents(self, documents: list) -> None:
        # Document 객체들을 LangChain Document 객체로 변환하여 추가
        lc_documents = [doc.to_langchain_document() for doc in documents]
        self.vectorstore.add_documents(lc_documents)

    def search(self, query: str, k: int = 3) -> list:
        # similarity_search()를 통해 유사 문서를 검색
        return self.vectorstore.similarity_search(query, k=k)


# --- EmbeddingStoreManager: 두 객체를 결합하는 Manager ---
class EmbeddingStoreManager:
    """
    EmbeddingStoreManager는 임베딩 모델과 벡터 스토어를 조합하여,
    Document 객체들의 임베딩 생성, 저장 및 유사도 검색 기능을 제공합니다.
    """
    def __init__(self, embedding_model: LangchainOpenAIEmbeddingModel, vector_store: LangchainFAISSVectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def embed_and_store(self, documents: list) -> None:
        self.vector_store.add_documents(documents)

    def search_chunks(self, query: str, k: int = 3) -> list:
        return self.vector_store.search(query, k=k)


# --- 사용 예시 ---
if __name__ == "__main__":
    # 테스트용 Document 객체들 생성
    docs = [
        Document("This is the content of the first document.", {"source": "https://example.com"}),
        Document("The second document contains more detailed information for testing.", {"source": "https://example.org"})
    ]

    # LangChain 기반 임베딩 모델과 FAISS 벡터스토어 생성
    embedding_model = LangchainOpenAIEmbeddingModel()
    vector_store = LangchainFAISSVectorStore(embedding_model)

    # EmbeddingStoreManager를 생성하여 두 객체를 결합
    manager = EmbeddingStoreManager(embedding_model, vector_store)

    # Document 임베딩 및 저장
    manager.embed_and_store(docs)

    # 사용자 질문에 대해 유사도 검색 실행
    query = "first document content"
    results = manager.search_chunks(query, k=2)

    print("Search results:")
    for res in results:
        print(f"Source: {res.metadata.get('source', '')}")
        print(f"Content: {res.page_content}\n")
