import os
import faiss
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from docmesh.embedding.BaseEmbeddingModel import BaseEmbeddingModel


class LangchainFAISSVectorStore:
    def __init__(self, embedding_model: BaseEmbeddingModel, path: str = None):
        self.path = path
        self.embedding_model = embedding_model
        # FAISS 인덱스 생성 (벡터 차원은 임베딩 모델에서 가져옴)
        index = faiss.IndexFlatL2(self.embedding_model.vector_dim)
        # FAISS 벡터 스토어 생성: InMemoryDocstore와 빈 매핑을 사용
        self.vectorstore = FAISS(
            embedding_function=lambda text: self.embedding_model.get_embedding(text),
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def set_path(self, path: str):
        self.path = path

    def add_documents(self, documents: list[Document]) -> None:
        self.vectorstore.add_documents(documents)

    def search(self, query: str, k: int = 3) -> list:
        return self.vectorstore.similarity_search(query, k=k)

    def save(self):
        if self.path is not None:
            # 저장 전, 경로(디렉토리)가 없으면 생성
            os.makedirs(self.path, exist_ok=True)
            try:
                self.vectorstore.save_local(self.path)
            except Exception as e:
                raise Exception(f"Error saving vectorstore to {self.path}: {e}")

    def load(self):
        if self.path is not None:
            try:
                self.vectorstore = FAISS.load_local(
                    self.path,
                    lambda text: self.embedding_model.get_embedding(text),
                    allow_dangerous_deserialization=True,
                )
            except Exception as e:
                raise Exception(f"Error loading vectorstore from {self.path}: {e}")
