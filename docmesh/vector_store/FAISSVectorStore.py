import os
import faiss
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

    def add_documents(self, documents: list) -> None:
        lc_documents = []
        for doc in documents:
            # 문서가 to_langchain_document() 메서드를 갖고 있으면 변환, 아니면 그대로 사용
            if hasattr(doc, "to_langchain_document"):
                lc_documents.append(doc.to_langchain_document())
            else:
                lc_documents.append(doc)
        self.vectorstore.add_documents(lc_documents)

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
