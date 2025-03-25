import os
import faiss
import json
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from docmesh.embedding import BaseEmbeddingModel
from docmesh.vector_store.BaseVectorStore import BaseVectorStore


# 간단한 JSON 기반의 Docstore 구현체
class DiskDocstore:
    def __init__(self, store=None):
        self.store = store or {}

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def items(self):
        return self.store.items()

    def get(self, key, default=None):
        return self.store.get(key, default)

    def to_dict(self):
        return self.store

    @classmethod
    def load(cls, file_path: str):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                store = json.load(f)
            return cls(store)
        else:
            return cls()

    def save(self, file_path: str):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.store, f)


# 로컬 저장소 기반 FAISS 벡터 스토어 구현
class DiskBasedLangchainFAISSVectorStore(BaseVectorStore):
    # Todo 아직 완성안됨
    def __init__(
        self, embedding_model: BaseEmbeddingModel, index_file: str, docstore_file: str
    ):
        self.embedding_model = embedding_model
        self.index_file = index_file
        self.docstore_file = docstore_file

        # FAISS 인덱스 로드 또는 생성
        if os.path.exists(self.index_file):
            index = faiss.read_index(self.index_file)
        else:
            index = faiss.IndexFlatL2(self.embedding_model.vector_dim)

        # 디스크 기반 Docstore 로드 또는 새로 생성 (여기서는 단순 JSON 방식)
        self.disk_docstore = DiskDocstore.load(self.docstore_file)
        # 저장된 인덱스-문서 매핑과 실제 docstore 데이터를 불러옵니다.
        self.index_to_docstore_id = self.disk_docstore.get("index_to_docstore_id", {})
        # JSON 저장 시 키는 문자열이므로 정수로 변환
        if self.index_to_docstore_id:
            self.index_to_docstore_id = {
                int(k): v for k, v in self.index_to_docstore_id.items()
            }

        # 실제 docstore 데이터가 저장되어 있으면 불러오고, 없으면 빈 dict 사용
        docstore_data = self.disk_docstore.get("docstore", {})

        # InMemoryDocstore 생성 시, 기존 데이터를 전달 (InMemoryDocstore는 dict로 초기화 가능)
        from langchain_community.docstore.in_memory import InMemoryDocstore

        in_memory_docstore = (
            InMemoryDocstore(docstore_data) if docstore_data else InMemoryDocstore()
        )

        # FAISS 래퍼를 이용해 벡터 스토어 생성
        self.vectorstore = FAISS(
            embedding_function=lambda text: self.embedding_model.get_embedding(text),
            index=index,
            docstore=in_memory_docstore,
            index_to_docstore_id=self.index_to_docstore_id,
        )

    def add_documents(self, documents: list) -> None:
        lc_documents = []
        for doc in documents:
            if hasattr(doc, "to_langchain_document"):
                lc_doc = doc.to_langchain_document()
            else:
                lc_doc = doc
            lc_documents.append(lc_doc)
        self.vectorstore.add_documents(lc_documents)

        # 만약 내부 매핑이 비어있다면 수동으로 생성 (테스트용 DummyEmbeddingModel에 대해서)
        if not self.vectorstore.index_to_docstore_id:
            self.vectorstore.index_to_docstore_id = {
                i: doc for i, doc in enumerate(lc_documents)
            }

        # FAISS 인덱스를 파일에 저장
        faiss.write_index(self.vectorstore.index, self.index_file)
        # 업데이트된 인덱스 매핑과 실제 docstore 데이터를 저장합니다.
        self.disk_docstore.store[
            "index_to_docstore_id"
        ] = self.vectorstore.index_to_docstore_id
        # 가정: InMemoryDocstore에 to_dict() 메서드가 있어 dict 형태로 얻을 수 있음.
        self.disk_docstore.store["docstore"] = (
            self.vectorstore.docstore.to_dict()
            if hasattr(self.vectorstore.docstore, "to_dict")
            else {}
        )
        self.disk_docstore.save(self.docstore_file)

    def search(self, query: str, k: int = 3) -> list:
        return self.vectorstore.similarity_search(query, k=k)


class LangchainFAISSVectorStore(BaseVectorStore):
    """
    LangchainFAISSVectorStore는 FAISS 인덱스, InMemoryDocstore, 빈 index_to_docstore_id 딕셔너리를 이용해
    벡터 스토어를 초기화하고, 임베딩 모델의 get_embedding()을 사용해 문서 임베딩 및 유사도 검색을 수행합니다.
    """

    def __init__(self, embedding_model: BaseEmbeddingModel):
        self.embedding_model = embedding_model
        index = faiss.IndexFlatL2(self.embedding_model.vector_dim)
        self.vectorstore = FAISS(
            embedding_function=lambda text: self.embedding_model.get_embedding(text),
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def add_documents(self, documents: list) -> None:
        lc_documents = []
        for doc in documents:
            if hasattr(doc, "to_langchain_document"):
                lc_documents.append(doc.to_langchain_document())
            else:
                lc_documents.append(doc)
        self.vectorstore.add_documents(lc_documents)

    def search(self, query: str, k: int = 3) -> list:
        return self.vectorstore.similarity_search(query, k=k)
