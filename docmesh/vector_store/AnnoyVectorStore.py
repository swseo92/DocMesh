import os
from docmesh.embedding import BaseEmbeddingModel
from docmesh.vector_store.BaseVectorStore import BaseVectorStore
from annoy import AnnoyIndex


class AnnoyVectorStore(BaseVectorStore):
    """
    AnnoyVectorStore는 AnnoyIndex를 사용하여 벡터를 저장하고 유사도 검색을 수행합니다.
    내부적으로 문서 객체를 별도의 dict에 저장하며, Annoy 인덱스에는 임베딩 벡터를 추가합니다.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        n_trees: int = 10,
        index_path: str = None,
    ):
        self.embedding_model = embedding_model
        self.dimension = embedding_model.vector_dim
        self.index = AnnoyIndex(self.dimension, metric="angular")
        self.docstore = {}
        self.next_doc_id = 0
        self.n_trees = n_trees
        self.built = False
        self.index_path = index_path
        if self.index_path and os.path.exists(self.index_path):
            self.index.load(self.index_path)
            self.built = True
            print(f"Annoy 인덱스를 '{self.index_path}'에서 로드했습니다.")

    def add_documents(self, documents: list) -> None:
        for doc in documents:
            if hasattr(doc, "to_langchain_document"):
                lc_doc = doc.to_langchain_document()
            else:
                lc_doc = doc
            vector = self.embedding_model.get_embedding(lc_doc.page_content)
            self.index.add_item(self.next_doc_id, vector)
            self.docstore[self.next_doc_id] = lc_doc
            self.next_doc_id += 1
        self.index.build(self.n_trees)
        self.built = True
        if self.index_path:
            self.index.save(self.index_path)
            print(f"Annoy 인덱스를 '{self.index_path}'에 저장했습니다.")

    def search(self, query: str, k: int = 3) -> list:
        query_vector = self.embedding_model.get_embedding(query)
        if not self.built:
            self.index.build(self.n_trees)
            self.built = True
        nearest_ids = self.index.get_nns_by_vector(
            query_vector, k, include_distances=False
        )
        return [self.docstore[i] for i in nearest_ids if i in self.docstore]
