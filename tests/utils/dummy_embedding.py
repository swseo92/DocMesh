from docmesh.embedding.BaseEmbeddingModel import BaseEmbeddingModel


class DummyEmbeddingModel(BaseEmbeddingModel):
    def __init__(self):
        self._vector_dim = 4

    def get_embedding(self, text: str) -> list:
        length = float(len(text))
        # 간단하게 [length, length+1, length+2, length+3] 형태의 벡터 반환
        return [length, length + 1, length + 2, length + 3]

    @property
    def vector_dim(self) -> int:
        return self._vector_dim
