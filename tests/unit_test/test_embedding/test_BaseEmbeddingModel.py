import pytest
from docmesh.format import Document
from docmesh.embedding.BaseEmbeddingModel import BaseEmbeddingModel
from langchain.schema import Document as LC_Document


# DummyEmbeddingModel은 BaseEmbeddingModel의 추상 메서드를 구현한 간단한 예시입니다.
class DummyEmbeddingModel(BaseEmbeddingModel):
    def get_embedding(self, text: str) -> list:
        return [len(text)]

    @property
    def vector_dim(self) -> int:
        return 1


def test_document_to_langchain_document():
    doc = Document("sample text", {"source": "dummy"})
    lc_doc = doc.to_langchain_document()
    assert isinstance(lc_doc, LC_Document)
    assert lc_doc.page_content == "sample text"
    assert lc_doc.metadata["source"] == "dummy"


def test_base_embedding_model_abstract():
    with pytest.raises(TypeError):
        _ = BaseEmbeddingModel()  # 직접 인스턴스화하면 추상 클래스이므로 에러가 발생해야 합니다.


def test_dummy_embedding_model():
    model = DummyEmbeddingModel()
    embedding = model.get_embedding("test")
    assert isinstance(embedding, list)
    # "test"의 길이는 4이므로 [4]가 반환되어야 합니다.
    assert embedding[0] == 4
    assert model.vector_dim == 1
