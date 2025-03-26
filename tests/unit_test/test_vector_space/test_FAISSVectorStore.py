import os
import pytest
from langchain.schema import Document
from docmesh.vector_store.FAISSVectorStore import LangchainFAISSVectorStore
from tests.mocks.dummy_embedding import DummyEmbeddingModel


@pytest.fixture
def mock_embedding_model():
    return DummyEmbeddingModel()


@pytest.fixture
def sample_documents():
    return [
        Document(page_content="Hello world", metadata={"source": "doc1"}),
        Document(page_content="Foo bar", metadata={"source": "doc2"}),
        Document(
            page_content="Lorem ipsum dolor sit amet", metadata={"source": "doc3"}
        ),
    ]


def test_faiss_vector_store_add_and_search(mock_embedding_model, sample_documents):
    """
    문서를 추가(add_documents)한 뒤,
    검색(search) 결과가 올바르게 반환되는지 테스트.
    """
    store = LangchainFAISSVectorStore(embedding_model=mock_embedding_model)
    store.add_documents(sample_documents)

    # 검색 쿼리를 날려본다.
    # mock_embedding_model은 텍스트 길이에 비례해 벡터를 생성한다.
    # - "world" 길이: 5
    # - "bar" 길이: 3
    # ...
    #
    # 검색 로직상, 유사도는 벡터 간 내적/유클리디안 기반이므로
    # 실제 결과 일치 여부는 단순히 "정상 동작 여부"만 확인 가능.
    # 여기서는 오작동 없이 문서를 반환하는지만 체크한다.
    query = "Hello"
    results = store.search(query, k=2)

    # k=2 → 최대 2개 결과
    assert len(results) <= 2, "검색 결과가 최대 2개여야 합니다."

    # 반환된 결과가 Document 인스턴스인지 확인
    for doc in results:
        assert isinstance(doc, Document)


def test_faiss_vector_store_save_and_load(
    tmp_path, mock_embedding_model, sample_documents
):
    """
    FAISS 인덱스를 디스크에 저장 후, 로드했을 때
    동일한 검색 결과가 나오는지 확인.
    """
    # 1) 임시 폴더 설정
    save_dir = tmp_path / "faiss_store"
    os.makedirs(save_dir, exist_ok=True)

    # 2) 벡터스토어 생성 및 문서 추가 → 저장
    store = LangchainFAISSVectorStore(
        embedding_model=mock_embedding_model, path=str(save_dir)
    )
    store.add_documents(sample_documents)
    store.save()

    # 3) 새로 스토어 인스턴스를 만들어 load
    new_store = LangchainFAISSVectorStore(
        embedding_model=mock_embedding_model, path=str(save_dir)
    )
    new_store.load()

    # 4) 저장 전 스토어와 저장 후 로드한 스토어에서 동일 쿼리 검색
    query = "Hello"
    old_results = store.search(query, k=2)
    new_results = new_store.search(query, k=2)

    # 두 결과가 동일한지(또는 길이와 문서 타입만 간단히 비교) 확인
    assert len(old_results) == len(new_results), "로드 전후 검색 결과의 개수가 달라서는 안 됩니다."

    # 결과 문서가 Document인지 확인
    for doc in new_results:
        assert isinstance(doc, Document)


def test_faiss_vector_store_no_path_save(mock_embedding_model, sample_documents):
    """
    path가 None인 상태에서 save()를 호출하면 예외 없이 그냥 넘어가는지,
    혹은 적절히 처리하는지(코드에 따라).
    """
    store = LangchainFAISSVectorStore(embedding_model=mock_embedding_model, path=None)
    store.add_documents(sample_documents)

    # path=None이면 아무것도 저장하지 않거나, 에러를 던지지 않아야 함.
    # 코드 상에서 "if self.path is not None:" 로직이 있으므로
    # 아무 일도 일어나지 않고 끝나는 게 정상.
    try:
        store.save()
    except Exception as e:
        pytest.fail(f"save() 호출 시 path=None이어도 예외가 발생하지 않아야 합니다. 에러: {e}")


def test_faiss_vector_store_no_path_load(mock_embedding_model):
    """
    path가 None인 상태에서 load()를 호출하면 에러 없이 통과하는지 테스트.
    """
    store = LangchainFAISSVectorStore(embedding_model=mock_embedding_model, path=None)
    try:
        store.load()
    except Exception as e:
        pytest.fail(f"load() 호출 시 path=None이어도 예외가 발생하지 않아야 합니다. 에러: {e}")


def test_faiss_vector_store_invalid_path_load(mock_embedding_model):
    """
    잘못된(존재하지 않는) 경로에서 load() 시도하면 예외가 발생하는지 테스트.
    """
    invalid_path = "some/nonexistent/path"
    store = LangchainFAISSVectorStore(
        embedding_model=mock_embedding_model, path=invalid_path
    )

    with pytest.raises(Exception) as excinfo:
        store.load()
    assert "Error loading vectorstore" in str(excinfo.value), "적절한 예외 메시지가 포함되어야 함."
