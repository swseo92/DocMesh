import os
import tempfile
import pytest

from docmesh.embedding.LangchainOpenAIEmbeddingModel import (
    LangchainOpenAIEmbeddingModel,
)
from docmesh.vector_store import LangchainFAISSVectorStore
from docmesh.format import Document
from tests.utils.dummy_embedding import DummyEmbeddingModel
from dotenv import load_dotenv


load_dotenv()

# 테스트를 위해 실제 API 키가 있는지 확인합니다.
pytestmark = pytest.mark.skipif(
    os.getenv("OPENAI_API_KEY") is None,
    reason="OPENAI_API_KEY not set, skipping tests using actual model.",
)


def create_actual_test_documents():
    # 실제 모델을 사용할 때는 다양한 주제의 문서를 생성합니다.
    doc1 = Document(
        "This is a test document about Python programming.", {"source": "example1"}
    )
    doc2 = Document(
        "Another document discussing AI and machine learning concepts.",
        {"source": "example2"},
    )
    doc3 = Document(
        "Yet another document covering language models and their applications.",
        {"source": "example3"},
    )
    return [doc1, doc2, doc3]


def create_length_test_documents():
    # 문서 3개 생성: 각각 길이가 20, 40, 60인 문자열과 source 메타데이터 포함
    doc1 = Document("A" * 20, {"source": "doc1"})
    doc2 = Document("B" * 40, {"source": "doc2"})
    doc3 = Document("C" * 60, {"source": "doc3"})
    return [doc1, doc2, doc3]


def test_add_and_search_documents_actual():
    # 실제 모델 생성 (예: text-embedding-ada-002)
    model = LangchainOpenAIEmbeddingModel(model_name="text-embedding-ada-002")
    # 저장 경로 없이 인스턴스 생성 (비영속적)
    store = LangchainFAISSVectorStore(model)
    docs = create_actual_test_documents()
    store.add_documents(docs)
    # "Python" 관련 쿼리로 검색
    results = store.search("What is Python?", k=2)
    assert (
        len(results) > 0
    ), "Search should return at least one document using the actual model."
    # 결과 문서의 메타데이터에 source가 포함되어 있는지 확인
    for doc in results:
        assert (
            "source" in doc.metadata
        ), "Returned document should contain a 'source' metadata."


def test_save_and_load():
    model = DummyEmbeddingModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "faiss_store")
        # 저장 경로를 지정하여 인스턴스 생성
        store = LangchainFAISSVectorStore(model, path=path)
        docs = create_actual_test_documents()
        store.add_documents(docs)
        # 저장 기능 호출
        store.save()
        # 새로운 인스턴스를 생성하고, load()를 통해 저장된 데이터를 복원
        new_store = LangchainFAISSVectorStore(model, path=path)
        new_store.load()
        results = new_store.search("What is Python?", k=2)
        print(results)
        assert (
            len(results) > 0
        ), "Loaded store should return search results using the actual model."


def test_new_instance_load_cycles():
    model = DummyEmbeddingModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "faiss_store")
        # 초기 store 인스턴스 생성 및 문서 추가
        initial_store = LangchainFAISSVectorStore(model, path=store_path)
        docs = create_actual_test_documents()
        initial_store.add_documents(docs)
        # 초기 검색 결과 확보 (예: "Python" 관련 검색)
        initial_results = initial_store.search("Python", k=2)
        assert len(initial_results) > 0, "Initial search returned no results."
        initial_contents = [doc.page_content for doc in initial_results]
        initial_store.save()

        cycles = 3
        for cycle in range(cycles):
            # 매 사이클마다 새로운 store 인스턴스를 생성하고 load()를 호출
            new_store = LangchainFAISSVectorStore(model, path=store_path)
            new_store.load()
            cycle_results = new_store.search("Python", k=2)
            assert (
                len(cycle_results) > 0
            ), f"Cycle {cycle + 1}: search returned no results."
            cycle_contents = [doc.page_content for doc in cycle_results]
            # 초기 결과와 동일한지 확인
            assert (
                cycle_contents == initial_contents
            ), f"Cycle {cycle + 1}: search results differ from initial."

            new_store.add_documents(docs)
            initial_results = new_store.search("Python", k=2)
            initial_contents = [doc.page_content for doc in initial_results]
            new_store.save()


def test_actual_data_expected_results():
    model = DummyEmbeddingModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = os.path.join(tmpdir, "faiss_store")
        # 저장 경로를 지정하여 LangchainFAISSVectorStore 인스턴스 생성
        store = LangchainFAISSVectorStore(model, path=store_path)
        docs = create_length_test_documents()
        store.add_documents(docs)
        store.save()

        # 쿼리 텍스트 길이 35 (예: "X" * 35)의 임베딩 벡터는 [35,36,37,38]입니다.
        # 각 문서의 임베딩은 각각 [20,21,22,23], [40,41,42,43], [60,61,62,63]이므로,
        # L2 거리는 doc1: 약 30, doc2: 약 10, doc3: 약 25가 되어, doc2가 가장 가까워야 합니다.
        query = "X" * 35
        results = store.search(query, k=2)
        assert len(results) > 0, "No search results returned."
        # 첫 번째 결과가 "B" * 40 인 doc2여야 함
        assert (
            results[0].page_content == "B" * 40
        ), "Expected doc2 to be closest to query of length 35."
        assert results[0].metadata.get("source") == "doc2", "Expected source 'doc2'."

        # 새로운 store 인스턴스를 생성하여 load() 후에도 동일한 결과가 나오는지 확인
        new_store = LangchainFAISSVectorStore(model, path=store_path)
        new_store.load()
        new_results = new_store.search(query, k=2)
        assert len(new_results) > 0, "No search results returned after load."
        assert (
            new_results[0].page_content == "B" * 40
        ), "After load, expected doc2 to be closest."
        assert (
            new_results[0].metadata.get("source") == "doc2"
        ), "After load, expected source 'doc2'."
