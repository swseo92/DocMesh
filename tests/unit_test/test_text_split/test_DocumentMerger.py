from langchain.schema import Document
from docmesh.text_split.DocumentMerger import DocumentMerger, default_length_function


def test_merge_documents_single_source_min_tokens():
    """
    하나의 source를 가진 여러 개의 짧은 문서가 순서대로 배치될 때,
    min_tokens 이하의 문서들이 연달아 병합되어
    최종적으로 충분한 길이를 가진 단일 문서가 되는지 테스트.
    """
    docs = [
        Document(page_content="첫 번째 " * 10, metadata={"source": "A"}),  # 길이 10
        Document(page_content="두 번째 " * 10, metadata={"source": "A"}),  # 길이 10
        Document(page_content="세 번째 " * 10, metadata={"source": "A"}),  # 길이 10
    ]

    total_length = 0
    for doc in docs:
        total_length += default_length_function(doc.page_content)

    # 총 길이 = 30 토큰
    merger = DocumentMerger(min_tokens=total_length)
    merged = merger.merge_documents(docs)

    # 최종적으로 한 번에 병합되어 길이가 total_length인 1개의 문서만 있어야 함
    assert len(merged) == 1, "모든 문서가 하나로 병합되어야 합니다."
    assert merged[0].metadata["source"] == "A"


def test_merge_documents_different_sources():
    """
    서로 다른 source를 가진 문서들은 병합되지 않고
    각각 별도의 문서로 유지되는지 테스트.
    """
    docs = [
        Document(page_content="문서1 " * 5, metadata={"source": "A"}),
        Document(page_content="문서2 " * 5, metadata={"source": "B"}),
        Document(page_content="문서3 " * 5, metadata={"source": "A"}),
    ]
    merger = DocumentMerger(min_tokens=50)
    merged = merger.merge_documents(docs)

    # 서로 다른 source이거나, 연속되지 않으므로 각각 병합되지 않아야 함
    assert len(merged) == 3, "서로 다른 source를 가진 문서는 병합되지 않아야 합니다."


def test_merge_documents_doc_already_long():
    """
    이미 min_tokens 이상인 문서는 단독으로 병합되지 않고, 그대로 추가되는지 테스트.
    """
    docs = [
        Document(page_content="충분히 긴 문서 " * 40, metadata={"source": "A"}),  # 길이 40
        Document(page_content="짧은 문서 " * 10, metadata={"source": "A"}),  # 길이 10
    ]
    # min_tokens = 20
    merger = DocumentMerger(min_tokens=20)
    merged = merger.merge_documents(docs)

    # 첫 번째 문서는 이미 40 토큰이므로 그대로 하나의 문서
    # 두 번째 문서는 10 토큰이어서 병합을 시도할 다음 문서가 없으므로 그대로 단일 문서로 저장
    # 최종 문서 수: 2
    assert len(merged) == 2, "이미 충분히 긴 문서는 병합되지 않고 유지되어야 합니다."


def test_merge_documents_no_docs():
    """
    빈 리스트가 들어왔을 때, 빈 리스트가 반환되는지 테스트.
    """
    merger = DocumentMerger(min_tokens=10)
    merged = merger.merge_documents([])
    assert merged == [], "빈 입력에 대해서는 빈 리스트를 반환해야 합니다."
