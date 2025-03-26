from langchain.schema import Document
from docmesh.text_split.DocumentChunkSplitter import DocumentChunkSplitter
from docmesh.text_split.tools import default_length_function


def test_chunkify_single_short_document():
    """
    문서가 chunk_size보다 짧은 경우,
    분할되지 않고 원본 문서 그대로 유지되는지 확인.
    """
    doc = Document(page_content="안녕하세요. 짧은 문서입니다.", metadata={"source": "A"})

    length = default_length_function(doc.page_content)

    splitter = DocumentChunkSplitter(chunk_size=length + 1, chunk_overlap=2)
    result = splitter.chunkify([doc])

    # 결과는 1개 문서, 원본 그대로여야 함
    assert len(result) == 1
    assert result[0].page_content == doc.page_content
    assert result[0].metadata == doc.metadata


def test_chunkify_single_document_multiple_chunks():
    """
    문서가 chunk_size보다 길어 여러 덩어리로 분할되는 경우를 테스트.
    chunk_size=5, chunk_overlap=2 로 설정하여,
    의도적으로 여러 번 chunk가 생기는지 확인.
    """
    # 이 문서는 토큰(단어) 단위로 5개씩 끊길 수 있도록, 단어를 20개 정도 넣어둔다.
    content_words = " ".join(f"단어{i}" for i in range(20))
    doc = Document(page_content=content_words, metadata={"source": "B"})
    splitter = DocumentChunkSplitter(chunk_size=5, chunk_overlap=2)

    result = splitter.chunkify([doc])

    # TokenTextSplitter의 기본 로직:
    # chunk_size=5, chunk_overlap=2 이므로,
    # 첫 번째 chunk는 단어 5개, 다음 chunk는 이전 chunk의 마지막 2개를 중복 포함한 뒤 3개를 더해서 5개, ...
    # 총 단어 20개 처리하면, 대략 아래와 같이 chunk가 만들어질 수 있음:
    # chunk1: 단어0 ~ 단어4 (5개)
    # chunk2: 단어3 ~ 단어7
    # chunk3: 단어5 ~ 단어9
    # chunk4: 단어7 ~ 단어11
    # ...
    #
    # 단어 총 20개 => chunk_size=5, overlap=2 => 실제 chunk 개수는
    # ceil((20 - 2) / (5 - 2)) = ceil(18/3) = 6 정도 나올 것
    #
    # 실제 TokenTextSplitter의 내부 처리 방식에 따라 +1정도 차이는 있을 수 있으니
    # 테스트에선 "5개 이상의 덩어리"가 생기는지만 확인하거나,
    # 혹은 실제 chunk 수 계산 공식을 따져 엄밀히 검사할 수 있음

    assert len(result) >= 5, f"5개 이상의 chunk로 분할되길 기대했지만, {len(result)}개가 반환됨."

    # 각 chunk의 metadata가 원본 metadata를 복사하고,
    # split_index가 0부터 순서대로 매겨지는지 확인
    for i, chunk_doc in enumerate(result):
        assert chunk_doc.metadata["source"] == "B"
        assert chunk_doc.metadata["split_index"] == i

    # 전체 chunk를 합쳤을 때 본문의 모든 단어가 포함되는지(중복 허용)는 확인 가능
    # 이 경우 overlap 때문에 중복되는 단어들이 있을 수 있음.
    # 다만, 이 테스트는 단순히 "문서가 여러 덩어리로 나뉜다"는 사실만 검증하면 충분.


def test_chunkify_multiple_documents():
    """
    여러 개의 Document가 입력될 때,
    각각에 대해 적절히 chunkify가 수행되는지 테스트.
    """
    doc1 = Document(
        page_content=" ".join(f"A{i}" for i in range(12)), metadata={"source": "A"}
    )
    doc2 = Document(
        page_content=" ".join(f"B{i}" for i in range(8)), metadata={"source": "B"}
    )
    splitter = DocumentChunkSplitter(chunk_size=5, chunk_overlap=2)

    result = splitter.chunkify([doc1, doc2])
    # doc1은 length=12, chunk_size=5, overlap=2 -> 최소 3개 chunk 예상
    # doc2는 length=8 , chunk_size=5, overlap=2 -> 최소 2개 chunk 예상
    # 총합 최소 5개 이상
    assert len(result) >= 5

    # doc1 관련 chunk부터 확인
    doc1_chunks = [r for r in result if r.metadata["source"] == "A"]
    assert len(doc1_chunks) >= 2, "doc1이 최소 2개의 chunk로 분할되어야 함"
    # split_index가 0부터 오름차순인지
    for i, chunk in enumerate(doc1_chunks):
        assert chunk.metadata["split_index"] == i, "split_index는 0부터 순서대로 매겨져야 함"

    # doc2 관련 chunk 확인
    doc2_chunks = [r for r in result if r.metadata["source"] == "B"]
    assert len(doc2_chunks) >= 2, "doc2도 최소 2개의 chunk로 분할되어야 함"
    for i, chunk in enumerate(doc2_chunks):
        assert chunk.metadata["split_index"] == i


def test_chunkify_when_split_result_is_single():
    """
    splitter.split_text() 결과가 1개 이하인 경우(예: 실제 chunk_size가 매우 큰 경우),
    원본 Document 그대로 반환되는지 테스트.
    """
    text = " ".join([f"Token{i}" for i in range(3)])
    doc = Document(page_content=text, metadata={"source": "C"})

    # chunk_size가 매우 커서 어떤 문서도 분할되지 않음
    splitter = DocumentChunkSplitter(chunk_size=100, chunk_overlap=0)
    result = splitter.chunkify([doc])

    assert len(result) == 1, "분할 결과가 1개라면, 원본 문서 그대로 사용."
    assert result[0].page_content == doc.page_content
    assert result[0].metadata == doc.metadata


def test_chunkify_no_documents():
    """
    빈 리스트를 입력했을 때,
    빈 리스트가 반환되는지 테스트.
    """
    splitter = DocumentChunkSplitter()
    result = splitter.chunkify([])
    assert result == [], "빈 문서 리스트를 입력하면 빈 결과가 나와야 함."
