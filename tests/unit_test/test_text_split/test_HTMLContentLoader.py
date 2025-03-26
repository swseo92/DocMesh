from langchain.schema import Document
from docmesh.text_split.HTMLContentLoader import HTMLContentLoader


def test_html_content_loader_basic():
    """
    HTMLContentLoader가 HTML 헤더 태그(h1~h6)를 기준으로
    문서를 분할하고, 각각의 Document가 metadata['source']를
    가지고 있는지 확인합니다.
    """
    # 1) 테스트용 HTML 문자열 구성
    sample_html = """
    <html>
        <head><title>Test Title</title></head>
        <body>
            <h1>Heading 1</h1>
            <p>This is a paragraph under Heading 1.</p>

            <h2>Subheading 1.1</h2>
            <p>This is a paragraph under Subheading 1.1.</p>

            <h2>Subheading 1.2</h2>
            <p>This is a paragraph under Subheading 1.2.</p>

            <h1>Heading 2</h1>
            <p>This is a paragraph under Heading 2.</p>
        </body>
    </html>
    """

    # 2) 로더 초기화
    test_url = "http://example.com"
    loader = HTMLContentLoader()

    # 3) 로드(=split) 수행
    documents = loader.load(url=test_url, html=sample_html)

    # 4) 결과 검증
    assert isinstance(documents, list), "documents는 리스트 형태여야 합니다."
    assert all(
        isinstance(doc, Document) for doc in documents
    ), "모든 항목은 langchain.schema.Document여야 합니다."

    # documents가 얼마나 분할되는지는 HTMLHeaderTextSplitter 로직에 따라 달라집니다.
    # h1 / h2 등 헤더마다 분할이 일어나므로, 아래의 예시는 (헤더 하나당 Document 1개) + (본문이 붙은 경우) 형태로 나뉩니다.
    # 실제로 몇 개로 분할되는지는 splitter의 내부 처리에 따라 달라질 수 있으니,
    # 테스트 목적에 맞춰 적절히 assert 값을 조정하세요.
    #
    # 여기서는 4개의 heading이 있으니, 최소 4개 이상의 Document가 생길 것으로 예상합니다.
    #
    # h1: Heading 1
    # h2: Subheading 1.1
    # h2: Subheading 1.2
    # h1: Heading 2
    #
    # 본문이 헤더와 하나의 chunk로 묶이는지, 아니면 별도의 chunk가 되는지는 splitter 옵션에 따라 달라질 수 있습니다.
    # default behavior로는 heading + paragraph가 하나의 chunk로 들어가게 됩니다.
    # 정확한 수치는 변경될 수 있으므로, 우선 4개 이상의 Document 존재를 확인합니다.

    assert (
        len(documents) >= 4
    ), f"헤더 기준으로 최소 4개 이상의 Document로 분할되길 기대했지만, {len(documents)}개가 반환되었습니다."

    # 모든 Document에 metadata['source']가 올바르게 들어있는지 확인
    for doc in documents:
        assert (
            doc.metadata.get("source") == test_url
        ), "metadata['source']가 URL과 일치해야 합니다."

    # 각 문서의 내용에 특정 키워드가 들어 있는지 확인 (예: Heading 1, Subheading 1.1 등)
    # 보통 첫 번째 Document에는 "Heading 1" 문구가 들어갈 것이고,
    # 그 이후 Document에는 "Subheading 1.1", "Subheading 1.2", "Heading 2" 등이 들어가야 할 것입니다.
    # splitter 옵션에 따라 실제 내용이 어떻게 분산되는지는 다를 수 있으므로,
    # 이 부분도 적절히 변경하여 검사할 수 있습니다.

    # 예시로 Heading 1 포함 여부 확인
    assert any(
        "Heading 1" in doc.page_content for doc in documents
    ), "분할된 문서 중 하나는 반드시 'Heading 1'을 포함해야 합니다."

    # 예시로 Subheading 1.1 포함 여부 확인
    assert any(
        "Subheading 1.1" in doc.page_content for doc in documents
    ), "분할된 문서 중 하나는 반드시 'Subheading 1.1'을 포함해야 합니다."

    # 예시로 Heading 2 포함 여부 확인
    assert any(
        "Heading 2" in doc.page_content for doc in documents
    ), "분할된 문서 중 하나는 반드시 'Heading 2'을 포함해야 합니다."

    # 실제 테스트 진행 시, Documents 각각의 content가 어떠한지 출력해보면 도움이 됩니다.
    # 디버깅을 위해 아래 코드를 잠시 추가해볼 수 있습니다. (최종 테스트 코드에서는 제거 가능)
    # for i, doc in enumerate(documents):
    #     print(f"Document {i}:\nContent: {doc.page_content}\nMetadata: {doc.metadata}\n")

    print(
        "테스트 통과: HTMLContentLoader가 의도대로 헤더 단위로 문서를 분할하고, metadata['source']를 정상적으로 설정합니다."
    )
