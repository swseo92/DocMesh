from unittest.mock import patch, Mock
from docmesh.crawler import (
    PageFetcher,
    LinkExtractor,
    LinkQueueManager,
    WebCrawler,
    main,
    HTMLContentLoader,
)

# 샘플 HTML
sample_html = """
<html>
  <body>
    <p>Hello, world!</p>
    <a href="/page2">Next page</a>
  </body>
</html>
"""

sample_html_2 = """
<html>
  <body>
    <p>This is page 2</p>
  </body>
</html>
"""


def test_main():
    main()


# -------- PageFetcher --------
@patch("docmesh.crawler.requests.get")
def test_page_fetcher_success(mock_get):
    mock_response = Mock(status_code=200, text="Mocked content")
    mock_response.headers = {"Content-Type": "text/html"}
    mock_get.return_value = mock_response

    fetcher = PageFetcher()
    result = fetcher.fetch("https://example.com")
    assert result is not None
    html, content_type = result
    assert html == "Mocked content"
    assert content_type.lower() == "text/html"


@patch("docmesh.crawler.requests.get", side_effect=Exception("Timeout"))
def test_page_fetcher_failure(mock_get):
    fetcher = PageFetcher()
    result = fetcher.fetch("https://example.com")
    assert result is None


# -------- LinkExtractor --------
def test_link_extractor_same_domain():
    extractor = LinkExtractor()
    base_url = "https://example.com"
    links = extractor.extract_links(sample_html, base_url)
    assert "https://example.com/page2" in links


def test_link_extractor_cross_domain_filtered():
    html = '<a href="https://othersite.com/page">Other</a>'
    extractor = LinkExtractor()
    links = extractor.extract_links(html, "https://example.com")
    assert links == []  # 필터링됨


# -------- LinkQueueManager --------
def test_link_queue_manager():
    q = LinkQueueManager("https://example.com")
    assert q.has_next()
    url = q.next()
    assert url == "https://example.com"
    assert url in q.visited

    q.add_links(["https://example.com/page2", "https://example.com/page2"])  # 중복 추가 안됨
    assert q.has_next()
    next_url = q.next()
    assert next_url == "https://example.com/page2"


# -------- WebCrawler (통합 테스트) --------
@patch("docmesh.crawler.requests.get")
def test_webcrawler_crawl(mock_get):
    # 두 페이지 순서로 mock
    def side_effect(url, timeout):
        mock_resp = Mock(status_code=200)
        if "page2" in url:
            mock_resp.text = sample_html_2
        else:
            mock_resp.text = sample_html
        # 모든 응답에 대해 Content-Type을 text/html로 설정
        mock_resp.headers = {"Content-Type": "text/html"}
        return mock_resp

    mock_get.side_effect = side_effect

    crawler = WebCrawler("https://example.com", use_max_pages=False)
    results = crawler.crawl()

    # 결과는 각 청크를 개별 딕셔너리로 저장
    urls = [result["source"] for result in results]
    assert "https://example.com" in urls
    assert "https://example.com/page2" in urls

    texts = [result["text"] for result in results]
    assert any("Hello" in t for t in texts)
    assert any("page 2" in t for t in texts)

    # 메타데이터가 함께 포함되었는지 확인
    for result in results:
        assert "metadata" in result
        assert result["metadata"].get("source") == result["source"]


# --- Test 1: HTMLContentLoader의 else 분기 검증 ---
def test_html_content_loader_else_branch(monkeypatch):
    test_url = "https://test.com"
    # 단순 HTML 예시; 여기서는 HTMLHeaderTextSplitter가 plain 문자열 리스트를 반환하도록 할 예정
    test_html = "<html><body><p>Paragraph one.</p><p>Paragraph two.</p></body></html>"

    # Fake split_text: 항상 plain 문자열 리스트를 반환하여, else 분기가 실행되도록 함.
    def fake_split_text(self, html: str):
        # plain 문자열 청크들을 반환
        return ["Fake chunk 1", "Fake chunk 2"]

    # monkeypatch: HTMLHeaderTextSplitter.split_text를 fake_split_text로 대체
    monkeypatch.setattr(
        "docmesh.crawler.HTMLHeaderTextSplitter.split_text", fake_split_text
    )

    loader = HTMLContentLoader(test_url, test_html)
    documents = loader.load()

    # load() 함수에서는 else 분기를 통해 새 Document 객체를 생성합니다.
    # 즉, 반환된 객체들은 langchain.schema.Document 타입이어야 합니다.
    from langchain.schema import Document as LC_Document

    assert len(documents) == 2, "반환된 청크 수가 예상과 다릅니다."
    for doc in documents:
        assert isinstance(doc, LC_Document), "반환된 객체가 langchain Document가 아닙니다."
        assert (
            doc.metadata.get("source") == test_url
        ), "metadata에 올바른 source가 설정되지 않았습니다."


# --- Test 2: WebCrawler의 max_pages 조건 검증 ---
def test_webcrawler_max_pages_stop(monkeypatch):
    # 가짜 PageFetcher: 항상 간단한 HTML과 "text/html" Content-Type을 반환
    class FakePageFetcher(PageFetcher):
        def fetch(self, url: str):
            fake_html = "<html><body><p>Dummy content.</p><a href='https://test.com/next'>Link</a></body></html>"
            return (fake_html, "text/html")

    # 가짜 LinkExtractor: 항상 새로운 링크(예: 기존 URL에 "-next"를 추가)를 반환
    class FakeLinkExtractor(LinkExtractor):
        def extract_links(
            self, html: str, base_url: str, same_domain_only: bool = True
        ):
            # 항상 base_url 뒤에 "-next"를 붙인 링크를 반환하여 큐에 추가
            return [base_url + "-next"]

    # WebCrawler 생성: max_pages를 2로 지정 (2페이지 방문 후 중단되어야 함)
    crawler = WebCrawler("https://test.com", use_max_pages=True, max_pages=2)

    # 가짜 PageFetcher와 LinkExtractor로 대체
    crawler.fetcher = FakePageFetcher()
    crawler.link_extractor = FakeLinkExtractor()

    results = crawler.crawl()

    # visited 집합의 크기가 max_pages(2) 이하인지 확인
    assert (
        len(crawler.queue.visited) <= 2
    ), f"방문한 URL 수가 max_pages를 초과했습니다: {len(crawler.queue.visited)}"

    # 결과가 반환되었는지 확인 (최소한 1페이지 이상)
    assert len(results) > 0, "크롤링 결과가 없습니다."

    # 디버그 출력 (옵션)
    for res in results:
        print(f"Visited: {res['source']}, Excerpt: {res['text'][:50]}")
