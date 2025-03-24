from unittest.mock import patch, Mock
from app.crawler import PageFetcher, LinkExtractor, LinkQueueManager, WebCrawler

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


# -------- PageFetcher --------
@patch("app.crawler.requests.get")
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


@patch("app.crawler.requests.get", side_effect=Exception("Timeout"))
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
@patch("app.crawler.requests.get")
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
