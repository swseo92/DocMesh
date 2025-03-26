from unittest.mock import patch, MagicMock

from docmesh.extractor.tools import (
    PageFetcher,
    LinkExtractor,
    LinkQueueManager,
    HTMLContentLoader,
)
from langchain.schema import Document


# =============================================================================
# 1. PageFetcher 테스트
# =============================================================================
@patch("requests.get")
def test_page_fetcher_success(mock_get):
    """정상적인 HTML 응답 시, (text, content-type)을 반환하는지 확인."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html><body>Test</body></html>"
    mock_resp.headers = {"Content-Type": "text/html; charset=UTF-8"}
    mock_get.return_value = mock_resp

    fetcher = PageFetcher()
    result = fetcher.fetch("https://example.com")
    assert result is not None
    html, ctype, url = result
    assert "<html>" in html
    assert "text/html" in ctype.lower()


@patch("requests.get")
def test_page_fetcher_http_error(mock_get):
    """status_code가 200이 아니면 None을 반환하는지 확인."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_get.return_value = mock_resp

    fetcher = PageFetcher()
    result = fetcher.fetch("https://example.com/notfound")
    assert result is None


@patch("requests.get", side_effect=Exception("Network Error"))
def test_page_fetcher_exception(mock_get):
    """예외 발생 시 None을 반환하는지 확인."""
    fetcher = PageFetcher()
    result = fetcher.fetch("https://example.com/error")
    assert result is None


# =============================================================================
# 2. LinkExtractor 테스트
# =============================================================================
def test_link_extractor_same_domain():
    """same_domain_only=True인 경우, 다른 도메인은 제외되는지 확인."""
    html = """
    <html><body>
      <a href="http://example.com/foo">Link1</a>
      <a href="http://another.com/bar">Link2</a>
    </body></html>
    """
    extractor = LinkExtractor()
    links = extractor.extract_links(html, "http://example.com", same_domain_only=True)
    assert len(links) == 1
    assert links[0] == "http://example.com/foo"


def test_link_extractor_all_domains():
    """same_domain_only=False인 경우, 모든 링크를 수집하는지 확인."""
    html = """
    <html><body>
      <a href="http://example.com/foo">Link1</a>
      <a href="http://another.com/bar">Link2</a>
    </body></html>
    """
    extractor = LinkExtractor()
    links = extractor.extract_links(html, "http://example.com", same_domain_only=False)
    assert len(links) == 2
    assert "http://example.com/foo" in links
    assert "http://another.com/bar" in links


# =============================================================================
# 4. LinkQueueManager 테스트
# =============================================================================
def test_link_queue_manager():
    """큐에 URL을 넣고, 순서대로 next()가 동작하는지 테스트."""
    manager = LinkQueueManager("https://example.com")
    assert manager.has_next() is True

    url1 = manager.next()
    assert url1 == "https://example.com"
    assert manager.has_next() is False

    # 새 링크 등록
    manager.add_links(["https://example.com/foo", "https://example.com/bar"])
    assert manager.has_next() is True

    url2 = manager.next()
    assert url2 == "https://example.com/foo"
    assert manager.has_next() is True

    url3 = manager.next()
    assert url3 == "https://example.com/bar"
    assert manager.has_next() is False


def test_link_queue_manager_no_duplicates():
    manager = LinkQueueManager("https://example.com")

    # start_url을 실제로 방문(큐에서 pop)
    first_url = manager.next()
    assert first_url == "https://example.com"
    # 이제 to_visit = [], visited = {"https://example.com"}

    # 중복 테스트: 이미 visited에 있으므로 새로 추가 안 됨
    manager.add_links(["https://example.com", "https://example.com/foo"])
    # "https://example.com"은 visited에 있음 -> 추가 안 됨
    # "https://example.com/foo"만 새로 들어감 -> to_visit = ["https://example.com/foo"]
    assert len(manager.to_visit) == 1

    # 한번 더 next()로 꺼내기
    next_url = manager.next()
    assert next_url == "https://example.com/foo"
    assert not manager.has_next()

    # 다시 중복 테스트
    manager.add_links(["https://example.com/foo"])
    # 이미 visited에 있으므로 -> 추가 안됨
    assert len(manager.to_visit) == 0


# =============================================================================
# 5. HTMLContentLoader 테스트
# =============================================================================
def test_html_content_loader_basic():
    """헤더 분할이 정상적으로 동작하는지 확인."""
    html = """
    <html>
      <h1>Title</h1>
      <p>Paragraph under H1.</p>
      <h2>Subtitle</h2>
      <p>Paragraph under H2.</p>
    </html>
    """
    loader = HTMLContentLoader(url="https://example.com", html=html)
    docs = loader.load()
    # docs는 [Document(page_content=...), Document(page_content=...)] 형태
    assert len(docs) >= 2  # h1, h2 각각에 대한 청크가 나올 것
    all_text = " ".join([d.page_content for d in docs])
    assert "Title" in all_text
    assert "Subtitle" in all_text


def test_html_content_loader_empty():
    """빈 HTML이면 문서 리스트가 빈 채로 리턴되는지 확인."""
    loader = HTMLContentLoader(url="https://example.com", html="")
    docs = loader.load()
    assert len(docs) == 0


def test_html_content_loader_metadata():
    """Document의 metadata['source']에 URL이 들어가는지 확인."""
    html = "<html><h1>Example</h1></html>"
    loader = HTMLContentLoader(url="https://example.com/test", html=html)
    docs = loader.load()
    assert len(docs) > 0
    for doc in docs:
        assert doc.metadata["source"] == "https://example.com/test"
        assert isinstance(doc, Document)
