from unittest.mock import patch, Mock
from docmesh.crawler import WebCrawler


@patch("docmesh.crawler.requests.get")
def test_webcrawler_cross_domain_links(mock_get):
    """
    이 테스트는 예시로, base_url(https://example.com) HTML 내에
    Cross-domain 링크(https://other.com/page)를 담아두고,
    실제 크롤 결과가 해당 외부 링크를 방문하지 않는지 확인합니다.
    """

    # Mock HTML
    mock_html_base = """
    <html>
      <body>
        <p>Base Page</p>
        <a href="https://example.com/page2">Local Link</a>
        <a href="https://other.com/page">External Link</a>
      </body>
    </html>
    """
    mock_html_page2 = """
    <html>
      <body>
        <p>Page2 Content</p>
      </body>
    </html>
    """

    def side_effect(url, timeout=5):
        # "메인 페이지" vs "page2"에 따라 다른 HTML을 반환하며, Content-Type 헤더 추가
        if url == "https://example.com/":
            response = Mock(status_code=200, text=mock_html_base)
            response.headers = {"Content-Type": "text/html"}
            return response
        elif url == "https://example.com/page2":
            response = Mock(status_code=200, text=mock_html_page2)
            response.headers = {"Content-Type": "text/html"}
            return response
        # 외부 도메인 주소나 그 외 URL은 404 처리, Content-Type 헤더 추가
        response = Mock(status_code=404, text="")
        response.headers = {"Content-Type": "text/html"}
        return response

    mock_get.side_effect = side_effect

    # WebCrawler 초기화
    # same_domain_only=True 동작은 LinkExtractor가 기본값으로 처리
    crawler = WebCrawler("https://example.com/", use_max_pages=False, max_pages=5)

    # 크롤 실행
    results = crawler.crawl()

    # 크롤 완료 후, 방문된 URL 목록
    visited_urls = list(crawler.queue.visited)

    # 검증:
    # 1) "https://example.com/" 과 "https://example.com/page2" 만 방문해야 함
    # 2) "https://other.com/page" 는 방문/큐 추가되지 않아야 함
    assert "https://example.com/" in visited_urls
    assert "https://example.com/page2" in visited_urls
    assert "https://other.com/page" not in visited_urls

    # collected 결과에서도 외부 링크를 수집하지 않았는지 확인
    collected_urls = [result["source"] for result in results]
    assert "https://other.com/page" not in collected_urls
    assert len(collected_urls) == 2  # base + page2

    print("Cross-domain link test passed.")
