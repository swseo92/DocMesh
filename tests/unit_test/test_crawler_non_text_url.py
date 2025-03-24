from unittest.mock import patch, Mock
from app.crawler import WebCrawler


@patch("app.crawler.requests.get")
def test_webcrawler_non_html(mock_get):
    # Mock 응답 구성
    start_html = """
    <html>
      <body>
        <a href="https://example.com/pdf">PDF Link</a>
        <a href="https://example.com/empty">Empty Link</a>
        <a href="https://example.com/normal">Normal Link</a>
      </body>
    </html>
    """
    # PDF 응답: PDF 콘텐츠, Content-Type이 application/pdf
    pdf_content = b"%PDF-1.4 some pdf content"
    # Empty 응답: 빈 HTML, Content-Type은 text/html
    empty_html = ""
    # Normal HTML 응답: 본문에 의미 있는 텍스트
    normal_html = """
    <html>
      <body>
        <p>Normal Content</p>
      </body>
    </html>
    """

    # 각 URL에 대해 반환할 데이터 매핑
    response_map = {
        "https://example.com": {
            "status_code": 200,
            "text": start_html,
            "content_type": "text/html",
        },
        "https://example.com/pdf": {
            "status_code": 200,
            "text": pdf_content.decode("utf-8", errors="ignore"),
            "content_type": "application/pdf",
        },
        "https://example.com/empty": {
            "status_code": 200,
            "text": empty_html,
            "content_type": "text/html",
        },
        "https://example.com/normal": {
            "status_code": 200,
            "text": normal_html,
            "content_type": "text/html",
        },
    }

    def side_effect(url, timeout=5):
        data = response_map.get(url)
        if not data:
            return Mock(status_code=404)
        mock_resp = Mock()
        mock_resp.status_code = data["status_code"]
        mock_resp.text = data["text"]
        mock_resp.headers = {"Content-Type": data["content_type"]}
        return mock_resp

    mock_get.side_effect = side_effect

    # WebCrawler 초기화; use_max_pages=False로 무한 크롤링(제한 없이)
    crawler = WebCrawler("https://example.com", use_max_pages=False)
    results = crawler.crawl()

    # 크롤 결과에 포함된 URL 추출
    collected_urls = [result["source"] for result in results]

    # 검증:
    # - PDF와 Empty 페이지는 Content-Type이나 텍스트 길이 조건에 의해 수집되지 않아야 함
    # - 시작 페이지("https://example.com")와 정상 페이지("https://example.com/normal")는 수집되어야 함
    assert "https://example.com/pdf" not in collected_urls
    assert "https://example.com/empty" not in collected_urls
    assert "https://example.com" in collected_urls
    assert "https://example.com/normal" in collected_urls

    print("Non-HTML / Empty document test passed.")
