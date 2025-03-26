import os
import tempfile
from unittest.mock import patch, MagicMock
from docmesh.extractor.FileSavingCrawler import FileSavingCrawler


@patch("docmesh.extractor.tools.requests.get")
def test_crawl_single_page_html(mock_get):
    """
    1. start_url 만 있는 경우
    2. HTML 응답이 오면 파일에 저장
    3. 링크는 없으므로 BFS 확장 X
    4. 결과는 방문 URL 1개, 파일 1개
    """
    # (1) mock requests response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "<html><body>Single Page Test</body></html>"
    mock_resp.headers = {"Content-Type": "text/html"}
    mock_get.return_value = mock_resp
    mock_resp.url = "https://example.com"

    with tempfile.TemporaryDirectory() as tmpdir:
        crawler = FileSavingCrawler(
            start_url="https://example.com",
            save_path=tmpdir,
            same_domain_only=True,
            excluded_patterns=[],
            use_max_pages=True,
            max_pages=5,
        )

        # (3) run crawl
        results = crawler.crawl()

        # (4) verify
        assert len(results) == 1
        assert results[0]["url"] == "https://example.com"
        file_path = results[0]["file_path"]
        assert os.path.exists(file_path)

        # 파일 내용 확인
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            assert "Single Page Test" in content

        # visited / to_visit
        assert len(crawler.visited) == 1
        assert crawler.start_url in crawler.visited


@patch("docmesh.extractor.tools.requests.get")
def test_crawl_non_html_page(mock_get):
    """
    1. 응답이 PDF 등 text/html이 아니면 파일 저장을 하지 않음
    2. results가 빈 리스트가 되어야 함
    """
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = "%PDF-1.4 (fake pdf data)"
    mock_resp.headers = {"Content-Type": "application/pdf"}
    mock_get.return_value = mock_resp

    with tempfile.TemporaryDirectory() as tmpdir:
        crawler = FileSavingCrawler(
            start_url="https://example.com/pdf",
            save_path=tmpdir,
            same_domain_only=True,
            excluded_patterns=[],
            use_max_pages=True,
            max_pages=5,
        )
        results = crawler.crawl()
        assert len(results) == 0
        assert len(crawler.visited) == 1  # 방문은 했으나 HTML이 아니므로 저장 X


@patch("docmesh.extractor.tools.requests.get")
def test_crawl_bfs_multi_pages(mock_get):
    """
    여러 링크가 있는 페이지를 BFS로 순회하는 시나리오:
    - page1 -> links to page2, page3
    - page2 -> no further links
    - page3 -> no further links
    - 모두 같은 도메인
    """
    # 가짜 HTML들
    html_page1 = """
    <html>
      <body>
        <a href="/page2">Page2</a>
        <a href="/page3">Page3</a>
      </body>
    </html>
    """
    html_page2 = "<html><body>Page 2 content</body></html>"
    html_page3 = "<html><body>Page 3 content</body></html>"

    def side_effect(url, timeout=5):
        # url에 따라 응답을 달리하기
        if url == "https://example.com":
            resp = MagicMock()
            resp.status_code = 200
            resp.text = html_page1
            resp.headers = {"Content-Type": "text/html"}
            return resp
        elif url == "https://example.com/page2":
            resp = MagicMock()
            resp.status_code = 200
            resp.text = html_page2
            resp.headers = {"Content-Type": "text/html"}
            return resp
        elif url == "https://example.com/page3":
            resp = MagicMock()
            resp.status_code = 200
            resp.text = html_page3
            resp.headers = {"Content-Type": "text/html"}
            return resp
        else:
            # 알 수 없는 URL => 404
            resp = MagicMock()
            resp.status_code = 404
            return resp

    mock_get.side_effect = side_effect

    with tempfile.TemporaryDirectory() as tmpdir:
        crawler = FileSavingCrawler(
            start_url="https://example.com",
            save_path=tmpdir,
            same_domain_only=True,
            excluded_patterns=[],
            use_max_pages=False,  # 페이지 수 제한 없음
        )

        results = crawler.crawl()
        # page1, page2, page3 방문
        # file_save 3개
        assert len(results) == 3
        visited_urls = set(r["url"] for r in results)
        assert "https://example.com" in visited_urls
        assert "https://example.com/page2" in visited_urls
        assert "https://example.com/page3" in visited_urls


@patch("docmesh.extractor.tools.requests.get")
def test_crawl_excluded_patterns(mock_get):
    """
    excluded_patterns 정규식에 걸리는 링크는 방문/저장 안 함
    """
    page_html = """
    <html>
      <body>
        <a href="/normal">NormalLink</a>
        <a href="/logout">LogoutLink</a>
      </body>
    </html>
    """

    def side_effect(url, timeout=5):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/html"}
        if url == "https://example.com":
            resp.text = page_html
        elif url == "https://example.com/normal":
            resp.text = "<html><body>Normal page</body></html>"
        elif url == "https://example.com/logout":
            resp.text = "<html><body>Logout page</body></html>"
        return resp

    mock_get.side_effect = side_effect

    with tempfile.TemporaryDirectory() as tmpdir:
        # logout에 해당하는 링크는 제외
        crawler = FileSavingCrawler(
            start_url="https://example.com",
            save_path=tmpdir,
            same_domain_only=True,
            excluded_patterns=[r"logout"],
            use_max_pages=False,
        )
        results = crawler.crawl()
        assert len(results) == 2
        # visited: [example.com, /normal]
        # /logout 은 제외
        visited_urls = set(r["url"] for r in results)
        assert "https://example.com/logout" not in visited_urls
        assert "https://example.com/normal" in visited_urls


@patch("docmesh.extractor.tools.requests.get")
def test_crawl_max_pages(mock_get):
    """
    max_pages=2 지정 시, 2개 페이지 방문 후 중단하는지 확인
    """
    html_index = """
    <html><body>
      <a href="/page2">Page2</a>
      <a href="/page3">Page3</a>
    </body></html>
    """
    html_page2 = "<html><body>Page 2 content</body></html>"
    html_page3 = "<html><body>Page 3 content</body></html>"

    def side_effect(url, timeout=5):
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"Content-Type": "text/html"}
        if url == "https://example.com":
            resp.text = html_index
        elif url == "https://example.com/page2":
            resp.text = html_page2
        elif url == "https://example.com/page3":
            resp.text = html_page3
        return resp

    mock_get.side_effect = side_effect

    with tempfile.TemporaryDirectory() as tmpdir:
        crawler = FileSavingCrawler(
            start_url="https://example.com",
            save_path=tmpdir,
            same_domain_only=True,
            excluded_patterns=[],
            use_max_pages=True,
            max_pages=2,
        )
        results = crawler.crawl()
        # 방문 가능한 페이지: index(/page2, /page3)
        # max_pages=2 -> index, page2만 방문하고 끝
        visited_urls = set(r["url"] for r in results)
        assert len(visited_urls) == 2
        assert "https://example.com" in visited_urls
        assert "https://example.com/page2" in visited_urls
        assert "https://example.com/page3" not in visited_urls  # 미방문


@patch("docmesh.extractor.tools.requests.get")
def test_crawl_cycle(mock_get):
    """
    a -> b -> c -> a (순환 구조)일 때, 중복 방문을 막아 무한 루프에 빠지지 않는지 확인.
    """
    # 가짜 HTML 페이지들
    html_a = """<html><body><a href="/b">LinkToB</a></body></html>"""
    html_b = """<html><body><a href="/c">LinkToC</a></body></html>"""
    html_c = """<html><body><a href="/a">LinkToA</a></body></html>"""

    def side_effect(url, timeout=5):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"Content-Type": "text/html"}
        # URL에 따라 HTML 반환
        if url == "https://example.com/a":
            mock_resp.text = html_a
            mock_resp.url = "https://example.com/a"
        elif url == "https://example.com/b":
            mock_resp.text = html_b
            mock_resp.url = "https://example.com/b"
        elif url == "https://example.com/c":
            mock_resp.text = html_c
            mock_resp.url = "https://example.com/c"
        else:
            # 그 외는 404 처리
            mock_resp.status_code = 404
        return mock_resp

    mock_get.side_effect = side_effect

    with tempfile.TemporaryDirectory() as tmpdir:
        # 시작점 a
        crawler = FileSavingCrawler(
            start_url="https://example.com/a",
            save_path=tmpdir,
            same_domain_only=True,
            excluded_patterns=[],
            use_max_pages=False,  # 페이지 제한 없이 순환 검사
        )

        results = crawler.crawl()
        # visited에 a, b, c가 각 1회만 들어가야 함. 무한 루프 X
        assert len(crawler.visited) == 3

        visited_urls = set(r["url"] for r in results)
        assert "https://example.com/a" in visited_urls
        assert "https://example.com/b" in visited_urls
        assert "https://example.com/c" in visited_urls

        # 파일도 3개 생성
        assert len(results) == 3

        # 혹시 to_visit이 비어있는지 확인(추가 링크가 없을 것)
        assert len(crawler.to_visit) == 0

        print("Cycle test passed: no infinite loop with a->b->c->a.")
