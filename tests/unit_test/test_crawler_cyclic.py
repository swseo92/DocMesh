import pytest
from unittest.mock import patch, Mock
from app.crawler import WebCrawler


# ======== CASE 1: a → b → c → a ========
@patch("app.crawler.requests.get")
def test_cyclic_links_case1(mock_get):
    html_map = {
        "https://example.com/a": '<a href="https://example.com/b">b</a>',
        "https://example.com/b": '<a href="https://example.com/c">c</a>',
        "https://example.com/c": '<a href="https://example.com/a">a</a>',
    }

    def get_side_effect(url, timeout):
        mock_resp = Mock(status_code=200)
        mock_resp.text = html_map.get(url, "")
        mock_resp.headers = {"Content-Type": "text/html"}
        return mock_resp

    mock_get.side_effect = get_side_effect

    crawler = WebCrawler("https://example.com/a", use_max_pages=False)
    results = crawler.crawl()

    visited_urls = [result["source"] for result in results]
    assert set(visited_urls) == {"https://example.com/a", "https://example.com/b", "https://example.com/c"}
    # 중복 방문 없이 3개만 크롤링됨


# ======== CASE 2: a → b → a ========
@patch("app.crawler.requests.get")
def test_cyclic_links_case2(mock_get):
    html_map = {
        "https://example.com/a": '<a href="https://example.com/b">b</a>',
        "https://example.com/b": '<a href="https://example.com/a">a</a>',
    }

    def get_side_effect(url, timeout):
        mock_resp = Mock(status_code=200)
        mock_resp.text = html_map.get(url, "")
        mock_resp.headers = {"Content-Type": "text/html"}
        return mock_resp

    mock_get.side_effect = get_side_effect

    crawler = WebCrawler("https://example.com/a", use_max_pages=False)
    results = crawler.crawl()

    visited_urls = [result["source"] for result in results]
    assert set(visited_urls) == {"https://example.com/a", "https://example.com/b"}
    # a는 처음에만 방문되고, b에서 다시 a를 참조해도 중복 방문되지 않음
