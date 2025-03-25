import pytest
import requests
from docmesh.crawler import WebCrawler
from docmesh.text_splitter import DocumentChunkPipeline, default_tokenizer
from langchain.schema import Document as LC_Document

# 더미 HTML: 두 개의 헤더와 본문이 포함되어 있음

content1 = "This is some text for header 1." * 100
content2 = "This is some text for header 2." * 100

dummy_html = f"""
<html>
  <body>
    <h1>Header 1</h1>
    <p>{content1}</p>
    <h2>Header 2</h2>
    <p>{content2}</p>
  </body>
</html>
"""


# Dummy Response 객체: requests.get이 반환할 객체
class DummyResponse:
    def __init__(self, text, headers):
        self.text = text
        self.status_code = 200
        self.headers = headers


# 더미 requests.get: 항상 dummy_html과 Content-Type을 text/html로 반환
def dummy_requests_get(url, timeout):
    return DummyResponse(dummy_html, {"Content-Type": "text/html"})


# 모든 테스트에 대해 requests.get을 dummy_requests_get으로 대체
@pytest.fixture(autouse=True)
def patch_requests_get(monkeypatch):
    monkeypatch.setattr(requests, "get", dummy_requests_get)


def test_crawler_text_splitter_integration():
    # 1. WebCrawler 생성: dummy URL 사용, 최대 페이지 제한 해제
    crawler = WebCrawler("https://dummy.com", use_max_pages=False)
    # 크롤링 실행: num_batch=1로 한 URL만 처리하도록 함
    results = crawler.crawl(num_batch=1)

    # crawler_results는 각 청크를 딕셔너리 형태로 반환해야 합니다.
    assert isinstance(results, list)
    assert len(results) > 0
    for res in results:
        assert "source" in res
        assert "text" in res
        assert "metadata" in res
        # 반환된 source는 dummy URL이어야 합니다.
        assert res["source"] == "https://dummy.com"

    # 2. DocumentChunkPipeline을 사용해 크롤러 결과를 LC_Document 객체로 변환
    pipeline = DocumentChunkPipeline(
        max_tokens=1000, min_tokens=500, desired_overlap=100
    )
    documents = pipeline.process(results)

    # documents는 LC_Document 객체 리스트여야 합니다.
    assert isinstance(documents, list)
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, LC_Document)
        # metadata에 source 정보가 포함되어 있어야 합니다.
        assert "source" in doc.metadata
        assert doc.metadata["source"] == "https://dummy.com"
        # page_content는 문자열이어야 합니다.
        assert isinstance(doc.page_content, str)

    # 3. 추가 검증: 토큰 단위로 분할된 청크의 길이 등 (선택 사항)
    # 예: 첫 번째 Document의 토큰 수가 min_tokens 이상인지 확인
    tokens = default_tokenizer(documents[0].page_content)
    assert len(tokens) >= 500, "첫 번째 청크의 토큰 수가 최소 요구량을 충족하지 않습니다."
