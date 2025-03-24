import os
import pytest
from fastapi.testclient import TestClient
import dotenv

dotenv.load_dotenv()


# Dummy WebCrawler: /crawl 호출 시 실제 크롤링 대신 고정된 결과를 반환합니다.
class DummyWebCrawler:
    def __init__(self, url, use_max_pages=True, max_pages=5):
        self.url = url
        self.use_max_pages = use_max_pages
        self.max_pages = max_pages

    def crawl(self):
        return [
            {
                "source": "https://dummy.com",
                "text": "dummy text",
                "metadata": {"source": "https://dummy.com"},
            }
        ]


# 조건에 따라 monkeypatch 적용 여부를 결정하는 fixture.
@pytest.fixture(autouse=True)
def override_crawler(monkeypatch, request):
    # 만약 테스트 함수에 "real_crawl" 마커가 있으면 실제 크롤러를 사용하도록 override하지 않습니다.
    if "real_crawl" not in request.keywords:
        monkeypatch.setattr("docmesh.crawler.WebCrawler", DummyWebCrawler)


@pytest.fixture
def client():
    from docmesh.main import app

    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_crawl_dummy(client):
    # Dummy 크롤러가 동작하는지 확인하는 테스트
    payload = {"url": "https://example.com"}
    response = client.post("/crawl", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data.get("status") == "crawl completed"
    results = json_data.get("results")
    assert isinstance(results, list)
    assert len(results) > 0
    assert results[0]["source"] == "https://dummy.com"
    assert results[0]["text"] == "dummy text"


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY is not set, skipping /ask integration test.",
)
def test_ask(client):
    payload = {"question": "What is the content of the first document?"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert "Sources:" in data["answer"]


# 실제 naver.com URL을 사용하여 크롤러가 정상 동작하는지 확인하는 테스트.
# 이 테스트는 실제 네트워크 호출을 수행하므로, 속도 및 결과가 다를 수 있습니다.
@pytest.mark.real_crawl
def test_crawl_naver(client):
    payload = {"url": "https://naver.com"}
    response = client.post("/crawl", json=payload)
    assert response.status_code == 200
    json_data = response.json()
    assert json_data.get("status") == "crawl completed"
    results = json_data.get("results")
    assert isinstance(results, list)
    # 실제 naver.com의 경우 문서 청크가 없을 수도 있으므로, 결과가 빈 리스트일 가능성도 염두에 두어야 합니다.
    # 여기서는 결과가 list 타입임을 확인합니다.
    # 추가로, 첫 번째 결과의 "source" 값에 "naver.com"이 포함되는지 간단히 체크합니다.
    if results:
        assert "naver.com" in results[0].get("source", "").lower()


if __name__ == "__main__":
    pytest.main()
