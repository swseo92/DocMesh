# tests/test_main.py
import pytest
from fastapi.testclient import TestClient

# app.main에 FastAPI 인스턴스(app)가 있다고 가정
# 예: from app.main import app
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_crawl_endpoint():
    # /crawl 엔드포인트에 POST 요청
    # "url": "https://example.com" 같은 간단한 URL로 테스트
    payload = {"url": "https://naver.com"}
    response = client.post("/crawl", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert data["status"] == "crawl completed"

def test_ask_endpoint():
    # /ask 엔드포인트에 POST로 질문을 던짐
    payload = {"question": "What is this project about?"}
    response = client.post("/ask", json=payload)
    assert response.status_code == 200

    data = response.json()
    # 예시: {"answer": "..."}
    assert "answer" in data
    # answer가 문자열인지 간단히 확인
    assert isinstance(data["answer"], str)
