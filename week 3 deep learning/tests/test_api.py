import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_valid(client):
    r = client.post("/predict", json={"text": "I was charged twice"})
    assert r.status_code == 200
    data = r.json()
    assert "category" in data
    assert "priority" in data
    assert 0 <= data["confidence"] <= 1


def test_predict_empty_text(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422


def test_predict_missing_field(client):
    r = client.post("/predict", json={})
    assert r.status_code == 422