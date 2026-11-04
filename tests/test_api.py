import pytest
from fastapi.testclient import TestClient

from src.app import app, state


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # the "with" runs startup, so the model loads
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_model_loaded_with_expected_classes(client):
    assert list(state["model"].classes_) == ["maybe", "no", "yes"]


def test_predict_returns_valid_label(client):
    r = client.post("/predict", json={"question": "Does aspirin reduce the risk of heart attack?"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"yes", "no", "maybe"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["context_source"] == "retrieved"


def test_predict_uses_provided_context(client):
    r = client.post(
        "/predict",
        json={"question": "Does exercise help depression?", "context": "Exercise reduced symptoms."},
    )
    assert r.status_code == 200
    assert r.json()["context_source"] == "provided"


def test_predict_rejects_short_question(client):
    assert client.post("/predict", json={"question": "hi"}).status_code == 422


def test_predict_rejects_missing_question(client):
    assert client.post("/predict", json={}).status_code == 422


def test_explain_returns_three_passages(client):
    r = client.post("/explain", json={"question": "Does smoking cause lung cancer?"})
    assert r.status_code == 200
    assert len(r.json()["passages"]) == 3


def test_metrics_counts_requests(client):
    before = client.get("/metrics").json()["request_count"]
    client.get("/health")
    after = client.get("/metrics").json()["request_count"]
    assert after > before
