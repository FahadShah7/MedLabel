"""Gemini endpoints and helpers, tested with a fake client so no API key or network is needed."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src import llm
from src.app import app, state
from src.retrieval import Retriever


class FakeGemini:
    def __init__(self, generated_text=None):
        self.decision_calls = []
        self.generate_calls = []
        self.generated_text = generated_text or json.dumps(
            {"label": "yes", "confidence": 0.8, "rationale": "The trial reduced events."}
        )
        self.embed_calls = []
        self.query_vector = None
        self.decisions = SimpleNamespace(create=self._decide)
        self.models = SimpleNamespace(
            generate_content=self._generate, embed_content=self._embed
        )

    def _embed(self, model, contents, config):
        self.embed_calls.append({"model": model, "contents": contents, "config": config})
        vec = self.query_vector if self.query_vector is not None else np.ones(8)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=list(vec)) for _ in contents])

    def _decide(self, model, input, questions):
        self.decision_calls.append({"model": model, "input": input, "questions": questions})
        key, question = next(iter(questions.items()))
        if question["type"] == "score":
            answer = SimpleNamespace(score=2.4, confidence=0.7, probabilities={})
        else:
            answer = SimpleNamespace(
                choice="no", confidence=0.71, probabilities={"no": 0.71, "yes": 0.2, "maybe": 0.09}
            )
        return SimpleNamespace(answers={key: answer}, usage=SimpleNamespace(total_input_tokens=42))

    def _generate(self, model, contents, config):
        self.generate_calls.append({"model": model, "contents": contents, "config": config})
        usage = SimpleNamespace(prompt_token_count=120, candidates_token_count=30)
        return SimpleNamespace(text=self.generated_text, usage_metadata=usage)


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def fake(client):
    fake = FakeGemini()
    state["gemini"] = fake
    yield fake
    state.pop("gemini", None)


def test_decide_returns_calibrated_choice(client, fake):
    r = client.post("/decide", json={"question": "Does vitamin D reduce depression?"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "no"
    assert body["model"] == "gemini-decision-1"
    assert body["context_source"] == "retrieved"
    assert set(body["probabilities"]) == {"yes", "no", "maybe"}
    question = fake.decision_calls[0]["questions"]["answer"]
    assert question["type"] == "choice"
    assert set(question["criteria"]) == {"yes", "no", "maybe"}


def test_decide_uses_provided_context(client, fake):
    r = client.post(
        "/decide", json={"question": "Does exercise help?", "context": "Exercise helped."}
    )
    assert r.json()["context_source"] == "provided"
    assert "Exercise helped." in fake.decision_calls[0]["input"]


def test_answer_returns_rationale_and_passages(client, fake):
    r = client.post("/answer", json={"question": "Does smoking cause lung cancer?"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "yes"
    assert body["rationale"]
    assert len(body["passages"]) == 3
    assert fake.generate_calls[0]["config"].temperature == 0.0


def test_answer_passes_temperature_and_model(client, fake):
    payload = {"question": "Does smoking cause cancer?", "temperature": 1.3,
               "model": "gemini-3.8-flash"}
    assert client.post("/answer", json=payload).status_code == 200
    call = fake.generate_calls[0]
    assert call["model"] == "gemini-3.8-flash"
    assert call["config"].temperature == 1.3


def test_answer_rejects_out_of_range_temperature(client, fake):
    payload = {"question": "Does smoking cause cancer?", "temperature": 3}
    assert client.post("/answer", json=payload).status_code == 422


def test_answer_rejects_unknown_model(client, fake):
    payload = {"question": "Does smoking cause cancer?", "model": "gpt-4"}
    assert client.post("/answer", json=payload).status_code == 422


def test_answer_invalid_model_output_returns_502(client):
    state["gemini"] = FakeGemini(generated_text='{"label": "perhaps"}')
    try:
        r = client.post("/answer", json={"question": "Does smoking cause cancer?"})
        assert r.status_code == 502
    finally:
        state.pop("gemini", None)


def test_gemini_endpoints_return_503_without_key(client):
    state.pop("gemini", None)
    assert client.post("/decide", json={"question": "Does X help Y?"}).status_code == 503
    assert client.post("/answer", json={"question": "Does X help Y?"}).status_code == 503


def test_prompt_variants_change_the_request():
    fake = FakeGemini()
    llm.decide(fake, "Does X help?", "ctx", prompt_id="plain")
    llm.decide(fake, "Does X help?", "ctx", prompt_id="expert")
    plain, expert = (c["questions"]["answer"] for c in fake.decision_calls)
    assert plain["criteria"]["yes"] == ""
    assert expert["criteria"]["yes"]
    assert "expert" in expert["instructions"].lower()


def test_judge_rationale_normalises_score():
    out = llm.judge_rationale(FakeGemini(), "abstract", "rationale")
    assert out["score"] == pytest.approx(2.4 / 3)


@pytest.fixture()
def semantic_index(client):
    """Random embeddings for the loaded corpus, replacing the real index for the test."""
    retriever = state["retriever"]
    saved = retriever.embeddings
    rng = np.random.default_rng(0)
    retriever.embeddings = rng.normal(size=(len(retriever.records), 8)).astype(np.float32)
    retriever.embeddings /= np.linalg.norm(retriever.embeddings, axis=1, keepdims=True)
    yield retriever
    retriever.embeddings = saved


def test_semantic_search_returns_nearest_abstract(client, fake, semantic_index):
    fake.query_vector = semantic_index.embeddings[4] * 2
    r = client.post(
        "/explain",
        json={"question": "Does smoking cause lung cancer?", "retrieval": "semantic", "k": 2},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["retrieval"] == "semantic"
    assert len(body["passages"]) == 2
    assert body["passages"][0]["pubid"] == int(semantic_index.records[4]["pubid"])
    assert body["passages"][0]["score"] == pytest.approx(1.0, abs=1e-3)
    assert fake.embed_calls[0]["config"].task_type == "RETRIEVAL_QUERY"


def test_hybrid_retrieval_feeds_decide(client, fake, semantic_index):
    r = client.post("/decide", json={"question": "Does vitamin D help?", "retrieval": "hybrid"})
    assert r.status_code == 200
    assert r.json()["context_source"] == "retrieved"
    assert len(fake.embed_calls) == 1


def test_semantic_search_needs_key(client, semantic_index):
    state.pop("gemini", None)
    r = client.post("/explain", json={"question": "Does X help Y?", "retrieval": "semantic"})
    assert r.status_code == 503


def test_semantic_search_needs_index(client, fake):
    retriever = state["retriever"]
    saved, retriever.embeddings = retriever.embeddings, None
    try:
        r = client.post("/explain", json={"question": "Does X help Y?", "retrieval": "hybrid"})
        assert r.status_code == 503
    finally:
        retriever.embeddings = saved


def test_unknown_retrieval_method_rejected(client):
    r = client.post("/explain", json={"question": "Does X help Y?", "retrieval": "magic"})
    assert r.status_code == 422


def test_hybrid_fuses_rankings_and_excludes_own_abstract():
    records = [
        {"pubid": 1, "context": "aspirin aspirin prevention trial"},
        {"pubid": 3, "context": "aspirin bleeding risk elderly"},
        {"pubid": 4, "context": "vitamin bone density"},
        {"pubid": 5, "context": "insulin diabetes control"},
        {"pubid": 2, "context": "statin cholesterol outcomes"},
    ]
    embeddings = np.array(
        [[1, 0], [0.6, 0.8], [0.9, 0.436], [0.95, 0.312], [0, 1]], dtype=np.float32
    )
    retriever = Retriever(records, embeddings=embeddings)
    query = np.array([0.0, 1.0])
    keyword = retriever.search("aspirin", k=2, method="bm25")
    assert [p["pubid"] for p in keyword] == [1, 3]
    semantic = retriever.search("aspirin", k=2, method="semantic", query_vector=query)
    assert [p["pubid"] for p in semantic] == [2, 3]
    hybrid = retriever.search("aspirin", k=3, method="hybrid", query_vector=query)
    assert hybrid[0]["pubid"] == 3  # second in both rankings beats first in only one
    excluded = retriever.search(
        "aspirin", k=3, method="hybrid", exclude_pubid=3, query_vector=query
    )
    assert 3 not in [p["pubid"] for p in excluded]
    assert retriever.rank_of("aspirin", 2, "semantic", query_vector=query) == 1


def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(llm.GeminiNotConfigured):
        llm.get_client()
