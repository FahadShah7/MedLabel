import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from src import llm
from src.retrieval import INDEX_PATH, Retriever, load_corpus, load_index

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/medlabel_v2.joblib"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
EMBEDDINGS_PATH = Path(os.getenv("EMBEDDINGS_PATH", str(INDEX_PATH)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("medlabel")

state: dict = {}
stats = {
    "requests": 0,
    "errors": 0,
    "total_latency": 0.0,
    "started_at": time.time(),
    "by_endpoint": {},
    "labels": {},
}


def embed_query(text: str):
    return llm.embed(gemini_client(), [text], task_type="RETRIEVAL_QUERY")[0]


def retrieve(question: str, k: int = 3, method: str = "bm25") -> list[dict]:
    retriever: Retriever = state["retriever"]
    if method != "bm25" and retriever.embeddings is None:
        raise HTTPException(status_code=503, detail="Semantic index is not available")
    try:
        return retriever.search(question, k=k, method=method)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Embedding request failed: %s", exc)
        raise HTTPException(status_code=502, detail="Gemini embedding request failed") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    state["model"] = joblib.load(MODEL_PATH)  # loaded once, at startup
    corpus = load_corpus(DATA_DIR)
    embeddings = load_index(EMBEDDINGS_PATH, [r["pubid"] for r in corpus])
    state["retriever"] = Retriever(corpus, embeddings=embeddings, embed_query=embed_query)
    try:
        state["gemini"] = llm.get_client()
    except llm.GeminiNotConfigured:
        logger.info("GEMINI_API_KEY not set: /decide, /answer and semantic search are disabled")
    logger.info(
        "Loaded %s, corpus of %d abstracts, semantic index %s",
        MODEL_PATH.name,
        len(corpus),
        "loaded" if embeddings is not None else "missing",
    )
    yield
    state.clear()


app = FastAPI(title="MedLabel", lifespan=lifespan)


@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
    path = request.url.path
    stats["requests"] += 1
    stats["total_latency"] += latency
    if response.status_code >= 400:
        stats["errors"] += 1
    ep = stats["by_endpoint"].setdefault(path, {"count": 0, "total_latency": 0.0})
    ep["count"] += 1
    ep["total_latency"] += latency
    response.headers["X-Response-Time-Ms"] = f"{round(latency * 1000, 2)}"
    logger.info(
        json.dumps(
            {
                "method": request.method,
                "path": path,
                "status": response.status_code,
                "latency_ms": round(latency * 1000, 2),
            }
        )
    )
    return response


class QuestionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(..., min_length=5, max_length=1000)
    retrieval: Literal["bm25", "semantic", "hybrid"] = Field(
        default="bm25", description="semantic and hybrid need GEMINI_API_KEY"
    )
    k: int = Field(default=3, ge=1, le=10, description="passages to retrieve")


class PredictRequest(QuestionRequest):
    context: str | None = Field(default=None, max_length=20000)


class AnswerRequest(PredictRequest):
    model: Literal[llm.GENERATIVE_MODELS] = llm.DEFAULT_GENERATIVE_MODEL
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    context_source: str
    probabilities: dict[str, float] = Field(default_factory=dict)


class DecideResponse(PredictResponse):
    model: str


class Passage(BaseModel):
    pubid: int = Field(..., description="PubMed identifier")
    score: float = Field(
        ..., description="BM25 score, cosine similarity or fused rank score, by method"
    )
    passage: str = Field(..., description="Passage context text")


class AnswerResponse(BaseModel):
    label: str
    confidence: float
    rationale: str
    model: str
    context_source: str
    passages: list[Passage] = Field(default_factory=list)


class ExplainResponse(BaseModel):
    question: str
    retrieval: str
    passages: list[Passage]


def resolve_context(req: PredictRequest) -> tuple[str, str, list[dict]]:
    if req.context:
        return req.context.strip(), "provided", []
    passages = retrieve(req.question, k=req.k, method=req.retrieval)
    return " ".join(p["passage"] for p in passages), "retrieved", passages


def count_label(label: str):
    stats["labels"][label] = stats["labels"].get(label, 0) + 1


def gemini_client():
    if "gemini" not in state:
        raise HTTPException(status_code=503, detail="Gemini is not configured (GEMINI_API_KEY)")
    return state["gemini"]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "model_version": MODEL_PATH.stem,
        "corpus_size": len(state["retriever"].records) if "retriever" in state else 0,
        "gemini_configured": "gemini" in state,
        "semantic_index_loaded": "retriever" in state
        and state["retriever"].embeddings is not None,
    }


@app.get("/stats")
def get_stats():
    n = stats["requests"]
    avg = stats["total_latency"] / n if n else 0.0
    return {
        "request_count": n,
        "error_count": stats["errors"],
        "avg_latency_ms": round(avg * 1000, 2),
        "uptime_seconds": round(time.time() - stats["started_at"], 1),
        "by_endpoint": {
            path: {
                "count": ep["count"],
                "avg_latency_ms": round(ep["total_latency"] / ep["count"] * 1000, 2),
            }
            for path, ep in stats["by_endpoint"].items()
        },
        "predicted_labels": stats["labels"],
    }


@app.get("/metrics")
def metrics():
    s = get_stats()
    return {"request_count": s["request_count"], "avg_latency_ms": s["avg_latency_ms"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    context, source, _ = resolve_context(req)
    model = state["model"]
    probs = model.predict_proba([f"{req.question.strip()} {context}"])[0]
    best = int(probs.argmax())
    classes = [str(c) for c in model.classes_]
    prob_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs)}
    count_label(classes[best])
    return PredictResponse(
        label=classes[best],
        confidence=round(float(probs[best]), 4),
        context_source=source,
        probabilities=prob_dict,
    )


@app.post("/decide", response_model=DecideResponse)
def decide(req: PredictRequest):
    """Yes / no / maybe from Gemini Decision 1, with calibrated probabilities."""
    client = gemini_client()
    context, source, _ = resolve_context(req)
    try:
        out = llm.decide(client, req.question, context)
    except Exception as exc:
        logger.warning("Gemini Decision call failed: %s", exc)
        raise HTTPException(status_code=502, detail="Gemini Decision request failed") from exc
    count_label(out["label"])
    return DecideResponse(
        label=out["label"],
        confidence=round(out["confidence"], 4),
        context_source=source,
        probabilities={k: round(v, 4) for k, v in out["probabilities"].items()},
        model=llm.DECISION_MODEL,
    )


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest):
    """Generated answer: label plus a one-sentence rationale grounded in the passages."""
    client = gemini_client()
    context, source, passages = resolve_context(req)
    try:
        out = llm.generate(
            client, req.question, context, model=req.model, temperature=req.temperature
        )
    except Exception as exc:
        logger.warning("Gemini generation failed: %s", exc)
        raise HTTPException(status_code=502, detail="Gemini generation request failed") from exc
    if out["label"] is None:
        raise HTTPException(status_code=502, detail="Gemini returned an invalid answer")
    count_label(out["label"])
    return AnswerResponse(
        label=out["label"],
        confidence=round(out["confidence"], 4),
        rationale=out["rationale"],
        model=req.model,
        context_source=source,
        passages=passages,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: QuestionRequest):
    passages = retrieve(req.question, k=req.k, method=req.retrieval)
    return ExplainResponse(question=req.question, retrieval=req.retrieval, passages=passages)
