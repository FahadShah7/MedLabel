import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field
from rank_bm25 import BM25Okapi

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/medlabel_v1.joblib"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

state: dict = {}
stats = {"requests": 0, "total_latency": 0.0}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_corpus(data_dir: Path) -> list[dict]:
    """Use the full dataset if it has been downloaded, else the committed sample."""
    files = [p for p in (data_dir / "train.csv", data_dir / "test.csv") if p.exists()]
    if not files:
        files = [data_dir / "sample.csv"]
    df = pd.concat([pd.read_csv(p) for p in files]).drop_duplicates("pubid")
    return df[["pubid", "context"]].to_dict("records")


def retrieve(question: str, k: int = 3) -> list[dict]:
    tokens = tokenize(question)
    if not tokens or "bm25" not in state or not state.get("corpus"):
        return []
    scores = state["bm25"].get_scores(tokens)
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {
            "pubid": int(state["corpus"][i]["pubid"]),
            "score": round(float(scores[i]), 3),
            "passage": state["corpus"][i]["context"],
        }
        for i in top
    ]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact not found at {MODEL_PATH}")
    state["model"] = joblib.load(MODEL_PATH)  # loaded once, at startup
    state["corpus"] = load_corpus(DATA_DIR)
    state["bm25"] = BM25Okapi([tokenize(r["context"]) for r in state["corpus"]])
    yield
    state.clear()


app = FastAPI(title="MedLabel", lifespan=lifespan)


@app.middleware("http")
async def track_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = time.perf_counter() - start
    stats["requests"] += 1
    stats["total_latency"] += latency
    response.headers["X-Response-Time-Ms"] = f"{round(latency * 1000, 2)}"
    return response


class QuestionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    question: str = Field(..., min_length=5, max_length=1000)


class PredictRequest(QuestionRequest):
    context: str | None = Field(default=None, max_length=20000)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    context_source: str
    probabilities: dict[str, float] = Field(default_factory=dict)


class Passage(BaseModel):
    pubid: int = Field(..., description="PubMed identifier")
    score: float = Field(..., description="BM25 relevance score")
    passage: str = Field(..., description="Passage context text")


class ExplainResponse(BaseModel):
    question: str
    passages: list[Passage]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.get("/metrics")
def metrics():
    n = stats["requests"]
    avg = stats["total_latency"] / n if n else 0.0
    return {"request_count": n, "avg_latency_ms": round(avg * 1000, 2)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if req.context:
        context, source = req.context.strip(), "provided"
    else:
        passages = retrieve(req.question)
        context = " ".join(p["passage"] for p in passages) if passages else ""
        source = "retrieved"
    model = state["model"]
    probs = model.predict_proba([f"{req.question.strip()} {context}"])[0]
    best = int(probs.argmax())
    classes = [str(c) for c in model.classes_]
    prob_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs)}
    return PredictResponse(
        label=str(model.classes_[best]),
        confidence=round(float(probs[best]), 4),
        context_source=source,
        probabilities=prob_dict,
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(req: QuestionRequest):
    return ExplainResponse(question=req.question, passages=retrieve(req.question))
