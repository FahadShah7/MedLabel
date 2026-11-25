"""Evaluate Gemini inference on PubMedQA and log every variation to MLflow.

Each sweep changes one thing from a fixed baseline, so its effect can be read directly:
prompts, models, temperature, context and other parameters. The TF-IDF model is scored on
the same questions as a reference point. Gemini answers are cached in .cache/ so a rerun
logs the same results without new API calls.

Usage: python -m src.evaluate_llm [--limit 50] [--sweeps prompts models ...]
"""
import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from src import llm
from src.app import tokenize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = Path("data")
CACHE_PATH = Path(".cache/llm_eval.jsonl")
EXPERIMENT = "medlabel-llm-eval"
BASELINE_MODEL = Path("models/medlabel_v2.joblib")
SEED = 42

GEN_BASE = {
    "model": llm.DEFAULT_GENERATIVE_MODEL,
    "prompt": "criteria",
    "temperature": 0.0,
    "top_p": 0.95,
    "context": "abstract",
    "k": 3,
    "repeats": 1,
}
DEC_BASE = {"model": llm.DECISION_MODEL, "prompt": "criteria", "context": "abstract", "k": 3}
TFIDF_BASE = {"model": "tfidf-logreg", "context": "abstract", "k": 3}


def sweeps() -> dict[str, list[tuple[str, dict]]]:
    """One-factor-at-a-time sweeps from a fixed baseline."""
    gen = lambda **kw: {**GEN_BASE, **kw}  # noqa: E731
    dec = lambda **kw: {**DEC_BASE, **kw}  # noqa: E731
    tfidf = lambda **kw: {**TFIDF_BASE, **kw}  # noqa: E731
    return {
        "prompts": [
            *[(f"gen_prompt_{p}", gen(prompt=p)) for p in llm.PROMPTS],
            *[(f"decision_prompt_{p}", dec(prompt=p)) for p in llm.PROMPTS],
        ],
        "models": [
            ("tfidf_logreg", tfidf()),
            ("gemini_decision_1", dec()),
            *[
                (m.replace("-", "_").replace(".", "_"), gen(model=m))
                for m in llm.EVAL_GENERATIVE_MODELS
            ],
        ],
        "temperature": [
            (f"gen_temp_{t}", gen(temperature=t, repeats=2)) for t in (0.0, 0.5, 1.0, 1.5, 2.0)
        ],
        "context": [
            *[(f"tfidf_context_{c}", tfidf(context=c)) for c in ("none", "retrieved", "abstract")],
            *[(f"decision_context_{c}", dec(context=c)) for c in ("none", "retrieved", "abstract")],
            *[(f"gen_context_{c}", gen(context=c)) for c in ("none", "retrieved", "abstract")],
        ],
        "parameters": [
            *[(f"gen_top_p_{p}", gen(temperature=1.0, top_p=p)) for p in (0.3, 0.95)],
            *[(f"decision_retrieved_k{k}", dec(context="retrieved", k=k)) for k in (1, 3, 5)],
        ],
    }


def engine(cfg: dict) -> str:
    if cfg["model"] == llm.DECISION_MODEL:
        return "decisions_api"
    if cfg["model"] == "tfidf-logreg":
        return "sklearn"
    return "generate_content"


class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, dict] = {}
        self.lock = threading.Lock()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut off by an interrupted run is simply re-requested
                self.data[row["key"]] = row["value"]

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        with self.lock:
            self.data[key] = value
            self.path.parent.mkdir(exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value}) + "\n")


class RateLimiter:
    """Spaces out calls across threads to stay under a requests-per-minute quota."""

    def __init__(self, rpm: float):
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self):
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval
        time.sleep(delay)


GENERATIVE_LIMITER = RateLimiter(0)


class Contexts:
    """Builds the context text for each context mode."""

    def __init__(self, corpus: pd.DataFrame):
        self.corpus = corpus.reset_index(drop=True)
        self.bm25 = BM25Okapi([tokenize(t) for t in self.corpus["context"]])

    def get(self, row, mode: str, k: int) -> str:
        if mode == "abstract":
            return row["context"]
        if mode == "none":
            return ""
        # Retrieved: BM25 top-k, excluding the question's own abstract, to mimic a new question
        scores = self.bm25.get_scores(tokenize(row["question"]))
        order = np.argsort(scores)[::-1]
        picked = [i for i in order if self.corpus.at[i, "pubid"] != row["pubid"]][:k]
        return "\n\n".join(self.corpus.at[i, "context"] for i in picked)


def load_eval_set(limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    test = pd.read_csv(DATA_DIR / "test.csv")
    corpus = pd.concat([pd.read_csv(DATA_DIR / "train.csv"), test]).drop_duplicates("pubid")
    if limit < len(test):
        test, _ = train_test_split(
            test, train_size=limit, stratify=test["label"], random_state=SEED
        )
    return test.reset_index(drop=True), corpus


def predict_one(client, cfg: dict, question: str, context: str, repeat: int, cache: Cache):
    call_cfg = {k: v for k, v in cfg.items() if k not in ("repeats", "k")}
    key = Cache.key("predict", call_cfg, question, context, repeat)
    cached = cache.get(key)
    if cached is not None:
        return {**cached, "cached": True}
    if engine(cfg) == "decisions_api":
        out = llm.decide(client, question, context, prompt_id=cfg["prompt"])
    else:
        GENERATIVE_LIMITER.wait()
        out = llm.generate(
            client,
            question,
            context,
            model=cfg["model"],
            prompt_id=cfg["prompt"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
        )
    cache.put(key, out)
    return {**out, "cached": False}


def judge_one(client, context: str, rationale: str, cache: Cache):
    key = Cache.key("judge", context, rationale)
    cached = cache.get(key)
    if cached is None:
        cached = llm.judge_rationale(client, context, rationale)
        cache.put(key, cached)
    return cached


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 5) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def score(rows: pd.DataFrame) -> dict:
    y_true, y_pred = rows["label"], rows["pred"].fillna("invalid")
    per_class = f1_score(y_true, y_pred, labels=list(llm.LABELS), average=None, zero_division=0)
    valid = rows["pred"].notna()
    metrics = {
        "macro_f1": f1_score(
            y_true, y_pred, labels=list(llm.LABELS), average="macro", zero_division=0
        ),
        "accuracy": accuracy_score(y_true, y_pred),
        **{f"f1_{lbl}": float(s) for lbl, s in zip(llm.LABELS, per_class)},
        "invalid_output_rate": float((~valid).mean()),
        "maybe_prediction_rate": float((y_pred == "maybe").mean()),
        "avg_latency_ms": float(rows["latency_ms"].mean()),
        "p95_latency_ms": float(rows["latency_ms"].quantile(0.95)),
        "avg_input_tokens": float(rows["input_tokens"].mean()),
        "avg_output_tokens": float(rows["output_tokens"].mean()),
    }
    if valid.any() and rows.loc[valid, "confidence"].notna().all():
        conf = rows.loc[valid, "confidence"].to_numpy(float)
        correct = (rows.loc[valid, "pred"] == rows.loc[valid, "label"]).to_numpy(float)
        metrics["mean_confidence"] = float(conf.mean())
        metrics["calibration_error"] = expected_calibration_error(conf, correct)
    return metrics


def confusion_figure(rows: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    labels = list(llm.LABELS) + (["invalid"] if rows["pred"].isna().any() else [])
    ConfusionMatrixDisplay.from_predictions(
        rows["label"], rows["pred"].fillna("invalid"), labels=labels, ax=ax, colorbar=False
    )
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def prompt_text(cfg: dict) -> str:
    if engine(cfg) == "sklearn":
        return "No prompt: TF-IDF features of question + context, logistic regression."
    p = llm.PROMPTS[cfg["prompt"]]
    lines = [f"prompt_id: {cfg['prompt']}", f"system: {p['system']}"]
    lines.append(f"instructions: {p['instructions']}")
    if p["criteria"]:
        lines += [f"criteria[{k}]: {v}" for k, v in llm.LABEL_CRITERIA.items()]
    return "\n".join(lines)


def run_config(name, sweep, cfg, items, contexts, client, cache, baseline, args):
    repeats = cfg.get("repeats", 1)
    jobs = [
        (i, r, row, contexts.get(row, cfg["context"], cfg["k"]))
        for i, row in items.iterrows()
        for r in range(repeats)
    ]

    if engine(cfg) == "sklearn":
        texts = [f"{row['question'].strip()} {ctx}".strip() for _, _, row, ctx in jobs]
        start = time.perf_counter()
        probs = baseline.predict_proba(texts)
        per_item_ms = (time.perf_counter() - start) * 1000 / len(texts)
        classes = list(baseline.classes_)
        outputs = [
            {
                "label": classes[int(p.argmax())],
                "confidence": float(p.max()),
                "rationale": None,
                "latency_ms": per_item_ms,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached": False,
            }
            for p in probs
        ]
    else:
        def call(job):
            _, r, row, ctx = job
            return predict_one(client, cfg, row["question"], ctx, r, cache)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            outputs = list(pool.map(call, jobs))

    records = []
    for (i, r, row, ctx), out in zip(jobs, outputs):
        records.append(
            {
                "pubid": int(row["pubid"]),
                "repeat": r,
                "question": row["question"],
                "label": row["label"],
                "pred": out["label"],
                "confidence": out["confidence"],
                "rationale": out.get("rationale"),
                "latency_ms": out["latency_ms"],
                "input_tokens": out["input_tokens"],
                "output_tokens": out["output_tokens"],
                "cached": out["cached"],
                "context": ctx,
            }
        )
    all_rows = pd.DataFrame(records)
    rows = all_rows[all_rows["repeat"] == 0].reset_index(drop=True)
    metrics = score(rows)

    if repeats > 1:
        agree = all_rows.groupby("pubid")["pred"].agg(lambda s: s.fillna("invalid").nunique() == 1)
        metrics["label_stability"] = float(agree.mean())

    if args.judge and engine(cfg) == "generate_content":
        to_judge = rows[rows["rationale"].fillna("").str.len() > 0]

        def judge(row):
            return judge_one(client, row["context"], row["rationale"], cache)["score"]

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            scores = list(pool.map(judge, [r for _, r in to_judge.iterrows()]))
        rows.loc[to_judge.index, "judge_faithfulness"] = scores
        if scores:
            metrics["judge_faithfulness"] = float(np.mean(scores))

    with mlflow.start_run(run_name=name):
        mlflow.set_tags(
            {
                "sweep": sweep,
                "engine": engine(cfg),
                "mlflow.note.content": f"Sweep '{sweep}': {json.dumps(cfg)}",
            }
        )
        mlflow.log_params({**cfg, "n_questions": len(rows), "eval_seed": SEED})
        mlflow.log_metrics(metrics)
        mlflow.log_text(prompt_text(cfg), "prompt.txt")
        mlflow.log_dict(cfg, "config.json")
        table = rows.drop(columns=["context"]).assign(
            correct=lambda d: d["pred"] == d["label"]
        )
        mlflow.log_table(table, "predictions.json")
        fig = confusion_figure(rows, name)
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        if engine(cfg) != "sklearn":
            for _, row in rows.head(args.trace_n).iterrows():
                with mlflow.start_span(name=name, span_type="LLM") as span:
                    span.set_inputs({"question": row["question"], "context": row["context"][:2000]})
                    span.set_outputs(
                        {"label": row["pred"], "confidence": row["confidence"],
                         "rationale": row["rationale"]}
                    )
                    span.set_attributes(
                        {"model": cfg["model"], "latency_ms": row["latency_ms"],
                         "cached": bool(row["cached"])}
                    )
    print(
        f"{sweep:<12} {name:<28} macro_f1={metrics['macro_f1']:.3f} "
        f"acc={metrics['accuracy']:.3f} invalid={metrics['invalid_output_rate']:.2f}"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50, help="questions from the test set")
    parser.add_argument("--sweeps", nargs="*", default=list(sweeps()), choices=list(sweeps()))
    parser.add_argument("--runs", nargs="*", help="only these run names (default: all in sweeps)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-judge", dest="judge", action="store_false")
    parser.add_argument("--trace-n", type=int, default=3, help="traces logged per run")
    parser.add_argument(
        "--gen-rpm", type=float, default=14, help="generative requests per minute (0 = no limit)"
    )
    args = parser.parse_args()
    GENERATIVE_LIMITER.__init__(args.gen_rpm)

    items, corpus = load_eval_set(args.limit)
    contexts = Contexts(corpus)
    cache = Cache(CACHE_PATH)
    baseline = joblib.load(BASELINE_MODEL)
    selected = [
        (sweep, name, cfg)
        for sweep in args.sweeps
        for name, cfg in sweeps()[sweep]
        if not args.runs or name in args.runs
    ]
    needs_gemini = any(engine(cfg) != "sklearn" for _, _, cfg in selected)
    client = llm.get_client() if needs_gemini else None

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    mix = items["label"].value_counts().to_dict()
    print(f"Evaluating {len(items)} test questions, label mix {mix}")
    for sweep, name, cfg in selected:
        run_config(name, sweep, cfg, items, contexts, client, cache, baseline, args)


if __name__ == "__main__":
    main()
