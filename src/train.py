"""Train TF-IDF classifiers, track every run in MLflow, save the best by macro-F1."""
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
VERSION = "v1"
LABELS = ["yes", "no", "maybe"]

CONFIGS = [
    {"name": "logreg_uni_C1", "model": "logreg", "ngram": (1, 1), "C": 1.0},
    {"name": "logreg_bi_C1", "model": "logreg", "ngram": (1, 2), "C": 1.0},
    {"name": "logreg_bi_C10", "model": "logreg", "ngram": (1, 2), "C": 10.0},
    {"name": "logreg_bi_C01", "model": "logreg", "ngram": (1, 2), "C": 0.1},
    {"name": "nb_bi_a05", "model": "nb", "ngram": (1, 2), "alpha": 0.5},
]


def make_text(df: pd.DataFrame) -> list[str]:
    # Same convention the API uses: question + " " + context
    return (df["question"].str.strip() + " " + df["context"].str.strip()).tolist()


def build_pipeline(cfg: dict):
    vec = TfidfVectorizer(ngram_range=cfg["ngram"], sublinear_tf=True, min_df=2)
    if cfg["model"] == "logreg":
        clf = LogisticRegression(
            C=cfg["C"], class_weight="balanced", max_iter=1000, random_state=42
        )
    else:
        clf = MultinomialNB(alpha=cfg["alpha"])
    return make_pipeline(vec, clf)


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    X_train, y_train = make_text(train), train["label"]
    X_test, y_test = make_text(test), test["label"]

    mlflow.set_experiment("medlabel")
    best = {"macro_f1": -1.0}

    for cfg in CONFIGS:
        pipe = build_pipeline(cfg)
        with mlflow.start_run(run_name=cfg["name"]):
            mlflow.log_params({k: str(v) for k, v in cfg.items() if k != "name"})
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            per_class = f1_score(y_test, preds, labels=LABELS, average=None, zero_division=0)
            metrics = {
                "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
                "accuracy": accuracy_score(y_test, preds),
                **{f"f1_{lbl}": float(s) for lbl, s in zip(LABELS, per_class)},
            }
            mlflow.log_metrics(metrics)
        print(f"{cfg['name']:<16} macro_f1={metrics['macro_f1']:.3f} acc={metrics['accuracy']:.3f}")

        if metrics["macro_f1"] > best["macro_f1"]:
            best = {**metrics, "config": cfg["name"], "pipeline": pipe}

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"medlabel_{VERSION}.joblib"
    joblib.dump(best.pop("pipeline"), model_path)
    (MODEL_DIR / f"medlabel_{VERSION}_metrics.json").write_text(json.dumps(best, indent=2))
    print(f"\nBest: {best['config']} (macro_f1={best['macro_f1']:.3f}) -> {model_path}")


if __name__ == "__main__":
    main()
