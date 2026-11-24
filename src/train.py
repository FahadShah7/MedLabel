"""Train TF-IDF classifiers, track every run in MLflow, save and register the best by macro-F1."""
import json
import os
from pathlib import Path

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
VERSION = "v2"
LABELS = ["yes", "no", "maybe"]
EXPERIMENT = "medlabel"
REGISTERED_MODEL = "medlabel-classifier"

DEFAULTS = {
    "input": "question+context",
    "analyzer": "word",
    "ngram": (1, 1),
    "max_features": None,
    "class_weight": "balanced",
}
CONFIGS = [
    # Milestone 1 runs
    {"name": "logreg_uni_C1", "model": "logreg", "C": 1.0},
    {"name": "logreg_bi_C1", "model": "logreg", "ngram": (1, 2), "C": 1.0},
    {"name": "logreg_bi_C10", "model": "logreg", "ngram": (1, 2), "C": 10.0},
    {"name": "logreg_bi_C01", "model": "logreg", "ngram": (1, 2), "C": 0.1},
    {"name": "nb_bi_a05", "model": "nb", "ngram": (1, 2), "alpha": 0.5},
    # Milestone 2 runs: one change each from the Milestone 1 winner
    {"name": "logreg_uni_C1_unbalanced", "model": "logreg", "C": 1.0, "class_weight": None},
    {"name": "logreg_uni_C1_5k_features", "model": "logreg", "C": 1.0, "max_features": 5000},
    {"name": "logreg_char_C1", "model": "logreg", "C": 1.0, "analyzer": "char_wb",
     "ngram": (3, 5)},
    {"name": "logreg_uni_C1_question_only", "model": "logreg", "C": 1.0, "input": "question"},
    {"name": "complement_nb_uni_a1", "model": "cnb", "alpha": 1.0},
    {"name": "linear_svm_uni_C1", "model": "svm", "C": 1.0},
]


def make_text(df: pd.DataFrame, input_mode: str = "question+context") -> list[str]:
    # Same convention the API uses: question + " " + context
    if input_mode == "question":
        return df["question"].str.strip().tolist()
    return (df["question"].str.strip() + " " + df["context"].str.strip()).tolist()


def build_pipeline(cfg: dict):
    vec = TfidfVectorizer(
        analyzer=cfg["analyzer"],
        ngram_range=cfg["ngram"],
        max_features=cfg["max_features"],
        sublinear_tf=True,
        min_df=2,
    )
    if cfg["model"] == "logreg":
        clf = LogisticRegression(
            C=cfg["C"], class_weight=cfg["class_weight"], max_iter=1000, random_state=42
        )
    elif cfg["model"] == "svm":
        # Calibration wraps the SVM so it still exposes predict_proba for the API
        svm = LinearSVC(C=cfg["C"], class_weight=cfg["class_weight"], random_state=42)
        clf = CalibratedClassifierCV(svm, cv=5)
    elif cfg["model"] == "cnb":
        clf = ComplementNB(alpha=cfg["alpha"])
    else:
        clf = MultinomialNB(alpha=cfg["alpha"])
    return make_pipeline(vec, clf)


def confusion_figure(y_true, y_pred, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=LABELS, ax=ax, colorbar=False)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    y_train, y_test = train["label"], test["label"]

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT)
    train_ds = mlflow.data.from_pandas(train, source="data/train.csv", name="pubmedqa-train")
    test_ds = mlflow.data.from_pandas(test, source="data/test.csv", name="pubmedqa-test")
    best = {"macro_f1": -1.0}

    for raw_cfg in CONFIGS:
        cfg = {**DEFAULTS, **raw_cfg}
        X_train, X_test = make_text(train, cfg["input"]), make_text(test, cfg["input"])
        pipe = build_pipeline(cfg)
        with mlflow.start_run(run_name=cfg["name"]) as run:
            mlflow.set_tags({"stage": "training", "model_family": cfg["model"]})
            mlflow.log_params({k: str(v) for k, v in cfg.items() if k != "name"})
            mlflow.log_input(train_ds, context="training")
            mlflow.log_input(test_ds, context="evaluation")
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            per_class = f1_score(y_test, preds, labels=LABELS, average=None, zero_division=0)
            metrics = {
                "macro_f1": f1_score(y_test, preds, average="macro", zero_division=0),
                "accuracy": accuracy_score(y_test, preds),
                **{f"f1_{lbl}": float(s) for lbl, s in zip(LABELS, per_class)},
            }
            mlflow.log_metrics(metrics)
            mlflow.log_text(
                classification_report(y_test, preds, labels=LABELS, zero_division=0),
                "classification_report.txt",
            )
            fig = confusion_figure(y_test, preds, cfg["name"])
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)
            model_info = mlflow.sklearn.log_model(
                pipe,
                name="model",
                signature=infer_signature(np.array(X_test[:5]), preds[:5]),
                serialization_format="cloudpickle",  # skops cannot serialize this pipeline
            )
        print(f"{cfg['name']:<28} macro_f1={metrics['macro_f1']:.3f} acc={metrics['accuracy']:.3f}")

        if metrics["macro_f1"] > best["macro_f1"]:
            best = {
                **metrics,
                "config": cfg["name"],
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                "pipeline": pipe,
            }

    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"medlabel_{VERSION}.joblib"
    joblib.dump(best.pop("pipeline"), model_path)

    version = mlflow.register_model(best.pop("model_uri"), REGISTERED_MODEL)
    client = MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", version.version)
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "artifact", str(model_path))
    best["registered_version"] = int(version.version)

    (MODEL_DIR / f"medlabel_{VERSION}_metrics.json").write_text(json.dumps(best, indent=2))
    print(f"\nBest: {best['config']} (macro_f1={best['macro_f1']:.3f}) -> {model_path}")
    print(f"Registered {REGISTERED_MODEL} v{version.version} with alias 'champion'")


if __name__ == "__main__":
    main()
