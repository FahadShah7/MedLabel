"""Passage retrieval over the PubMedQA abstracts: BM25, semantic and hybrid.

- bm25: keyword matching (rank_bm25), needs nothing else.
- semantic: cosine similarity between Gemini embeddings of the question and of each abstract.
  Abstract embeddings are precomputed by `python -m src.build_index`; the question is embedded
  at query time, so this needs GEMINI_API_KEY.
- hybrid: reciprocal rank fusion of the bm25 and semantic rankings.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

METHODS = ("bm25", "semantic", "hybrid")
INDEX_PATH = Path("models/corpus_embeddings.npz")
RRF_K = 60  # standard reciprocal rank fusion constant


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_corpus(data_dir: Path) -> list[dict]:
    """Use the full dataset if it has been downloaded, else the committed sample."""
    files = [p for p in (data_dir / "train.csv", data_dir / "test.csv") if p.exists()]
    if not files:
        files = [data_dir / "sample.csv"]
    df = pd.concat([pd.read_csv(p) for p in files]).drop_duplicates("pubid")
    return df[["pubid", "context"]].to_dict("records")


def normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def load_index(path: Path, pubids: list[int]) -> np.ndarray | None:
    """Embeddings aligned to `pubids`, or None if the index is missing or does not cover them."""
    if not path.exists():
        return None
    data = np.load(path)
    row_of = {int(p): i for i, p in enumerate(data["pubids"])}
    if any(int(p) not in row_of for p in pubids):
        return None
    return normalize(data["embeddings"][[row_of[int(p)] for p in pubids]])


class Retriever:
    def __init__(self, records: list[dict], embeddings: np.ndarray | None = None, embed_query=None):
        """records: dicts with pubid and context. embed_query: text -> vector, for semantic."""
        self.records = records
        self.pubids = np.array([int(r["pubid"]) for r in records])
        self.bm25 = BM25Okapi([tokenize(r["context"]) for r in records]) if records else None
        self.embeddings = embeddings
        self.embed_query = embed_query

    @property
    def semantic_ready(self) -> bool:
        return self.embeddings is not None and self.embed_query is not None

    def scores(self, question: str, method: str, query_vector=None) -> np.ndarray:
        """One relevance score per abstract; higher is better."""
        if method == "bm25":
            tokens = tokenize(question)
            return self.bm25.get_scores(tokens) if tokens else np.zeros(len(self.records))
        if not self.semantic_ready and query_vector is None:
            raise RuntimeError("Semantic search needs the embedding index and a Gemini client.")
        if method == "semantic":
            vec = query_vector if query_vector is not None else self.embed_query(question)
            return self.embeddings @ normalize(vec)
        if method == "hybrid":
            n = len(self.records)
            fused = np.zeros(n)
            for sub in ("bm25", "semantic"):
                ranks = np.empty(n)
                ranks[np.argsort(-self.scores(question, sub, query_vector))] = np.arange(1, n + 1)
                fused += 1.0 / (RRF_K + ranks)
            return fused
        raise ValueError(f"Unknown retrieval method {method!r}")

    def search(
        self, question: str, k: int = 3, method: str = "bm25", exclude_pubid=None, query_vector=None
    ) -> list[dict]:
        if not self.records:
            return []
        scores = self.scores(question, method, query_vector)
        order = np.argsort(-scores, kind="stable")
        if exclude_pubid is not None:
            order = order[self.pubids[order] != int(exclude_pubid)]
        return [
            {
                "pubid": int(self.records[i]["pubid"]),
                "score": round(float(scores[i]), 4),
                "passage": self.records[i]["context"],
            }
            for i in order[:k]
        ]

    def rank_of(self, question: str, pubid: int, method: str, query_vector=None) -> int:
        """1-based rank of a given abstract for this question."""
        order = np.argsort(-self.scores(question, method, query_vector), kind="stable")
        return int(np.where(self.pubids[order] == int(pubid))[0][0]) + 1
