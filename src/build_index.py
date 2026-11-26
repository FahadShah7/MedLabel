"""Embed every abstract with Gemini and save the semantic search index.

Progress is saved after each batch, so a run stopped by the daily embedding quota (1,000 texts
per key on the free tier) resumes where it left off.

Usage: python -m src.build_index [--max-new N]   (needs GEMINI_API_KEY and the full dataset)
"""
import argparse
import logging
from pathlib import Path

import numpy as np

from src import llm
from src.retrieval import INDEX_PATH, load_corpus

PARTIAL_PATH = Path(".cache/index_partial.npz")
BATCH = 20


def load_partial() -> dict[int, np.ndarray]:
    if not PARTIAL_PATH.exists():
        return {}
    data = np.load(PARTIAL_PATH)
    if str(data["model"]) != llm.EMBEDDING_MODEL:
        return {}
    return dict(zip(data["pubids"].tolist(), data["embeddings"]))


def save_partial(done: dict[int, np.ndarray]):
    PARTIAL_PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        PARTIAL_PATH,
        pubids=np.array(list(done)),
        embeddings=np.stack(list(done.values())),
        model=llm.EMBEDDING_MODEL,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new", type=int, help="embed at most this many abstracts this run")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    corpus = load_corpus(Path("data"))
    done = load_partial()
    todo = [r for r in corpus if int(r["pubid"]) not in done][: args.max_new]
    client = llm.get_client() if todo else None
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        vectors = llm.embed(
            client,
            [r["context"] for r in batch],
            task_type="RETRIEVAL_DOCUMENT",
            batch_size=BATCH,
            tokens_per_minute=25_000,
        )
        done.update({int(r["pubid"]): np.asarray(v, np.float16) for r, v in zip(batch, vectors)})
        save_partial(done)
        print(f"{len(done)}/{len(corpus)} abstracts embedded", flush=True)

    if len(done) < len(corpus):
        print("Index incomplete: rerun to continue (another key or after the quota resets).")
        return
    INDEX_PATH.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        INDEX_PATH,
        pubids=np.array([int(r["pubid"]) for r in corpus]),
        embeddings=np.stack([done[int(r["pubid"])] for r in corpus]),
        model=llm.EMBEDDING_MODEL,
    )
    print(f"Saved {len(corpus)} embeddings to {INDEX_PATH}")


if __name__ == "__main__":
    main()
