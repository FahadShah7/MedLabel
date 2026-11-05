"""Download PubMedQA (pqa_labeled), validate it, and write a stratified 80/20 split."""
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

OUT_DIR = Path("data")
DATASET_NAME = "qiaojin/PubMedQA"
DATASET_CONFIG = "pqa_labeled"
VALID_LABELS = {"yes", "no", "maybe"}
SEED = 42
TEST_SIZE = 0.2


def load_raw() -> pd.DataFrame:
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    rows = [
        {
            "pubid": ex["pubid"],
            "question": ex["question"],
            "context": " ".join(ex["context"]["contexts"]),
            "label": ex["final_decision"],
        }
        for ex in ds
    ]
    return pd.DataFrame(rows)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Raise ValueError if the data is unusable; otherwise return it unchanged."""
    required_cols = {"pubid", "question", "context", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[["question", "context", "label"]].isnull().any().any():
        raise ValueError("Missing values found")
    if df["pubid"].duplicated().any():
        raise ValueError("Duplicate pubid values found")
    for col in ("question", "context"):
        if (df[col].str.strip() == "").any():
            raise ValueError(f"Empty text in column '{col}'")
    bad = set(df["label"]) - VALID_LABELS
    if bad:
        raise ValueError(f"Unexpected labels: {bad}")
    return df


def split_and_save(df: pd.DataFrame):
    train, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=SEED
    )
    OUT_DIR.mkdir(exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    df.sample(50, random_state=SEED).to_csv(OUT_DIR / "sample.csv", index=False)
    return train, test


def profile(df: pd.DataFrame) -> dict:
    """Summarise row count, class balance and average text length."""
    return {
        "rows": len(df),
        "label_counts": df["label"].value_counts().to_dict(),
        "avg_question_words": round(df["question"].str.split().str.len().mean(), 1),
        "avg_context_words": round(df["context"].str.split().str.len().mean(), 1),
    }


def main():
    df = validate(load_raw())
    print("profile:", profile(df))
    train, test = split_and_save(df)
    print(f"train={len(train)} test={len(test)}")
    print("train labels:", train["label"].value_counts(normalize=True).round(2).to_dict())
    print("test labels: ", test["label"].value_counts(normalize=True).round(2).to_dict())


if __name__ == "__main__":
    main()
