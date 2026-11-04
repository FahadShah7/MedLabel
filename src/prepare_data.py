"""Download PubMedQA (pqa_labeled), validate it, and write a stratified 80/20 split."""
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

OUT_DIR = Path("data")
VALID_LABELS = {"yes", "no", "maybe"}
SEED = 42


def load_raw() -> pd.DataFrame:
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
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
    if df[["question", "context", "label"]].isnull().any().any():
        raise ValueError("Missing values found")
    for col in ("question", "context"):
        if (df[col].str.strip() == "").any():
            raise ValueError(f"Empty text in column '{col}'")
    bad = set(df["label"]) - VALID_LABELS
    if bad:
        raise ValueError(f"Unexpected labels: {bad}")
    return df


def split_and_save(df: pd.DataFrame):
    train, test = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=SEED
    )
    OUT_DIR.mkdir(exist_ok=True)
    train.to_csv(OUT_DIR / "train.csv", index=False)
    test.to_csv(OUT_DIR / "test.csv", index=False)
    df.sample(50, random_state=SEED).to_csv(OUT_DIR / "sample.csv", index=False)
    return train, test


def main():
    df = validate(load_raw())
    train, test = split_and_save(df)
    print(f"train={len(train)} test={len(test)}")
    print("train labels:", train["label"].value_counts(normalize=True).round(2).to_dict())
    print("test labels: ", test["label"].value_counts(normalize=True).round(2).to_dict())


if __name__ == "__main__":
    main()
