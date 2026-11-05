import pandas as pd
import pytest

from src import prepare_data
from src.prepare_data import split_and_save, validate


def make_df(**overrides):
    row = {"pubid": 1, "question": "Does X help?", "context": "Some abstract text.", "label": "yes"}
    row.update(overrides)
    return pd.DataFrame([row])


def test_validate_accepts_good_data():
    assert len(validate(make_df())) == 1


def test_validate_rejects_missing_column():
    df = make_df().drop(columns=["context"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate(df)


def test_validate_rejects_missing_value():
    with pytest.raises(ValueError, match="Missing"):
        validate(make_df(question=None))


def test_validate_rejects_duplicate_pubid():
    df = pd.concat([make_df(), make_df()], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        validate(df)


def test_validate_rejects_empty_text():
    with pytest.raises(ValueError, match="Empty"):
        validate(make_df(context="   "))


def test_validate_rejects_unexpected_label():
    with pytest.raises(ValueError, match="Unexpected"):
        validate(make_df(label="definitely"))


def test_split_is_stratified_and_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_data, "OUT_DIR", tmp_path)
    labels = ["yes"] * 60 + ["no"] * 30 + ["maybe"] * 10
    df = pd.DataFrame({"pubid": range(100), "question": "q", "context": "c", "label": labels})
    train, test = split_and_save(df)
    assert len(train) == 80 and len(test) == 20
    assert test["label"].value_counts().to_dict() == {"yes": 12, "no": 6, "maybe": 2}
    train_again, _ = split_and_save(df)
    assert list(train["pubid"]) == list(train_again["pubid"])
