import pandas as pd
import pytest

from src.prepare_data import validate


def make_df(**overrides):
    row = {"pubid": 1, "question": "Does X help?", "context": "Some abstract text.", "label": "yes"}
    row.update(overrides)
    return pd.DataFrame([row])


def test_validate_accepts_good_data():
    assert len(validate(make_df())) == 1


def test_validate_rejects_missing_value():
    with pytest.raises(ValueError, match="Missing"):
        validate(make_df(question=None))


def test_validate_rejects_empty_text():
    with pytest.raises(ValueError, match="Empty"):
        validate(make_df(context="   "))


def test_validate_rejects_unexpected_label():
    with pytest.raises(ValueError, match="Unexpected"):
        validate(make_df(label="definitely"))
