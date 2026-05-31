"""Tests for dataset loading and the leakage-free contract."""
import pandas as pd
import pytest

from heart.config import DEFAULT_DATASET, FEATURES, TARGET
from heart.data import load_dataset, split_xy


def test_load_dataset_shape():
    df = load_dataset(DEFAULT_DATASET)
    assert set(FEATURES + [TARGET]).issubset(df.columns)
    assert len(df) > 0


def test_target_is_binary():
    df = load_dataset(DEFAULT_DATASET)
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_split_xy_separates_target():
    X, y = split_xy(load_dataset(DEFAULT_DATASET))
    assert TARGET not in X.columns
    assert list(X.columns) == FEATURES
    assert len(X) == len(y)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_dataset("does_not_exist.csv")


def test_uci_string_encoding():
    """String/categorical UCI values should be decoded to integers."""
    raw = pd.DataFrame({
        "age": [60], "sex": ["Male"], "cp": ["asymptomatic"], "trestbps": [140],
        "chol": [250], "fbs": [True], "restecg": ["normal"], "thalch": [120],
        "exang": [False], "oldpeak": [1.0], "slope": ["flat"], "num": [2],
    })
    tmp = pd.DataFrame(raw)
    path = "_tmp_uci.csv"
    tmp.to_csv(path, index=False)
    try:
        df = load_dataset(path)
        assert df.loc[0, "sex"] == 1
        assert df.loc[0, "cp"] == 3
        assert df.loc[0, TARGET] == 1  # num>0 -> disease
    finally:
        import os
        os.remove(path)
