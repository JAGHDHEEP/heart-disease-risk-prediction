"""Tests for input validation — the project's first line of defence."""
import pandas as pd
import pytest

from heart.validation import ValidationError, validate_dataframe, validate_features


def test_valid_patient_passes(valid_patient):
    clean = validate_features(valid_patient)
    assert clean["age"] == 63.0
    assert set(clean) == set(valid_patient)


def test_missing_feature_raises(valid_patient):
    del valid_patient["chol"]
    with pytest.raises(ValidationError, match="Missing required feature"):
        validate_features(valid_patient)


def test_non_numeric_raises(valid_patient):
    valid_patient["age"] = "old"
    with pytest.raises(ValidationError, match="must be numeric"):
        validate_features(valid_patient)


def test_out_of_range_raises(valid_patient):
    valid_patient["age"] = 900
    with pytest.raises(ValidationError, match="outside the plausible range"):
        validate_features(valid_patient)


def test_dataframe_missing_column(valid_patient):
    df = pd.DataFrame([valid_patient]).drop(columns=["chol"])
    with pytest.raises(ValidationError, match="missing column"):
        validate_dataframe(df)


def test_dataframe_roundtrip(valid_patient):
    df = pd.DataFrame([valid_patient, valid_patient])
    rows = validate_dataframe(df)
    assert len(rows) == 2
