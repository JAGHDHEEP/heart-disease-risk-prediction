"""Shared test fixtures."""
import pytest


@pytest.fixture
def valid_patient() -> dict:
    """A well-formed, in-range patient record."""
    return {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalch": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 2,
    }
