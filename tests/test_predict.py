"""Tests for the inference service (requires a trained model artifact)."""
import pytest

from heart.config import PIPELINE_PATH
from heart.predict import predict

pytestmark = pytest.mark.skipif(
    not PIPELINE_PATH.exists(),
    reason="Model not trained — run `python -m heart.train` first.",
)


def test_predict_structure(valid_patient):
    result = predict(valid_patient)
    assert result["prediction"] in {"Disease", "No Disease"}
    assert 0 <= result["disease_probability"] <= 100
    assert result["risk_level"] in {"Very Low", "Low", "Moderate", "High"}
    # probabilities should sum to ~100
    assert abs(result["disease_probability"] + result["no_disease_probability"] - 100) < 0.1


def test_predict_is_deterministic(valid_patient):
    assert predict(valid_patient) == predict(valid_patient)


def test_high_risk_profile_scores_higher():
    low = {"age": 40, "sex": 0, "cp": 0, "trestbps": 110, "chol": 180, "fbs": 0,
           "restecg": 0, "thalch": 180, "exang": 0, "oldpeak": 0.0, "slope": 0}
    high = {"age": 67, "sex": 1, "cp": 3, "trestbps": 160, "chol": 286, "fbs": 0,
            "restecg": 2, "thalch": 108, "exang": 1, "oldpeak": 2.5, "slope": 2}
    assert predict(high)["disease_probability"] > predict(low)["disease_probability"]
