"""Inference service.

Loads the fitted Pipeline once (cached) and exposes a clean ``predict`` API.
Because the saved object is a full Pipeline, callers pass *raw* feature values —
imputation and scaling happen internally, identically to training. There is no
separate scaler to keep in sync (a fragile pattern in the original project).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from .config import FEATURES, PIPELINE_PATH
from .logging_conf import get_logger
from .validation import validate_features

logger = get_logger(__name__)


class ModelNotTrainedError(RuntimeError):
    """Raised when no trained pipeline artifact exists on disk."""


@lru_cache(maxsize=1)
def load_pipeline(path: str | Path = PIPELINE_PATH):
    """Load and cache the fitted sklearn Pipeline."""
    path = Path(path)
    if not path.exists():
        raise ModelNotTrainedError(
            f"No trained model at {path}. Run `python -m heart.train` first."
        )
    logger.info("Loading model pipeline from %s", path)
    return joblib.load(path)


def _risk_level(disease_prob: float) -> str:
    if disease_prob >= 0.75:
        return "High"
    if disease_prob >= 0.50:
        return "Moderate"
    if disease_prob >= 0.25:
        return "Low"
    return "Very Low"


def predict(features: dict) -> dict:
    """Predict heart-disease risk for a single patient.

    Args:
        features: mapping of the 11 feature names to numeric values.

    Returns:
        dict with prediction label, probabilities, confidence and risk level.

    Raises:
        ValidationError: if any feature is missing or out of range.
        ModelNotTrainedError: if the model artifact is absent.
    """
    clean = validate_features(features)
    pipeline = load_pipeline()

    row = pd.DataFrame([clean])[FEATURES]
    proba = pipeline.predict_proba(row)[0]
    disease_prob = float(proba[1])
    label = int(disease_prob >= 0.5)

    return {
        "prediction": "Disease" if label else "No Disease",
        "label": label,
        "disease_probability": round(disease_prob * 100, 2),
        "no_disease_probability": round(float(proba[0]) * 100, 2),
        "confidence": round(float(max(proba)) * 100, 2),
        "risk_level": _risk_level(disease_prob),
    }


def predict_batch(rows: list[dict]) -> list[dict]:
    """Vectorised prediction for many patients (used by CSV upload / API batch)."""
    return [predict(r) for r in rows]
