"""Per-patient explainability with SHAP (with a graceful fallback).

The original project mislabelled *global* feature importances as "SHAP" and
returned the same explanation for every patient. This module computes real
per-prediction SHAP values for the full trained pipeline using SHAP's
model-agnostic explainer (robust across xgboost/shap versions), falling back to
global importances only if SHAP is not installed.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .config import DEFAULT_DATASET, FEATURES
from .logging_conf import get_logger
from .predict import load_pipeline
from .validation import validate_features

logger = get_logger(__name__)

_BACKGROUND_SIZE = 50


@lru_cache(maxsize=1)
def _explainer():
    """Build (and cache) a model-agnostic SHAP explainer over the pipeline.

    Returns ``None`` if SHAP (or the background dataset) is unavailable, so
    callers transparently fall back to global importances.
    """
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed — using global importances fallback")
        return None

    try:
        from .data import load_dataset, split_xy
        background, _ = split_xy(load_dataset(DEFAULT_DATASET))
        background = shap.sample(background, _BACKGROUND_SIZE, random_state=42)
    except Exception as exc:  # missing dataset, etc.
        logger.warning("Could not build SHAP background (%s) — fallback", exc)
        return None

    pipeline = load_pipeline()
    predict_fn = lambda data: pipeline.predict_proba(data)[:, 1]  # noqa: E731
    return shap.Explainer(predict_fn, background, feature_names=FEATURES)


def explain(features: dict, top_k: int = 11) -> list[dict]:
    """Return per-feature contributions for a single patient, ranked by impact.

    Each item: ``{feature, value, contribution, direction}`` where
    ``contribution`` is the signed SHAP value (positive => pushes toward
    'Disease'). Falls back to unsigned global importance if SHAP is unavailable.
    """
    clean = validate_features(features)
    pipeline = load_pipeline()
    explainer = _explainer()

    if explainer is None:
        importances = pipeline.named_steps["clf"].feature_importances_
        contributions = {f: float(v) for f, v in zip(FEATURES, importances, strict=False)}
        signed = False
    else:
        row = pd.DataFrame([clean])[FEATURES]
        shap_values = explainer(row)
        values = np.asarray(shap_values.values).reshape(-1)
        contributions = {f: float(v) for f, v in zip(FEATURES, values, strict=False)}
        signed = True

    explanation = [
        {
            "feature": f,
            "value": clean[f],
            "contribution": round(contributions[f], 4),
            "direction": (
                ("increases risk" if contributions[f] > 0 else "decreases risk")
                if signed else "important"
            ),
        }
        for f in FEATURES
    ]
    explanation.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return explanation[:top_k]
