"""Input validation shared by the API, web app and inference service.

Centralising validation means every entry point rejects the same bad input the
same way — and unit tests can target one place.
"""
from __future__ import annotations

import pandas as pd

from .config import FEATURES, FEATURE_RANGES


class ValidationError(ValueError):
    """Raised when patient features are missing, non-numeric or out of range."""


def validate_features(features: dict) -> dict:
    """Validate and coerce a single patient's features.

    Returns a clean dict of ``{feature: float}`` limited to FEATURES, or raises
    ``ValidationError`` with a human-readable message.
    """
    if not isinstance(features, dict):
        raise ValidationError("Features must be provided as a mapping.")

    missing = [f for f in FEATURES if f not in features or features[f] in ("", None)]
    if missing:
        raise ValidationError(f"Missing required feature(s): {', '.join(missing)}")

    clean: dict[str, float] = {}
    for f in FEATURES:
        try:
            value = float(features[f])
        except (TypeError, ValueError):
            raise ValidationError(f"Feature '{f}' must be numeric, got {features[f]!r}")

        rng = FEATURE_RANGES[f]
        if not (rng.low <= value <= rng.high):
            raise ValidationError(
                f"Feature '{f}'={value} is outside the plausible range "
                f"[{rng.low}, {rng.high}]"
            )
        clean[f] = value
    return clean


def validate_dataframe(df: pd.DataFrame) -> list[dict]:
    """Validate an uploaded CSV (one or more patient rows).

    Returns a list of clean feature dicts. Raises ``ValidationError`` on the
    first problem found.
    """
    if df.empty:
        raise ValidationError("Uploaded file contains no rows.")

    missing_cols = [c for c in FEATURES if c not in df.columns]
    if missing_cols:
        raise ValidationError(f"CSV is missing column(s): {', '.join(missing_cols)}")

    return [validate_features(row._asdict() if hasattr(row, "_asdict") else dict(row))
            for _, row in df[FEATURES].iterrows()]
