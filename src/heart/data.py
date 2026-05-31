"""Dataset loading and validation.

Handles both the bundled Cleveland CSV (already a binary ``target``) and the
combined UCI dataset (multi-class ``num`` column), normalising either into a
clean ``(X, y)`` pair. Crucially, **no imputation or scaling happens here** —
those steps live inside the modelling Pipeline so they are fit on training
folds only (see ``pipeline.py``). This is what prevents data leakage.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import FEATURES, TARGET
from .logging_conf import get_logger

logger = get_logger(__name__)

# Maps the categorical UCI encoding to the integer encoding the model expects.
_UCI_MAPS = {
    "sex": {"Male": 1, "Female": 0},
    "cp": {"typical angina": 0, "atypical angina": 1, "non-anginal": 2, "asymptomatic": 3},
    "fbs": {True: 1, False: 0, "TRUE": 1, "FALSE": 0},
    "restecg": {"normal": 0, "st-t abnormality": 1, "lv hypertrophy": 2},
    "exang": {True: 1, False: 0, "TRUE": 1, "FALSE": 0},
    "slope": {"upsloping": 0, "flat": 1, "downsloping": 2},
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a heart-disease CSV and return a frame with FEATURES + binary TARGET.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if required feature columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    df = df.rename(columns={"thalach": "thalch"})  # UCI/Cleveland naming drift
    logger.info("Loaded %d rows from %s", len(df), path.name)

    # Derive a binary target from whichever column is present.
    if TARGET not in df.columns:
        if "num" in df.columns:
            df[TARGET] = (df["num"] > 0).astype(int)
        else:
            raise ValueError("Dataset has neither a 'target' nor a 'num' column")
    df[TARGET] = (df[TARGET] > 0).astype(int)

    # Decode categorical UCI values where they appear as strings/bools.
    for col, mapping in _UCI_MAPS.items():
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(mapping)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required feature columns: {missing}")

    # Coerce features to numeric; unparseable values become NaN and are imputed
    # later *inside the pipeline* (no leakage).
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[FEATURES + [TARGET]]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a loaded frame into the feature matrix X and target vector y."""
    return df[FEATURES].copy(), df[TARGET].copy()
