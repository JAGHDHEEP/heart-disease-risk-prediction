"""Central configuration: paths, feature schema, and validation ranges.

Keeping every "magic value" here means the training pipeline, the API and the
web app all agree on the exact same feature contract — a common source of bugs
when these live in three different files.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (resolved relative to the repository root, not the CWD)
# --------------------------------------------------------------------------- #
ROOT_DIR: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
REPORTS_DIR: Path = ROOT_DIR / "reports"

DEFAULT_DATASET: Path = DATA_DIR / "heart_cleveland.csv"
PIPELINE_PATH: Path = MODELS_DIR / "heart_pipeline.joblib"
METRICS_PATH: Path = MODELS_DIR / "metrics.json"

# --------------------------------------------------------------------------- #
# Feature schema — the single source of truth for the model contract
# --------------------------------------------------------------------------- #
FEATURES: list[str] = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalch", "exang", "oldpeak", "slope",
]
TARGET: str = "target"

# Human-readable labels used by the web app / API documentation.
FEATURE_LABELS: dict[str, str] = {
    "age": "Age (years)",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (0–3)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1/0)",
    "restecg": "Resting ECG results (0–2)",
    "thalch": "Max heart rate achieved",
    "exang": "Exercise-induced angina (1/0)",
    "oldpeak": "ST depression induced by exercise",
    "slope": "Slope of peak exercise ST segment (0–2)",
}


@dataclass(frozen=True)
class FeatureRange:
    """Plausible clinical range used to reject obviously invalid inputs."""

    low: float
    high: float


# Guards against garbage input (e.g. age=900). These are deliberately wide
# clinical bounds, not the training-data min/max.
FEATURE_RANGES: dict[str, FeatureRange] = {
    "age": FeatureRange(1, 120),
    "sex": FeatureRange(0, 1),
    "cp": FeatureRange(0, 3),
    "trestbps": FeatureRange(50, 260),
    "chol": FeatureRange(0, 700),
    "fbs": FeatureRange(0, 1),
    "restecg": FeatureRange(0, 2),
    "thalch": FeatureRange(40, 250),
    "exang": FeatureRange(0, 1),
    "oldpeak": FeatureRange(-3, 10),
    "slope": FeatureRange(0, 2),
}

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5


@dataclass
class TrainConfig:
    """Tunable knobs for a training run (overridable from the CLI)."""

    dataset_path: Path = DEFAULT_DATASET
    test_size: float = TEST_SIZE
    cv_folds: int = CV_FOLDS
    random_state: int = RANDOM_STATE
    search_iter: int = 40  # RandomizedSearchCV iterations for XGBoost
    features: list[str] = field(default_factory=lambda: list(FEATURES))
