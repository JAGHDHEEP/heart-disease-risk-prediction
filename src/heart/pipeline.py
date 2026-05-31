"""Model pipeline factory.

Every estimator is wrapped in a single sklearn ``Pipeline`` of
``[imputer -> scaler -> classifier]``. Because the imputer and scaler are
*steps in the pipeline*, ``cross_val_score`` / ``GridSearchCV`` re-fit them on
each training fold only — eliminating the data leakage present in the original
notebook (which imputed and scaled the full dataset before splitting).
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import RANDOM_STATE


def _wrap(classifier) -> Pipeline:
    """Wrap a classifier with median imputation + standardisation."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )


def candidate_models(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Return the set of models compared during training."""
    return {
        "logistic_regression": _wrap(
            LogisticRegression(max_iter=1000, random_state=random_state)
        ),
        "random_forest": _wrap(
            RandomForestClassifier(n_estimators=300, random_state=random_state)
        ),
        "svm": _wrap(SVC(probability=True, random_state=random_state)),
        "xgboost": _wrap(
            XGBClassifier(
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )
        ),
    }


def xgb_search_space() -> dict[str, list]:
    """RandomizedSearchCV space for the XGBoost step of the pipeline.

    Keys are prefixed with ``clf__`` to address the classifier inside the
    pipeline.
    """
    return {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [3, 4, 5, 6, 8],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "clf__min_child_weight": [1, 3, 5],
        "clf__gamma": [0, 0.1, 0.3],
    }
