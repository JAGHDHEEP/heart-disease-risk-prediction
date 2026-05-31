"""Training entry point.

Run with:  ``python -m heart.train``  (or ``make train``)

Pipeline of work:
    1. Load + validate data (no leakage: imputation/scaling live in the model).
    2. Stratified train/test split.
    3. Compare candidate models with stratified CV on the *training* set.
    4. Hyperparameter-tune XGBoost with RandomizedSearchCV.
    5. Refit the winner, evaluate once on the held-out test set.
    6. Persist the fitted Pipeline + a metrics.json report + evaluation plots.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import __version__, config
from .data import load_dataset, split_xy
from .logging_conf import get_logger
from .pipeline import candidate_models, xgb_search_space

logger = get_logger(__name__)


def _evaluate(pipeline, X_test, y_test) -> dict:
    """Compute the held-out test metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "specificity": round(tn / (tn + fp), 4),
        "sensitivity": round(tp / (tp + fn), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train(cfg: config.TrainConfig) -> dict:
    """Execute the full training run and return the metrics report."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(cfg.dataset_path)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )
    logger.info("Train=%d  Test=%d  Positives(train)=%.1f%%",
                len(X_train), len(X_test), 100 * y_train.mean())

    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)

    # 1) Model comparison on the training set (CV only — test stays untouched).
    comparison: dict[str, dict] = {}
    for name, pipe in candidate_models(cfg.random_state).items():
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        comparison[name] = {"cv_roc_auc_mean": round(scores.mean(), 4),
                            "cv_roc_auc_std": round(scores.std(), 4)}
        logger.info("CV  %-20s ROC-AUC %.4f (+/- %.4f)",
                    name, scores.mean(), scores.std())

    # 2) Tune XGBoost (best base learner on this data).
    logger.info("Tuning XGBoost with RandomizedSearchCV (%d iters)...", cfg.search_iter)
    search = RandomizedSearchCV(
        estimator=candidate_models(cfg.random_state)["xgboost"],
        param_distributions=xgb_search_space(),
        n_iter=cfg.search_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=cfg.random_state,
        refit=True,
    )
    search.fit(X_train, y_train)
    best_pipeline = search.best_estimator_  # the TUNED model is what we keep
    logger.info("Best CV ROC-AUC: %.4f", search.best_score_)
    logger.info("Best params: %s", search.best_params_)

    # 3) Final, single evaluation on the held-out test set.
    test_metrics = _evaluate(best_pipeline, X_test, y_test)
    logger.info("TEST  acc=%.4f  roc_auc=%.4f  recall=%.4f",
                test_metrics["accuracy"], test_metrics["roc_auc"], test_metrics["recall"])

    report = {
        "package_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": cfg.dataset_path.name,
        "n_rows": int(len(df)),
        "features": cfg.features,
        "best_model": "xgboost",
        "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
        "cv_best_roc_auc": round(search.best_score_, 4),
        "model_comparison": comparison,
        "test_metrics": test_metrics,
        "classification_report": classification_report(
            y_test, best_pipeline.predict(X_test),
            target_names=["No Disease", "Disease"], output_dict=True,
        ),
    }

    joblib.dump(best_pipeline, config.PIPELINE_PATH)
    config.METRICS_PATH.write_text(json.dumps(report, indent=2))
    logger.info("Saved pipeline -> %s", config.PIPELINE_PATH)
    logger.info("Saved metrics  -> %s", config.METRICS_PATH)

    _save_plots(best_pipeline, X_test, y_test)
    return report


def _save_plots(pipeline, X_test, y_test) -> None:
    """Persist ROC curve, confusion matrix and feature-importance figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
    except ImportError:  # plotting is optional, never block training
        logger.warning("matplotlib not available — skipping plots")
        return

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
    plt.title("ROC Curve — Tuned XGBoost")
    plt.savefig(config.REPORTS_DIR / "roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test, display_labels=["No Disease", "Disease"], cmap="Blues"
    )
    plt.title("Confusion Matrix — Test Set")
    plt.savefig(config.REPORTS_DIR / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()

    importances = pipeline.named_steps["clf"].feature_importances_
    order = np.argsort(importances)
    plt.figure(figsize=(8, 5))
    plt.barh([config.FEATURES[i] for i in order], importances[order], color="teal")
    plt.xlabel("Importance")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(config.REPORTS_DIR / "feature_importance.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("Saved plots    -> %s", config.REPORTS_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the heart-disease model.")
    parser.add_argument("--dataset", type=str, default=str(config.DEFAULT_DATASET))
    parser.add_argument("--search-iter", type=int, default=40)
    parser.add_argument("--test-size", type=float, default=config.TEST_SIZE)
    args = parser.parse_args()

    cfg = config.TrainConfig(
        dataset_path=config.Path(args.dataset),
        search_iter=args.search_iter,
        test_size=args.test_size,
    )
    train(cfg)


if __name__ == "__main__":
    main()
