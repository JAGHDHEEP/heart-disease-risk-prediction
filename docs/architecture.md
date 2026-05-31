# Architecture

## Design goals
1. **No data leakage** — the single most important correctness property.
2. **One feature contract** shared by training, the API and the UI.
3. **Separation of concerns** — a reusable `heart` library, with thin serving
   layers (FastAPI, Streamlit) on top.
4. **Reproducibility** — one command regenerates the model and all reports.

## Layers

| Layer | Module(s) | Responsibility |
| --- | --- | --- |
| Config | `heart.config` | Paths, feature schema, valid ranges, train config |
| Data | `heart.data` | Load + normalise CSV → `(X, y)`; **no** imputation/scaling |
| Modelling | `heart.pipeline` | `Pipeline[impute→scale→clf]` factory + search space |
| Training | `heart.train` | Compare → tune → evaluate once → persist artifacts |
| Inference | `heart.predict` | Cached pipeline load, single/batch prediction |
| Explainability | `heart.explain` | Model-agnostic per-patient SHAP (+ fallback) |
| Validation | `heart.validation` | Shared guardrails for all entry points |
| Serving | `api/main.py`, `app/streamlit_app.py` | REST + UI |

## Why a single `Pipeline` artifact?
Bundling the imputer + scaler + classifier into one serialized object means:
- preprocessing is **fit on training folds only** (leakage-free),
- there is **no separate scaler file** to keep in sync at inference time
  (a fragile pattern in the original project),
- inference code passes **raw** feature values and the artifact does the rest.

## Request flow (FastAPI `/predict`)
```
client JSON ─► Pydantic schema (type/range) ─► heart.validation (shared rules)
           ─► heart.predict.predict()  ─► Pipeline.predict_proba
           ─► heart.explain.explain()  ─► SHAP contributions
           ─► PredictionResponse JSON
```

## Trade-offs
- **Small dataset (303 rows):** linear models are competitive; XGBoost was kept
  for its strong test ROC-AUC but the gap is small and honestly reported.
- **Committed model artifact:** chosen for "clone-and-run" convenience over the
  purist "never commit binaries" rule, because the file is tiny (~110 KB) and
  fully reproducible.
- **Model-agnostic SHAP:** slightly slower than `TreeExplainer` but robust
  across xgboost/shap versions (TreeExplainer breaks on xgboost 3.x base_score).
