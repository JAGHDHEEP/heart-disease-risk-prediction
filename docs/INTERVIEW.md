# Interview Preparation

Questions a reviewer is likely to ask about *this* project, with strong answers.

---

## Project & ML fundamentals

**Q1. Walk me through the project.**
A pipeline that predicts heart disease from 11 clinical features. Data is loaded
and validated, then preprocessing (median imputation + standardisation) and an
XGBoost classifier are bundled in one scikit-learn `Pipeline`. I compare four
models with stratified cross-validation, tune the best with `RandomizedSearchCV`,
evaluate once on a held-out test set (0.94 ROC-AUC), and serve it via FastAPI and
Streamlit with per-patient SHAP explanations.

**Q2. What is data leakage and how did you prevent it?**
Leakage is when information from outside the training set influences the model —
here the original code computed imputation medians and scaling stats on the
*whole* dataset before splitting, so test statistics bled into training and
inflated the metrics. I fixed it by putting the imputer and scaler *inside* the
Pipeline, so during cross-validation and the final fit they're learned from the
training portion only. The test set is used exactly once.

**Q3. Why ROC-AUC instead of accuracy?**
Accuracy is threshold-dependent and misleading on even mildly imbalanced data.
ROC-AUC measures ranking quality across all thresholds and is more robust. For a
medical use case I also report **recall/sensitivity** because **false negatives
(missing a sick patient) are the costliest error**.

**Q4. Your dataset is only 303 rows — concerns?**
Yes — small data means high variance in estimates, so I use *stratified*
CV and report the standard deviation across folds. It also explains why simpler
models (logistic regression) are competitive with XGBoost. I'd address it by
training on the combined UCI dataset (~920 rows) and adding nested CV.

**Q5. XGBoost's base CV score was below logistic regression — why keep it?**
Honest answer: on raw CV they're within noise. XGBoost gave the best held-out
**test** ROC-AUC after tuning, so I shipped it, but I explicitly report the
comparison rather than hiding it. On a larger dataset I'd expect the gap to widen
in XGBoost's favour; on this one I'd genuinely consider logistic regression for
its simplicity and interpretability.

---

## Explainability

**Q6. How is your SHAP different from feature importance?**
Global feature importance (e.g. XGBoost gain) is one number per feature for the
*whole model*. SHAP gives a signed contribution *per feature, per prediction* —
it tells a specific patient *why* they were flagged. The original project
mislabelled global importances as "SHAP" and returned identical values for
everyone; I compute genuine per-patient values.

**Q7. Why a model-agnostic SHAP explainer instead of TreeExplainer?**
`TreeExplainer` is faster, but the installed xgboost 3.x stores `base_score` in a
format the installed SHAP's tree loader can't parse. The model-agnostic
explainer wraps the pipeline's `predict_proba` with a background sample — robust
across versions, and fast enough for 11 features. I also kept a global-importance
fallback if SHAP isn't installed.

---

## Engineering & deployment

**Q8. Why save a Pipeline instead of model + scaler separately?**
A single artifact can't get out of sync. The original saved the model and scaler
as two pickles; if feature order or scaling logic drifted between training and
serving, predictions would silently be wrong. The Pipeline applies the exact same
fitted transforms at inference.

**Q9. How do you validate inputs?**
One shared `validation` module enforces the feature contract and clinical ranges
(e.g. age 1–120). FastAPI adds a typed Pydantic layer on top (HTTP 422 on bad
input). This prevents garbage-in predictions and is unit-tested.

**Q10. How would you deploy and monitor this in production?**
Containerised with Docker; `/health` for load-balancer probes; CI gates merges
with lint + train + test. For monitoring I'd log every request + prediction,
track input **drift** (e.g. PSI on feature distributions) and prediction
distribution, alert on latency/error rates, and schedule periodic retraining
with model-registry versioning (MLflow).

**Q11. How is the model loaded efficiently?**
`load_pipeline` is cached with `functools.lru_cache`, so the ~110 KB artifact is
deserialised once per process and reused across requests, not reloaded per call.

---

## Advanced / system-design

**Q12. The cost of false negatives vs false positives differs. How would you act on that?**
I'd move off the default 0.5 threshold: pick the operating point on the ROC/PR
curve that achieves a target recall (say ≥0.90), accepting more false positives.
I'd also calibrate probabilities (`CalibratedClassifierCV`) so the reported
risk % is trustworthy, and surface the chosen threshold as a config value.

**Q13. How would you scale the API to high traffic?**
Stateless containers behind a load balancer, horizontal autoscaling, multiple
uvicorn workers, model loaded once per worker (already cached). Add a request
queue / batch endpoint for bulk scoring, and a CDN/cache for idempotent reads.

**Q14. How would you guarantee reproducibility?**
Pinned dependencies, a fixed `random_state` everywhere, a single `train`
entry point that writes `metrics.json` with the params and timestamp, and CI that
retrains on every push. For full rigor I'd add data versioning (DVC) and an
MLflow run per experiment.

**Q15. What are the ethical/clinical risks?**
It's not a medical device and must not drive diagnosis. Risks: dataset bias
(Cleveland cohort isn't representative), automation bias by clinicians, and
fairness gaps across sex/age. Mitigations: a model card, subgroup performance
reporting, clear disclaimers, and human-in-the-loop use only.

---

## Rapid-fire concepts
- **Precision vs recall:** precision = of predicted-positive, how many are right;
  recall = of actual-positive, how many we caught.
- **StandardScaler:** centres to mean 0, scales to unit variance; needed for
  SVM/LogReg, harmless for trees.
- **Stratified split/CV:** preserves class ratio in each fold — important for
  imbalanced or small data.
- **Why median imputation:** robust to outliers vs mean.
- **Soft vs hard voting:** soft averages probabilities (used in the notebook's
  ensemble), hard takes majority class vote.
