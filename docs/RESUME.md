# Resume, LinkedIn & Recruiter Material

## ATS-friendly resume bullet points
Pick 3–4. They lead with impact and include keywords ATS scanners look for.

- Built an **end-to-end machine-learning system** to predict heart-disease risk
  from 11 clinical features, achieving **0.94 ROC-AUC** and **86% recall** on a
  held-out test set with a tuned **XGBoost** model.
- **Eliminated data leakage** by encapsulating imputation and scaling inside a
  scikit-learn `Pipeline`, ensuring preprocessing was fit on training folds only
  — correcting inflated metrics from the original implementation.
- Engineered a reproducible training pipeline that **compares 4 algorithms**
  (Logistic Regression, Random Forest, SVM, XGBoost) via stratified
  cross-validation and tunes the best with `RandomizedSearchCV`.
- Implemented **explainable AI** using per-patient **SHAP** values, exposing the
  top risk drivers for each prediction.
- Shipped the model as a typed **FastAPI** REST service and an interactive
  **Streamlit** app, **containerised with Docker** and deployed to the cloud.
- Added a **17-test pytest suite** and **GitHub Actions CI** (lint + train +
  test), reaching production-grade reliability and reproducibility.

## Project summary for LinkedIn

> 🫀 **Heart Disease Risk Prediction (Python, XGBoost, FastAPI, Streamlit)**
> Built a production-ready ML system that predicts heart-disease risk from
> routine clinical data, reaching 0.94 ROC-AUC. I focused on what real ML work
> demands beyond model accuracy: a **leakage-free pipeline**, rigorous model
> comparison and hyperparameter tuning, **per-patient SHAP explainability**, a
> tested codebase with CI, and dual deployment as a REST API and a Streamlit
> demo (Dockerised). Code + live demo 👇
> #MachineLearning #MLOps #Python #ExplainableAI #DataScience

## GitHub repository description (the one-liner under the repo name)

> Production-ready heart-disease risk prediction — leakage-free XGBoost pipeline
> with SHAP explainability, served via FastAPI + Streamlit, Dockerised and
> tested (CI).

## Recruiter-friendly explanation (non-technical)

> This project predicts a person's risk of heart disease from standard medical
> test results. What makes it stand out isn't just the prediction — it's that
> it's built the way real production software is: the model explains *why* it
> made each prediction, the code is automatically tested every time it changes,
> mistakes that would secretly inflate accuracy were found and fixed, and it can
> be run as a website or plugged into other systems through an API. It shows I
> can take a machine-learning idea all the way from raw data to something
> reliable, explainable and deployable.

## 30-second verbal pitch (for interviews)

> "I took a heart-disease classifier from a notebook to a deployable system. The
> headline result is 0.94 ROC-AUC, but the part I'm most proud of is catching a
> **data-leakage bug** — preprocessing was being fit on the whole dataset before
> the split — and fixing it by moving imputation and scaling into the pipeline.
> I compared four models, tuned the best with randomized search, added **real
> per-patient SHAP explanations**, wrote a test suite with CI, and shipped it as
> both a FastAPI service and a Streamlit app in Docker."
