# GitHub Setup Guide

## 1. Initialise and push

```bash
cd heart-disease-risk-prediction
git init
git add .
git commit -m "feat: production-ready heart disease prediction system"
git branch -M main
git remote add origin https://github.com/JAGHDHEEP/heart-disease-risk-prediction.git
git push -u origin main
```

## 2. Repository settings
- **Description:** see `docs/RESUME.md` → "GitHub repository description".
- **Topics/tags:** `machine-learning`, `xgboost`, `shap`, `fastapi`,
  `streamlit`, `healthcare`, `scikit-learn`, `mlops`, `explainable-ai`, `python`.
- **About → Website:** paste your deployed Streamlit URL.
- Enable **Issues** and **Actions** (CI runs automatically on push).

## 3. Make it look professional
- ✅ Pin this repo on your GitHub profile.
- ✅ Confirm the CI badge in the README turns green after first push.
- ✅ Add screenshots/GIF to `docs/SCREENSHOTS.md` and link them in the README
  demo section.
- ✅ Create a **Release** `v1.0.0` with a short changelog.
- ✅ Add a deployed-demo link at the top of the README.

## 4. Suggested commit history (if rebuilding incrementally)
A clean, story-telling history reads well to reviewers:
```
feat: project scaffold + config and feature schema
feat: leakage-free data loading and modelling pipeline
feat: training CLI with model comparison + tuning
feat: inference service and real per-patient SHAP
feat: FastAPI service and Streamlit demo
test: validation, data, predict and API tests
docs: README, architecture, deployment, interview prep
ci: lint + train + test workflow
chore: Docker + compose for one-command deploy
```

## 5. What NOT to commit
Already handled by `.gitignore`: `__pycache__/`, `.venv/`, uploaded CSVs,
`*.log`, IDE folders. The original project had committed `__pycache__`, a
binary `.db`, a 15 MB `Demo.mp4` and a `pip freeze` requirements file — all
cleaned up here. Keep large media (the demo video) in a GitHub Release asset or
link to it, not in the repo tree.
