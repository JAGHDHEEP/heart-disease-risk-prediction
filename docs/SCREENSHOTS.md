# Screenshots

Add images here and reference them from the README's Demo section.

Suggested shots to capture once running locally:

1. **Streamlit — prediction result** (`streamlit run app/streamlit_app.py`)
   - The metrics row (prediction / probability / risk) + SHAP bar chart.
   - Save as `docs/img/streamlit_result.png`.

2. **Streamlit — CSV upload tab**
   - Save as `docs/img/streamlit_csv.png`.

3. **FastAPI Swagger UI** (`uvicorn api.main:app` → `/docs`)
   - The `/predict` endpoint expanded with an example response.
   - Save as `docs/img/api_docs.png`.

4. **Evaluation plots** — already generated in `reports/`:
   - `roc_curve.png`, `confusion_matrix.png`, `feature_importance.png`.

Embed example:
```markdown
![Streamlit result](docs/img/streamlit_result.png)
```

> Tip: a short animated GIF of the Streamlit flow at the top of the README is the
> single highest-impact thing for recruiters skimming the repo.
