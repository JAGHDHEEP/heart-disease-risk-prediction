"""Streamlit demo — the recommended way to *show* this project.

Run:  ``streamlit run app/streamlit_app.py``

A single-file, dependency-light UI: enter a patient's features (or upload a
CSV), get a risk prediction plus a SHAP explanation chart. Ideal for a
portfolio demo and a deployed Streamlit Community Cloud link.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make ``src`` importable when run via `streamlit run app/streamlit_app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from heart.config import FEATURE_LABELS, FEATURES  # noqa: E402
from heart.explain import explain  # noqa: E402
from heart.predict import ModelNotTrainedError, predict  # noqa: E402
from heart.validation import ValidationError, validate_dataframe  # noqa: E402

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="wide")

st.title("❤️ Heart Disease Risk Predictor")
st.caption(
    "Tuned XGBoost pipeline (leakage-free) with per-patient SHAP explanations. "
    "Educational demo — not a medical device."
)

RISK_COLORS = {"Very Low": "🟢", "Low": "🟢", "Moderate": "🟠", "High": "🔴"}


def _render_result(result: dict, features: dict) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction", result["prediction"])
    c2.metric("Disease probability", f"{result['disease_probability']}%")
    c3.metric("Risk level", f"{RISK_COLORS.get(result['risk_level'], '')} {result['risk_level']}")
    st.progress(min(result["disease_probability"] / 100, 1.0))

    st.subheader("Why? Top contributing factors (SHAP)")
    exp = explain(features, top_k=8)
    chart_df = pd.DataFrame(exp).set_index("feature")["contribution"]
    st.bar_chart(chart_df)
    st.dataframe(pd.DataFrame(exp), use_container_width=True, hide_index=True)


tab_form, tab_csv = st.tabs(["Single patient", "CSV upload"])

with tab_form:
    with st.form("patient_form"):
        cols = st.columns(3)
        values: dict[str, float] = {}
        defaults = {"age": 54, "sex": 1, "cp": 0, "trestbps": 130, "chol": 240,
                    "fbs": 0, "restecg": 1, "thalch": 150, "exang": 0,
                    "oldpeak": 1.0, "slope": 1}
        for i, feat in enumerate(FEATURES):
            with cols[i % 3]:
                values[feat] = st.number_input(
                    FEATURE_LABELS[feat], value=float(defaults[feat]), step=1.0
                )
        submitted = st.form_submit_button("Predict", use_container_width=True)
    if submitted:
        try:
            _render_result(predict(values), values)
        except (ValidationError, ModelNotTrainedError) as exc:
            st.error(str(exc))

with tab_csv:
    st.write("Upload a CSV with columns:", ", ".join(FEATURES))
    uploaded = st.file_uploader("Patient CSV", type="csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            rows = validate_dataframe(df)
            results = pd.DataFrame([predict(r) for r in rows])
            st.dataframe(pd.concat([df.reset_index(drop=True), results], axis=1),
                         use_container_width=True)
        except (ValidationError, ModelNotTrainedError) as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not process file: {exc}")
