"""FastAPI inference service.

Run locally:  ``uvicorn api.main:app --reload``
Interactive docs at  http://localhost:8000/docs

Exposes a typed, validated JSON API around the trained pipeline — the
"production" face of the project, suitable for integration into other systems.
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from heart import __version__
from heart.config import FEATURES
from heart.explain import explain
from heart.predict import ModelNotTrainedError, load_pipeline, predict
from heart.validation import ValidationError, validate_dataframe


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model on startup; fail fast (in logs) if it is missing."""
    try:
        load_pipeline()
        app.state.model_error = None
    except ModelNotTrainedError as exc:  # pragma: no cover
        app.state.model_error = str(exc)
    yield


app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predict heart-disease risk from 11 clinical features, with "
    "per-patient SHAP explanations.",
    version=__version__,
    lifespan=lifespan,
)


class PatientFeatures(BaseModel):
    """Schema for a single patient. Ranges are enforced by the model layer."""

    age: float = Field(..., examples=[63], ge=1, le=120)
    sex: int = Field(..., examples=[1], ge=0, le=1)
    cp: int = Field(..., examples=[3], ge=0, le=3)
    trestbps: float = Field(..., examples=[145], ge=50, le=260)
    chol: float = Field(..., examples=[233], ge=0, le=700)
    fbs: int = Field(..., examples=[1], ge=0, le=1)
    restecg: int = Field(..., examples=[0], ge=0, le=2)
    thalch: float = Field(..., examples=[150], ge=40, le=250)
    exang: int = Field(..., examples=[0], ge=0, le=1)
    oldpeak: float = Field(..., examples=[2.3], ge=-3, le=10)
    slope: int = Field(..., examples=[2], ge=0, le=2)


class PredictionResponse(BaseModel):
    prediction: str
    label: int
    disease_probability: float
    no_disease_probability: float
    confidence: float
    risk_level: str
    explanation: list[dict]


@app.get("/health")
def health() -> dict:
    """Liveness/readiness probe for containers and load balancers."""
    ready = getattr(app.state, "model_error", None) is None
    return {"status": "ok" if ready else "model_not_loaded", "version": __version__}


@app.post("/predict", response_model=PredictionResponse)
def predict_one(patient: PatientFeatures) -> PredictionResponse:
    """Predict risk for a single patient and return a SHAP explanation."""
    try:
        result = predict(patient.model_dump())
        result["explanation"] = explain(patient.model_dump(), top_k=5)
        return PredictionResponse(**result)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict/batch")
async def predict_csv(file: UploadFile = File(...)) -> dict:
    """Predict risk for every row of an uploaded CSV (columns = the 11 features)."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    try:
        df = pd.read_csv(io.BytesIO(await file.read()))
        rows = validate_dataframe(df)
        results = [predict(r) for r in rows]
        return {"count": len(results), "results": results}
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # malformed CSV, etc.
        raise HTTPException(status_code=400, detail=f"Could not process file: {exc}") from exc


@app.get("/features")
def feature_schema() -> dict:
    """Return the ordered feature contract for API consumers."""
    return {"features": FEATURES}
