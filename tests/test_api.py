"""Smoke tests for the FastAPI service."""
import pytest

from heart.config import PIPELINE_PATH

pytestmark = pytest.mark.skipif(
    not PIPELINE_PATH.exists(),
    reason="Model not trained — run `python -m heart.train` first.",
)


@pytest.fixture
def client():
    from api.main import app
    from fastapi.testclient import TestClient
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_endpoint(client, valid_patient):
    resp = client.post("/predict", json=valid_patient)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in {"Disease", "No Disease"}
    assert len(body["explanation"]) == 5


def test_predict_validation_error(client, valid_patient):
    valid_patient["age"] = 900  # out of pydantic range
    resp = client.post("/predict", json=valid_patient)
    assert resp.status_code == 422
