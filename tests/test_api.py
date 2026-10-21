"""
Automated API Test Suite for Iris Classification API.

Tests cover:
- Root endpoint navigation and status
- Health check and model readiness verification
- Model metadata inspection
- Model inference with valid payload and response structure validation
- Input validation rejection on negative measurements (HTTP 422)
- Input validation rejection on missing required fields (HTTP 422)
"""

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="module")
def client():
    """
    TestClient fixture that enters the application lifespan context,
    triggering the startup event to load 'artifacts/iris_model.joblib'.
    """
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client: TestClient):
    """Test GET / returns HTTP 200 with documentation and endpoint links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == "Welcome to the Iris Classification API"
    assert "documentation" in data
    assert "health" in data
    assert "model_info" in data
    assert "predict" in data


def test_health_endpoint(client: TestClient):
    """Test GET /health returns HTTP 200 and confirms the model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "healthy"
    assert data.get("model_loaded") is True


def test_model_info_endpoint(client: TestClient):
    """Test GET /model-info returns HTTP 200 with correct model metadata."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data.get("model_type") == "RandomForestClassifier"
    assert data.get("model_version") == "1.0.0"
    assert "feature_names" in data
    assert isinstance(data["feature_names"], list)
    assert len(data["feature_names"]) == 4
    assert "target_names" in data
    assert isinstance(data["target_names"], list)
    assert len(data["target_names"]) == 3
    assert "test_accuracy" in data
    assert isinstance(data["test_accuracy"], float)


def test_predict_valid_input(client: TestClient):
    """Test POST /predict with valid measurements returns HTTP 200 and valid prediction."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data.get("predicted_species") == "setosa"
    assert data.get("predicted_class") == 0
    assert isinstance(data.get("confidence"), float)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data.get("probabilities"), dict)
    assert set(data["probabilities"].keys()) == {"setosa", "versicolor", "virginica"}
    assert data.get("model_version") == "1.0.0"


def test_predict_negative_measurement(client: TestClient):
    """Test POST /predict with a negative measurement returns HTTP 422."""
    payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_missing_measurement(client: TestClient):
    """Test POST /predict with a missing required measurement returns HTTP 422."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        # petal_length is missing
        "petal_width": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
