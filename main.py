"""
FastAPI Application for Iris Flower Species Classification.

This module exposes a RESTful API to serve predictions from a pre-trained
Random Forest model bundle (serialized via Joblib).
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class PredictionRequest(BaseModel):
    """
    Schema for flower measurement input payload.
    All dimensions are in centimetres, strictly positive, and <= 20.0 cm.
    """
    sepal_length: float = Field(
        ...,
        gt=0,
        le=20.0,
        description="Sepal length in centimetres",
        json_schema_extra={"example": 5.1},
    )
    sepal_width: float = Field(
        ...,
        gt=0,
        le=20.0,
        description="Sepal width in centimetres",
        json_schema_extra={"example": 3.5},
    )
    petal_length: float = Field(
        ...,
        gt=0,
        le=20.0,
        description="Petal length in centimetres",
        json_schema_extra={"example": 1.4},
    )
    petal_width: float = Field(
        ...,
        gt=0,
        le=20.0,
        description="Petal width in centimetres",
        json_schema_extra={"example": 0.2},
    )


class PredictionResponse(BaseModel):
    """
    Schema for classification prediction response.
    """
    predicted_class: int = Field(..., description="Numerical class label (0, 1, or 2)")
    predicted_species: str = Field(..., description="Common species name (setosa, versicolor, virginica)")
    confidence: float = Field(..., description="Highest prediction probability rounded to 4 decimal places")
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probabilities mapped to each target species, rounded to 4 decimal places",
    )
    model_version: str = Field(..., description="Semantic version of the loaded model")


# ---------------------------------------------------------------------------
# Application Lifespan (Startup & Shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown lifecycle.

    Loads the serialized model bundle into memory once at startup,
    ensuring paths are resolved relative to this file's directory.
    Fails startup with an informative message if the artifact is missing or corrupted.
    """
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "artifacts" / "iris_model.joblib"

    if not model_path.is_file():
        raise RuntimeError(
            f"Model artifact not found at '{model_path}'. "
            "Please run 'train_model.py' to generate and save the model bundle before starting the API."
        )

    try:
        model_bundle = joblib.load(model_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model artifact from '{model_path}': {exc}. "
            "Ensure the file is a valid, readable Joblib bundle."
        ) from exc

    required_keys = {"model", "target_names", "feature_names", "test_accuracy", "model_version"}
    missing_keys = required_keys - set(model_bundle.keys())
    if missing_keys:
        raise RuntimeError(
            f"Model bundle at '{model_path}' is missing required keys: {sorted(missing_keys)}"
        )

    # Attach bundle to application state for safe access across requests
    app.state.model_bundle = model_bundle

    yield

    # Clean up state on application shutdown
    app.state.model_bundle = None


# ---------------------------------------------------------------------------
# FastAPI Application Instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Iris Classification API",
    version="1.0.0",
    description=(
        "A production-ready REST API for classifying Iris flower species using "
        "a trained Random Forest classifier. Provides health checks, model metadata, "
        "and real-time inference endpoints."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get(
    "/",
    tags=["General"],
    summary="Root Welcome Endpoint",
    description="Returns a welcome message with navigation links to documentation and endpoints.",
)
async def root() -> Dict[str, str]:
    """Provides general information and links to key API endpoints."""
    return {
        "message": "Welcome to the Iris Classification API",
        "documentation": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "predict": "/predict",
    }


@app.get(
    "/health",
    tags=["Diagnostics"],
    summary="Health Check Endpoint",
    description="Reports the service status and whether the ML model is currently loaded in memory.",
)
async def health(request: Request) -> Dict[str, Any]:
    """Checks service health and confirms model bundle readiness."""
    bundle = getattr(request.app.state, "model_bundle", None)
    model_loaded = bundle is not None and "model" in bundle
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
    }


@app.get(
    "/model-info",
    tags=["Diagnostics"],
    summary="Model Metadata Endpoint",
    description="Returns metadata about the active model bundle, including type, version, features, and test accuracy.",
)
async def model_info(request: Request) -> Dict[str, Any]:
    """Returns metadata and evaluation metrics from the saved model bundle."""
    bundle = getattr(request.app.state, "model_bundle", None)
    if not bundle or "model" not in bundle:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded in memory.",
        )

    model = bundle["model"]
    return {
        "model_type": model.__class__.__name__,
        "model_version": str(bundle.get("model_version", "unknown")),
        "feature_names": [str(f) for f in bundle.get("feature_names", [])],
        "target_names": [str(t) for t in bundle.get("target_names", [])],
        "test_accuracy": float(bundle.get("test_accuracy", 0.0)),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Classify Iris Species",
    description=(
        "Receives 4 continuous flower measurements, formats them into a 2D array, "
        "runs model inference, and returns predicted class, species, confidence, "
        "and individual class probabilities."
    ),
)
async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    """
    Generates Iris species predictions from input physical dimensions.

    - Validates feature values against physical biological boundaries (0 < x <= 20).
    - Structures inputs into a 2-dimensional (1, 4) NumPy array.
    - Computes discrete classification and class probabilities.
    - Casts NumPy scalar types to standard Python primitives with 4-decimal precision.
    """
    bundle = getattr(request.app.state, "model_bundle", None)
    if not bundle or "model" not in bundle:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded in memory.",
        )

    try:
        model = bundle["model"]
        target_names = bundle["target_names"]
        model_version = str(bundle["model_version"])

        # Arrange features into a 2D NumPy array with 1 row and 4 features:
        # [sepal length, sepal width, petal length, petal width]
        features_array = np.array(
            [[
                payload.sepal_length,
                payload.sepal_width,
                payload.petal_length,
                payload.petal_width,
            ]],
            dtype=np.float64,
        )

        # Run model inference
        raw_pred = model.predict(features_array)
        raw_probs = model.predict_proba(features_array)[0]

        # Convert NumPy primitives to native Python types
        predicted_class = int(raw_pred[0])
        predicted_species = str(target_names[predicted_class])

        # Map class names to probabilities rounded to 4 decimal places
        probabilities = {
            str(target_names[idx]): round(float(prob), 4)
            for idx, prob in enumerate(raw_probs)
        }
        confidence = round(float(np.max(raw_probs)), 4)

        return PredictionResponse(
            predicted_class=predicted_class,
            predicted_species=predicted_species,
            confidence=confidence,
            probabilities=probabilities,
            model_version=model_version,
        )

    except HTTPException:
        # Re-raise explicit HTTP exceptions untouched
        raise
    except Exception as exc:
        # Catch unexpected errors, log or chain for debugging, but shield internal details from clients
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed due to an internal server error.",
        ) from exc


# ---------------------------------------------------------------------------
# Direct Execution Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
