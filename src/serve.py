"""
Serving Module
==============
FastAPI inference service with health checks, batch prediction,
and prediction logging.  Designed for Docker deployment.

Usage:
    python src/serve.py --model_path models/production
    # Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

import argparse
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, description="Review text for sentiment prediction"
    )
    review_length: Optional[int] = None
    word_count: Optional[int] = None


class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    latency_ms: float
    model_version: str


class BatchPredictResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    total_latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    total_predictions: int


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class AppState:
    """Holds model artifacts and runtime stats."""

    def __init__(self) -> None:
        self.model: Optional[Any] = None
        self.feature_transformers: Optional[Dict[str, Any]] = None
        self.model_version: str = "unknown"
        self.start_time: float = time.time()
        self.prediction_count: int = 0
        self.prediction_log: List[Dict[str, Any]] = []

    def load(self, model_path: str) -> None:
        model_file = os.path.join(model_path, "model.joblib")
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        self.model = joblib.load(model_file)
        logger.info("Model loaded from %s", model_file)

        # Load feature transformers
        transformers_path = os.path.join("models", "feature_transformers.pkl")
        if os.path.exists(transformers_path):
            import pickle

            with open(transformers_path, "rb") as f:
                self.feature_transformers = pickle.load(f)
            logger.info("Feature transformers loaded.")

        # Read version from metrics if available
        metrics_file = os.path.join(model_path, "metrics.json")
        if os.path.exists(metrics_file):
            import json

            with open(metrics_file) as f:
                metrics = json.load(f)
            self.model_version = f"acc={metrics.get('accuracy', '?'):.4f}"
        else:
            self.model_version = "latest"

    def predict_single(
        self, text: str, numeric_values: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Run prediction on a single text input."""
        from src.data_pipeline import preprocess_text

        if self.model is None or self.feature_transformers is None:
            raise RuntimeError("Model or transformers not loaded.")

        clean = preprocess_text(text)
        tfidf = self.feature_transformers["tfidf"]
        scaler = self.feature_transformers.get("scaler")
        metadata = self.feature_transformers.get("metadata", {})

        X_tfidf = tfidf.transform([clean])

        # Add numeric features if scaler exists and columns are known
        numeric_cols = metadata.get("numeric_columns", [])
        if scaler is not None and numeric_cols:
            num_arr = np.array(
                [[(numeric_values or {}).get(c, 0) for c in numeric_cols]]
            )
            X_num = scaler.transform(num_arr)
            from scipy.sparse import csr_matrix, hstack

            X = hstack([X_tfidf, csr_matrix(X_num)])
        else:
            X = X_tfidf

        t0 = time.perf_counter()
        prediction = self.model.predict(X)[0]
        confidence = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = float(max(proba))
        latency_ms = (time.perf_counter() - t0) * 1000

        self.prediction_count += 1
        return {
            "prediction": str(prediction),
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 3),
        }


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    model_path = os.environ.get("MODEL_PATH", "models/production")
    if not os.path.exists(os.path.join(model_path, "model.joblib")):
        model_path = "models/latest"
    try:
        state.load(model_path)
    except FileNotFoundError:
        logger.warning("No model found at startup. Load manually or retrain.")
    yield


app = FastAPI(
    title="MLOps Sentiment API",
    description="Production sentiment analysis inference service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for monitoring and Docker HEALTHCHECK."""
    return HealthResponse(
        status="healthy" if state.model is not None else "degraded",
        model_loaded=state.model is not None,
        model_version=state.model_version,
        uptime_seconds=round(time.time() - state.start_time, 1),
        total_predictions=state.prediction_count,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict sentiment for a single text."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    numeric_values = {}
    if request.review_length is not None:
        numeric_values["review_length"] = request.review_length
    if request.word_count is not None:
        numeric_values["word_count"] = request.word_count

    result = state.predict_single(request.text, numeric_values or None)

    # Log prediction
    state.prediction_log.append(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "text_length": len(request.text),
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "latency_ms": result["latency_ms"],
        }
    )

    return PredictResponse(
        prediction=result["prediction"],
        confidence=result["confidence"],
        latency_ms=result["latency_ms"],
        model_version=state.model_version,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(request: BatchPredictRequest):
    """Predict sentiment for multiple texts."""
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0 = time.perf_counter()
    predictions = []
    for text in request.texts:
        result = state.predict_single(text)
        predictions.append(result)
    total_ms = (time.perf_counter() - t0) * 1000

    return BatchPredictResponse(
        predictions=predictions,
        total_latency_ms=round(total_ms, 3),
        model_version=state.model_version,
    )


@app.get("/metrics")
async def metrics():
    """Return prediction statistics for monitoring."""
    recent = state.prediction_log[-100:] if state.prediction_log else []
    latencies = [p["latency_ms"] for p in recent]
    return {
        "total_predictions": state.prediction_count,
        "recent_avg_latency_ms": round(float(np.mean(latencies)), 3)
        if latencies
        else 0,
        "recent_p95_latency_ms": round(float(np.percentile(latencies, 95)), 3)
        if latencies
        else 0,
        "uptime_seconds": round(time.time() - state.start_time, 1),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FastAPI inference server")
    parser.add_argument("--model_path", type=str, default="models/production")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ["MODEL_PATH"] = args.model_path
    import uvicorn

    uvicorn.run(
        "src.serve:app", host=args.host, port=args.port, reload=False, workers=1
    )


if __name__ == "__main__":
    main()
