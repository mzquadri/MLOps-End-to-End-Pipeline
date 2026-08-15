"""
Serving Module
==============
FastAPI inference service with health checks, batch prediction,
and prediction logging.  Designed for Docker deployment.

Usage:
    python -m src.serve --bundle_path models/registry/sentiment-classifier/v1
    # Then visit http://127.0.0.1:8000/docs for interactive API docs.
"""

import argparse
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    texts: List[str] = Field(..., min_length=1, max_length=100)


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
        self.feature_store: Optional[Any] = None
        self.model_version: str = "unknown"
        self.start_time: float = time.time()
        self.prediction_count: int = 0
        self.prediction_log: List[Dict[str, Any]] = []

    @property
    def ready(self) -> bool:
        return self.model is not None and self.feature_store is not None

    def load(self, bundle_path: str) -> None:
        """Load one complete, passing bundle from a trusted local registry."""
        from src.feature_store import FeatureStore
        from src.model_bundle import LINEAGE_FILE, load_trusted_bundle, read_bundle_json

        self.model = None
        self.feature_store = None
        bundle = Path(bundle_path).resolve()
        if len(bundle.parents) < 2:
            raise RuntimeError("Production bundle path is not inside a registry")
        registry_root = bundle.parents[1]
        index_path = registry_root / "registry.json"
        if not index_path.is_file():
            raise RuntimeError("Production registry index is missing")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        relative_bundle = bundle.relative_to(registry_root).as_posix()
        entries = [
            entry
            for model in index.get("models", {}).values()
            for entry in model.get("versions", [])
            if entry.get("bundle_path") == relative_bundle
        ]
        if len(entries) != 1 or entries[0].get("stage") != "production":
            raise RuntimeError("Serving requires a production registry bundle")

        model, transformers, _ = load_trusted_bundle(
            str(bundle), require_gate_passed=True
        )
        lineage = read_bundle_json(str(bundle), LINEAGE_FILE)
        schema = lineage.get("feature_schema", {})
        config = {
            "data": {
                "label_column": schema.get("label_column", "sentiment"),
                "text_column": schema.get("text_column", "review_text"),
            }
        }
        feature_store = FeatureStore(config)
        feature_store.import_transformers(transformers)

        self.model = model
        self.feature_store = feature_store
        self.model_version = bundle.name
        logger.info("Validated production model bundle loaded from %s", bundle)

    def predict_single(
        self, text: str, numeric_values: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Run prediction on a single text input."""
        if not self.ready:
            raise RuntimeError("A complete model bundle is not loaded.")

        model = self.model
        feature_store = self.feature_store
        if model is None or feature_store is None:
            raise RuntimeError("A complete model bundle is not loaded.")
        values = dict(numeric_values or {})
        numeric_columns = feature_store.metadata.get("numeric_columns", [])
        if "review_length" in numeric_columns and "review_length" not in values:
            values["review_length"] = len(text)
        if "word_count" in numeric_columns and "word_count" not in values:
            values["word_count"] = len(text.split())
        X = feature_store.transform_single(text, values)

        t0 = time.perf_counter()
        prediction = model.predict(X)[0]
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
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
    bundle_path = os.environ.get(
        "MODEL_BUNDLE_PATH", "models/registry/sentiment-classifier/v1"
    )
    state.load(bundle_path)
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
        status="healthy" if state.ready else "degraded",
        model_loaded=state.ready,
        model_version=state.model_version,
        uptime_seconds=round(time.time() - state.start_time, 1),
        total_predictions=state.prediction_count,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Predict sentiment for a single text."""
    if not state.ready:
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
    if not state.ready:
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
    parser.add_argument("--bundle_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ["MODEL_BUNDLE_PATH"] = args.bundle_path
    import uvicorn

    uvicorn.run(
        "src.serve:app", host=args.host, port=args.port, reload=False, workers=1
    )


if __name__ == "__main__":
    main()
