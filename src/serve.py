"""FastAPI inference service backed by exactly one validated production bundle.

Liveness and readiness are separate here, and the distinction matters:

* `/health` always answers while the process is alive. It reports whether a model is
  loaded, and why not if it isn't.
* `/ready` answers 503 until a model is actually usable. This is the endpoint a load
  balancer or orchestrator should gate traffic on.

The service starts even when no bundle can be loaded. That is a deliberate change: it
used to raise during startup, which killed the process, made the "degraded" branch of
`/health` unreachable, and left an operator with a crash loop and no endpoint to ask
what was wrong. Set `REQUIRE_MODEL_AT_STARTUP=1` to restore fail-fast behaviour where a
deployment would rather not start at all than start unready.

Usage:
    python -m src.serve --bundle_path models/registry/sentiment-classifier/v1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

PREDICTION_LOG_CAPACITY = 500

#: Longest accepted review, in characters.
#:
#: The served model is trained on the UCI sentiment-labelled sentences, which are single
#: sentences: median 68 characters, p99 265, longest 479. This cap rejects none of them and
#: still leaves four times the headroom of the longest training example, so it bounds cost
#: without narrowing what the model was built to answer.
#:
#: Vectorising is linear in input length, and nothing else bounded it: a 10 MB body measured
#: at 12.2 s of CPU and 209 MB of peak memory for one request, and the batch endpoint would
#: take a hundred of those. Text far beyond the training distribution is out-of-domain
#: anyway, so refusing it at the edge costs no real prediction.
MAX_TEXT_LENGTH = 2000

ReviewText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_LENGTH)]

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class PredictRequest(BaseModel):
    text: ReviewText = Field(..., description="Review text for sentiment prediction")
    review_length: int | None = None
    word_count: int | None = None


class BatchPredictRequest(BaseModel):
    # max_length on the list bounds how many texts arrive; the item type bounds how large
    # each one may be. Without the second the first is not a limit on anything.
    texts: list[ReviewText] = Field(..., min_length=1, max_length=100)


class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    latency_ms: float
    model_version: str


class BatchPredictResponse(BaseModel):
    predictions: list[dict[str, Any]]
    total_latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float
    total_predictions: int
    detail: str | None = None


class ReadyResponse(BaseModel):
    ready: bool
    model_version: str
    detail: str | None = None


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class AppState:
    """Holds model artifacts and runtime stats."""

    def __init__(self) -> None:
        self.model: Any | None = None
        self.feature_store: Any | None = None
        self.model_version: str = "unknown"
        self.start_time: float = time.time()
        self.prediction_count: int = 0
        # Bounded on purpose. The previous unbounded list grew for the lifetime of the
        # process while only ever being read as the last hundred entries.
        self.prediction_log: deque[dict[str, Any]] = deque(maxlen=PREDICTION_LOG_CAPACITY)
        self.load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.feature_store is not None

    def reset(self) -> None:
        """Return to the unloaded state. Used by tests and by a failed reload."""
        self.model = None
        self.feature_store = None
        self.model_version = "unknown"
        self.load_error = None
        self.prediction_count = 0
        self.prediction_log.clear()

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
        self, text: str, numeric_values: dict[str, float] | None = None
    ) -> dict[str, Any]:
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
    """Attempt to load the production bundle, then serve either way.

    A load failure is recorded rather than raised, so the process stays up and can
    explain itself over `/health`. Set REQUIRE_MODEL_AT_STARTUP=1 for fail-fast.
    """
    bundle_path = os.environ.get(
        "MODEL_BUNDLE_PATH", "models/registry/sentiment-classifier/v1"
    )
    try:
        state.load(bundle_path)
    except Exception as exc:  # noqa: BLE001 - reported through /health, not swallowed
        state.load_error = f"{type(exc).__name__}: {exc}"
        logger.error("Model bundle could not be loaded: %s", state.load_error)
        if os.environ.get("REQUIRE_MODEL_AT_STARTUP") == "1":
            raise
        logger.warning("Starting unready. /ready will report 503 until a model loads.")
    yield


app = FastAPI(
    title="MLOps Sentiment API",
    description="Production sentiment analysis inference service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness. Always 200 while the process is running, with the model state in body."""
    return HealthResponse(
        status="healthy" if state.ready else "degraded",
        model_loaded=state.ready,
        model_version=state.model_version,
        uptime_seconds=round(time.time() - state.start_time, 1),
        total_predictions=state.prediction_count,
        detail=None if state.ready else (state.load_error or "no model loaded"),
    )


@app.get("/ready", response_model=ReadyResponse)
async def ready(response: Response):
    """Readiness. 503 until a validated bundle is actually usable.

    Route traffic on this, not on /health: a live process with no model can accept a
    connection and fail every prediction, which is the failure mode worth avoiding.
    """
    if not state.ready:
        response.status_code = 503
    return ReadyResponse(
        ready=state.ready,
        model_version=state.model_version,
        detail=None if state.ready else (state.load_error or "no model loaded"),
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

    # Structured record for monitoring. Deliberately stores the *length* of the input,
    # never the input itself, so the log cannot become a copy of user content.
    state.prediction_log.append(
        {
            "timestamp": datetime.now(UTC).isoformat(),
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
    """In-process serving statistics over a bounded recent window.

    This is a JSON summary, not a Prometheus endpoint, and it resets when the process
    restarts. docs/PRODUCTION.md describes what a real monitoring setup would add and
    why none of it is pretended at here.
    """
    recent = list(state.prediction_log)[-100:]
    latencies = [entry["latency_ms"] for entry in recent]
    predicted = [entry["prediction"] for entry in recent]
    distribution: dict[str, int] = {}
    for label in predicted:
        distribution[label] = distribution.get(label, 0) + 1

    return {
        "ready": state.ready,
        "model_version": state.model_version,
        "total_predictions": state.prediction_count,
        "window_size": len(recent),
        "recent_avg_latency_ms": round(float(np.mean(latencies)), 3) if latencies else 0,
        "recent_p95_latency_ms": (
            round(float(np.percentile(latencies, 95)), 3) if latencies else 0
        ),
        "recent_prediction_distribution": distribution,
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
