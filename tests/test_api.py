"""
Tests for the FastAPI serving endpoints.
"""

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    """Create a test client with the app (model may not be loaded)."""
    from src.serve import app

    return TestClient(app)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "uptime_seconds" in data
        assert "total_predictions" in data

    def test_health_reports_model_status(self, client):
        response = client.get("/health")
        data = response.json()
        # Model may or may not be loaded in test; just check field exists
        assert isinstance(data["model_loaded"], bool)


# ---------------------------------------------------------------------------
# Predict endpoint (without model loaded)
# ---------------------------------------------------------------------------


class TestPredictEndpointNoModel:
    def test_predict_returns_503_without_model(self, client):
        """When no model is loaded, predict should return 503."""
        from src.serve import state

        # Ensure model is not loaded
        if state.model is not None:
            pytest.skip("Model is loaded in this environment")
        response = client.post("/predict", json={"text": "This is a test review"})
        assert response.status_code == 503

    def test_batch_predict_returns_503_without_model(self, client):
        from src.serve import state

        if state.model is not None:
            pytest.skip("Model is loaded in this environment")
        response = client.post("/predict/batch", json={"texts": ["test review"]})
        assert response.status_code == 503


# ---------------------------------------------------------------------------
# Predict endpoint (with mock model)
# ---------------------------------------------------------------------------


class TestPredictEndpointWithModel:
    @pytest.fixture(autouse=True)
    def setup_mock_model(self):
        """Set up a minimal mock model + transformers for prediction tests."""
        import pickle
        import tempfile

        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        from src.serve import state

        # Train a tiny model
        texts = [
            "great product love it",
            "excellent quality good",
            "terrible awful bad",
            "horrible waste money",
        ]
        labels = ["positive", "positive", "negative", "negative"]

        tfidf = TfidfVectorizer(max_features=50)
        X = tfidf.fit_transform(texts)
        model = LogisticRegression(max_iter=200)
        model.fit(X, labels)

        # Set state manually
        state.model = model
        state.feature_transformers = {
            "tfidf": tfidf,
            "scaler": None,
            "metadata": {"numeric_columns": []},
        }
        state.model_version = "test-v1"
        state.prediction_count = 0

        yield

        # Teardown
        state.model = None
        state.feature_transformers = None

    def test_single_prediction(self, client):
        response = client.post("/predict", json={"text": "This is a great product"})
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in ("positive", "negative")
        assert "confidence" in data
        assert "latency_ms" in data
        assert data["confidence"] > 0

    def test_batch_prediction(self, client):
        response = client.post(
            "/predict/batch",
            json={"texts": ["great quality", "terrible product", "love it"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 3
        assert "total_latency_ms" in data

    def test_prediction_increments_count(self, client):
        from src.serve import state

        initial = state.prediction_count
        client.post("/predict", json={"text": "test review"})
        assert state.prediction_count == initial + 1


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "total_predictions" in data
        assert "uptime_seconds" in data


# ---------------------------------------------------------------------------
# Request validation
# ---------------------------------------------------------------------------


class TestRequestValidation:
    def test_empty_text_rejected(self, client):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422  # Pydantic validation error

    def test_missing_text_rejected(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422
