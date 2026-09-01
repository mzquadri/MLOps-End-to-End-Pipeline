"""Tests for the inference service.

Every client here is built with `with TestClient(app)`, which is what actually runs the
application lifespan. The previous suite used a bare `TestClient(app)`, so startup never
executed and the tests silently exercised a service that had never tried to load a model.
Two of them would have failed immediately if it had.
"""

from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

from src.serve import MAX_TEXT_LENGTH


@pytest.fixture
def unready_client(monkeypatch, tmp_path):
    """A service pointed at a bundle that does not exist.

    It must start anyway and be able to explain itself, rather than dying during
    startup and leaving an operator with a crash loop and no endpoint to query.
    """
    from src.serve import app

    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(tmp_path / "no-such-bundle"))
    monkeypatch.delenv("REQUIRE_MODEL_AT_STARTUP", raising=False)
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def production_bundle(tmp_path_factory):
    """Run the real pipeline once and promote a bundle to production."""
    from src.pipeline import run

    workspace = tmp_path_factory.mktemp("serving")
    with open("configs/ci_config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    summary = run(
        config,
        candidate_dir=str(workspace / "candidates"),
        registry_dir=str(workspace / "registry"),
        results_dir=str(workspace / "results"),
    )
    return workspace, summary


@pytest.fixture
def ready_client(monkeypatch, production_bundle):
    """A service loading a genuinely promoted bundle through the real code path."""
    from src.serve import app

    workspace, summary = production_bundle
    bundle = workspace / "registry" / "sentiment-classifier" / summary["registry"]["version"]
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(bundle))
    with TestClient(app) as client:
        yield client


class TestUnreadyService:
    def test_health_is_live_but_reports_degraded(self, unready_client):
        response = unready_client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False
        assert body["detail"], "a degraded service must say why"

    def test_ready_returns_503(self, unready_client):
        response = unready_client.get("/ready")
        assert response.status_code == 503
        assert response.json()["ready"] is False

    def test_predict_returns_503(self, unready_client):
        response = unready_client.post("/predict", json={"text": "a review"})
        assert response.status_code == 503

    def test_batch_predict_returns_503(self, unready_client):
        response = unready_client.post("/predict/batch", json={"texts": ["a review"]})
        assert response.status_code == 503

    def test_fail_fast_mode_refuses_to_start(self, monkeypatch, tmp_path):
        """The opt-in behaviour for deployments that prefer not to start at all."""
        from src.serve import app

        monkeypatch.setenv("MODEL_BUNDLE_PATH", str(tmp_path / "missing"))
        monkeypatch.setenv("REQUIRE_MODEL_AT_STARTUP", "1")
        with pytest.raises(RuntimeError):
            with TestClient(app):
                pass


class TestReadyService:
    def test_ready_returns_200_with_the_promoted_version(self, ready_client):
        response = ready_client.get("/ready")
        assert response.status_code == 200
        assert response.json()["ready"] is True
        assert response.json()["model_version"] == "v1"

    def test_health_is_healthy(self, ready_client):
        body = ready_client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["model_loaded"] is True
        assert body["detail"] is None

    def test_single_prediction_uses_the_bundled_transformers(self, ready_client):
        response = ready_client.post(
            "/predict", json={"text": "this product is excellent and works great"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["prediction"] in {"positive", "negative"}
        assert 0.0 < body["confidence"] <= 1.0
        assert body["latency_ms"] >= 0
        assert body["model_version"] == "v1"

    def test_batch_prediction(self, ready_client):
        response = ready_client.post(
            "/predict/batch",
            json={"texts": ["excellent quality", "terrible waste of money", "i love it"]},
        )
        assert response.status_code == 200
        assert len(response.json()["predictions"]) == 3

    def test_prediction_count_increments(self, ready_client):
        before = ready_client.get("/metrics").json()["total_predictions"]
        ready_client.post("/predict", json={"text": "a review"})
        after = ready_client.get("/metrics").json()["total_predictions"]
        assert after == before + 1

    def test_metrics_reports_the_recent_window(self, ready_client):
        ready_client.post("/predict", json={"text": "excellent product"})
        body = ready_client.get("/metrics").json()
        assert body["ready"] is True
        assert body["window_size"] >= 1
        assert sum(body["recent_prediction_distribution"].values()) == body["window_size"]


class TestPrivacy:
    def test_prediction_log_never_stores_request_text(self, ready_client):
        from src.serve import state

        secret = "an unusually distinctive private review sentence"
        assert ready_client.post("/predict", json={"text": secret}).status_code == 200
        assert secret not in repr(list(state.prediction_log))
        assert state.prediction_log[-1]["text_length"] == len(secret)

    def test_prediction_log_is_bounded(self, ready_client):
        """An unbounded log is a slow memory leak in a long-lived service."""
        from src.serve import PREDICTION_LOG_CAPACITY, state

        for _ in range(PREDICTION_LOG_CAPACITY + 25):
            ready_client.post("/predict", json={"text": "short review"})
        assert len(state.prediction_log) == PREDICTION_LOG_CAPACITY


class TestRequestValidation:
    def test_empty_text_rejected(self, unready_client):
        assert unready_client.post("/predict", json={"text": ""}).status_code == 422

    def test_missing_text_rejected(self, unready_client):
        assert unready_client.post("/predict", json={}).status_code == 422

    def test_batch_size_is_capped(self, unready_client):
        response = unready_client.post(
            "/predict/batch", json={"texts": ["review"] * 101}
        )
        assert response.status_code == 422

    def test_text_length_is_capped(self, unready_client):
        """Vectorising is linear in input length, so an unbounded field is a cost hole.

        Measured before the cap existed: a 10 MB body took 12.2 s of CPU and 209 MB of
        peak memory to transform, for one request.
        """
        response = unready_client.post(
            "/predict", json={"text": "x" * (MAX_TEXT_LENGTH + 1)}
        )
        assert response.status_code == 422

    def test_text_at_the_cap_is_accepted(self, unready_client):
        """The cap must bound cost without narrowing the domain.

        The longest training sentence is 479 characters, so a review at the limit is far
        outside anything the model was fitted on and still has to be accepted -- a 503 from
        an unloaded model, not a 422 from validation.
        """
        response = unready_client.post(
            "/predict", json={"text": "x" * MAX_TEXT_LENGTH}
        )
        assert response.status_code == 503

    def test_batch_caps_each_item_not_just_the_count(self, unready_client):
        """A hundred-item limit bounds nothing if each item may be arbitrarily large."""
        response = unready_client.post(
            "/predict/batch", json={"texts": ["ok", "x" * (MAX_TEXT_LENGTH + 1)]}
        )
        assert response.status_code == 422
