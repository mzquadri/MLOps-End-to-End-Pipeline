"""Shared fixtures and bundle builders.

The lineage and gate-report shapes are constructed in one place so that tightening the
bundle contract means updating one helper rather than hunting through four test files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

DATASET_PROVENANCE = {
    "kind": "synthetic-fixture",
    "key": "test-fixture",
    "name": "Deterministic test fixture",
    "license": "Not applicable (generated in-process)",
    "citation": "Not applicable",
}


def make_lineage(
    data_hash: str = "test-data",
    *,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
    numeric_columns: list | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """A lineage record that satisfies the bundle contract."""
    lineage: dict[str, Any] = {
        "data_hash": data_hash,
        "dataset": dict(DATASET_PROVENANCE),
        "feature_schema": {
            "label_column": "sentiment",
            "numeric_columns": numeric_columns if numeric_columns is not None else [],
            "text_column": "review_text",
        },
        "split": {
            "random_state": random_state,
            "stratified": True,
            "test_size": test_size,
            "validation_size": validation_size,
        },
    }
    lineage.update(extra)
    return lineage


def make_gate_report(
    data_hash: str = "test-data",
    *,
    accuracy: float = 1.0,
    f1_weighted: float = 1.0,
    accuracy_over_baseline: float = 0.5,
    p95_latency_ms: float = 1.0,
    accuracy_threshold: float = 0.5,
    f1_threshold: float = 0.5,
    baseline_threshold: float = 0.2,
    latency_threshold: float = 500.0,
) -> dict[str, Any]:
    """A self-consistent evaluation report covering every required gate check."""
    checks = {
        "accuracy": {
            "value": accuracy,
            "threshold": accuracy_threshold,
            "passed": accuracy >= accuracy_threshold,
        },
        "f1_weighted": {
            "value": f1_weighted,
            "threshold": f1_threshold,
            "passed": f1_weighted >= f1_threshold,
        },
        "accuracy_over_baseline": {
            "value": accuracy_over_baseline,
            "threshold": baseline_threshold,
            "passed": accuracy_over_baseline >= baseline_threshold,
        },
        "latency_p95": {
            "value": p95_latency_ms,
            "threshold": latency_threshold,
            "passed": p95_latency_ms <= latency_threshold,
        },
    }
    overall = all(check["passed"] for check in checks.values())
    return {
        "data_hash": data_hash,
        "evaluation_status": "passed" if overall else "failed",
        "metrics": {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "accuracy_over_baseline": accuracy_over_baseline,
        },
        "latency": {"p95_latency_ms": p95_latency_ms},
        "performance_gate": {"overall_passed": overall, "checks": checks},
    }


def create_passing_bundle(path: Path, data_hash: str = "test-data") -> None:
    """Build a minimal, complete, gate-passing bundle at `path`."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from src.model_bundle import create_candidate_bundle, update_evaluation_report

    texts = ["great product", "excellent item", "bad product", "awful item"]
    labels = ["positive", "positive", "negative", "negative"]
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=100).fit(features, labels)

    create_candidate_bundle(
        str(path),
        model,
        {
            "tfidf": vectorizer,
            "scaler": None,
            "metadata": {"numeric_columns": [], "total_features": features.shape[1]},
        },
        {"validation_accuracy": 1.0, "validation_f1_weighted": 1.0},
        make_lineage(data_hash),
        features.shape[1],
    )
    update_evaluation_report(str(path), make_gate_report(data_hash))


@pytest.fixture
def sample_config() -> dict[str, Any]:
    return {
        "experiment": {"name": "test-experiment"},
        "data": {
            "dataset": "uci-sentiment-labelled-sentences",
            "cache_dir": "data/cache",
            "allow_download": False,
            "source": None,
            "allow_synthetic_fallback": True,
            "prefer_synthetic": True,
            "synthetic_rows": 200,
            "test_size": 0.2,
            "validation_size": 0.2,
            "random_state": 42,
            "text_column": "review_text",
            "label_column": "sentiment",
            "max_features": 200,
        },
        "model": {
            "type": "logistic_regression",
            "hyperparameters": {
                "logistic_regression": {"C": 1.0, "max_iter": 200, "solver": "lbfgs"},
                "random_forest": {"n_estimators": 10, "max_depth": 5},
                "svm": {"C": 1.0, "kernel": "rbf"},
            },
        },
        "training": {"cv_folds": 3, "scoring": "f1_weighted"},
        "validation": {
            "data": {
                "on_failure": "warn",
                "max_null_pct": 0.05,
                "max_duplicate_pct": 0.60,
                "min_class_ratio": 0.10,
            },
            "min_accuracy": 0.5,
            "min_f1": 0.5,
            "min_accuracy_over_baseline": 0.2,
            "max_latency_ms": 500,
        },
        "mlflow": {"enabled": False, "tracking_uri": "sqlite:///test_mlruns.db"},
    }


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """A small frame with enough vocabulary variation to be learnable but not trivial."""
    from src.data_pipeline import add_derived_columns
    from src.datasets import synthetic_fixture

    return add_derived_columns(synthetic_fixture(300, seed=0), "review_text", "sentiment")


@pytest.fixture(autouse=True)
def reset_serving_state():
    """Keep serving state from leaking between tests in either direction."""
    from src.serve import state

    state.reset()
    yield
    state.reset()
