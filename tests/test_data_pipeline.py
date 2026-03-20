"""
Tests for the data pipeline module.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers – create a minimal config & sample DataFrame
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    return {
        "data": {
            "source": "data/reviews.csv",
            "test_size": 0.2,
            "random_state": 42,
            "text_column": "review_text",
            "label_column": "sentiment",
            "max_features": 500,
        },
        "validation": {
            "min_accuracy": 0.80,
            "min_f1": 0.78,
            "max_drift_psi": 0.25,
            "max_latency_ms": 200,
        },
        "mlflow": {"tracking_uri": "sqlite:///test_mlruns.db"},
    }


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    n = 200
    texts = [f"sample review text number {i}" for i in range(n)]
    sentiments = rng.choice(["positive", "negative"], size=n, p=[0.6, 0.4])
    return pd.DataFrame(
        {
            "review_text": texts,
            "sentiment": sentiments,
            "review_length": [len(t) for t in texts],
            "word_count": [len(t.split()) for t in texts],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPreprocessText:
    def test_lowercase(self):
        from src.data_pipeline import preprocess_text

        assert preprocess_text("HELLO WORLD") == "hello world"

    def test_remove_url(self):
        from src.data_pipeline import preprocess_text

        result = preprocess_text("Visit http://example.com for info")
        assert "http" not in result

    def test_remove_html(self):
        from src.data_pipeline import preprocess_text

        result = preprocess_text("Hello <b>world</b>")
        assert "<b>" not in result

    def test_special_chars(self):
        from src.data_pipeline import preprocess_text

        result = preprocess_text("great!!! product... #1")
        assert "!" not in result
        assert "#" not in result


class TestDataHash:
    def test_deterministic(self, sample_df):
        from src.data_pipeline import compute_data_hash

        h1 = compute_data_hash(sample_df)
        h2 = compute_data_hash(sample_df)
        assert h1 == h2

    def test_different_for_different_data(self, sample_df):
        from src.data_pipeline import compute_data_hash

        h1 = compute_data_hash(sample_df)
        modified = sample_df.copy()
        modified.iloc[0, 0] = "changed text"
        h2 = compute_data_hash(modified)
        assert h1 != h2


class TestDataManifest:
    def test_saves_json(self, sample_df):
        from src.data_pipeline import compute_data_hash, save_data_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            data_hash = compute_data_hash(sample_df)
            path = save_data_manifest(data_hash, "test.csv", 200, 4, output_dir=tmpdir)
            assert os.path.exists(path)
            import json

            with open(path) as f:
                manifest = json.load(f)
            assert manifest["data_hash"] == data_hash
            assert manifest["n_rows"] == 200


class TestDataValidator:
    def test_null_check_passes(self, sample_config, sample_df):
        from src.data_pipeline import DataValidator

        validator = DataValidator(sample_config)
        result = validator.check_nulls(sample_df)
        assert result["passed"] is True

    def test_null_check_fails(self, sample_config):
        from src.data_pipeline import DataValidator

        df = pd.DataFrame({"a": [None] * 50 + [1] * 50})
        validator = DataValidator(sample_config)
        result = validator.check_nulls(df, max_null_pct=0.05)
        assert result["passed"] is False

    def test_duplicate_check(self, sample_config, sample_df):
        from src.data_pipeline import DataValidator

        validator = DataValidator(sample_config)
        result = validator.check_duplicates(sample_df)
        assert "duplicate_percentage" in result

    def test_class_balance(self, sample_config, sample_df):
        from src.data_pipeline import DataValidator

        validator = DataValidator(sample_config)
        result = validator.check_class_balance(sample_df["sentiment"])
        assert result["passed"] is True

    def test_psi_identical_distributions(self, sample_config):
        from src.data_pipeline import DataValidator

        validator = DataValidator(sample_config)
        rng = np.random.default_rng(42)
        ref = rng.normal(0, 1, 1000)
        psi = validator.compute_psi(ref, ref)
        assert psi < 0.01  # nearly zero for identical


class TestDemoData:
    def test_generate_demo_data(self):
        from src.data_pipeline import _generate_demo_data

        df = _generate_demo_data(500)
        assert len(df) == 500
        assert "review_text" in df.columns
        assert "sentiment" in df.columns
        assert set(df["sentiment"].unique()) == {"positive", "negative"}


class TestLoadAndPreprocess:
    def test_runs_with_synthetic_data(self, sample_config):
        """When source file doesn't exist, it should generate demo data."""
        from src.data_pipeline import load_and_preprocess

        df, data_hash = load_and_preprocess(sample_config)
        assert len(df) > 0
        assert isinstance(data_hash, str)
        assert len(data_hash) == 12
