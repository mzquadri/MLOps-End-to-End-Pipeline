"""Tests for text cleaning, content hashing, validation and its failure policy."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from src.data_pipeline import (
    DataValidationError,
    DataValidator,
    add_derived_columns,
    compute_data_hash,
    enforce_validation,
    load_and_preprocess,
    preprocess_text,
    save_data_manifest,
)


class TestPreprocessText:
    def test_lowercase(self):
        assert preprocess_text("HELLO WORLD") == "hello world"

    def test_removes_url(self):
        assert "http" not in preprocess_text("Visit http://example.com for info")

    def test_removes_html(self):
        assert "<b>" not in preprocess_text("Hello <b>world</b>")

    def test_drops_punctuation(self):
        result = preprocess_text("great!!! product... #1")
        assert "!" not in result
        assert "#" not in result

    def test_is_stateless(self):
        """Cleaning must not depend on any other row, or it could leak across a split."""
        assert preprocess_text("Great Product") == preprocess_text("Great Product")


class TestDataHash:
    def test_deterministic(self, sample_df):
        assert compute_data_hash(sample_df) == compute_data_hash(sample_df)

    def test_changes_with_content(self, sample_df):
        modified = sample_df.copy()
        modified.iloc[0, 0] = "changed text"
        assert compute_data_hash(sample_df) != compute_data_hash(modified)


class TestDataManifest:
    def test_writes_to_the_requested_directory(self, sample_df, tmp_path):
        """The manifest goes where the caller says, never into the working tree."""
        data_hash = compute_data_hash(sample_df)
        path = save_data_manifest(data_hash, "fixture", 200, 4, output_dir=str(tmp_path))
        assert os.path.exists(path)
        assert str(tmp_path) in path
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        assert manifest["data_hash"] == data_hash
        assert manifest["n_rows"] == 200


class TestDataValidator:
    def test_null_check_passes(self, sample_config, sample_df):
        assert DataValidator(sample_config).check_nulls(sample_df)["passed"] is True

    def test_null_check_fails(self, sample_config):
        frame = pd.DataFrame({"a": [None] * 50 + [1] * 50})
        result = DataValidator(sample_config).check_nulls(frame, max_null_pct=0.05)
        assert result["passed"] is False

    def test_duplicate_check_reports_percentage(self, sample_config, sample_df):
        result = DataValidator(sample_config).check_duplicates(sample_df)
        assert "duplicate_percentage" in result

    def test_class_balance(self, sample_config, sample_df):
        assert DataValidator(sample_config).check_class_balance(
            sample_df["sentiment"]
        )["passed"] is True

    def test_overall_fails_when_any_check_fails(self, sample_config):
        frame = pd.DataFrame(
            {"review_text": [None] * 10, "sentiment": ["positive"] * 10}
        )
        result = DataValidator(sample_config).run_all_checks(frame, "sentiment")
        assert result["overall_passed"] is False

    def test_psi_is_zero_for_identical_distributions(self, sample_config):
        import numpy as np

        reference = np.random.default_rng(42).normal(0, 1, 1000)
        assert DataValidator(sample_config).compute_psi(reference, reference) < 0.01

    def test_psi_is_finite_when_a_bucket_is_empty(self, sample_config):
        """Laplace smoothing keeps a disjoint comparison from returning infinity."""
        import numpy as np

        left = np.zeros(100)
        right = np.ones(100) * 50
        psi = DataValidator(sample_config).compute_psi(left, right)
        assert np.isfinite(psi)


class TestValidationPolicy:
    """The behaviour that made the original synthetic run look credible."""

    def _failing_report(self):
        return {
            "overall_passed": False,
            "null_check": {"passed": True},
            "duplicate_check": {"passed": False, "duplicate_percentage": 0.99},
            "class_balance": {"passed": True},
        }

    def test_error_policy_stops_the_run(self):
        with pytest.raises(DataValidationError, match="duplicate_check"):
            enforce_validation(self._failing_report(), "error")

    def test_warn_policy_continues(self):
        enforce_validation(self._failing_report(), "warn")

    def test_passing_report_is_a_no_op(self):
        enforce_validation({"overall_passed": True}, "error")


class TestDerivedColumns:
    def test_adds_clean_text_and_length_features(self):
        frame = pd.DataFrame(
            {"review_text": ["Great Product!"], "sentiment": ["positive"]}
        )
        result = add_derived_columns(frame, "review_text", "sentiment")
        assert result.loc[0, "review_text_clean"] == "great product"
        assert result.loc[0, "review_length"] == len("Great Product!")
        assert result.loc[0, "word_count"] == 2


class TestLoadAndPreprocess:
    def test_synthetic_path_returns_data_hash_provenance_and_report(self, sample_config):
        frame, data_hash, provenance, report = load_and_preprocess(sample_config)
        assert len(frame) > 0
        assert len(data_hash) == 12
        assert provenance["kind"] == "synthetic-fixture"
        assert provenance["license"].startswith("Not applicable")
        assert "overall_passed" in report

    def test_writes_nothing_to_the_working_tree(self, sample_config, tmp_path, monkeypatch):
        """Loading data is a pure read. It used to drop a manifest into ./data."""
        monkeypatch.chdir(tmp_path)
        load_and_preprocess(sample_config)
        assert not (tmp_path / "data").exists()

    def test_refuses_to_substitute_synthetic_data_when_not_allowed(
        self, sample_config, tmp_path
    ):
        """The reference run must fail loudly rather than degrade silently.

        The cache directory is redirected at tmp_path so the test is hermetic. Pointing
        it at the real `data/cache` made the result depend on whether the developer had
        run the pipeline before, which is how a test quietly stops testing anything.
        """
        config = dict(sample_config)
        config["data"] = {
            **sample_config["data"],
            "cache_dir": str(tmp_path / "empty-cache"),
            "allow_synthetic_fallback": False,
            "prefer_synthetic": False,
            "allow_download": False,
        }
        with pytest.raises(DataValidationError, match="Refusing to fall back"):
            load_and_preprocess(config)
