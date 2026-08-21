"""End-to-end tests for the reference pipeline.

These run the same code path the documented reference run uses, on the offline fixture
so they stay fast and network-free. They assert the properties that are easy to claim in
a README and hard to keep true: no leakage, a gate that can actually refuse, provenance
that survives into the published bundle, and a run that reproduces.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from src.model_bundle import (
    EVALUATION_REPORT_FILE,
    LINEAGE_FILE,
    MANIFEST_FILE,
    file_sha256,
    validate_bundle,
)
from src.pipeline import GateFailure, run


@pytest.fixture
def ci_config():
    with open("configs/ci_config.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_in(workspace: Path, config: dict) -> dict:
    return run(
        config,
        candidate_dir=str(workspace / "candidates"),
        registry_dir=str(workspace / "registry"),
        results_dir=str(workspace / "results"),
    )


class TestReferenceRun:
    def test_completes_and_promotes_to_production(self, tmp_path, ci_config):
        summary = run_in(tmp_path, ci_config)

        assert summary["performance_gate"]["overall_passed"] is True
        assert summary["registry"]["registered"] is True
        assert summary["registry"]["stage"] == "production"
        assert summary["registry"]["version"] == "v1"

        bundle = tmp_path / "registry" / "sentiment-classifier" / "v1"
        validate_bundle(bundle, require_evaluated=True, require_gate_passed=True)

    def test_writes_a_machine_readable_summary(self, tmp_path, ci_config):
        run_in(tmp_path, ci_config)
        summary = json.loads(
            (tmp_path / "results" / "reference_run.json").read_text(encoding="utf-8")
        )
        for key in ("dataset", "split", "training_metrics", "test_metrics", "registry"):
            assert key in summary
        assert summary["split"]["train_rows"] > summary["split"]["test_rows"]

    def test_curve_points_are_persisted(self, tmp_path, ci_config):
        run_in(tmp_path, ci_config)
        curves = json.loads(
            (tmp_path / "results" / "evaluation_curves.json").read_text(encoding="utf-8")
        )
        assert len(curves["roc"]["fpr"]) == len(curves["roc"]["tpr"])
        assert curves["precision_recall"]["precision"]


class TestSplitDiscipline:
    def test_three_splits_are_disjoint_and_sized_as_configured(self, ci_config):
        import numpy as np

        from src.data_pipeline import load_and_preprocess
        from src.train import three_way_split

        frame, _, _, _ = load_and_preprocess(ci_config)
        labels = frame[ci_config["data"]["label_column"]].to_numpy()
        train, validation, test, *_ = three_way_split(frame, labels, 0.2, 0.2, 42)

        total = len(frame)
        assert len(train) + len(validation) + len(test) == total
        assert len(test) == pytest.approx(total * 0.2, abs=2)
        assert len(validation) == pytest.approx(total * 0.2, abs=2)

        indices = [set(part.index) for part in (train, validation, test)]
        assert not indices[0] & indices[1]
        assert not indices[0] & indices[2]
        assert not indices[1] & indices[2]
        assert np.array_equal(
            three_way_split(frame, labels, 0.2, 0.2, 42)[2].index.to_numpy(),
            test.index.to_numpy(),
        ), "the same seed must reproduce the same test split"

    def test_vocabulary_is_fitted_on_training_rows_only(self, ci_config):
        """A token that occurs only outside training must not enter the vocabulary.

        This is the concrete form of "no leakage". If TF-IDF were fitted on the full
        frame, the marker below would appear in the fitted feature names.
        """
        from src.data_pipeline import load_and_preprocess
        from src.feature_store import FeatureStore
        from src.train import three_way_split

        frame, _, _, _ = load_and_preprocess(ci_config)
        labels = frame[ci_config["data"]["label_column"]].to_numpy()
        train, validation, test, *_ = three_way_split(frame, labels, 0.2, 0.2, 42)

        marker = "zzhapaxmarker"
        test = test.copy()
        test.loc[test.index[0], "review_text_clean"] += f" {marker}"

        store = FeatureStore(ci_config)
        store.fit_transform_train(train)
        assert marker not in set(store.tfidf.get_feature_names_out())

        # And the transform-only path still accepts the row, ignoring the unseen token.
        assert store.transform(test).shape[0] == len(test)


class TestGate:
    def test_an_unreachable_threshold_blocks_registration(self, tmp_path, ci_config):
        config = copy.deepcopy(ci_config)
        config["validation"]["min_accuracy"] = 1.01

        with pytest.raises(GateFailure, match="accuracy"):
            run_in(tmp_path, config)

        # Nothing was published, and the failure is recorded on the candidate.
        assert not (tmp_path / "registry" / "sentiment-classifier").exists()
        report = json.loads(
            (tmp_path / "candidates" / "sentiment-classifier" / EVALUATION_REPORT_FILE)
            .read_text(encoding="utf-8")
        )
        assert report["evaluation_status"] == "failed"
        summary = json.loads(
            (tmp_path / "results" / "reference_run.json").read_text(encoding="utf-8")
        )
        assert summary["registry"]["registered"] is False

    def test_baseline_margin_is_part_of_the_gate(self, tmp_path, ci_config):
        config = copy.deepcopy(ci_config)
        config["validation"]["min_accuracy_over_baseline"] = 0.99

        with pytest.raises(GateFailure, match="accuracy_over_baseline"):
            run_in(tmp_path, config)


class TestProvenance:
    def test_dataset_licence_reaches_the_published_bundle(self, tmp_path, ci_config):
        run_in(tmp_path, ci_config)
        lineage = json.loads(
            (tmp_path / "registry" / "sentiment-classifier" / "v1" / LINEAGE_FILE)
            .read_text(encoding="utf-8")
        )
        assert lineage["dataset"]["license"]
        assert lineage["dataset"]["kind"] == "synthetic-fixture"
        assert lineage["data_validation"]["overall_passed"] is True

    def test_bundle_lineage_contains_no_absolute_paths(self, tmp_path, ci_config):
        run_in(tmp_path, ci_config)
        raw = (
            tmp_path / "registry" / "sentiment-classifier" / "v1" / LINEAGE_FILE
        ).read_text(encoding="utf-8")
        assert "C:\\" not in raw and "/home/" not in raw and "/Users/" not in raw

    def test_manifest_checksums_match_the_published_files(self, tmp_path, ci_config):
        run_in(tmp_path, ci_config)
        bundle = tmp_path / "registry" / "sentiment-classifier" / "v1"
        manifest = json.loads((bundle / MANIFEST_FILE).read_text(encoding="utf-8"))
        for filename, digest in manifest["artifact_checksums"].items():
            assert file_sha256(bundle / filename) == digest


class TestReproducibility:
    def test_two_runs_agree_on_data_hash_and_metrics(self, tmp_path, ci_config):
        first = run_in(tmp_path / "run-a", ci_config)
        second = run_in(tmp_path / "run-b", ci_config)

        assert first["data_hash"] == second["data_hash"]
        assert first["test_metrics"]["accuracy"] == second["test_metrics"]["accuracy"]
        assert first["test_metrics"]["f1_weighted"] == second["test_metrics"]["f1_weighted"]
        assert first["training_metrics"]["cv_mean"] == second["training_metrics"]["cv_mean"]
        assert first["model"]["feature_dimension"] == second["model"]["feature_dimension"]

    def test_model_artifact_is_byte_identical_across_runs(self, tmp_path, ci_config):
        """Same seed, same data, same code should serialise to the same bytes.

        Latency and timestamps vary between runs, which is why they live in the report
        rather than in the model artifact.
        """
        first = run_in(tmp_path / "run-a", ci_config)
        second = run_in(tmp_path / "run-b", ci_config)
        assert (
            first["registry"]["artifact_checksums"]["model.joblib"]
            == second["registry"]["artifact_checksums"]["model.joblib"]
        )
