"""Regression tests for atomic, promotion-gated model bundles."""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.data_pipeline import compute_data_hash, preprocess_text
from src.feature_store import FeatureStore
from src.model_bundle import (
    ARTIFACT_FILES,
    EVALUATION_REPORT_FILE,
    MANIFEST_FILE,
    TRANSFORMERS_FILE,
    BundleIntegrityError,
    create_candidate_bundle,
    load_trusted_bundle,
    update_evaluation_report,
    validate_bundle,
)
from tests.conftest import DATASET_PROVENANCE, make_gate_report, make_lineage

FAKE_VALIDATION = {
    "overall_passed": True,
    "duplicate_check": {"duplicate_percentage": 0.0, "passed": True},
    "class_balance": {"min_class_percentage": 0.5, "passed": True},
    "null_check": {"passed": True},
}


@pytest.fixture
def bundle_config(tmp_path):
    return {
        "experiment": {"name": "bundle-test"},
        "data": {
            "source": str(tmp_path / "missing.csv"),
            "test_size": 0.25,
            "random_state": 42,
            "text_column": "review_text",
            "label_column": "sentiment",
            "max_features": 100,
            "validation_size": 0.2,
            "allow_synthetic_fallback": True,
            "prefer_synthetic": True,
        },
        "model": {
            "type": "logistic_regression",
            "hyperparameters": {
                "logistic_regression": {
                    "C": 1.0,
                    "max_iter": 200,
                    "solver": "lbfgs",
                }
            },
        },
        "training": {"cv_folds": 2, "scoring": "f1_weighted"},
        "validation": {
            "data": {"on_failure": "warn", "max_duplicate_pct": 0.95},
            "min_accuracy": 0.5,
            "min_f1": 0.5,
            "min_accuracy_over_baseline": 0.2,
            "max_latency_ms": 500,
        },
    }


@pytest.fixture
def bundle_df():
    texts = [
        "great product love it",
        "great quality love it",
        "excellent product works well",
        "excellent quality works well",
        "terrible product avoid it",
        "terrible quality avoid it",
        "awful product waste money",
        "awful quality waste money",
    ] * 4
    labels = ["positive"] * 4 + ["negative"] * 4
    labels *= 4
    return pd.DataFrame(
        {
            "review_text": texts,
            "review_text_clean": [preprocess_text(text) for text in texts],
            "sentiment": labels,
            "review_length": [len(text) for text in texts],
            "word_count": [len(text.split()) for text in texts],
        }
    )


def make_candidate(path: Path, config, df: pd.DataFrame) -> Path:
    labels = df["sentiment"].to_numpy()
    train_df, test_df, y_train, _ = train_test_split(
        df,
        labels,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=labels,
    )
    store = FeatureStore(config, cache_dir=str(path.parent / "cache"))
    X_train, _ = store.get_features(train_df, test_df, use_cache=False)
    model = LogisticRegression(max_iter=200).fit(X_train, y_train)
    create_candidate_bundle(
        str(path),
        model,
        store.export_transformers(),
        {"accuracy": 1.0, "f1_weighted": 1.0},
        make_lineage(
            compute_data_hash(df),
            test_size=config["data"]["test_size"],
            validation_size=config["data"].get("validation_size", 0.2),
            random_state=config["data"]["random_state"],
            numeric_columns=store.metadata["numeric_columns"],
            data_rows=len(df),
            evaluation_rows=len(test_df),
            training_rows=len(train_df),
        ),
        X_train.shape[1],
    )
    return path


def passing_report(data_hash: str) -> dict:
    return make_gate_report(data_hash)


def failed_report(data_hash: str) -> dict:
    return make_gate_report(
        data_hash, accuracy=0.1, f1_weighted=0.1, accuracy_over_baseline=-0.05
    )


def test_transform_only_path_preserves_fitted_state(bundle_config, bundle_df, tmp_path):
    train_df, test_df = train_test_split(bundle_df, test_size=0.25, random_state=42)
    store = FeatureStore(bundle_config, cache_dir=str(tmp_path / "cache"))
    store.get_features(train_df, use_cache=False)
    vocabulary = dict(store.tfidf.vocabulary_)
    mean = store.scaler.mean_.copy()
    scale = store.scaler.scale_.copy()

    transformed = store.transform(test_df)

    assert transformed.shape[1] == store.metadata["total_features"]
    assert store.tfidf.vocabulary_ == vocabulary
    np.testing.assert_array_equal(store.scaler.mean_, mean)
    np.testing.assert_array_equal(store.scaler.scale_, scale)


def test_bundle_manifest_and_dimension_are_valid(bundle_config, bundle_df, tmp_path):
    bundle = make_candidate(tmp_path / "candidate", bundle_config, bundle_df)
    manifest = validate_bundle(bundle, require_evaluated=False)

    assert set(manifest["artifact_checksums"]) == set(ARTIFACT_FILES)
    assert manifest["required_files"][-1] == MANIFEST_FILE
    model, transformers, loaded_manifest = load_trusted_bundle(str(bundle))
    assert model.n_features_in_ == loaded_manifest["expected_feature_dimension"]
    assert transformers["metadata"]["total_features"] == model.n_features_in_


def test_checksum_tampering_is_rejected_before_loading(
    bundle_config, bundle_df, tmp_path
):
    bundle = make_candidate(tmp_path / "candidate", bundle_config, bundle_df)
    with (bundle / TRANSFORMERS_FILE).open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(BundleIntegrityError, match="Checksum mismatch"):
        load_trusted_bundle(str(bundle))


def test_registry_requires_passed_staging_then_production(
    bundle_config, bundle_df, tmp_path
):
    from src.model_registry import LocalModelRegistry

    bundle = make_candidate(tmp_path / "candidate", bundle_config, bundle_df)
    registry = LocalModelRegistry(str(tmp_path / "registry"))
    with pytest.raises(BundleIntegrityError, match="not been evaluated"):
        registry.register_model("sentiment", str(bundle), stage="staging")
    with pytest.raises(ValueError, match="requires promotion"):
        registry.register_model("sentiment", str(bundle), stage="production")

    update_evaluation_report(
        str(bundle), failed_report(compute_data_hash(bundle_df))
    )
    with pytest.raises(BundleIntegrityError, match="did not pass"):
        registry.register_model("sentiment", str(bundle), stage="staging")

    update_evaluation_report(
        str(bundle), passing_report(compute_data_hash(bundle_df))
    )
    version = registry.register_model("sentiment", str(bundle), stage="staging")
    registry.transition_stage("sentiment", version, "production")
    production = registry.get_production_model("sentiment")

    assert production["stage"] == "production"
    assert not Path(production["bundle_path"]).is_absolute()
    registered = tmp_path / "registry" / production["bundle_path"]
    validate_bundle(registered, require_gate_passed=True)


def test_serving_loads_one_complete_bundle_and_rejects_missing_transformers(
    bundle_config, bundle_df, tmp_path
):
    from src.serve import AppState

    bundle = make_candidate(tmp_path / "candidate", bundle_config, bundle_df)
    update_evaluation_report(
        str(bundle), passing_report(compute_data_hash(bundle_df))
    )
    state = AppState()
    with pytest.raises(RuntimeError, match="registry index"):
        state.load(str(bundle))

    from src.model_registry import LocalModelRegistry

    registry = LocalModelRegistry(str(tmp_path / "registry"))
    version = registry.register_model("sentiment", str(bundle), stage="staging")
    registry.transition_stage("sentiment", version, "production")
    production_bundle = tmp_path / "registry" / "sentiment" / version
    state.load(str(production_bundle))
    result = state.predict_single("great product")
    assert state.ready is True
    assert result["prediction"] in {"positive", "negative"}

    (production_bundle / TRANSFORMERS_FILE).unlink()
    with pytest.raises(BundleIntegrityError, match="Missing bundle artifact"):
        state.load(str(production_bundle))
    assert state.ready is False


def test_self_attested_gate_without_checks_is_rejected(
    bundle_config, bundle_df, tmp_path
):
    bundle = make_candidate(tmp_path / "candidate", bundle_config, bundle_df)
    report = passing_report(compute_data_hash(bundle_df))
    report["performance_gate"]["checks"] = {}

    with pytest.raises(BundleIntegrityError, match="gate inventory"):
        update_evaluation_report(str(bundle), report)


def test_registry_rolls_back_bundle_when_index_write_fails(
    bundle_config, bundle_df, tmp_path, monkeypatch
):
    from src.model_registry import LocalModelRegistry

    bundle = make_candidate(tmp_path / "candidate", bundle_config, bundle_df)
    update_evaluation_report(
        str(bundle), passing_report(compute_data_hash(bundle_df))
    )
    registry = LocalModelRegistry(str(tmp_path / "registry"))

    def fail_index_write():
        raise OSError("simulated index failure")

    monkeypatch.setattr(registry, "_save_index", fail_index_write)
    with pytest.raises(OSError, match="simulated index failure"):
        registry.register_model("sentiment", str(bundle), stage="staging")

    assert registry.list_versions("sentiment") == []
    assert not (tmp_path / "registry" / "sentiment" / "v1").exists()


def test_interrupted_candidate_replacement_is_recovered(
    bundle_config, bundle_df, tmp_path
):
    bundle = make_candidate(tmp_path / "candidate", bundle_config, bundle_df)
    backup = bundle.with_name(".candidate.backup")
    bundle.rename(backup)

    update_evaluation_report(
        str(bundle), passing_report(compute_data_hash(bundle_df))
    )

    assert bundle.is_dir()
    assert not backup.exists()
    validate_bundle(bundle, require_gate_passed=True)


def test_registry_reconciles_complete_orphan_bundle(
    bundle_config, bundle_df, tmp_path
):
    from src.model_registry import LocalModelRegistry

    bundle = make_candidate(tmp_path / "candidate", bundle_config, bundle_df)
    update_evaluation_report(
        str(bundle), passing_report(compute_data_hash(bundle_df))
    )
    orphan = tmp_path / "registry" / "sentiment" / "v1"
    shutil.copytree(bundle, orphan)

    registry = LocalModelRegistry(str(tmp_path / "registry"))
    version = registry.register_model("sentiment", str(bundle), stage="staging")

    assert version == "v1"
    assert registry.list_versions("sentiment")[0]["stage"] == "staging"


def test_stage_transition_rolls_back_when_index_write_fails(
    bundle_config, bundle_df, tmp_path, monkeypatch
):
    from src.model_registry import LocalModelRegistry

    bundle = make_candidate(tmp_path / "candidate", bundle_config, bundle_df)
    update_evaluation_report(
        str(bundle), passing_report(compute_data_hash(bundle_df))
    )
    registry = LocalModelRegistry(str(tmp_path / "registry"))
    version = registry.register_model("sentiment", str(bundle), stage="staging")

    def fail_index_write():
        raise OSError("simulated transition failure")

    monkeypatch.setattr(registry, "_save_index", fail_index_write)
    with pytest.raises(OSError, match="simulated transition failure"):
        registry.transition_stage("sentiment", version, "production")

    assert registry.list_versions("sentiment")[0]["stage"] == "staging"


def test_mlflow_helper_cannot_register_directly_to_production():
    from src.model_registry import register_with_mlflow

    with pytest.raises(ValueError, match="staging-only"):
        register_with_mlflow("sentiment", "run-id", stage="Production")


def test_failed_evaluation_uses_transform_only_and_exits_nonzero(
    bundle_config, bundle_df, tmp_path, monkeypatch
):
    from src import data_pipeline, evaluate

    bundle = make_candidate(tmp_path / "candidate", bundle_config, bundle_df)
    config_path = tmp_path / "config.yaml"
    evaluation_config = yaml.safe_load(yaml.safe_dump(bundle_config))
    evaluation_config["data"]["random_state"] = 7
    evaluation_config["data"]["test_size"] = 0.5
    evaluation_config["data"]["label_column"] = "wrong_runtime_label"
    evaluation_config["data"]["text_column"] = "wrong_runtime_text"
    config_path.write_text(yaml.safe_dump(evaluation_config), encoding="utf-8")
    data_hash = compute_data_hash(bundle_df)
    monkeypatch.setattr(
        data_pipeline,
        "load_and_preprocess",
        lambda config: (bundle_df, data_hash, dict(DATASET_PROVENANCE), FAKE_VALIDATION),
    )

    def refit_is_forbidden(*args, **kwargs):
        raise AssertionError("Evaluation attempted to refit preprocessing state")

    monkeypatch.setattr(FeatureStore, "get_features", refit_is_forbidden)
    monkeypatch.setattr(FeatureStore, "fit_transform_train", refit_is_forbidden)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--bundle_path",
            str(bundle),
            "--config",
            str(config_path),
            "--min_accuracy",
            "1.1",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        evaluate.main()
    assert exc_info.value.code == 1
    report = json.loads((bundle / EVALUATION_REPORT_FILE).read_text(encoding="utf-8"))
    assert report["evaluation_status"] == "failed"
    assert report["performance_gate"]["overall_passed"] is False
