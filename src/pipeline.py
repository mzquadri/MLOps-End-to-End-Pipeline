"""The reference run: one command that exercises the whole lifecycle.

    python -m src.pipeline --config configs/train_config.yaml

Train, evaluate against the gate, register as staging, promote to production, and write
a run summary. Each stage is still available on its own; this module only sequences them
so a reader can reproduce the documented result without copying five commands correctly.

It stops at the first failure and returns a non-zero exit code, because a pipeline that
continues past a failed gate is not a gate.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from src.data_pipeline import load_and_preprocess
from src.evaluate import (
    ModelEvaluator,
    PerformanceGate,
    build_evaluation_report,
    write_curves,
)
from src.feature_store import FeatureStore
from src.model_bundle import (
    LINEAGE_FILE,
    MANIFEST_FILE,
    create_candidate_bundle,
    load_trusted_bundle,
    read_bundle_json,
    update_evaluation_report,
)
from src.model_registry import LocalModelRegistry
from src.train import three_way_split, train_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GateFailure(RuntimeError):
    """Raised when the candidate does not clear the promotion gate."""


def run(
    config: dict[str, Any],
    *,
    model_name: str = "sentiment-classifier",
    candidate_dir: str = "models/candidates",
    registry_dir: str = "models/registry",
    results_dir: str = "results",
    experiment: str | None = None,
) -> dict[str, Any]:
    """Execute the full lifecycle and return a machine-readable run summary."""
    label_column = config["data"].get("label_column", "sentiment")
    seed = int(config["data"].get("random_state", 42))
    test_size = float(config["data"].get("test_size", 0.2))
    validation_size = float(config["data"].get("validation_size", 0.2))

    logger.info("[1/5] Acquiring and validating data")
    df, data_hash, provenance, validation_report = load_and_preprocess(config)
    labels = df[label_column].to_numpy()

    train_df, validation_df, test_df, y_train, y_validation, y_test = three_way_split(
        df, labels, test_size, validation_size, seed
    )
    logger.info(
        "Split: train=%d validation=%d test=%d", len(train_df), len(validation_df), len(test_df)
    )

    logger.info("[2/5] Fitting features on train only, then training")
    store = FeatureStore(config)
    X_train = store.fit_transform_train(train_df)
    X_validation = store.transform(validation_df)
    model, training_metrics = train_model(
        X_train, y_train, X_validation, y_validation, config, experiment
    )

    bundle_path = str(Path(candidate_dir) / model_name)
    lineage = {
        "data_hash": data_hash,
        "data_rows": int(len(df)),
        "dataset": provenance,
        "data_validation": {
            "overall_passed": validation_report["overall_passed"],
            "duplicate_percentage": validation_report["duplicate_check"][
                "duplicate_percentage"
            ],
            "min_class_percentage": (
                validation_report["class_balance"]["min_class_percentage"]
                if validation_report["class_balance"]
                else None
            ),
        },
        "evaluation_rows": int(len(test_df)),
        "experiment": experiment or config.get("experiment", {}).get("name", "default"),
        "feature_schema": {
            "label_column": label_column,
            "numeric_columns": store.metadata.get("numeric_columns", []),
            "text_column": store.text_column,
        },
        "model_type": config["model"]["type"],
        "split": {
            "random_state": seed,
            "stratified": True,
            "test_size": test_size,
            "validation_size": validation_size,
        },
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
    }
    create_candidate_bundle(
        bundle_path=bundle_path,
        model=model,
        transformers=store.export_transformers(),
        training_metrics=training_metrics,
        lineage=lineage,
        expected_feature_dimension=int(X_train.shape[1]),
    )
    logger.info("Candidate bundle written to %s", bundle_path)

    logger.info("[3/5] Evaluating on the held-out test split")
    # Reload through the bundle rather than reusing the in-memory objects: this is the
    # same path evaluation and serving take, so a broken bundle fails here too.
    loaded_model, transformers, manifest = load_trusted_bundle(bundle_path)
    eval_store = FeatureStore(config)
    eval_store.import_transformers(transformers)
    X_test = eval_store.transform(test_df)
    if X_test.shape[1] != manifest["expected_feature_dimension"]:
        raise RuntimeError("Test feature dimension does not match the bundle manifest")

    evaluator = ModelEvaluator(loaded_model, config)
    metrics = evaluator.compute_metrics(X_test, y_test)
    latency = evaluator.measure_latency(X_test)
    write_curves(evaluator.compute_curves(X_test, y_test), results_dir)

    gate_result = PerformanceGate(config).evaluate(metrics, latency)
    update_evaluation_report(
        bundle_path, build_evaluation_report(metrics, latency, gate_result, data_hash)
    )
    logger.info(
        "Test accuracy %.4f | baseline %.4f | margin %.4f | F1 %.4f | p95 %.3f ms",
        metrics["accuracy"],
        metrics["baseline_accuracy"],
        metrics["accuracy_over_baseline"],
        metrics["f1_weighted"],
        latency["p95_latency_ms"],
    )

    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": provenance,
        "data_hash": data_hash,
        "data_validation": lineage["data_validation"],
        "split": {**lineage["split"], "train_rows": len(train_df),
                  "validation_rows": len(validation_df), "test_rows": len(test_df)},
        "model": {
            "type": config["model"]["type"],
            "hyperparameters": config["model"]["hyperparameters"].get(
                config["model"]["type"], {}
            ),
            "feature_dimension": int(X_train.shape[1]),
        },
        "training_metrics": training_metrics,
        "test_metrics": {
            key: value
            for key, value in metrics.items()
            if key not in ("classification_report",)
        },
        "latency": latency,
        "performance_gate": gate_result,
    }

    if not gate_result["overall_passed"]:
        failed = [n for n, c in gate_result["checks"].items() if not c["passed"]]
        summary["registry"] = {"registered": False, "reason": f"gate failed: {failed}"}
        _write_summary(summary, results_dir)
        raise GateFailure(
            f"Candidate failed the promotion gate ({', '.join(failed)}); not registered."
        )

    logger.info("[4/5] Registering the passing candidate as staging")
    registry = LocalModelRegistry(registry_dir=registry_dir)
    version = registry.register_model(
        model_name=model_name,
        model_path=bundle_path,
        stage="staging",
        description=f"Reference run on {provenance.get('key')}",
    )

    logger.info("[5/5] Promoting %s to production", version)
    registry.transition_stage(model_name, version, "production")

    registered_path = str(Path(registry_dir) / model_name / version)
    registered_manifest = json.loads(
        (Path(registered_path) / MANIFEST_FILE).read_text(encoding="utf-8")
    )
    try:
        bundle_path = os.path.relpath(registered_path)
    except ValueError:
        # On Windows, relpath raises when the target sits on a different drive
        # than the working directory. An absolute path is still usable here.
        bundle_path = registered_path
    summary["registry"] = {
        "registered": True,
        "model_name": model_name,
        "version": version,
        "stage": "production",
        "bundle_path": bundle_path.replace("\\", "/"),
        "artifact_checksums": registered_manifest["artifact_checksums"],
        "bundle_format_version": registered_manifest["bundle_format_version"],
    }
    summary["lineage"] = read_bundle_json(registered_path, LINEAGE_FILE)

    _write_summary(summary, results_dir)
    logger.info(
        "Reference run complete. Serve it with:\n"
        "  python -m src.serve --bundle_path %s --host 127.0.0.1 --port 8000",
        summary["registry"]["bundle_path"],
    )
    return summary


def _write_summary(summary: dict[str, Any], results_dir: str) -> str:
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "reference_run.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    logger.info("Run summary written -> %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full reference pipeline")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--model_name", type=str, default="sentiment-classifier")
    parser.add_argument("--candidate_dir", type=str, default="models/candidates")
    parser.add_argument("--registry_dir", type=str, default="models/registry")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--experiment", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    try:
        run(
            config,
            model_name=args.model_name,
            candidate_dir=args.candidate_dir,
            registry_dir=args.registry_dir,
            results_dir=args.results_dir,
            experiment=args.experiment,
        )
    except GateFailure as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
