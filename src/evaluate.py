"""Held-out evaluation and the promotion gate.

This is the only module that reads the test split, and it reads it once. It reconstructs
the split recorded in the candidate's lineage rather than accepting split settings passed
in later, so an operator cannot re-roll the split until a model passes.

The gate is deliberately not a single accuracy floor. A floor alone is meaningless
without knowing the class balance, so the gate also requires a margin over the
majority-class baseline measured on the same rows.

Usage:
    python -m src.evaluate --bundle_path models/candidates/sentiment-classifier
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Compute held-out metrics, a baseline comparison, and latency."""

    def __init__(self, model: Any, config: dict[str, Any] | None = None) -> None:
        self.model = model
        self.config = config or {}

    def compute_metrics(
        self,
        X_test: Any,
        y_test: np.ndarray,
        baseline_prediction: str | None = None,
    ) -> dict[str, Any]:
        """Classification metrics plus the baseline margin the gate depends on."""
        y_pred = self.model.predict(X_test)

        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            "precision_weighted": float(
                precision_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "confusion_matrix_labels": sorted({str(label) for label in y_test}),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            ),
            "n_test_rows": int(len(y_test)),
        }

        # Majority-class comparator on exactly these rows.
        if baseline_prediction is None:
            values, counts = np.unique(y_test, return_counts=True)
            baseline_prediction = str(values[int(np.argmax(counts))])
        baseline_pred = np.full(len(y_test), baseline_prediction, dtype=object)
        baseline_accuracy = float(accuracy_score(y_test, baseline_pred))
        metrics["baseline_strategy"] = "most_frequent"
        metrics["baseline_prediction"] = baseline_prediction
        metrics["baseline_accuracy"] = baseline_accuracy
        metrics["baseline_f1_weighted"] = float(
            f1_score(y_test, baseline_pred, average="weighted", zero_division=0)
        )
        metrics["accuracy_over_baseline"] = float(metrics["accuracy"] - baseline_accuracy)

        classes = np.unique(y_test)
        if len(classes) == 2 and hasattr(self.model, "predict_proba"):
            positive_class = self.model.classes_[1]
            y_binary = (y_test == positive_class).astype(int)
            y_proba = self.model.predict_proba(X_test)[:, 1]
            metrics["positive_class"] = str(positive_class)
            metrics["roc_auc"] = float(roc_auc_score(y_binary, y_proba))
            precision, recall, _ = precision_recall_curve(y_binary, y_proba)
            metrics["pr_auc"] = float(auc(recall, precision))

        return metrics

    def compute_curves(self, X_test: Any, y_test: np.ndarray) -> dict[str, Any] | None:
        """ROC and PR curve points, written outside the bundle as a report artifact.

        These used to be computed and then dropped on the floor before the report was
        written. They are useful, but they are report-sized rather than contract-sized,
        so they live in `results/` and the bundle keeps the scalar areas.
        """
        classes = np.unique(y_test)
        if len(classes) != 2 or not hasattr(self.model, "predict_proba"):
            return None

        positive_class = self.model.classes_[1]
        y_binary = (y_test == positive_class).astype(int)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, roc_thresholds = roc_curve(y_binary, y_proba)
        precision, recall, pr_thresholds = precision_recall_curve(y_binary, y_proba)

        def rounded(values: np.ndarray) -> list:
            return [round(float(value), 6) for value in values]

        return {
            "positive_class": str(positive_class),
            "roc": {
                "fpr": rounded(fpr),
                "tpr": rounded(tpr),
                "thresholds": rounded(np.clip(roc_thresholds, -1e9, 1e9)),
            },
            "precision_recall": {
                "precision": rounded(precision),
                "recall": rounded(recall),
                "thresholds": rounded(pr_thresholds),
            },
        }

    def measure_latency(self, X_sample: Any, n_runs: int = 100) -> dict[str, float]:
        """Single-row inference latency, as a rough serving-cost signal.

        Measured in-process on one machine, so it is a regression guard, not a service
        level objective. A real latency budget is measured at the service boundary under
        concurrency, which this repository does not attempt.
        """
        single = X_sample[:1]
        timings = []
        for _ in range(n_runs):
            started = time.perf_counter()
            self.model.predict(single)
            timings.append((time.perf_counter() - started) * 1000)

        return {
            "mean_latency_ms": round(float(np.mean(timings)), 3),
            "p95_latency_ms": round(float(np.percentile(timings, 95)), 3),
            "p99_latency_ms": round(float(np.percentile(timings, 99)), 3),
            "n_runs": n_runs,
        }


class PerformanceGate:
    """Promotion criteria. A candidate that fails any check cannot be registered.

    Thresholds live in config so the bar is reviewable in version control rather than
    hidden in an argument default. See docs/EVALUATION.md for how the current values
    were derived - the short version is that they come from the validation split, never
    from the test split this gate is applied to.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        rules = config.get("validation", {})
        self.min_accuracy = rules.get("min_accuracy", 0.85)
        self.min_f1 = rules.get("min_f1", 0.83)
        self.min_accuracy_over_baseline = rules.get("min_accuracy_over_baseline", 0.20)
        self.max_latency_ms = rules.get("max_latency_ms", 100)

    def evaluate(
        self, metrics: dict[str, Any], latency: dict[str, float]
    ) -> dict[str, Any]:
        checks = {
            "accuracy": {
                "value": metrics["accuracy"],
                "threshold": self.min_accuracy,
                "passed": metrics["accuracy"] >= self.min_accuracy,
            },
            "f1_weighted": {
                "value": metrics["f1_weighted"],
                "threshold": self.min_f1,
                "passed": metrics["f1_weighted"] >= self.min_f1,
            },
            "accuracy_over_baseline": {
                "value": metrics["accuracy_over_baseline"],
                "threshold": self.min_accuracy_over_baseline,
                "passed": metrics["accuracy_over_baseline"]
                >= self.min_accuracy_over_baseline,
            },
            "latency_p95": {
                "value": latency["p95_latency_ms"],
                "threshold": self.max_latency_ms,
                "passed": latency["p95_latency_ms"] <= self.max_latency_ms,
            },
        }
        return {
            "overall_passed": all(check["passed"] for check in checks.values()),
            "checks": checks,
        }


def build_evaluation_report(
    metrics: dict[str, Any],
    latency: dict[str, float],
    gate_result: dict[str, Any],
    data_hash: str,
) -> dict[str, Any]:
    """The deterministic report embedded in the bundle."""
    return {
        "data_hash": data_hash,
        "evaluation_status": "passed" if gate_result["overall_passed"] else "failed",
        "latency": latency,
        "metrics": metrics,
        "performance_gate": gate_result,
    }


def write_curves(curves: dict[str, Any] | None, output_dir: str) -> str | None:
    """Persist curve points next to the run, outside the promotion contract."""
    if curves is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "evaluation_curves.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(curves, handle, indent=2, sort_keys=True)
        handle.write("\n")
    logger.info("Curve points written -> %s", path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate bundle against the promotion gate"
    )
    parser.add_argument(
        "--bundle_path", type=str, default="models/candidates/sentiment-classifier"
    )
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument(
        "--min_accuracy",
        type=float,
        default=None,
        help="Override the configured accuracy gate (config value is used when omitted)",
    )
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # Only override when the operator actually asked. The previous version assigned the
    # argparse default unconditionally, so the configured gate was never used.
    if args.min_accuracy is not None:
        config.setdefault("validation", {})["min_accuracy"] = args.min_accuracy
        logger.warning(
            "Accuracy gate overridden from the command line: %.4f", args.min_accuracy
        )

    from src.model_bundle import (
        LINEAGE_FILE,
        load_trusted_bundle,
        read_bundle_json,
        update_evaluation_report,
    )

    model, transformers, manifest = load_trusted_bundle(args.bundle_path)
    lineage = read_bundle_json(args.bundle_path, LINEAGE_FILE)
    logger.info("Loaded candidate bundle from %s", args.bundle_path)

    from src.data_pipeline import load_and_preprocess
    from src.feature_store import FeatureStore
    from src.train import three_way_split

    schema = lineage["feature_schema"]
    split = lineage["split"]
    evaluation_config = copy.deepcopy(config)
    evaluation_config.setdefault("data", {})
    evaluation_config["data"]["label_column"] = schema["label_column"]
    evaluation_config["data"]["text_column"] = schema["text_column"]
    evaluation_config["data"]["random_state"] = split["random_state"]
    evaluation_config["data"]["test_size"] = split["test_size"]
    evaluation_config["data"]["validation_size"] = split.get("validation_size", 0.0)

    df, data_hash, _, _ = load_and_preprocess(evaluation_config)
    if lineage.get("data_hash") != data_hash:
        raise RuntimeError(
            "Evaluation data does not match the candidate lineage. The bundle was "
            "trained on a different dataset version."
        )

    label_column = schema["label_column"]
    labels = df[label_column].to_numpy()
    _, _, test_df, _, _, y_test = three_way_split(
        df,
        labels,
        float(split["test_size"]),
        float(split.get("validation_size", 0.0)),
        int(split["random_state"]),
    )
    expected_rows = lineage.get("evaluation_rows")
    if expected_rows is not None and len(test_df) != expected_rows:
        raise RuntimeError("Test row count does not match the bundle lineage")

    store = FeatureStore(evaluation_config)
    store.import_transformers(transformers)
    X_test = store.transform(test_df)
    if X_test.shape[1] != manifest["expected_feature_dimension"]:
        raise RuntimeError("Test feature dimension does not match the bundle manifest")

    evaluator = ModelEvaluator(model, config)
    metrics = evaluator.compute_metrics(X_test, y_test)
    latency = evaluator.measure_latency(X_test)
    write_curves(evaluator.compute_curves(X_test, y_test), args.results_dir)

    gate_result = PerformanceGate(config).evaluate(metrics, latency)
    update_evaluation_report(
        args.bundle_path, build_evaluation_report(metrics, latency, gate_result, data_hash)
    )

    logger.info(
        "Test accuracy %.4f | baseline %.4f | margin %.4f | weighted F1 %.4f",
        metrics["accuracy"],
        metrics["baseline_accuracy"],
        metrics["accuracy_over_baseline"],
        metrics["f1_weighted"],
    )

    if gate_result["overall_passed"]:
        logger.info("Passed every promotion gate; the candidate is promotable.")
        return

    failed = [name for name, check in gate_result["checks"].items() if not check["passed"]]
    logger.error("Failed gates: %s. The candidate is not promotable.", failed)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
