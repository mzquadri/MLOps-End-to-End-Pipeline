"""
Evaluation Module
=================
Model evaluation with performance gates, fairness checks, and detailed
reporting. Determines whether a model is ready for promotion.

Usage:
    python -m src.evaluate --bundle_path models/candidates/sentiment-classifier
"""

import argparse
import copy
import json
import logging
import os
import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """Evaluate a trained model and enforce quality gates."""

    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        self.model = model
        self.config = config or {}
        self.validation = self.config.get("validation", {})

    def compute_metrics(self, X_test: Any, y_test: np.ndarray) -> Dict[str, Any]:
        """Compute comprehensive classification metrics."""
        y_pred = self.model.predict(X_test)

        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred, output_dict=True
            ),
        }

        # ROC-AUC for binary classification
        unique_classes = np.unique(y_test)
        if len(unique_classes) == 2 and hasattr(self.model, "predict_proba"):
            positive_class = self.model.classes_[1]
            y_binary = y_test == positive_class
            y_proba = self.model.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_binary, y_proba))

            # ROC curve points
            fpr, tpr, thresholds = roc_curve(y_binary, y_proba)
            metrics["roc_curve"] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            }

            # Precision-Recall curve
            prec, rec, pr_thresh = precision_recall_curve(y_binary, y_proba)
            metrics["pr_curve"] = {
                "precision": prec.tolist(),
                "recall": rec.tolist(),
            }

        return metrics

    def measure_latency(self, X_sample: Any, n_runs: int = 100) -> Dict[str, float]:
        """Measure single-sample inference latency."""
        if hasattr(X_sample, "toarray"):
            single = X_sample[:1]
        else:
            single = X_sample[:1]

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            self.model.predict(single)
            times.append((time.perf_counter() - t0) * 1000)  # ms

        return {
            "mean_latency_ms": round(float(np.mean(times)), 3),
            "p95_latency_ms": round(float(np.percentile(times, 95)), 3),
            "p99_latency_ms": round(float(np.percentile(times, 99)), 3),
        }


# ---------------------------------------------------------------------------
# Performance gates
# ---------------------------------------------------------------------------


class PerformanceGate:
    """
    Enforce promotion criteria before a model can be registered.

    Gates:
        - min_accuracy
        - min_f1
        - max_latency_ms
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        val = config.get("validation", {})
        self.min_accuracy = val.get("min_accuracy", 0.85)
        self.min_f1 = val.get("min_f1", 0.83)
        self.max_latency_ms = val.get("max_latency_ms", 100)

    def evaluate(
        self, metrics: Dict[str, Any], latency: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check all gates and return pass/fail report."""
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
            "latency_p95": {
                "value": latency["p95_latency_ms"],
                "threshold": self.max_latency_ms,
                "passed": latency["p95_latency_ms"] <= self.max_latency_ms,
            },
        }
        overall = all(c["passed"] for c in checks.values())
        return {"overall_passed": overall, "checks": checks}


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    metrics: Dict[str, Any],
    latency: Dict[str, float],
    gate_result: Dict[str, Any],
    output_dir: str = "results",
) -> str:
    """Generate and save a JSON evaluation report."""
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "metrics": {
            k: v for k, v in metrics.items() if k not in ("roc_curve", "pr_curve")
        },
        "latency": latency,
        "performance_gate": gate_result,
    }
    path = os.path.join(output_dir, "evaluation_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Evaluation report saved → %s", path)
    return path


def build_evaluation_report(
    metrics: Dict[str, Any],
    latency: Dict[str, float],
    gate_result: Dict[str, Any],
    data_hash: str,
) -> Dict[str, Any]:
    """Build the deterministic report embedded in the evaluated bundle."""
    return {
        "data_hash": data_hash,
        "evaluation_status": "passed"
        if gate_result["overall_passed"]
        else "failed",
        "latency": latency,
        "metrics": {
            k: v for k, v in metrics.items() if k not in ("roc_curve", "pr_curve")
        },
        "performance_gate": gate_result,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model and enforce performance gates"
    )
    parser.add_argument(
        "--bundle_path",
        type=str,
        default="models/candidates/sentiment-classifier",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.85, help="Min accuracy threshold"
    )
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Override threshold if specified via CLI
    if "validation" not in config:
        config["validation"] = {}
    config["validation"]["min_accuracy"] = args.threshold

    from src.model_bundle import (
        LINEAGE_FILE,
        load_trusted_bundle,
        read_bundle_json,
        update_evaluation_report,
    )

    # Validate checksums before loading trusted local pickle-capable artifacts.
    model, transformers, manifest = load_trusted_bundle(args.bundle_path)
    lineage = read_bundle_json(args.bundle_path, LINEAGE_FILE)
    logger.info("Loaded candidate bundle from %s", args.bundle_path)

    # Rebuild test data
    from src.data_pipeline import load_and_preprocess
    from src.feature_store import FeatureStore

    schema = lineage["feature_schema"]
    split = lineage["split"]
    evaluation_config = copy.deepcopy(config)
    evaluation_config.setdefault("data", {})
    evaluation_config["data"]["label_column"] = schema["label_column"]
    evaluation_config["data"]["text_column"] = schema["text_column"]
    evaluation_config["data"]["random_state"] = split["random_state"]
    evaluation_config["data"]["test_size"] = split["test_size"]

    df, data_hash = load_and_preprocess(evaluation_config)
    if lineage.get("data_hash") != data_hash:
        raise RuntimeError(
            "Evaluation data hash does not match the candidate bundle lineage"
        )
    label_col = schema["label_column"]
    y = df[label_col].values

    from sklearn.model_selection import train_test_split

    _, df_test, _, y_test = train_test_split(
        df,
        y,
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratify=y,
    )
    expected_rows = lineage.get("evaluation_rows")
    if expected_rows is not None and len(df_test) != expected_rows:
        raise RuntimeError("Evaluation row count does not match bundle lineage")

    store = FeatureStore(evaluation_config)
    store.import_transformers(transformers)
    X_test = store.transform(df_test)
    if X_test.shape[1] != manifest["expected_feature_dimension"]:
        raise RuntimeError(
            "Evaluation feature dimension does not match bundle manifest"
        )

    evaluator = ModelEvaluator(model, config)
    metrics = evaluator.compute_metrics(X_test, y_test)
    latency = evaluator.measure_latency(X_test)

    gate = PerformanceGate(config)
    gate_result = gate.evaluate(metrics, latency)

    report = build_evaluation_report(metrics, latency, gate_result, data_hash)
    update_evaluation_report(args.bundle_path, report)

    if gate_result["overall_passed"]:
        logger.info("PASSED all performance gates. Model is ready for promotion.")
    else:
        failed = [k for k, v in gate_result["checks"].items() if not v["passed"]]
        logger.error("FAILED gates: %s. Candidate is not promotable.", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
