"""Check a completed reference run against the result this repository documents.

The README quotes numbers. Numbers in a README rot silently: someone changes a default,
the model shifts by two points, and the documentation quietly becomes fiction. This
script turns the documented result into an assertion that CI evaluates.

    python scripts/check_reference_run.py [results/reference_run.json]

Exact equality is not required. The run is seeded and the dependencies are pinned, so on
one machine it reproduces to the digit - but a different CPU, BLAS build or platform can
move a linear model by a fraction of a point. The tolerance below is wide enough to
absorb that and far too narrow to hide a real regression.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

METRIC_TOLERANCE = 0.02

EXPECTED = {
    "dataset": {
        "key": "uci-sentiment-labelled-sentences",
        "kind": "licensed-download",
        "license": "CC BY 4.0",
        "sha256": "afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343",
    },
    "split": {
        "train_rows": 1800,
        "validation_rows": 600,
        "test_rows": 600,
        "random_state": 42,
        "stratified": True,
    },
    "metrics": {
        "accuracy": 0.8067,
        "f1_weighted": 0.8067,
        "roc_auc": 0.8795,
        "pr_auc": 0.8895,
        "baseline_accuracy": 0.5000,
        "accuracy_over_baseline": 0.3067,
    },
    "bundle_format_version": "1.1",
}


def check(failures: list[str], condition: bool, message: str) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/reference_run.json")
    if not path.is_file():
        print(f"No run summary at {path}. Run: python -m src.pipeline", file=sys.stderr)
        return 1

    summary = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []

    dataset = summary.get("dataset", {})
    for field, expected in EXPECTED["dataset"].items():
        check(
            failures,
            dataset.get(field) == expected,
            f"dataset.{field}: expected {expected!r}, got {dataset.get(field)!r}",
        )

    split = summary.get("split", {})
    for field, expected in EXPECTED["split"].items():
        check(
            failures,
            split.get(field) == expected,
            f"split.{field}: expected {expected!r}, got {split.get(field)!r}",
        )

    metrics = summary.get("test_metrics", {})
    for field, expected in EXPECTED["metrics"].items():
        actual = metrics.get(field)
        check(
            failures,
            isinstance(actual, (int, float)) and abs(actual - expected) <= METRIC_TOLERANCE,
            f"test_metrics.{field}: expected {expected} +/- {METRIC_TOLERANCE}, got {actual}",
        )

    gate = summary.get("performance_gate", {})
    check(failures, gate.get("overall_passed") is True, "the promotion gate did not pass")
    check(
        failures,
        set(gate.get("checks", {})) == {
            "accuracy",
            "f1_weighted",
            "accuracy_over_baseline",
            "latency_p95",
        },
        f"unexpected gate inventory: {sorted(gate.get('checks', {}))}",
    )

    registry = summary.get("registry", {})
    check(failures, registry.get("registered") is True, "no model was registered")
    check(failures, registry.get("stage") == "production", "the model was not promoted")
    check(
        failures,
        registry.get("bundle_format_version") == EXPECTED["bundle_format_version"],
        f"bundle format: expected {EXPECTED['bundle_format_version']}, "
        f"got {registry.get('bundle_format_version')}",
    )
    check(
        failures,
        len(registry.get("artifact_checksums", {})) == 5,
        "the published bundle does not carry five artifact checksums",
    )

    validation = summary.get("data_validation", {})
    check(failures, validation.get("overall_passed") is True, "data validation did not pass")

    if failures:
        print("Reference run does not match the documented result:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        "Reference run matches the documented result: "
        f"accuracy {metrics['accuracy']:.4f}, "
        f"baseline {metrics['baseline_accuracy']:.4f}, "
        f"margin {metrics['accuracy_over_baseline']:.4f}, "
        f"{registry['model_name']} {registry['version']} in {registry['stage']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
