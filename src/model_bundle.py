"""Atomic, self-contained model bundle creation and validation.

Joblib artifacts are pickle-capable. Only call the loading functions in this
module for bundles produced locally or received through a trusted channel.
Checksums detect corruption and incomplete copies; they do not authenticate an
untrusted bundle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib

BUNDLE_FORMAT_VERSION = "1.1"
MODEL_FILE = "model.joblib"
TRANSFORMERS_FILE = "feature_transformers.joblib"
TRAINING_METRICS_FILE = "training_metrics.json"
LINEAGE_FILE = "lineage.json"
EVALUATION_REPORT_FILE = "evaluation_report.json"
MANIFEST_FILE = "manifest.json"
ARTIFACT_FILES = (
    MODEL_FILE,
    TRANSFORMERS_FILE,
    TRAINING_METRICS_FILE,
    LINEAGE_FILE,
    EVALUATION_REPORT_FILE,
)
REQUIRED_GATE_CHECKS = {
    "accuracy": ("metrics", "accuracy", "minimum"),
    "f1_weighted": ("metrics", "f1_weighted", "minimum"),
    # A raw accuracy floor says nothing on its own: 0.95 is poor on a 99%-imbalanced
    # task and excellent on a balanced one. Requiring a margin over the majority-class
    # baseline makes the gate mean the same thing across datasets.
    "accuracy_over_baseline": ("metrics", "accuracy_over_baseline", "minimum"),
    "latency_p95": ("latency", "p95_latency_ms", "maximum"),
}


class BundleIntegrityError(RuntimeError):
    """Raised when a bundle is incomplete, corrupt, or dimensionally invalid."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(bundle_dir: Path, expected_dimension: int) -> dict[str, Any]:
    return {
        "artifact_checksums": {
            filename: file_sha256(bundle_dir / filename)
            for filename in ARTIFACT_FILES
        },
        "bundle_format_version": BUNDLE_FORMAT_VERSION,
        "expected_feature_dimension": expected_dimension,
        "required_files": list(ARTIFACT_FILES) + [MANIFEST_FILE],
    }


def _atomic_replace_directory(source: Path, destination: Path) -> None:
    """Replace a candidate directory transactionally, with crash recovery."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.backup")
    if backup.exists():
        if destination.exists():
            shutil.rmtree(backup)
        else:
            os.replace(backup, destination)

    if destination.exists():
        os.replace(destination, backup)
    try:
        os.replace(source, destination)
    except BaseException:
        if backup.exists():
            os.replace(backup, destination)
        raise
    else:
        if backup.exists():
            shutil.rmtree(backup)


def _recover_interrupted_replacement(destination: Path) -> None:
    """Restore the last complete candidate after an interrupted replacement."""
    backup = destination.with_name(f".{destination.name}.backup")
    if not destination.exists() and backup.exists():
        os.replace(backup, destination)


def _has_absolute_path(value: Any) -> bool:
    if isinstance(value, str):
        return Path(value).is_absolute()
    if isinstance(value, Mapping):
        return any(_has_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_absolute_path(item) for item in value)
    return False


def _validate_lineage(lineage: Any) -> None:
    if not isinstance(lineage, dict):
        raise BundleIntegrityError("Bundle lineage must be a JSON object")
    if not isinstance(lineage.get("data_hash"), str) or not lineage["data_hash"]:
        raise BundleIntegrityError("Bundle lineage requires a data_hash")
    if _has_absolute_path(lineage):
        raise BundleIntegrityError("Bundle lineage must not contain absolute paths")

    schema = lineage.get("feature_schema")
    if not isinstance(schema, dict):
        raise BundleIntegrityError("Bundle lineage requires a feature_schema")
    if not isinstance(schema.get("text_column"), str):
        raise BundleIntegrityError("Feature schema requires a text_column")
    if not isinstance(schema.get("label_column"), str):
        raise BundleIntegrityError("Feature schema requires a label_column")
    if not isinstance(schema.get("numeric_columns"), list):
        raise BundleIntegrityError("Feature schema requires numeric_columns")

    # A model that cannot say where its training data came from, or under what terms,
    # is not publishable. Enforcing that here means no code path can quietly skip it.
    dataset = lineage.get("dataset")
    if not isinstance(dataset, dict):
        raise BundleIntegrityError("Bundle lineage requires a dataset provenance record")
    for field_name in ("kind", "key", "license"):
        if not isinstance(dataset.get(field_name), str) or not dataset[field_name]:
            raise BundleIntegrityError(
                f"Dataset provenance requires a non-empty {field_name}"
            )

    split = lineage.get("split")
    if not isinstance(split, dict):
        raise BundleIntegrityError("Bundle lineage requires a split contract")
    if not isinstance(split.get("random_state"), int):
        raise BundleIntegrityError("Split contract requires an integer random_state")
    test_size = split.get("test_size")
    if not isinstance(test_size, (int, float)) or not 0 < test_size < 1:
        raise BundleIntegrityError("Split test_size must be between zero and one")
    if split.get("stratified") is not True:
        raise BundleIntegrityError("Bundle split must be stratified")


def _validated_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BundleIntegrityError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BundleIntegrityError(f"{label} must be finite")
    return result


def _validate_evaluation_report(report: Any, lineage: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        raise BundleIntegrityError("Evaluation report must be a JSON object")
    status = report.get("evaluation_status")
    if status not in {"passed", "failed"}:
        raise BundleIntegrityError("Bundle has not been evaluated")
    if report.get("data_hash") != lineage.get("data_hash"):
        raise BundleIntegrityError("Evaluation data hash does not match lineage")

    metrics = report.get("metrics")
    latency = report.get("latency")
    gate = report.get("performance_gate")
    if not isinstance(metrics, dict) or not isinstance(latency, dict):
        raise BundleIntegrityError("Evaluation report requires metrics and latency")
    if not isinstance(gate, dict) or not isinstance(gate.get("checks"), dict):
        raise BundleIntegrityError("Evaluation report requires gate checks")
    checks = gate["checks"]
    if set(checks) != set(REQUIRED_GATE_CHECKS):
        raise BundleIntegrityError("Evaluation report gate inventory does not match")

    check_results = []
    for name, (section_name, value_name, direction) in REQUIRED_GATE_CHECKS.items():
        check = checks[name]
        if not isinstance(check, dict):
            raise BundleIntegrityError(f"Gate check {name} must be an object")
        source = metrics if section_name == "metrics" else latency
        source_value = _validated_number(source.get(value_name), value_name)
        value = _validated_number(check.get("value"), f"{name} value")
        threshold = _validated_number(check.get("threshold"), f"{name} threshold")
        if value != source_value:
            raise BundleIntegrityError(f"Gate check {name} does not match report")
        expected_pass = (
            value >= threshold if direction == "minimum" else value <= threshold
        )
        if check.get("passed") is not expected_pass:
            raise BundleIntegrityError(f"Gate check {name} result is inconsistent")
        check_results.append(expected_pass)

    overall_passed = all(check_results)
    if gate.get("overall_passed") is not overall_passed:
        raise BundleIntegrityError("Overall gate result is inconsistent")
    if (status == "passed") is not overall_passed:
        raise BundleIntegrityError("Evaluation status and gate result disagree")


def create_candidate_bundle(
    bundle_path: str,
    model: Any,
    transformers: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
    lineage: Mapping[str, Any],
    expected_feature_dimension: int,
) -> Path:
    """Atomically create a complete candidate bundle with a pending gate."""
    destination = Path(bundle_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        joblib.dump(model, temp_dir / MODEL_FILE)
        joblib.dump(dict(transformers), temp_dir / TRANSFORMERS_FILE)
        _write_json(temp_dir / TRAINING_METRICS_FILE, dict(training_metrics))
        _write_json(temp_dir / LINEAGE_FILE, dict(lineage))
        _write_json(
            temp_dir / EVALUATION_REPORT_FILE,
            {
                "evaluation_status": "pending",
                "performance_gate": {"overall_passed": False, "checks": {}},
            },
        )
        _write_json(
            temp_dir / MANIFEST_FILE,
            _manifest(temp_dir, expected_feature_dimension),
        )
        validate_bundle(temp_dir, require_evaluated=False)
        _atomic_replace_directory(temp_dir, destination)
    except BaseException:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    return destination


def update_evaluation_report(
    bundle_path: str, report: Mapping[str, Any]
) -> Path:
    """Atomically finalize a candidate with a recomputed evaluation report."""
    destination = Path(bundle_path)
    manifest = validate_bundle(destination, require_evaluated=False)
    temp_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.evaluation.", dir=destination.parent
        )
    )
    shutil.rmtree(temp_dir)
    try:
        shutil.copytree(destination, temp_dir)
        _write_json(temp_dir / EVALUATION_REPORT_FILE, dict(report))
        _write_json(
            temp_dir / MANIFEST_FILE,
            _manifest(temp_dir, manifest["expected_feature_dimension"]),
        )
        validate_bundle(temp_dir, require_evaluated=True)
        _atomic_replace_directory(temp_dir, destination)
    except BaseException:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
    return destination


def validate_bundle(
    bundle_path: os.PathLike[str] | str,
    *,
    require_evaluated: bool = True,
    require_gate_passed: bool = False,
) -> dict[str, Any]:
    """Validate required files, manifest schema, checksums, and gate state."""
    bundle_dir = Path(bundle_path)
    _recover_interrupted_replacement(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise BundleIntegrityError(f"Missing bundle manifest: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise BundleIntegrityError(f"Invalid bundle manifest: {manifest_path}") from exc

    if manifest.get("bundle_format_version") != BUNDLE_FORMAT_VERSION:
        raise BundleIntegrityError("Unsupported bundle format version")
    if manifest.get("required_files") != list(ARTIFACT_FILES) + [MANIFEST_FILE]:
        raise BundleIntegrityError("Bundle required_files contract does not match")
    expected_dimension = manifest.get("expected_feature_dimension")
    if not isinstance(expected_dimension, int) or expected_dimension <= 0:
        raise BundleIntegrityError(
            "Expected feature dimension must be a positive integer"
        )

    checksums = manifest.get("artifact_checksums")
    if not isinstance(checksums, dict) or set(checksums) != set(ARTIFACT_FILES):
        raise BundleIntegrityError("Bundle checksum inventory does not match")
    for filename in ARTIFACT_FILES:
        artifact = bundle_dir / filename
        if not artifact.is_file():
            raise BundleIntegrityError(f"Missing bundle artifact: {filename}")
        if file_sha256(artifact) != checksums[filename]:
            raise BundleIntegrityError(f"Checksum mismatch for {filename}")

    try:
        report = json.loads(
            (bundle_dir / EVALUATION_REPORT_FILE).read_text(encoding="utf-8")
        )
        lineage = json.loads((bundle_dir / LINEAGE_FILE).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise BundleIntegrityError("Bundle JSON artifact is invalid") from exc
    _validate_lineage(lineage)
    if not isinstance(report, dict):
        raise BundleIntegrityError("Evaluation report must be a JSON object")
    status = report.get("evaluation_status")
    passed = report.get("performance_gate", {}).get("overall_passed") is True
    if require_evaluated and status not in {"passed", "failed"}:
        raise BundleIntegrityError("Bundle has not been evaluated")
    if status in {"passed", "failed"}:
        _validate_evaluation_report(report, lineage)
    if require_gate_passed and (status != "passed" or not passed):
        raise BundleIntegrityError("Bundle did not pass its promotion gate")
    return manifest


def load_trusted_bundle(
    bundle_path: str,
    *,
    require_gate_passed: bool = False,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load a checksum-valid bundle known to come from a trusted source."""
    bundle_dir = Path(bundle_path)
    manifest = validate_bundle(
        bundle_dir,
        require_evaluated=require_gate_passed,
        require_gate_passed=require_gate_passed,
    )
    model = joblib.load(bundle_dir / MODEL_FILE)
    transformers = joblib.load(bundle_dir / TRANSFORMERS_FILE)

    expected = manifest["expected_feature_dimension"]
    if not isinstance(transformers, dict):
        raise BundleIntegrityError("Transformer artifact must contain a dictionary")
    metadata = transformers.get("metadata", {})
    tfidf = transformers.get("tfidf")
    if tfidf is None or not hasattr(tfidf, "get_feature_names_out"):
        raise BundleIntegrityError("Transformer artifact is missing fitted TF-IDF")
    tfidf_dimension = len(tfidf.get_feature_names_out())
    numeric_columns = metadata.get("numeric_columns", [])
    if not isinstance(numeric_columns, list):
        raise BundleIntegrityError("Transformer numeric_columns must be a list")
    scaler = transformers.get("scaler")
    if numeric_columns:
        scaler_dimension = getattr(scaler, "n_features_in_", None)
        if scaler_dimension != len(numeric_columns):
            raise BundleIntegrityError("Numeric scaler dimension does not match schema")
    elif scaler is not None:
        raise BundleIntegrityError("Unexpected numeric scaler without numeric columns")
    if tfidf_dimension + len(numeric_columns) != expected:
        raise BundleIntegrityError(
            "Fitted transformer dimension does not match manifest"
        )
    if metadata.get("total_features") != expected:
        raise BundleIntegrityError(
            "Transformer metadata dimension does not match manifest"
        )
    model_dimension = getattr(model, "n_features_in_", expected)
    if int(model_dimension) != expected:
        raise BundleIntegrityError("Model feature dimension does not match manifest")
    return model, transformers, manifest


def read_bundle_json(bundle_path: str, filename: str) -> dict[str, Any]:
    """Read a checksum-validated JSON artifact without loading pickle content."""
    if filename not in {TRAINING_METRICS_FILE, LINEAGE_FILE, EVALUATION_REPORT_FILE}:
        raise ValueError(f"Unsupported JSON bundle artifact: {filename}")
    validate_bundle(bundle_path, require_evaluated=False)
    return json.loads((Path(bundle_path) / filename).read_text(encoding="utf-8"))
