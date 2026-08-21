"""Data acquisition, validation, cleaning and content-hash versioning.

Three responsibilities, deliberately kept separate:

* *Acquisition* decides where rows come from and records that decision as provenance.
* *Validation* decides whether those rows are fit to train on, and can refuse.
* *Versioning* fingerprints the exact frame so evaluation can prove it saw the same data.

The validator used to detect a broken dataset, log a warning, and let training continue.
That is how a pipeline reports 100% accuracy on 99% duplicate rows and nobody notices.
Validation now has a configurable failure mode, and the reference configuration sets it
to `error` so a degenerate dataset stops the run instead of decorating it.

Usage:
    python -m src.data_pipeline --config configs/train_config.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src import datasets

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

URL_PATTERN = re.compile(r"http\S+")
HTML_PATTERN = re.compile(r"<.*?>")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


class DataValidationError(RuntimeError):
    """Raised when input data fails a blocking quality check."""


# ---------------------------------------------------------------------------
# Versioning
# ---------------------------------------------------------------------------


def compute_data_hash(df: pd.DataFrame) -> str:
    """SHA-256 over row-wise content hashes, truncated for readability.

    Row order matters here, which is what we want: evaluation must reconstruct the
    identical frame before it is allowed to reuse a recorded split.
    """
    content = pd.util.hash_pandas_object(df).values.tobytes()
    return hashlib.sha256(content).hexdigest()[:12]


def save_data_manifest(
    data_hash: str,
    source_path: str,
    n_rows: int,
    n_cols: int,
    output_dir: str = "data",
) -> str:
    """Write a manifest describing one data version to an explicit directory.

    Callers pass the directory. Nothing writes into the repository working tree as a
    side effect of loading data.
    """
    manifest = {
        "data_hash": data_hash,
        "source_path": source_path,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, f"manifest_{data_hash}.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    logger.info("Data manifest saved -> %s", manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class DataValidator:
    """Quality checks over an input frame: nulls, duplicates, balance, drift."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        rules = config.get("validation", {}).get("data", {})
        self.max_null_pct = rules.get("max_null_pct", 0.05)
        self.max_duplicate_pct = rules.get("max_duplicate_pct", 0.10)
        self.min_class_ratio = rules.get("min_class_ratio", 0.10)

    def check_nulls(
        self, df: pd.DataFrame, max_null_pct: float | None = None
    ) -> dict[str, Any]:
        threshold = self.max_null_pct if max_null_pct is None else max_null_pct
        null_pcts = df.isnull().mean()
        violations = null_pcts[null_pcts > threshold]
        return {
            "passed": violations.empty,
            "threshold": threshold,
            "null_percentages": {k: round(float(v), 4) for k, v in null_pcts.items()},
            "violations": {k: round(float(v), 4) for k, v in violations.items()},
        }

    def check_duplicates(
        self, df: pd.DataFrame, max_dup_pct: float | None = None
    ) -> dict[str, Any]:
        threshold = self.max_duplicate_pct if max_dup_pct is None else max_dup_pct
        dup_pct = float(df.duplicated().mean())
        return {
            "passed": dup_pct <= threshold,
            "threshold": threshold,
            "duplicate_percentage": round(dup_pct, 4),
        }

    def check_class_balance(
        self, labels: pd.Series, min_ratio: float | None = None
    ) -> dict[str, Any]:
        threshold = self.min_class_ratio if min_ratio is None else min_ratio
        counts = labels.value_counts(normalize=True)
        min_class_pct = float(counts.min())
        return {
            "passed": min_class_pct >= threshold,
            "threshold": threshold,
            "class_distribution": {k: round(float(v), 4) for k, v in counts.items()},
            "min_class_percentage": round(min_class_pct, 4),
        }

    def compute_psi(
        self, reference: np.ndarray, current: np.ndarray, buckets: int = 10
    ) -> float:
        """Population Stability Index between two numeric distributions.

        Counts are Laplace-smoothed by one so an empty bucket cannot produce an
        infinite score, which would make the whole comparison useless.
        """
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            buckets + 1,
        )
        ref_counts = np.histogram(reference, bins=breakpoints)[0] + 1
        cur_counts = np.histogram(current, bins=breakpoints)[0] + 1
        ref_pcts = ref_counts / ref_counts.sum()
        cur_pcts = cur_counts / cur_counts.sum()
        return float(np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts)))

    def check_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        numeric_cols: list[str] | None = None,
        max_psi: float = 0.2,
    ) -> dict[str, Any]:
        """KS test and PSI per numeric column, for comparing two data versions."""
        if numeric_cols is None:
            numeric_cols = reference_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        results: dict[str, Any] = {}
        overall_pass = True
        for col in numeric_cols:
            if col not in current_df.columns:
                continue
            ks_stat, ks_p = stats.ks_2samp(
                reference_df[col].dropna(), current_df[col].dropna()
            )
            psi = self.compute_psi(
                reference_df[col].dropna().values, current_df[col].dropna().values
            )
            col_pass = psi <= max_psi
            overall_pass = overall_pass and col_pass
            results[col] = {
                "ks_statistic": round(float(ks_stat), 4),
                "ks_pvalue": round(float(ks_p), 4),
                "psi": round(psi, 4),
                "passed": col_pass,
            }
        return {"passed": overall_pass, "columns": results}

    def run_all_checks(self, df: pd.DataFrame, label_col: str) -> dict[str, Any]:
        null_check = self.check_nulls(df)
        duplicate_check = self.check_duplicates(df)
        class_balance = (
            self.check_class_balance(df[label_col]) if label_col in df.columns else None
        )
        checks = [null_check, duplicate_check]
        if class_balance is not None:
            checks.append(class_balance)
        return {
            "null_check": null_check,
            "duplicate_check": duplicate_check,
            "class_balance": class_balance,
            "overall_passed": all(check["passed"] for check in checks),
        }


def enforce_validation(report: dict[str, Any], on_failure: str) -> None:
    """Apply the configured failure policy to a validation report.

    `error` is the reference-run policy. `warn` exists for exploratory work where you
    knowingly want to look at data that would not be fit to train on.
    """
    if report["overall_passed"]:
        return

    failed = [
        name
        for name in ("null_check", "duplicate_check", "class_balance")
        if report.get(name) and not report[name]["passed"]
    ]
    summary = f"Data validation failed: {', '.join(failed)}"

    if on_failure == "error":
        raise DataValidationError(
            f"{summary}. Set validation.data.on_failure to 'warn' to proceed anyway."
        )
    logger.warning("%s (continuing because on_failure='warn')", summary)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_text(text: str) -> str:
    """Lowercase, strip URLs and markup, keep alphanumerics, collapse whitespace.

    Deliberately simple and stateless. Anything learned from the corpus belongs in the
    feature store, where it can be fitted on the training split alone.
    """
    text = str(text).lower().strip()
    text = URL_PATTERN.sub("", text)
    text = HTML_PATTERN.sub("", text)
    text = NON_ALPHANUMERIC_PATTERN.sub(" ", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def add_derived_columns(
    df: pd.DataFrame, text_column: str, label_column: str
) -> pd.DataFrame:
    """Add the cleaned text and the two length features the model uses.

    These are computed per row from the row's own text, so they carry no information
    from any other row and cannot leak across a split boundary.
    """
    frame = df.copy()
    frame[f"{text_column}_clean"] = frame[text_column].apply(preprocess_text)
    frame["review_length"] = frame[text_column].astype(str).str.len()
    frame["word_count"] = frame[text_column].astype(str).str.split().str.len()
    columns = [text_column, label_column, f"{text_column}_clean", "review_length", "word_count"]
    if "source" in frame.columns:
        columns.append("source")
    return frame[columns].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def acquire_dataframe(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, str]]:
    """Resolve the data source and return rows plus their provenance record.

    Resolution order is explicit and never silently degrades:
      1. a local CSV, when `data.source` is set and present
      2. the licensed reference dataset
      3. the synthetic fixture, only when `data.allow_synthetic_fallback` is true

    The third step used to be automatic. That is what let a template-generated toy
    stand in for evidence, so it is now opt-in.
    """
    data_cfg = config["data"]
    text_column = data_cfg.get("text_column", "review_text")
    label_column = data_cfg.get("label_column", "sentiment")
    allow_synthetic = bool(data_cfg.get("allow_synthetic_fallback", False))

    source = data_cfg.get("source")
    if source and os.path.exists(source):
        logger.info("Loading local CSV source")
        frame = datasets.read_local_csv(source, text_column, label_column)
        return frame, datasets.local_provenance(source, frame)

    if allow_synthetic and data_cfg.get("prefer_synthetic", False):
        rows = int(data_cfg.get("synthetic_rows", 400))
        seed = int(data_cfg.get("random_state", 42))
        logger.info("Using the synthetic fixture (explicitly requested)")
        return datasets.synthetic_fixture(rows, seed), datasets.synthetic_provenance(
            rows, seed
        )

    dataset_key = data_cfg.get("dataset", datasets.UCI_SENTIMENT.key)
    try:
        frame, provenance = datasets.load_reference_dataset(
            dataset_key,
            cache_dir=data_cfg.get("cache_dir", "data/cache"),
            allow_download=bool(data_cfg.get("allow_download", True)),
        )
        return frame, provenance
    except (datasets.DatasetIntegrityError, OSError) as exc:
        if not allow_synthetic:
            raise DataValidationError(
                f"Could not obtain the reference dataset ({exc}). Refusing to fall back "
                "to synthetic data: set data.allow_synthetic_fallback to true if you "
                "explicitly want a fixture run."
            ) from exc
        rows = int(data_cfg.get("synthetic_rows", 400))
        seed = int(data_cfg.get("random_state", 42))
        logger.warning("Reference dataset unavailable (%s); using synthetic fixture", exc)
        return datasets.synthetic_fixture(rows, seed), datasets.synthetic_provenance(
            rows, seed
        )


def load_and_preprocess(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, str, dict[str, str], dict[str, Any]]:
    """Acquire, clean, validate and fingerprint the dataset.

    Returns (frame, data_hash, provenance, validation_report). The caller decides what
    to persist; this function writes nothing.
    """
    data_cfg = config["data"]
    text_column = data_cfg.get("text_column", "review_text")
    label_column = data_cfg.get("label_column", "sentiment")

    raw, provenance = acquire_dataframe(config)
    frame = add_derived_columns(raw, text_column, label_column)

    validator = DataValidator(config)
    report = validator.run_all_checks(frame, label_column)
    on_failure = config.get("validation", {}).get("data", {}).get("on_failure", "warn")
    enforce_validation(report, on_failure)

    data_hash = compute_data_hash(frame)
    logger.info(
        "Data ready: %d rows, %d columns, hash=%s, source=%s",
        len(frame),
        len(frame.columns),
        data_hash,
        provenance.get("key"),
    )
    return frame, data_hash, provenance, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and validate the dataset")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default=None,
        help="Optional directory to write a data manifest into",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    frame, data_hash, provenance, report = load_and_preprocess(config)
    logger.info("Provenance: %s", json.dumps(provenance, indent=2))
    logger.info("Validation passed: %s", report["overall_passed"])

    if args.manifest_dir:
        save_data_manifest(
            data_hash, provenance.get("key", "unknown"), len(frame), len(frame.columns),
            output_dir=args.manifest_dir,
        )


if __name__ == "__main__":
    main()
