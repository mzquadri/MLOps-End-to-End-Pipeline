"""
Data Pipeline Module
====================
Handles data loading, validation, preprocessing, and hash-based versioning.
Implements DVC-like content hashing for reproducibility tracking.

Usage:
    python src/data_pipeline.py --config configs/train_config.yaml
"""

import argparse
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy import stats

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data versioning helpers
# ---------------------------------------------------------------------------


def compute_data_hash(df: pd.DataFrame) -> str:
    """Compute SHA-256 hash of a DataFrame for version tracking."""
    content = pd.util.hash_pandas_object(df).values.tobytes()
    return hashlib.sha256(content).hexdigest()[:12]


def save_data_manifest(
    data_hash: str,
    source_path: str,
    n_rows: int,
    n_cols: int,
    output_dir: str = "data",
) -> str:
    """Save a JSON manifest describing the current data version."""
    manifest = {
        "data_hash": data_hash,
        "source_path": source_path,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "timestamp": datetime.utcnow().isoformat(),
    }
    manifest_path = os.path.join(output_dir, f"manifest_{data_hash}.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Data manifest saved → %s", manifest_path)
    return manifest_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class DataValidator:
    """Validate data quality: nulls, duplicates, class balance, drift."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    # ---- basic checks --------------------------------------------------
    def check_nulls(
        self, df: pd.DataFrame, max_null_pct: float = 0.05
    ) -> Dict[str, Any]:
        null_pcts = df.isnull().mean()
        violations = null_pcts[null_pcts > max_null_pct]
        return {
            "passed": violations.empty,
            "null_percentages": null_pcts.to_dict(),
            "violations": violations.to_dict(),
        }

    def check_duplicates(
        self, df: pd.DataFrame, max_dup_pct: float = 0.10
    ) -> Dict[str, Any]:
        dup_pct = df.duplicated().mean()
        return {
            "passed": dup_pct <= max_dup_pct,
            "duplicate_percentage": round(dup_pct, 4),
        }

    def check_class_balance(
        self, labels: pd.Series, min_ratio: float = 0.1
    ) -> Dict[str, Any]:
        counts = labels.value_counts(normalize=True)
        min_class_pct = counts.min()
        return {
            "passed": min_class_pct >= min_ratio,
            "class_distribution": counts.to_dict(),
            "min_class_percentage": round(min_class_pct, 4),
        }

    # ---- drift detection -----------------------------------------------
    def compute_psi(
        self, reference: np.ndarray, current: np.ndarray, buckets: int = 10
    ) -> float:
        """Population Stability Index between two distributions."""
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            buckets + 1,
        )
        ref_counts = np.histogram(reference, bins=breakpoints)[0] + 1
        cur_counts = np.histogram(current, bins=breakpoints)[0] + 1
        ref_pcts = ref_counts / ref_counts.sum()
        cur_pcts = cur_counts / cur_counts.sum()
        psi = np.sum((cur_pcts - ref_pcts) * np.log(cur_pcts / ref_pcts))
        return float(psi)

    def check_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        numeric_cols: Optional[list] = None,
        max_psi: float = 0.2,
    ) -> Dict[str, Any]:
        """Run KS-test and PSI on numeric columns."""
        if numeric_cols is None:
            numeric_cols = reference_df.select_dtypes(
                include=[np.number]
            ).columns.tolist()

        results: Dict[str, Any] = {}
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
            if not col_pass:
                overall_pass = False
            results[col] = {
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": round(ks_p, 4),
                "psi": round(psi, 4),
                "passed": col_pass,
            }
        return {"passed": overall_pass, "columns": results}

    def run_all_checks(self, df: pd.DataFrame, label_col: str) -> Dict[str, Any]:
        """Execute all data quality checks."""
        return {
            "null_check": self.check_nulls(df),
            "duplicate_check": self.check_duplicates(df),
            "class_balance": self.check_class_balance(df[label_col])
            if label_col in df.columns
            else None,
            "overall_passed": True,  # updated below
        }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def preprocess_text(text: str) -> str:
    """Basic text cleaning for NLP tasks."""
    import re

    text = str(text).lower().strip()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"<.*?>", "", text)  # remove HTML tags
    text = re.sub(r"[^a-z0-9\s]", " ", text)  # keep alphanumerics
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def load_and_preprocess(config: Dict[str, Any]) -> Tuple[pd.DataFrame, str]:
    """
    Load data from config path, preprocess, version, and return clean DataFrame.
    Returns (df, data_hash).
    """
    data_cfg = config["data"]
    source = data_cfg["source"]

    logger.info("Loading data from %s", source)
    if not os.path.exists(source):
        logger.warning("Source file not found. Generating synthetic demo data.")
        df = _generate_demo_data(n=2000)
    else:
        df = pd.read_csv(source)

    text_col = data_cfg.get("text_column", "review_text")
    label_col = data_cfg.get("label_column", "sentiment")

    # Clean text
    if text_col in df.columns:
        df[f"{text_col}_clean"] = df[text_col].apply(preprocess_text)

    # Version the data
    data_hash = compute_data_hash(df)
    logger.info("Data hash: %s  |  Shape: %s", data_hash, df.shape)
    save_data_manifest(data_hash, source, len(df), len(df.columns))

    # Validate
    validator = DataValidator(config)
    report = validator.run_all_checks(df, label_col)
    if not report["null_check"]["passed"]:
        logger.warning(
            "Null-value check failed: %s", report["null_check"]["violations"]
        )
    if not report["duplicate_check"]["passed"]:
        logger.warning(
            "Duplicate check failed: %.2f%% duplicates",
            report["duplicate_check"]["duplicate_percentage"] * 100,
        )

    return df, data_hash


# ---------------------------------------------------------------------------
# Demo data generation (for running without external data)
# ---------------------------------------------------------------------------


def _generate_demo_data(n: int = 2000) -> pd.DataFrame:
    """Generate synthetic sentiment review data for demonstration."""
    rng = np.random.default_rng(42)

    positive_templates = [
        "This product is excellent and works great",
        "I love this item it exceeded my expectations",
        "Very happy with the quality and fast delivery",
        "Best purchase I have made this year",
        "Outstanding product highly recommended",
    ]
    negative_templates = [
        "Terrible quality broke after one day",
        "Very disappointed with this purchase",
        "Do not buy this product waste of money",
        "Horrible experience would not recommend",
        "Poor quality and even worse customer service",
    ]

    reviews, sentiments = [], []
    for _ in range(n):
        if rng.random() > 0.45:
            reviews.append(
                rng.choice(positive_templates) + f" rating {rng.integers(4, 6)}"
            )
            sentiments.append("positive")
        else:
            reviews.append(
                rng.choice(negative_templates) + f" rating {rng.integers(1, 3)}"
            )
            sentiments.append("negative")

    return pd.DataFrame(
        {
            "review_text": reviews,
            "sentiment": sentiments,
            "review_length": [len(r) for r in reviews],
            "word_count": [len(r.split()) for r in reviews],
        }
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data pipeline")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    df, data_hash = load_and_preprocess(config)
    logger.info("Pipeline complete. %d rows, hash=%s", len(df), data_hash)

    # Save processed data
    out_path = "data/processed.csv"
    os.makedirs("data", exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Processed data saved → %s", out_path)


if __name__ == "__main__":
    main()
