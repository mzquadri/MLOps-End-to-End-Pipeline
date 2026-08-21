"""Model training: split, fit features on train only, cross-validate, emit a candidate.

Split policy, which is the part worth reading carefully:

    train (60%)       fits TF-IDF, the scaler, and the model
    validation (20%)  the only split training is allowed to look at
    test (20%)        never touched here; `src.evaluate` opens it exactly once

Everything reported by this module is a *training-time* number: cross-validation on the
training split and metrics on validation. The test split is not loaded, not transformed
and not scored, so no amount of iterating on this file can leak it. That separation is
why `training_metrics.json` no longer contains test metrics - it used to, under a name
that implied otherwise.

Usage:
    python -m src.train --config configs/train_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import numpy as np
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "svm": SVC,
}


def build_model(config: dict[str, Any]) -> Any:
    """Instantiate the configured estimator."""
    model_type = config["model"]["type"]
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(MODEL_REGISTRY)}"
        )

    hyperparams = dict(config["model"]["hyperparameters"].get(model_type, {}))
    if model_type == "svm":
        # Needed so the serving layer can report a confidence alongside the label.
        hyperparams["probability"] = True

    model = MODEL_REGISTRY[model_type](**hyperparams)
    logger.info("Built %s with %s", model_type, hyperparams)
    return model


def three_way_split(
    df: Any, labels: np.ndarray, test_size: float, validation_size: float, seed: int
) -> tuple[Any, Any, Any, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/validation/test split, derived from one seed.

    `validation_size` is expressed as a fraction of the whole dataset, then converted to
    a fraction of the post-test remainder. Stating it relative to the original dataset
    is what a reader expects, and doing the conversion here keeps that expectation true.
    """
    train_pool, test_df, y_pool, y_test = train_test_split(
        df, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    relative_validation = validation_size / (1.0 - test_size)
    train_df, validation_df, y_train, y_validation = train_test_split(
        train_pool,
        y_pool,
        test_size=relative_validation,
        random_state=seed,
        stratify=y_pool,
    )
    return train_df, validation_df, test_df, y_train, y_validation, y_test


def cross_validate_model(
    model: Any, X: Any, y: np.ndarray, config: dict[str, Any]
) -> dict[str, float]:
    """Stratified k-fold cross-validation on the training split only."""
    cv_folds = config.get("training", {}).get("cv_folds", 5)
    scoring = config.get("training", {}).get("scoring", "f1_weighted")
    seed = config["data"].get("random_state", 42)

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring, n_jobs=None)
    results = {
        "cv_metric": scoring,
        "cv_mean": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
        "cv_scores": [float(score) for score in scores],
    }
    logger.info("CV %s: %.4f +/- %.4f", scoring, results["cv_mean"], results["cv_std"])
    return results


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
    }


def majority_baseline(
    X_train: Any, y_train: np.ndarray, X_eval: Any, y_eval: np.ndarray
) -> dict[str, float]:
    """Always-predict-the-largest-class comparator.

    Without it, an accuracy number has no scale. On a balanced binary task this lands
    near 0.5, which is precisely the context a reader needs before being impressed by
    anything else.
    """
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    return classification_metrics(y_eval, baseline.predict(X_eval))


def train_model(
    X_train: Any,
    y_train: np.ndarray,
    X_validation: Any,
    y_validation: np.ndarray,
    config: dict[str, Any],
    experiment_name: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Cross-validate, fit, and score on validation. Returns (model, training metrics)."""
    model = build_model(config)
    cv_results = cross_validate_model(model, X_train, y_train, config)

    logger.info("Fitting final model on %d training rows", X_train.shape[0])
    started = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - started

    validation_metrics = classification_metrics(y_validation, model.predict(X_validation))
    baseline_metrics = majority_baseline(X_train, y_train, X_validation, y_validation)

    metrics: dict[str, Any] = {
        **{f"validation_{k}": v for k, v in validation_metrics.items()},
        **{f"baseline_{k}": v for k, v in baseline_metrics.items()},
        "train_time_seconds": round(train_seconds, 3),
        **cv_results,
    }
    logger.info(
        "Validation accuracy %.4f (majority baseline %.4f) | weighted F1 %.4f",
        validation_metrics["accuracy"],
        baseline_metrics["accuracy"],
        validation_metrics["f1_weighted"],
    )

    _try_mlflow_log(config, model, metrics, experiment_name)
    return model, metrics


def _try_mlflow_log(
    config: dict[str, Any],
    model: Any,
    metrics: dict[str, Any],
    experiment_name: str | None,
) -> None:
    """Log to MLflow when it is installed and enabled; otherwise say so and move on.

    Tracking is optional on purpose. The bundle, not MLflow, is the source of truth for
    promotion, so a missing tracker degrades observability rather than correctness.
    """
    mlflow_cfg = config.get("mlflow", {})
    if not mlflow_cfg.get("enabled", False):
        logger.info("MLflow logging disabled in config; skipping")
        return

    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        logger.warning("mlflow is enabled in config but not installed; skipping")
        return

    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "sqlite:///mlruns.db"))
    mlflow.set_experiment(
        experiment_name or config.get("experiment", {}).get("name", "default")
    )
    with mlflow.start_run():
        model_type = config["model"]["type"]
        mlflow.log_param("model_type", model_type)
        for key, value in config["model"]["hyperparameters"].get(model_type, {}).items():
            mlflow.log_param(key, value)
        mlflow.log_param("cv_folds", config.get("training", {}).get("cv_folds", 5))
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(name, value)
        mlflow.sklearn.log_model(model, name="model")
        logger.info("Logged run to MLflow")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a candidate model bundle")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument(
        "--bundle_path",
        type=str,
        default="models/candidates/sentiment-classifier",
        help="Candidate bundle directory to create",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    from src.data_pipeline import load_and_preprocess
    from src.feature_store import FeatureStore
    from src.model_bundle import create_candidate_bundle

    df, data_hash, provenance, validation_report = load_and_preprocess(config)
    label_column = config["data"].get("label_column", "sentiment")
    labels = df[label_column].to_numpy()

    test_size = float(config["data"].get("test_size", 0.2))
    validation_size = float(config["data"].get("validation_size", 0.2))
    seed = int(config["data"].get("random_state", 42))
    train_df, validation_df, test_df, y_train, y_validation, _ = three_way_split(
        df, labels, test_size, validation_size, seed
    )
    logger.info(
        "Split sizes - train %d, validation %d, test %d (test is not read here)",
        len(train_df),
        len(validation_df),
        len(test_df),
    )

    store = FeatureStore(config)
    X_train = store.fit_transform_train(train_df)
    X_validation = store.transform(validation_df)

    model, metrics = train_model(
        X_train, y_train, X_validation, y_validation, config, args.experiment
    )

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
        "experiment": args.experiment
        or config.get("experiment", {}).get("name", "default"),
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
        bundle_path=args.bundle_path,
        model=model,
        transformers=store.export_transformers(),
        training_metrics=metrics,
        lineage=lineage,
        expected_feature_dimension=int(X_train.shape[1]),
    )

    logger.info(
        "Candidate bundle written to %s\n%s",
        args.bundle_path,
        json.dumps(metrics, indent=2, sort_keys=True),
    )


if __name__ == "__main__":
    main()
