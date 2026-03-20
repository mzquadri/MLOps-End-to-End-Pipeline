"""
Training Module
===============
Trains ML models with MLflow experiment tracking, cross-validation,
and hyperparameter logging.  Supports Logistic Regression, Random Forest,
and SVM classifiers for the sentiment-analysis demo task.

Usage:
    python src/train.py --config configs/train_config.yaml --experiment sentiment-v1
"""

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "svm": SVC,
}


def build_model(config: Dict[str, Any]) -> Any:
    """Instantiate a model from config."""
    model_type = config["model"]["type"]
    hyperparams = config["model"]["hyperparameters"].get(model_type, {})

    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. Choose from {list(MODEL_REGISTRY)}"
        )

    # SVM needs probability=True for calibration/soft predictions
    if model_type == "svm":
        hyperparams["probability"] = True

    model = MODEL_REGISTRY[model_type](**hyperparams)
    logger.info("Built model: %s with params %s", model_type, hyperparams)
    return model


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------


def cross_validate_model(
    model: Any,
    X: Any,
    y: np.ndarray,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Run stratified k-fold cross-validation."""
    cv_folds = config.get("training", {}).get("cv_folds", 5)
    scoring = config.get("training", {}).get("scoring", "f1_weighted")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring, n_jobs=-1)

    results = {
        "cv_mean": float(np.mean(scores)),
        "cv_std": float(np.std(scores)),
        "cv_scores": scores.tolist(),
    }
    logger.info("CV %s: %.4f ± %.4f", scoring, results["cv_mean"], results["cv_std"])
    return results


def train_model(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
    config: Dict[str, Any],
    experiment_name: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train model, evaluate, and optionally log to MLflow.

    Returns (fitted_model, metrics_dict).
    """
    model = build_model(config)

    # Cross-validation
    cv_results = cross_validate_model(model, X_train, y_train, config)

    # Final fit
    logger.info("Training final model on %d samples …", X_train.shape[0])
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Predict
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "precision_weighted": float(
            precision_score(y_test, y_pred, average="weighted")
        ),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted")),
        "train_time_seconds": round(train_time, 2),
        **cv_results,
    }
    logger.info(
        "Test accuracy: %.4f  |  F1: %.4f", metrics["accuracy"], metrics["f1_weighted"]
    )

    # MLflow logging (optional – graceful fallback if MLflow is unavailable)
    _try_mlflow_log(config, model, metrics, experiment_name)

    return model, metrics


# ---------------------------------------------------------------------------
# MLflow integration
# ---------------------------------------------------------------------------


def _try_mlflow_log(
    config: Dict[str, Any],
    model: Any,
    metrics: Dict[str, Any],
    experiment_name: Optional[str],
) -> None:
    """Attempt to log run to MLflow; skip silently if not installed."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        logger.warning("mlflow not installed – skipping experiment tracking.")
        return

    tracking_uri = config.get("mlflow", {}).get("tracking_uri", "sqlite:///mlruns.db")
    mlflow.set_tracking_uri(tracking_uri)

    exp_name = experiment_name or config.get("experiment", {}).get("name", "default")
    mlflow.set_experiment(exp_name)

    with mlflow.start_run():
        # Log parameters
        model_type = config["model"]["type"]
        mlflow.log_param("model_type", model_type)
        hyperparams = config["model"]["hyperparameters"].get(model_type, {})
        for k, v in hyperparams.items():
            mlflow.log_param(k, v)
        mlflow.log_param("cv_folds", config.get("training", {}).get("cv_folds", 5))

        # Log metrics
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(name, value)

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        logger.info("MLflow run logged to experiment '%s'", exp_name)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(
    model: Any, metrics: Dict[str, Any], path: str = "models/latest"
) -> None:
    """Save trained model and metrics to disk."""
    import joblib

    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, "model.joblib"))

    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Model and metrics saved → %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train model with experiment tracking")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Import sibling modules
    from src.data_pipeline import load_and_preprocess
    from src.feature_store import FeatureStore

    # Data
    df, data_hash = load_and_preprocess(config)
    label_col = config["data"].get("label_column", "sentiment")
    y = df[label_col].values

    # Train/test split
    test_size = config["data"].get("test_size", 0.2)
    random_state = config["data"].get("random_state", 42)
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Features
    store = FeatureStore(config)
    X_train, X_test = store.get_features(df_train, df_test, use_cache=False)
    store.save_transformers()

    # Train
    model, metrics = train_model(
        X_train, y_train, X_test, y_test, config, args.experiment
    )
    save_model(model, metrics)

    logger.info(
        "Training pipeline complete. Metrics:\n%s", json.dumps(metrics, indent=2)
    )


if __name__ == "__main__":
    main()
