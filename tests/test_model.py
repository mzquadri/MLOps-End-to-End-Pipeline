"""
Tests for model training, feature store, and evaluation.
"""

import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_config():
    return {
        "experiment": {"name": "test-experiment"},
        "data": {
            "source": "data/reviews.csv",
            "test_size": 0.2,
            "random_state": 42,
            "text_column": "review_text",
            "label_column": "sentiment",
            "max_features": 200,
        },
        "model": {
            "type": "logistic_regression",
            "hyperparameters": {
                "logistic_regression": {"C": 1.0, "max_iter": 200, "solver": "lbfgs"},
                "random_forest": {"n_estimators": 10, "max_depth": 5},
                "svm": {"C": 1.0, "kernel": "rbf"},
            },
        },
        "training": {"cv_folds": 3, "scoring": "f1_weighted"},
        "validation": {
            "min_accuracy": 0.5,
            "min_f1": 0.5,
            "max_drift_psi": 0.3,
            "max_latency_ms": 500,
        },
        "mlflow": {"tracking_uri": "sqlite:///test_mlruns.db"},
    }


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    n = 300
    positive_templates = [
        "great product love it",
        "excellent quality highly recommend",
        "best purchase ever made",
    ]
    negative_templates = [
        "terrible quality broke fast",
        "waste of money do not buy",
        "horrible experience avoid",
    ]
    texts, sentiments = [], []
    for i in range(n):
        if rng.random() > 0.45:
            texts.append(rng.choice(positive_templates) + f" extra words {i}")
            sentiments.append("positive")
        else:
            texts.append(rng.choice(negative_templates) + f" extra words {i}")
            sentiments.append("negative")
    return pd.DataFrame(
        {
            "review_text": texts,
            "sentiment": sentiments,
            "review_length": [len(t) for t in texts],
            "word_count": [len(t.split()) for t in texts],
        }
    )


# ---------------------------------------------------------------------------
# Build Model
# ---------------------------------------------------------------------------


class TestBuildModel:
    def test_logistic_regression(self, sample_config):
        from src.train import build_model

        model = build_model(sample_config)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_random_forest(self, sample_config):
        from src.train import build_model

        sample_config["model"]["type"] = "random_forest"
        model = build_model(sample_config)
        assert hasattr(model, "fit")

    def test_svm(self, sample_config):
        from src.train import build_model

        sample_config["model"]["type"] = "svm"
        model = build_model(sample_config)
        assert hasattr(model, "predict_proba")  # probability=True added

    def test_invalid_type(self, sample_config):
        from src.train import build_model

        sample_config["model"]["type"] = "nonexistent"
        with pytest.raises(ValueError, match="Unknown model type"):
            build_model(sample_config)


# ---------------------------------------------------------------------------
# Feature Store
# ---------------------------------------------------------------------------


class TestFeatureStore:
    def test_tfidf_computation(self, sample_config, sample_df):
        from src.feature_store import FeatureStore

        store = FeatureStore(sample_config, cache_dir=tempfile.mkdtemp())
        df_train, df_test = train_test_split(sample_df, test_size=0.2, random_state=42)
        X_train, X_test = store.compute_tfidf(
            df_train["review_text"], df_test["review_text"]
        )
        assert X_train.shape[0] == len(df_train)
        assert X_test.shape[0] == len(df_test)
        assert X_train.shape[1] == X_test.shape[1]

    def test_numeric_features(self, sample_config, sample_df):
        from src.feature_store import FeatureStore

        store = FeatureStore(sample_config, cache_dir=tempfile.mkdtemp())
        df_train, df_test = train_test_split(sample_df, test_size=0.2, random_state=42)
        X_train, X_test = store.compute_numeric_features(df_train, df_test)
        assert X_train.shape[0] == len(df_train)
        assert X_train.shape[1] == 2  # review_length, word_count

    def test_get_features_combined(self, sample_config, sample_df):
        from src.feature_store import FeatureStore

        store = FeatureStore(sample_config, cache_dir=tempfile.mkdtemp())
        df_train, df_test = train_test_split(sample_df, test_size=0.2, random_state=42)
        X_train, X_test = store.get_features(df_train, df_test, use_cache=False)
        assert X_train.shape[0] == len(df_train)
        assert X_test.shape[0] == len(df_test)
        # TF-IDF + 2 numeric columns
        assert X_train.shape[1] > 2

    def test_save_and_load_transformers(self, sample_config, sample_df):
        from src.feature_store import FeatureStore

        store = FeatureStore(sample_config, cache_dir=tempfile.mkdtemp())
        df_train, _ = train_test_split(sample_df, test_size=0.2, random_state=42)
        store.get_features(df_train, use_cache=False)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        store.save_transformers(path)

        store2 = FeatureStore(sample_config, cache_dir=tempfile.mkdtemp())
        store2.load_transformers(path)
        assert store2.tfidf is not None


# ---------------------------------------------------------------------------
# Training end-to-end
# ---------------------------------------------------------------------------


class TestTrainModel:
    def test_train_and_metrics(self, sample_config, sample_df):
        from src.feature_store import FeatureStore
        from src.train import train_model

        df_train, df_test = train_test_split(sample_df, test_size=0.2, random_state=42)
        y_train = df_train["sentiment"].values
        y_test = df_test["sentiment"].values

        store = FeatureStore(sample_config, cache_dir=tempfile.mkdtemp())
        X_train, X_test = store.get_features(df_train, df_test, use_cache=False)

        model, metrics = train_model(X_train, y_train, X_test, y_test, sample_config)
        assert "accuracy" in metrics
        assert "f1_weighted" in metrics
        assert metrics["accuracy"] > 0
        assert hasattr(model, "predict")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestModelEvaluator:
    def test_compute_metrics(self, sample_config, sample_df):
        from src.evaluate import ModelEvaluator
        from src.feature_store import FeatureStore
        from src.train import train_model

        df_train, df_test = train_test_split(sample_df, test_size=0.2, random_state=42)
        y_train = df_train["sentiment"].values
        y_test = df_test["sentiment"].values

        store = FeatureStore(sample_config, cache_dir=tempfile.mkdtemp())
        X_train, X_test = store.get_features(df_train, df_test, use_cache=False)
        model, _ = train_model(X_train, y_train, X_test, y_test, sample_config)

        evaluator = ModelEvaluator(model, sample_config)
        metrics = evaluator.compute_metrics(X_test, y_test)
        assert "accuracy" in metrics
        assert "confusion_matrix" in metrics
        assert "classification_report" in metrics

    def test_latency_measurement(self, sample_config, sample_df):
        from src.evaluate import ModelEvaluator
        from src.feature_store import FeatureStore
        from src.train import train_model

        df_train, df_test = train_test_split(sample_df, test_size=0.2, random_state=42)
        store = FeatureStore(sample_config, cache_dir=tempfile.mkdtemp())
        X_train, X_test = store.get_features(df_train, df_test, use_cache=False)
        model, _ = train_model(
            X_train,
            df_train["sentiment"].values,
            X_test,
            df_test["sentiment"].values,
            sample_config,
        )

        evaluator = ModelEvaluator(model, sample_config)
        latency = evaluator.measure_latency(X_test, n_runs=10)
        assert "mean_latency_ms" in latency
        assert latency["mean_latency_ms"] > 0


# ---------------------------------------------------------------------------
# Performance Gate
# ---------------------------------------------------------------------------


class TestPerformanceGate:
    def test_passes_when_above_threshold(self, sample_config):
        from src.evaluate import PerformanceGate

        gate = PerformanceGate(sample_config)
        result = gate.evaluate(
            {"accuracy": 0.90, "f1_weighted": 0.88},
            {"p95_latency_ms": 5.0},
        )
        assert result["overall_passed"] is True

    def test_fails_when_below_threshold(self, sample_config):
        from src.evaluate import PerformanceGate

        gate = PerformanceGate(sample_config)
        result = gate.evaluate(
            {"accuracy": 0.40, "f1_weighted": 0.35},
            {"p95_latency_ms": 5.0},
        )
        assert result["overall_passed"] is False


# ---------------------------------------------------------------------------
# Model Registry
# ---------------------------------------------------------------------------


class TestModelRegistry:
    def test_register_and_list(self, sample_config):
        import tempfile

        from src.model_registry import LocalModelRegistry

        with (
            tempfile.TemporaryDirectory() as reg_dir,
            tempfile.TemporaryDirectory() as model_dir,
        ):
            # Create a dummy model file
            import joblib
            from sklearn.linear_model import LogisticRegression

            m = LogisticRegression()
            m.fit([[1], [2]], [0, 1])
            joblib.dump(m, f"{model_dir}/model.joblib")

            registry = LocalModelRegistry(registry_dir=reg_dir)
            v = registry.register_model(
                "test-model", model_dir, {"accuracy": 0.9}, stage="staging"
            )
            assert v == "v1"

            versions = registry.list_versions("test-model")
            assert len(versions) == 1
            assert versions[0]["stage"] == "staging"

    def test_promote_to_production(self):
        import tempfile

        from src.model_registry import LocalModelRegistry

        with (
            tempfile.TemporaryDirectory() as reg_dir,
            tempfile.TemporaryDirectory() as model_dir,
        ):
            import joblib
            from sklearn.linear_model import LogisticRegression

            m = LogisticRegression()
            m.fit([[1], [2]], [0, 1])
            joblib.dump(m, f"{model_dir}/model.joblib")

            registry = LocalModelRegistry(registry_dir=reg_dir)
            registry.register_model(
                "test-model", model_dir, {"accuracy": 0.9}, stage="staging"
            )
            registry.transition_stage("test-model", "v1", "production")

            prod = registry.get_production_model("test-model")
            assert prod is not None
            assert prod["stage"] == "production"
