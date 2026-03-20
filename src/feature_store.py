"""
Feature Store Module
====================
Computes, caches, and retrieves features for the ML pipeline.
Supports TF-IDF text features and numeric features with hash-based caching.

Usage:
    from src.feature_store import FeatureStore
    store = FeatureStore(config)
    X_train, X_test = store.get_features(df_train, df_test)
"""

import hashlib
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse, spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Feature computation and caching for the MLOps pipeline.

    Supports:
        - TF-IDF text vectorization (with configurable max_features)
        - Numeric feature scaling
        - Hash-based caching to avoid redundant computation
    """

    def __init__(
        self, config: Dict[str, Any], cache_dir: str = "data/feature_cache"
    ) -> None:
        self.config = config
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        data_cfg = config.get("data", {})
        self.text_column = data_cfg.get("text_column", "review_text")
        self.label_column = data_cfg.get("label_column", "sentiment")
        self.max_features = data_cfg.get("max_features", 10_000)

        self.tfidf: Optional[TfidfVectorizer] = None
        self.scaler: Optional[StandardScaler] = None
        self._feature_metadata: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, df: pd.DataFrame, prefix: str) -> str:
        """Deterministic cache key from DataFrame contents."""
        content_hash = hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:10]
        return f"{prefix}_{content_hash}"

    def _load_cache(self, key: str) -> Optional[Any]:
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(path):
            logger.info("Cache HIT: %s", key)
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, key: str, obj: Any) -> None:
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info("Cached → %s", path)

    # ------------------------------------------------------------------
    # TF-IDF features
    # ------------------------------------------------------------------

    def compute_tfidf(
        self, train_texts: pd.Series, test_texts: Optional[pd.Series] = None
    ) -> Tuple[spmatrix, Optional[spmatrix]]:
        """Fit TF-IDF on training texts and optionally transform test texts."""
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            min_df=2,
        )
        X_train_tfidf = self.tfidf.fit_transform(train_texts)
        logger.info(
            "TF-IDF fitted: %d features from %d documents",
            X_train_tfidf.shape[1],
            X_train_tfidf.shape[0],
        )

        X_test_tfidf = None
        if test_texts is not None:
            X_test_tfidf = self.tfidf.transform(test_texts)

        self._feature_metadata["tfidf_features"] = X_train_tfidf.shape[1]
        self._feature_metadata["tfidf_vocab_size"] = len(self.tfidf.vocabulary_)
        return X_train_tfidf, X_test_tfidf

    # ------------------------------------------------------------------
    # Numeric features
    # ------------------------------------------------------------------

    def compute_numeric_features(
        self,
        df_train: pd.DataFrame,
        df_test: Optional[pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Scale numeric columns with StandardScaler."""
        if columns is None:
            exclude = {self.text_column, f"{self.text_column}_clean", self.label_column}
            columns = [
                c
                for c in df_train.select_dtypes(include=[np.number]).columns
                if c not in exclude
            ]

        if not columns:
            return np.empty((len(df_train), 0)), (
                np.empty((len(df_test), 0)) if df_test is not None else None
            )

        self.scaler = StandardScaler()
        X_train_num = self.scaler.fit_transform(df_train[columns].fillna(0))
        logger.info("Numeric features scaled: %d columns", len(columns))

        X_test_num = None
        if df_test is not None:
            X_test_num = self.scaler.transform(df_test[columns].fillna(0))

        self._feature_metadata["numeric_columns"] = columns
        return X_train_num, X_test_num

    # ------------------------------------------------------------------
    # Combined feature pipeline
    # ------------------------------------------------------------------

    def get_features(
        self,
        df_train: pd.DataFrame,
        df_test: Optional[pd.DataFrame] = None,
        use_cache: bool = True,
    ) -> Tuple[Any, Optional[Any]]:
        """
        Build full feature matrix (TF-IDF + numeric) for train and test sets.

        Returns sparse matrices combining text and numeric features.
        """
        cache_key = self._cache_key(df_train, "features")

        if use_cache:
            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        # Text column (use cleaned version if available)
        text_col_clean = f"{self.text_column}_clean"
        train_texts = (
            df_train[text_col_clean]
            if text_col_clean in df_train.columns
            else df_train[self.text_column].astype(str)
        )
        test_texts = None
        if df_test is not None:
            test_texts = (
                df_test[text_col_clean]
                if text_col_clean in df_test.columns
                else df_test[self.text_column].astype(str)
            )

        X_train_tfidf, X_test_tfidf = self.compute_tfidf(train_texts, test_texts)
        X_train_num, X_test_num = self.compute_numeric_features(df_train, df_test)

        # Combine sparse TF-IDF with dense numeric
        from scipy.sparse import csr_matrix

        if X_train_num.shape[1] > 0:
            X_train = hstack([X_train_tfidf, csr_matrix(X_train_num)])
            X_test = (
                hstack([X_test_tfidf, csr_matrix(X_test_num)])
                if X_test_tfidf is not None
                else None
            )
        else:
            X_train = X_train_tfidf
            X_test = X_test_tfidf

        logger.info("Final feature matrix: %s", X_train.shape)
        self._feature_metadata["total_features"] = X_train.shape[1]
        self._feature_metadata["timestamp"] = datetime.utcnow().isoformat()

        result = (X_train, X_test)
        if use_cache:
            self._save_cache(cache_key, result)
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_transformers(self, path: str = "models/feature_transformers.pkl") -> None:
        """Save fitted TF-IDF vectorizer and scaler for inference."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "tfidf": self.tfidf,
                    "scaler": self.scaler,
                    "metadata": self._feature_metadata,
                },
                f,
            )
        logger.info("Feature transformers saved → %s", path)

    def load_transformers(self, path: str = "models/feature_transformers.pkl") -> None:
        """Load pre-fitted transformers for inference."""
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        self.tfidf = bundle["tfidf"]
        self.scaler = bundle["scaler"]
        self._feature_metadata = bundle.get("metadata", {})
        logger.info("Feature transformers loaded from %s", path)

    def transform_single(
        self, text: str, numeric_values: Optional[Dict[str, float]] = None
    ) -> Any:
        """Transform a single input for inference."""
        if self.tfidf is None:
            raise RuntimeError(
                "Transformers not fitted. Call load_transformers() first."
            )
        from src.data_pipeline import preprocess_text

        clean = preprocess_text(text)
        X_tfidf = self.tfidf.transform([clean])

        if self.scaler is not None and numeric_values:
            num_cols = self._feature_metadata.get("numeric_columns", [])
            num_arr = np.array([[numeric_values.get(c, 0) for c in num_cols]])
            X_num = self.scaler.transform(num_arr)
            from scipy.sparse import csr_matrix

            return hstack([X_tfidf, csr_matrix(X_num)])
        return X_tfidf

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._feature_metadata)
