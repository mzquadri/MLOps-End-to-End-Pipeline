"""
Model Registry Module
=====================
Version-controlled model storage with stage management.
Supports transitions: None → Staging → Production, with optional MLflow backend.

Usage:
    python src/model_registry.py --register --stage production
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

VALID_STAGES = {"none", "staging", "production", "archived"}


# ---------------------------------------------------------------------------
# Local registry
# ---------------------------------------------------------------------------


class LocalModelRegistry:
    """
    File-system-based model registry with JSON metadata.

    Models are stored under ``registry_dir/<model_name>/<version>/``,
    and a ``registry.json`` keeps the index of all versions and stages.
    """

    def __init__(self, registry_dir: str = "models/registry") -> None:
        self.registry_dir = registry_dir
        self.index_path = os.path.join(registry_dir, "registry.json")
        os.makedirs(registry_dir, exist_ok=True)
        self._index = self._load_index()

    # ---- index management -----------------------------------------------

    def _load_index(self) -> Dict[str, Any]:
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                return json.load(f)
        return {"models": {}}

    def _save_index(self) -> None:
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2, default=str)

    # ---- registration ----------------------------------------------------

    def register_model(
        self,
        model_name: str,
        model_path: str,
        metrics: Dict[str, Any],
        stage: str = "none",
        description: str = "",
    ) -> str:
        """
        Copy a trained model to the registry and record metadata.
        Returns the assigned version string (e.g., "v3").
        """
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Choose from {VALID_STAGES}")

        # Determine version number
        if model_name not in self._index["models"]:
            self._index["models"][model_name] = {"versions": []}
        versions = self._index["models"][model_name]["versions"]
        version_num = len(versions) + 1
        version_tag = f"v{version_num}"

        # Copy model artifacts
        dest_dir = os.path.join(self.registry_dir, model_name, version_tag)
        os.makedirs(dest_dir, exist_ok=True)
        for fname in os.listdir(model_path):
            src = os.path.join(model_path, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dest_dir)

        # Compute model hash for lineage
        model_file = os.path.join(dest_dir, "model.joblib")
        model_hash = (
            self._file_hash(model_file) if os.path.exists(model_file) else "unknown"
        )

        # Record metadata
        entry = {
            "version": version_tag,
            "stage": stage,
            "registered_at": datetime.utcnow().isoformat(),
            "model_hash": model_hash,
            "metrics": {
                k: v for k, v in metrics.items() if isinstance(v, (int, float, str))
            },
            "description": description,
            "artifacts_path": dest_dir,
        }
        versions.append(entry)
        self._save_index()

        logger.info("Registered %s %s (stage=%s)", model_name, version_tag, stage)
        return version_tag

    # ---- stage transitions -----------------------------------------------

    def transition_stage(self, model_name: str, version: str, new_stage: str) -> None:
        """Move a model version to a new stage."""
        if new_stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{new_stage}'.")
        entry = self._get_version(model_name, version)
        old_stage = entry["stage"]
        entry["stage"] = new_stage
        entry["stage_transition"] = {
            "from": old_stage,
            "to": new_stage,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Demote previous production model if promoting to production
        if new_stage == "production":
            for v in self._index["models"][model_name]["versions"]:
                if v["version"] != version and v["stage"] == "production":
                    v["stage"] = "archived"
                    logger.info("Archived previous production model: %s", v["version"])

        self._save_index()
        logger.info(
            "Transitioned %s %s: %s → %s", model_name, version, old_stage, new_stage
        )

    # ---- querying --------------------------------------------------------

    def get_production_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Return metadata for the current production model (if any)."""
        if model_name not in self._index["models"]:
            return None
        for v in reversed(self._index["models"][model_name]["versions"]):
            if v["stage"] == "production":
                return v
        return None

    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        if model_name not in self._index["models"]:
            return []
        return self._index["models"][model_name]["versions"]

    def load_model(self, model_name: str, version: Optional[str] = None) -> Any:
        """Load a model artifact from the registry."""
        if version is None:
            entry = self.get_production_model(model_name)
            if entry is None:
                raise RuntimeError(f"No production model found for '{model_name}'")
        else:
            entry = self._get_version(model_name, version)

        model_file = os.path.join(entry["artifacts_path"], "model.joblib")
        logger.info("Loading model from %s", model_file)
        return joblib.load(model_file)

    # ---- helpers ---------------------------------------------------------

    def _get_version(self, model_name: str, version: str) -> Dict[str, Any]:
        if model_name not in self._index["models"]:
            raise KeyError(f"Model '{model_name}' not found in registry.")
        for v in self._index["models"][model_name]["versions"]:
            if v["version"] == version:
                return v
        raise KeyError(f"Version '{version}' not found for model '{model_name}'.")

    @staticmethod
    def _file_hash(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()[:12]


# ---------------------------------------------------------------------------
# MLflow registry wrapper (optional)
# ---------------------------------------------------------------------------


def register_with_mlflow(
    model_name: str,
    run_id: str,
    stage: str = "Staging",
    tracking_uri: str = "sqlite:///mlruns.db",
) -> None:
    """Register a logged MLflow model by run-id and transition its stage."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        logger.warning("mlflow not installed – using local registry only.")
        return

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    client.transition_model_version_stage(
        name=model_name, version=mv.version, stage=stage
    )
    logger.info("MLflow: registered %s v%s → %s", model_name, mv.version, stage)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Model Registry management")
    parser.add_argument("--register", action="store_true", help="Register a model")
    parser.add_argument("--model_path", type=str, default="models/latest")
    parser.add_argument("--model_name", type=str, default="sentiment-classifier")
    parser.add_argument(
        "--stage", type=str, default="staging", choices=list(VALID_STAGES)
    )
    parser.add_argument("--list", action="store_true", help="List registered versions")
    parser.add_argument(
        "--promote", type=str, default=None, help="Promote version to production"
    )
    args = parser.parse_args()

    registry = LocalModelRegistry()

    if args.register:
        # Load metrics from model directory
        metrics_file = os.path.join(args.model_path, "metrics.json")
        metrics = {}
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics = json.load(f)

        registry.register_model(
            model_name=args.model_name,
            model_path=args.model_path,
            metrics=metrics,
            stage=args.stage,
        )

    if args.list:
        versions = registry.list_versions(args.model_name)
        for v in versions:
            print(
                f"  {v['version']}  stage={v['stage']}  acc={v['metrics'].get('accuracy', 'N/A')}"
            )

    if args.promote:
        registry.transition_stage(args.model_name, args.promote, "production")


if __name__ == "__main__":
    main()
