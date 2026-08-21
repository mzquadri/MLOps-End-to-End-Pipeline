"""
Model Registry Module
=====================
Version-controlled model storage with stage management.
Supports transitions: None → Staging → Production, with optional MLflow backend.

Usage:
    python -m src.model_registry --register --stage staging
    python -m src.model_registry --promote v1
"""

import argparse
import copy
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        os.makedirs(self.registry_dir, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".registry.", dir=self.registry_dir)
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._index, f, indent=2, sort_keys=True, default=str)
                f.write("\n")
            os.replace(temp_path, self.index_path)
        except BaseException:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise

    # ---- registration ----------------------------------------------------

    def register_model(
        self,
        model_name: str,
        model_path: str,
        stage: str = "none",
        description: str = "",
    ) -> str:
        """
        Copy a trained model to the registry and record metadata.
        Returns the assigned version string (e.g., "v3").
        """
        if stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{stage}'. Choose from {VALID_STAGES}")
        if stage in {"production", "archived"}:
            raise ValueError(
                "Register as 'none' or 'staging'; production requires promotion"
            )
        if not model_name or any(part in model_name for part in ("/", "\\", "..")):
            raise ValueError("model_name must be a safe directory name")

        from src.model_bundle import (
            EVALUATION_REPORT_FILE,
            TRAINING_METRICS_FILE,
            load_trusted_bundle,
            read_bundle_json,
            validate_bundle,
        )

        validate_bundle(
            model_path,
            require_evaluated=stage == "staging",
            require_gate_passed=stage == "staging",
        )
        if stage == "staging":
            load_trusted_bundle(model_path, require_gate_passed=True)
        bundle_metrics = read_bundle_json(model_path, TRAINING_METRICS_FILE)
        evaluation = read_bundle_json(model_path, EVALUATION_REPORT_FILE)

        # Determine version number
        if model_name not in self._index["models"]:
            self._index["models"][model_name] = {"versions": []}
        versions = self._index["models"][model_name]["versions"]
        version_num = len(versions) + 1
        version_tag = f"v{version_num}"

        # Prepare immutable registry metadata and bundle destination.
        relative_bundle_path = os.path.join(model_name, version_tag).replace("\\", "/")
        dest_dir = os.path.join(self.registry_dir, model_name, version_tag)
        orphan_reused = False
        if os.path.exists(dest_dir):
            source_manifest = validate_bundle(
                model_path,
                require_evaluated=stage == "staging",
                require_gate_passed=stage == "staging",
            )
            orphan_manifest = validate_bundle(
                dest_dir,
                require_evaluated=stage == "staging",
                require_gate_passed=stage == "staging",
            )
            if source_manifest != orphan_manifest:
                raise FileExistsError(
                    "Registry version already exists with different content: "
                    f"{dest_dir}"
                )
            if stage == "staging":
                load_trusted_bundle(dest_dir, require_gate_passed=True)
            orphan_reused = True
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        entry = {
            "version": version_tag,
            "stage": stage,
            "registered_at": datetime.utcnow().isoformat(),
            "metrics": {
                k: v
                for k, v in bundle_metrics.items()
                if isinstance(v, (int, float, str))
            },
            "evaluation_status": evaluation.get("evaluation_status", "pending"),
            "description": description,
            "bundle_path": relative_bundle_path,
        }

        temp_dir = ""
        if not orphan_reused:
            temp_dir = tempfile.mkdtemp(
                prefix=f".{version_tag}.", dir=os.path.dirname(dest_dir)
            )
            shutil.rmtree(temp_dir)
        published = False
        indexed = False
        try:
            if not orphan_reused:
                shutil.copytree(model_path, temp_dir)
                validate_bundle(
                    temp_dir,
                    require_evaluated=stage == "staging",
                    require_gate_passed=stage == "staging",
                )
                if stage == "staging":
                    load_trusted_bundle(temp_dir, require_gate_passed=True)
                os.replace(temp_dir, dest_dir)
                published = True
            versions.append(entry)
            indexed = True
            self._save_index()
        except BaseException:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            if indexed:
                versions.remove(entry)
            if published and os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            raise

        logger.info("Registered %s %s (stage=%s)", model_name, version_tag, stage)
        return version_tag

    # ---- stage transitions -----------------------------------------------

    def transition_stage(self, model_name: str, version: str, new_stage: str) -> None:
        """Move a model version to a new stage."""
        if new_stage not in VALID_STAGES:
            raise ValueError(f"Invalid stage '{new_stage}'.")
        original_index = copy.deepcopy(self._index)
        entry = self._get_version(model_name, version)
        old_stage = entry["stage"]
        if new_stage == "production" and old_stage != "staging":
            raise ValueError("Production promotion requires a staging version")
        if new_stage in {"staging", "production"}:
            from src.model_bundle import load_trusted_bundle, validate_bundle

            validate_bundle(
                self._bundle_path(entry),
                require_evaluated=True,
                require_gate_passed=True,
            )
            load_trusted_bundle(
                self._bundle_path(entry), require_gate_passed=True
            )
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

        try:
            self._save_index()
        except BaseException:
            self._index = original_index
            raise
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
        """Load a model from a trusted, checksum-valid registry bundle."""
        if version is None:
            entry = self.get_production_model(model_name)
            if entry is None:
                raise RuntimeError(f"No production model found for '{model_name}'")
        else:
            entry = self._get_version(model_name, version)

        from src.model_bundle import load_trusted_bundle

        bundle_path = self._bundle_path(entry)
        logger.info("Loading model bundle from %s", bundle_path)
        model, _, _ = load_trusted_bundle(
            bundle_path, require_gate_passed=entry["stage"] == "production"
        )
        return model

    # ---- helpers ---------------------------------------------------------

    def _get_version(self, model_name: str, version: str) -> Dict[str, Any]:
        if model_name not in self._index["models"]:
            raise KeyError(f"Model '{model_name}' not found in registry.")
        for v in self._index["models"][model_name]["versions"]:
            if v["version"] == version:
                return v
        raise KeyError(f"Version '{version}' not found for model '{model_name}'.")

    def _bundle_path(self, entry: Dict[str, Any]) -> str:
        relative_path = entry.get("bundle_path")
        if not isinstance(relative_path, str):
            raise RuntimeError("Registry entry is missing bundle_path")
        root = Path(self.registry_dir).resolve()
        resolved = (root / relative_path).resolve()
        if root not in resolved.parents:
            raise RuntimeError("Registry bundle path escapes the registry root")
        return str(resolved)


# ---------------------------------------------------------------------------
# MLflow registry wrapper (optional)
# ---------------------------------------------------------------------------


def register_with_mlflow(
    model_name: str,
    run_id: str,
    stage: str = "Staging",
    tracking_uri: str = "sqlite:///mlruns.db",
) -> None:
    """Register a logged MLflow model in staging only.

    Production promotion belongs to the validated local bundle workflow. This
    optional compatibility helper cannot bypass that gate.
    """
    if stage.lower() != "staging":
        raise ValueError("MLflow compatibility registration is staging-only")
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
    parser.add_argument(
        "--bundle_path",
        type=str,
        default="models/candidates/sentiment-classifier",
    )
    parser.add_argument("--model_name", type=str, default="sentiment-classifier")
    parser.add_argument(
        "--stage", type=str, default="staging", choices=["none", "staging"]
    )
    parser.add_argument("--list", action="store_true", help="List registered versions")
    parser.add_argument(
        "--promote", type=str, default=None, help="Promote version to production"
    )
    args = parser.parse_args()

    registry = LocalModelRegistry()

    if args.register:
        registry.register_model(
            model_name=args.model_name,
            model_path=args.bundle_path,
            stage=args.stage,
        )

    if args.list:
        versions = registry.list_versions(args.model_name)
        for v in versions:
            accuracy = v["metrics"].get("accuracy", "N/A")
            print(
                f"  {v['version']}  stage={v['stage']}  acc={accuracy}"
            )

    if args.promote:
        registry.transition_stage(args.model_name, args.promote, "production")


if __name__ == "__main__":
    main()
