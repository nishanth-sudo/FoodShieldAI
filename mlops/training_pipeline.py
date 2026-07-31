import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.data_dir = config.get("data_dir", "mlops/datasets")
        self.prepared_dir = config.get("prepared_dir", "mlops/datasets/prepared")
        self.experiment_dir = config.get("experiment_dir", "mlops/experiments")
        self.model_registry_dir = config.get("model_registry_dir", "mlops/model_registry")
        self.mlflow_uri = config.get("mlflow_tracking_uri")
        self.dvc_enabled = config.get("dvc_enabled", True)

    def prepare_data(self, dataset_name: str = "Freshness44") -> dict:
        from mlops.pipelines.prepare_data import prepare_food101, prepare_freshness44

        if dataset_name == "Freshness44":
            return prepare_freshness44(
                data_dir=f"{self.data_dir}/Freshness44",
                output_dir=f"{self.prepared_dir}/Freshness44",
            )
        elif dataset_name == "Food101":
            return prepare_food101(
                data_dir=f"{self.data_dir}/FoodImages",
                output_dir=f"{self.prepared_dir}/Food101",
            )
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def pull_data_via_dvc(self, dataset_path: str | None = None) -> bool:
        if not self.dvc_enabled:
            return True
        try:
            target = dataset_path or f"{self.data_dir}/Freshness44"
            result = subprocess.run(
                [sys.executable, "-m", "dvc", "pull", target],
                capture_output=True, text=True, timeout=300,
            )
            if result.returncode != 0:
                logger.warning(f"DVC pull failed: {result.stderr}")
                return False
            logger.info(f"DVC pull successful: {target}")
            return True
        except Exception as e:
            logger.warning(f"DVC pull error (non-fatal): {e}")
            return False

    def run(
        self,
        model_name: str = "spoilage_detection",
        dataset_name: str = "Freshness44",
        dataset_version: str = "latest",
        hyperparams: dict | None = None,
    ) -> dict:
        logger.info(f"Starting pipeline: model={model_name}, dataset={dataset_name}")

        self.pull_data_via_dvc()

        data_info = self.prepare_data(dataset_name)
        logger.info(f"Data prepared: {data_info.get('splits', {})}")

        if model_name == "spoilage_detection":
            from mlops.pipelines.spoilage_training import run_spoilage_training

            hp = hyperparams or {}
            result = run_spoilage_training(
                data_dir=f"{self.prepared_dir}/{dataset_name}",
                checkpoint_dir=f"{self.experiment_dir}/{model_name}",
                epochs=hp.get("epochs", 30),
                batch_size=hp.get("batch_size", 32),
                learning_rate=hp.get("learning_rate", 5e-4),
                backbone=hp.get("backbone", "resnet34"),
                mlflow_uri=self.mlflow_uri,
            )

            if result.get("checkpoint_path"):
                self._register_model(model_name, result["checkpoint_path"], result)

            return result
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _register_model(self, model_name: str, checkpoint_path: str, metrics: dict) -> None:
        registry_dir = Path(self.model_registry_dir) / model_name
        registry_dir.mkdir(parents=True, exist_ok=True)

        import datetime
        import shutil

        version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dest = registry_dir / f"v_{version}.pt"
        shutil.copy2(checkpoint_path, model_dest)

        manifest = {
            "model_name": model_name,
            "version": version,
            "timestamp": datetime.datetime.now().isoformat(),
            "source_checkpoint": checkpoint_path,
            "metrics": {
                "best_metric": metrics.get("best_metric"),
                "best_epoch": metrics.get("best_epoch"),
            },
            "status": "staged",
        }
        (registry_dir / f"v_{version}.json").write_text(json.dumps(manifest, indent=2))

        latest_symlink = registry_dir / "latest"
        if latest_symlink.exists():
            latest_symlink.unlink()
        latest_symlink.symlink_to(f"v_{version}.pt") if not latest_symlink.exists() else None

        logger.info(f"Model registered: {model_name} v_{version} at {model_dest}")
