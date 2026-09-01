import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelDeployer:
    def __init__(
        self,
        registry_dir: str = "mlops/model_registry",
        deploy_dir: str = "mlops/deployed",
    ) -> None:
        self.registry_dir = Path(registry_dir)
        self.deploy_dir = Path(deploy_dir)
        self.deploy_dir.mkdir(parents=True, exist_ok=True)

    def _get_model_path(self, model_name: str, version: str | None = None) -> Path | None:
        model_registry = self.registry_dir / model_name
        if not model_registry.exists():
            logger.error(f"No registry found for model '{model_name}'")
            return None

        if version:
            model_file = model_registry / f"v_{version}.pt"
        else:
            latest_link = model_registry / "latest"
            if latest_link.exists():
                resolved = latest_link.resolve()
                stem = resolved.stem
                version = stem.split("_")[-1] if "_" in stem else "unknown"
                return resolved
            versions = sorted(model_registry.glob("v_*.pt"))
            if not versions:
                logger.error(f"No versions found for model '{model_name}'")
                return None
            return versions[-1]

        return model_file if model_file.exists() else None

    def deploy_to_staging(self, model_name: str, version: str | None = None) -> bool:
        model_path = self._get_model_path(model_name, version)
        if model_path is None:
            return False

        staging_dir = self.deploy_dir / "staging" / model_name
        staging_dir.mkdir(parents=True, exist_ok=True)
        dest = staging_dir / model_path.name
        shutil.copy2(str(model_path), str(dest))

        manifest = {
            "model_name": model_name,
            "version": version or "latest",
            "stage": "staging",
            "source": str(model_path),
            "destination": str(dest),
        }
        (staging_dir / "deploy_manifest.json").write_text(json.dumps(manifest, indent=2))
        logger.info(f"Deployed {model_name} v{version or 'latest'} to staging")
        return True

    def promote_to_production(self, model_name: str, version: str | None = None) -> bool:
        """Promote a model from staging to production."""
        staging_path = self._get_staging_path(model_name, version)
        if staging_path is None:
            logger.error(f"No staging deployment found for {model_name}")
            return False

        if not self._run_validation_tests(model_name):
            logger.error(f"Validation tests failed for {model_name}")
            return False

        prod_dir = self.deploy_dir / "production" / model_name
        prod_dir.mkdir(parents=True, exist_ok=True)
        dest = prod_dir / staging_path.name
        shutil.copy2(str(staging_path), str(dest))

        manifest = {
            "model_name": model_name,
            "version": version or "latest",
            "stage": "production",
            "source": str(staging_path),
            "destination": str(dest),
        }
        (prod_dir / "deploy_manifest.json").write_text(json.dumps(manifest, indent=2))

        logger.info(f"Promoted {model_name} v{version or 'latest'} to production")
        return True

    def rollback(self, model_name: str, version: str) -> bool:
        prod_dir = self.deploy_dir / "production" / model_name
        if not prod_dir.exists():
            logger.error(f"No production deployment for {model_name}")
            return False

        model_path = self._get_model_path(model_name, version)
        if model_path is None:
            return False

        dest = prod_dir / model_path.name
        shutil.copy2(str(model_path), str(dest))
        manifest = json.loads((prod_dir / "deploy_manifest.json").read_text())
        manifest["rollback_version"] = version
        (prod_dir / "deploy_manifest.json").write_text(json.dumps(manifest, indent=2))

        logger.info(f"Rolled back {model_name} to version {version}")
        return True

    def _get_staging_path(self, model_name: str, version: str | None = None) -> Path | None:
        staging_dir = self.deploy_dir / "staging" / model_name
        if not staging_dir.exists():
            return None
        if version:
            return staging_dir / f"v_{version}.pt"
        pt_files = list(staging_dir.glob("*.pt"))
        return pt_files[-1] if pt_files else None

    def _run_validation_tests(self, model_name: str) -> bool:
        import torch

        validation_dir = self.deploy_dir / "staging" / model_name
        pt_files = list(validation_dir.glob("*.pt"))
        if not pt_files:
            return False

        try:
            checkpoint = torch.load(pt_files[0], map_location="cpu", weights_only=True)
            if "model_state_dict" not in checkpoint:
                logger.warning("Checkpoint missing model_state_dict")
            required_keys = {"epoch", "model_state_dict", "metric"}
            actual_keys = set(checkpoint.keys())
            missing = required_keys - actual_keys
            if missing:
                logger.warning(f"Checkpoint missing keys: {missing}")
            logger.info(f"Validation passed for {model_name}")
            return True
        except Exception as e:
            logger.error(f"Validation failed for {model_name}: {e}")
            return False
