import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai_engine.training.config import AugmentationConfig, TrainingConfig
from ai_engine.training.dataset import SpoilageDataset
from ai_engine.training.trainer import ModelTrainer

logger = logging.getLogger(__name__)


def load_prepared_data(split_dir: str) -> tuple[list[str], list[int], list[float]]:
    meta_path = Path(split_dir) / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found at {meta_path}. Run prepare_data first.")

    data = json.loads(meta_path.read_text())
    image_paths = [r["image_path"] for r in data["records"]]
    spoilage_labels = [r["label"] for r in data["records"]]
    severity_scores = [float(r.get("severity", 0.5)) for r in data["records"]]
    return image_paths, spoilage_labels, severity_scores


def run_spoilage_training(
    data_dir: str = "mlops/datasets/prepared/Freshness44",
    checkpoint_dir: str = "mlops/experiments/spoilage_detection",
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 5e-4,
    backbone: str = "resnet34",
    mlflow_uri: str | None = None,
    device: str = "cpu",
) -> dict:
    config = TrainingConfig(
        model_type="spoilage_detection",
        backbone=backbone,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        save_dir=checkpoint_dir,
        experiment_name="spoilage_freshness44",
        early_stopping_patience=10,
        mlflow_tracking_uri=mlflow_uri,
        mixed_precision=torch.cuda.is_available(),
    )
    aug_config = AugmentationConfig(enabled=True)

    logger.info("Loading Freshness44 dataset...")
    train_paths, train_labels, train_severity = load_prepared_data(f"{data_dir}/train")
    val_paths, val_labels, val_severity = load_prepared_data(f"{data_dir}/val")

    logger.info(f"Train: {len(train_paths)} samples, Val: {len(val_paths)} samples")

    train_dataset = SpoilageDataset(
        image_paths=train_paths,
        spoilage_labels=train_labels,
        severity_scores=train_severity,
        augment=True,
        aug_config=aug_config,
    )
    val_dataset = SpoilageDataset(
        image_paths=val_paths,
        spoilage_labels=val_labels,
        severity_scores=val_severity,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    trainer = ModelTrainer(config)
    history = trainer.train(train_loader, val_loader)

    best_ckpt = Path(checkpoint_dir) / "spoilage_freshness44" / "best_model.pt"
    if best_ckpt.exists():
        logger.info(f"Best model saved to {best_ckpt}")

    return {
        "best_metric": trainer.best_metric,
        "best_epoch": trainer.best_epoch,
        "history": history,
        "checkpoint_path": str(best_ckpt) if best_ckpt.exists() else None,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_spoilage_training()
    print(f"Training completed. Best metric: {result['best_metric']:.4f} at epoch {result['best_epoch']}")
