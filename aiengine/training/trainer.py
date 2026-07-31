import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from aiengine.training.config import TrainingConfig
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_seed()
        self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.loss_fn = self._create_loss_fn()
        use_amp = config.mixed_precision and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None
        self.writer = None
        self.best_metric = float("-inf")
        self.best_epoch = -1
        self.patience_counter = 0
        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [], "val_metric": [],
        }

    def _setup_seed(self) -> None:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def _build_model(self) -> nn.Module:
        config = self.config
        if config.model_type == "food_classification":
            from aiengine.models.food_classification.model import FoodClassifier
            return FoodClassifier(
                num_classes=config.num_classes,
                backbone=config.backbone,
            )
        elif config.model_type == "spoilage_detection":
            from aiengine.models.spoilage_detection.model import SpoilageDetector
            return SpoilageDetector(backbone=config.backbone)
        elif config.model_type == "shelf_life_prediction":
            from aiengine.models.shelf_life_prediction.model import ShelfLifePredictor
            return ShelfLifePredictor(label_feature_dim=config.label_feature_dim)
        elif config.model_type == "contamination_risk":
            from aiengine.models.contamination_risk.model import ContaminationRiskAssessor
            return ContaminationRiskAssessor(
                num_risk_categories=config.num_risk_categories,
                backbone=config.backbone,
            )
        elif config.model_type == "packaging_defect":
            try:
                from ultralytics import YOLO
                return YOLO("yolov8n.pt")
            except ImportError as err:
                raise ImportError("ultralytics is required for packaging defect training") from err
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            return torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

        lr = self.config.learning_rate
        wd = self.config.weight_decay
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
        raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")

    def _create_scheduler(self) -> object:
        t = self.config.scheduler_type
        if t == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        elif t == "step":
            return StepLR(self.optimizer, step_size=self.config.step_size, gamma=self.config.gamma)
        elif t == "plateau":
            return ReduceLROnPlateau(
                self.optimizer, mode="max",
                patience=self.config.early_stopping_patience // 2,
            )
        return None

    def _create_loss_fn(self) -> nn.Module | dict:
        config = self.config
        if config.model_type == "food_classification":
            return nn.CrossEntropyLoss()
        elif config.model_type in ("spoilage_detection",):
            return {
                "spoilage": nn.BCEWithLogitsLoss(),
                "severity": nn.MSELoss(),
            }
        elif config.model_type == "shelf_life_prediction":
            return nn.MSELoss()
        elif config.model_type == "contamination_risk":
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

    def save_checkpoint(self, path: str, epoch: int, metric: float, is_best: bool = False) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "metric": metric,
            "history": self.history,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, path)
        if is_best:
            best_path = str(Path(path).parent / "best_model.pt")
            torch.save(state, best_path)
            logger.info(f"Saved best model (epoch={epoch}, metric={metric:.4f}) to {best_path}")

    def load_checkpoint(self, path: str, load_optimizer: bool = True) -> dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint.get("history", self.history)
        logger.info(f"Loaded checkpoint from {path} (epoch={checkpoint.get('epoch', '?')})")
        return checkpoint

    def _train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            if isinstance(self.loss_fn, dict):
                loss = self._train_multi_output_batch(batch)
            else:
                loss = self._train_standard_batch(batch)

            total_loss += loss

            if batch_idx % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                logger.debug(
                    f"Train batch [{batch_idx}/{num_batches}] "
                    f"Loss: {loss:.4f} ({elapsed:.1f}s)"
                )

        return total_loss / num_batches

    def _apply_grad_clip(self) -> None:
        if self.config.gradient_clip_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)

    def _train_standard_batch(self, batch: list | tuple) -> float:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            images, label_feats, targets = batch
            images = images.to(self.device)
            label_feats = label_feats.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images, label_feats)
                    loss = self.loss_fn(outputs, targets)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                self._apply_grad_clip()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, label_feats)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self._apply_grad_clip()
                self.optimizer.step()
            return loss.item()

        images, targets = batch
        images = images.to(self.device)
        targets = targets.to(self.device)

        self.optimizer.zero_grad()
        if self.scaler:
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.loss_fn(outputs, targets)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self._apply_grad_clip()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self._apply_grad_clip()
            self.optimizer.step()

        return loss.item()

    def _train_multi_output_batch(self, batch: list | tuple) -> float:
        images, targets = batch
        images = images.to(self.device)
        spoilage_target = targets["spoilage_label"].to(self.device).view(-1, 1)
        severity_target = targets["severity"].to(self.device).view(-1, 1)

        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss_spoilage = self.loss_fn["spoilage"](outputs["spoilage_logit"], spoilage_target)
        loss_severity = self.loss_fn["severity"](outputs["severity"], severity_target)
        loss = loss_spoilage + 0.5 * loss_severity
        loss.backward()
        self._apply_grad_clip()
        self.optimizer.step()
        return loss.item()

    def _validate(self, dataloader: DataLoader) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds: list = []
        all_targets: list = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(self.loss_fn, dict):
                    images, targets = batch
                    images = images.to(self.device)
                    spoilage_target = targets["spoilage_label"].to(self.device).view(-1, 1)
                    severity_target = targets["severity"].to(self.device).view(-1, 1)
                    outputs = self.model(images)
                    loss_spoilage = self.loss_fn["spoilage"](
                        outputs["spoilage_logit"], spoilage_target,
                    )
                    loss_severity = self.loss_fn["severity"](
                        outputs["severity"], severity_target,
                    )
                    loss = loss_spoilage + 0.5 * loss_severity
                    all_preds.append(torch.sigmoid(outputs["spoilage_logit"]))
                    all_targets.append(spoilage_target)
                else:
                    if len(batch) == 3:
                        images, label_feats, targets = batch
                        images = images.to(self.device)
                        label_feats = label_feats.to(self.device)
                        targets = targets.to(self.device)
                        outputs = self.model(images, label_feats)
                    else:
                        images, targets = batch
                        images = images.to(self.device)
                        targets = targets.to(self.device)
                        outputs = self.model(images)

                    if outputs.ndim == 1 or outputs.size(-1) == 1:
                        loss = self.loss_fn(outputs.squeeze(), targets.squeeze())
                    else:
                        loss = self.loss_fn(outputs, targets)

                    if outputs.ndim > 1 and outputs.size(-1) > 1:
                        all_preds.append(torch.softmax(outputs, dim=1))
                    else:
                        all_preds.append(outputs)
                    all_targets.append(targets)

                total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if not all_preds:
            return avg_loss, 0.0

        preds = torch.cat(all_preds, dim=0)
        tgts = torch.cat(all_targets, dim=0)

        if self.config.model_type == "food_classification":
            pred_classes = preds.argmax(dim=1)
            if tgts.ndim > 1 and tgts.size(-1) > 1:
                tgts = tgts.argmax(dim=1)
            correct = (pred_classes == tgts).sum().item()
            metric = correct / tgts.size(0)
        elif self.config.model_type == "contamination_risk":
            pred_labels = (preds > 0.3).float()
            correct = (pred_labels == tgts).float().mean().item()
            metric = correct
        elif self.config.model_type in ("spoilage_detection",):
            pred_classes = (preds > 0.5).float()
            correct = (pred_classes == tgts).sum().item()
            metric = correct / tgts.size(0)
        elif self.config.model_type == "shelf_life_prediction":
            mae = nn.L1Loss()(preds.squeeze(), tgts.squeeze()).item()
            metric = -mae
        else:
            metric = -avg_loss

        return avg_loss, metric

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
    ) -> dict[str, list[float]]:
        config = self.config
        logger.info(f"Starting training for {config.experiment_name}")
        logger.info(f"Device: {self.device}, Epochs: {config.epochs}, Batch: {config.batch_size}")
        logger.info(f"Optimizer: {config.optimizer}, LR: {config.learning_rate}")

        save_dir = Path(config.save_dir) / config.experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "config.json", "w") as f:
            json.dump(vars(config), f, indent=2, default=str)

        self.writer = SummaryWriter(log_dir=str(save_dir / "logs"))

        for epoch in range(1, config.epochs + 1):
            start_time = time.time()

            train_loss = self._train_epoch(train_loader)
            val_loss, val_metric = (
                self._validate(val_loader) if val_loader else (train_loss, 0.0)
            )

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_metric"].append(val_metric)

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", val_loss, epoch)
            self.writer.add_scalar("Metric/val", val_metric, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            if isinstance(self.scheduler, (CosineAnnealingLR, StepLR)):
                self.scheduler.step()
            elif isinstance(self.scheduler, ReduceLROnPlateau) and val_loader:
                self.scheduler.step(val_metric)

            elapsed = time.time() - start_time
            logger.info(
                f"Epoch [{epoch}/{config.epochs}] "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Metric: {val_metric:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

            is_best = val_metric > self.best_metric
            if is_best:
                self.best_metric = val_metric
                self.best_epoch = epoch
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if epoch % 5 == 0 or is_best or epoch == config.epochs:
                checkpoint_path = str(save_dir / f"checkpoint_epoch_{epoch}.pt")
                self.save_checkpoint(checkpoint_path, epoch, val_metric, is_best=is_best)

            if self.patience_counter >= config.early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch}. Best epoch: {self.best_epoch}"
                )
                break

            self._log_mlflow(epoch, train_loss, val_loss, val_metric)

        self.writer.close()
        logger.info(
            f"Training complete. Best metric: {self.best_metric:.4f} "
            f"at epoch {self.best_epoch}"
        )
        return self.history

    def evaluate(self, model: nn.Module, test_loader: DataLoader) -> dict:
        model.eval()
        model.to(self.device)
        total_loss = 0.0
        all_preds: list = []
        all_targets: list = []

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 3:
                    images, label_feats, targets = batch
                    images = images.to(self.device)
                    label_feats = label_feats.to(self.device)
                    targets = targets.to(self.device)
                    outputs = model(images, label_feats)
                else:
                    images, targets = batch
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    outputs = model(images)

                all_preds.append(outputs.cpu())
                all_targets.append(targets.cpu())

        preds = torch.cat(all_preds, dim=0)
        tgts = torch.cat(all_targets, dim=0)

        total = total_loss / len(test_loader) if total_loss else 0.0
        results: dict = {"loss": total}

        if self.config.model_type == "food_classification":
            pred_classes = preds.argmax(dim=1)
            if tgts.ndim > 1 and tgts.size(-1) > 1:
                tgts = tgts.argmax(dim=1)
            correct = (pred_classes == tgts).sum().item()
            results["accuracy"] = correct / tgts.size(0)
        elif self.config.model_type == "contamination_risk":
            pred_labels = (preds > 0.3).float()
            correct = (pred_labels == tgts).float().mean().item()
            results["accuracy"] = correct
        elif self.config.model_type == "spoilage_detection":
            pred_classes = (preds > 0.5).float()
            correct = (pred_classes == tgts).sum().item()
            results["accuracy"] = correct / tgts.size(0)
        elif self.config.model_type == "shelf_life_prediction":
            results["mae"] = nn.L1Loss()(preds.squeeze(), tgts.squeeze()).item()
            results["mse"] = nn.MSELoss()(preds.squeeze(), tgts.squeeze()).item()

        return results

    def _log_mlflow(
        self, epoch: int, train_loss: float, val_loss: float, val_metric: float,
    ) -> None:
        if not self.config.mlflow_tracking_uri:
            return
        try:
            import mlflow
            if not mlflow.active_run():
                mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
                mlflow.start_run(run_name=self.config.experiment_name, nested=True)
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_metric": val_metric,
            }, step=epoch)
        except ImportError:
            pass
