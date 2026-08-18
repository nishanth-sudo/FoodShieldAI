from dataclasses import dataclass
from typing import Literal

ModelType = Literal[
    "food_classification",
    "spoilage_detection",
    "shelf_life_prediction",
    "contamination_risk",
    "packaging_defect",
]


@dataclass
class TrainingConfig:
    model_type: ModelType = "food_classification"
    backbone: str = "efficientnet_b0"
    num_classes: int = 100
    num_risk_categories: int = 10
    label_feature_dim: int = 64
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    scheduler_type: Literal["cosine", "step", "plateau", "none"] = "cosine"
    step_size: int = 10
    gamma: float = 0.1
    warmup_epochs: int = 3
    early_stopping_patience: int = 10
    early_stopping_delta: float = 1e-4
    mixed_precision: bool = False
    gradient_clip_norm: float | None = 1.0
    log_interval: int = 10
    save_dir: str = "checkpoints"
    experiment_name: str = "foodshield_experiment"
    mlflow_tracking_uri: str | None = None
    seed: int = 42


@dataclass
class AugmentationConfig:
    enabled: bool = True
    horizontal_flip_prob: float = 0.5
    rotation_degrees: float = 15.0
    brightness_jitter: float = 0.2
    contrast_jitter: float = 0.2
    saturation_jitter: float = 0.2
    hue_jitter: float = 0.1
    random_crop_scale: tuple[float, float] = (0.8, 1.0)
    random_affine_degrees: float = 10.0
    random_affine_translate: float = 0.1


DEFAULT_CONFIGS: dict[str, TrainingConfig] = {
    "food_classification": TrainingConfig(
        model_type="food_classification",
        backbone="efficientnet_b0",
        num_classes=100,
        learning_rate=1e-3,
    ),
    "spoilage_detection": TrainingConfig(
        model_type="spoilage_detection",
        backbone="resnet34",
        num_classes=2,
        learning_rate=5e-4,
    ),
    "shelf_life_prediction": TrainingConfig(
        model_type="shelf_life_prediction",
        backbone="efficientnet_b0",
        label_feature_dim=64,
        learning_rate=5e-4,
    ),
    "contamination_risk": TrainingConfig(
        model_type="contamination_risk",
        backbone="resnet50",
        num_risk_categories=10,
        learning_rate=5e-4,
    ),
    "packaging_defect": TrainingConfig(
        model_type="packaging_defect",
        backbone="yolov8",
        learning_rate=1e-4,
    ),
}
