import random
from pathlib import Path

import torch
from ai_engine.training.config import AugmentationConfig
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class FoodDataset(Dataset):
    def __init__(
        self,
        image_paths: list[str],
        targets: list,
        transform: T.Compose | None = None,
        augment: bool = False,
        aug_config: AugmentationConfig | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.targets = targets
        self.aug_config = aug_config or AugmentationConfig()

        base = self._make_base_transform()
        if augment:
            aug_list = [
                T.RandomHorizontalFlip(p=self.aug_config.horizontal_flip_prob),
                T.RandomRotation(degrees=self.aug_config.rotation_degrees),
                T.ColorJitter(
                    brightness=self.aug_config.brightness_jitter,
                    contrast=self.aug_config.contrast_jitter,
                    saturation=self.aug_config.saturation_jitter,
                    hue=self.aug_config.hue_jitter,
                ),
                T.RandomAffine(
                    degrees=self.aug_config.random_affine_degrees,
                    translate=(
                        self.aug_config.random_affine_translate,
                        self.aug_config.random_affine_translate,
                    ),
                ),
                T.RandomResizedCrop(
                    size=(224, 224),
                    scale=self.aug_config.random_crop_scale,
                ),
            ]
            self.transform = T.Compose([T.RandomOrder(aug_list), base])
        else:
            self.transform = base if transform is None else transform

    def _make_base_transform(self) -> T.Compose:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        tensor = self.transform(image)
        target = self.targets[idx]
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.long)
        return tensor, target


class ClassificationDataset(FoodDataset):
    pass


class SpoilageDataset(Dataset):
    def __init__(
        self,
        image_paths: list[str],
        spoilage_labels: list[int],
        severity_scores: list[float],
        transform: T.Compose | None = None,
        augment: bool = False,
        aug_config: AugmentationConfig | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.spoilage_labels = spoilage_labels
        self.severity_scores = severity_scores
        self.aug_config = aug_config or AugmentationConfig()

        base = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        if augment:
            aug_list = [
                T.RandomHorizontalFlip(p=self.aug_config.horizontal_flip_prob),
                T.RandomRotation(degrees=self.aug_config.rotation_degrees),
                T.ColorJitter(
                    brightness=self.aug_config.brightness_jitter,
                    contrast=self.aug_config.contrast_jitter,
                    saturation=self.aug_config.saturation_jitter,
                ),
            ]
            self.transform = T.Compose([T.RandomOrder(aug_list), base])
        else:
            self.transform = base if transform is None else transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        tensor = self.transform(image)
        return tensor, {
            "spoilage_label": torch.tensor(self.spoilage_labels[idx], dtype=torch.float32),
            "severity": torch.tensor(self.severity_scores[idx], dtype=torch.float32),
        }


class ShelfLifeDataset(Dataset):
    def __init__(
        self,
        image_paths: list[str],
        days_remaining: list[float],
        label_features: list[list[float]] | None = None,
        transform: T.Compose | None = None,
        augment: bool = False,
    ) -> None:
        self.image_paths = image_paths
        self.days_remaining = days_remaining
        self.label_features = label_features or [[0.5] * 10 for _ in image_paths]

        base = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.transform = base if transform is None else transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        tensor = self.transform(image)
        days = torch.tensor(self.days_remaining[idx], dtype=torch.float32).unsqueeze(0)
        label_feats = torch.tensor(self.label_features[idx], dtype=torch.float32)
        return tensor, label_feats, days


class ContaminationDataset(FoodDataset):
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        tensor = self.transform(image)
        target = self.targets[idx]
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.float32)
        return tensor, target


def create_datasets(
    image_dir: str,
    model_type: str,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    augment: bool = True,
) -> dict[str, Dataset]:
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    image_paths = sorted([
        str(p) for p in Path(image_dir).rglob("*")
        if p.suffix.lower() in image_extensions
    ])

    random.seed(seed)
    random.shuffle(image_paths)

    n = len(image_paths)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    train_paths = image_paths[:n_train]
    val_paths = image_paths[n_train:n_train + n_val]
    test_paths = image_paths[n_train + n_val:]

    return {
        "train_paths": train_paths,
        "val_paths": val_paths,
        "test_paths": test_paths,
    }
