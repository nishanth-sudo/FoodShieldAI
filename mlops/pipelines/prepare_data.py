import json
import random
from pathlib import Path


def prepare_freshness44(
    data_dir: str = "mlops/datasets/Freshness44",
    output_dir: str = "mlops/datasets/prepared/Freshness44",
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    image_size: tuple[int, int] = (224, 224),
) -> dict:
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    classes: list[str] = []
    image_paths: list[str] = []
    labels: list[int] = []
    class_to_idx: dict[str, int] = {}

    class_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    for cls_dir in class_dirs:
        class_name = cls_dir.name
        if class_name not in class_to_idx:
            class_to_idx[class_name] = len(classes)
            classes.append(class_name)

    for cls_dir in class_dirs:
        cls_idx = class_to_idx[cls_dir.name]
        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                image_paths.append(str(img_path))
                labels.append(cls_idx)

    combined = list(zip(image_paths, labels, strict=False))
    random.seed(seed)
    random.shuffle(combined)
    image_paths[:], labels[:] = zip(*combined, strict=False) if combined else ([], [])

    n = len(image_paths)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    splits = {
        "train": (image_paths[:n_train], labels[:n_train]),
        "val": (image_paths[n_train : n_train + n_val], labels[n_train : n_train + n_val]),
        "test": (image_paths[n_train + n_val :], labels[n_train + n_val :]),
    }

    metadata = {
        "dataset": "Freshness44",
        "num_classes": len(classes),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "splits": {k: len(v[0]) for k, v in splits.items()},
        "image_size": list(image_size),
    }

    for split_name, (paths, lbls) in splits.items():
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for src_path, label in zip(paths, lbls, strict=False):
            records.append({"image_path": src_path, "label": label, "class_name": classes[label]})
        with open(split_dir / "metadata.json", "w") as f:
            json.dump({"records": records, "num_samples": len(records)}, f, indent=2)

    (output_path / "dataset.json").write_text(json.dumps(metadata, indent=2))

    print(f"Freshness44 prepared: {n_train} train, {n_val} val, {n_test} test")
    print(f"Classes ({len(classes)}): {classes}")
    return metadata


def prepare_food101(
    data_dir: str = "mlops/datasets/FoodImages",
    output_dir: str = "mlops/datasets/prepared/Food101",
    image_size: tuple[int, int] = (224, 224),
) -> dict:
    import json as _json

    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    meta_dir = data_path / "meta" / "meta"
    classes_path = meta_dir / "classes.txt"
    labels_path = meta_dir / "labels.txt"
    train_json_path = meta_dir / "train.json"
    test_json_path = meta_dir / "test.json"

    classes = [
        line.strip().split(maxsplit=1)[1] for line in classes_path.read_text().strip().splitlines()
    ]
    human_labels = [
        line.strip().split(maxsplit=1)[1] for line in labels_path.read_text().strip().splitlines()
    ]

    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    train_data = _json.loads(train_json_path.read_text())
    test_data = _json.loads(test_json_path.read_text())

    image_dir = data_path / "images"

    for split_name, split_data in [("train", train_data), ("test", test_data)]:
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for cls_name, img_paths in split_data.items():
            cls_idx = class_to_idx[cls_name]
            for rel_path in img_paths:
                full_path = str(image_dir / rel_path)
                records.append({"image_path": full_path, "label": cls_idx, "class_name": cls_name})
        with open(split_dir / "metadata.json", "w") as f:
            json.dump({"records": records, "num_samples": len(records)}, f, indent=2)

    metadata = {
        "dataset": "Food101",
        "num_classes": len(classes),
        "classes": classes,
        "human_labels": human_labels,
        "class_to_idx": class_to_idx,
        "splits": {"train": len(train_data), "test": len(test_data)},
        "image_size": list(image_size),
    }
    (output_path / "dataset.json").write_text(json.dumps(metadata, indent=2))

    train_count = sum(len(v) for v in train_data.values())
    test_count = sum(len(v) for v in test_data.values())
    print(f"Food101 prepared: {train_count} train, {test_count} test")
    print(f"Classes: {len(classes)}")
    return metadata


if __name__ == "__main__":
    prepare_freshness44()
