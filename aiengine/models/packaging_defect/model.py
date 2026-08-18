from PIL import Image

DEFECT_CLASSES = [
    "dent",
    "tear",
    "leak",
    "seal_failure",
    "crack",
    "bulge",
    "mislabel",
    "dirt_contamination",
]


class PackagingDefectDetector:
    def __init__(
        self, model_type: str = "yolov8", model_path: str = "", confidence_threshold: float = 0.25
    ) -> None:
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.defect_classes = DEFECT_CLASSES
        self.model = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, model_path: str) -> None:
        if self.model_type == "yolov8":
            try:
                from ultralytics import YOLO

                self.model = YOLO(model_path)
            except ImportError:
                raise ImportError(
                    "ultralytics is required for YOLOv8. Install with: pip install ultralytics"
                ) from None
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def detect(self, image: Image.Image) -> list[dict]:
        if self.model is None:
            return self._dummy_predictions(image)
        results = self.model(image, conf=self.confidence_threshold)
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    {
                        "defect_type": self.defect_classes[cls_id]
                        if cls_id < len(self.defect_classes)
                        else f"class_{cls_id}",
                        "confidence": round(confidence, 4),
                        "bbox": [round(c, 1) for c in [x1, y1, x2, y2]],
                        "class_id": cls_id,
                    }
                )
        return detections

    def _dummy_predictions(self, image: Image.Image) -> list[dict]:
        return [
            {
                "defect_type": "seal_failure",
                "confidence": 0.87,
                "bbox": [50.0, 60.0, 200.0, 150.0],
                "class_id": 3,
            },
            {
                "defect_type": "dent",
                "confidence": 0.72,
                "bbox": [180.0, 30.0, 260.0, 100.0],
                "class_id": 0,
            },
        ]
