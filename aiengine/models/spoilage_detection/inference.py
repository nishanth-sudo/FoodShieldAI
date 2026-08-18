import torch
from PIL import Image

from aiengine.models.spoilage_detection.model import SpoilageDetector
from aiengine.preprocessing.pipeline import PreprocessingPipeline


class SpoilageDetectionInference:
    def __init__(self, model_path: str, backbone: str = "resnet34", device: str = "cpu") -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = SpoilageDetector(backbone=backbone)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        self.model.to(self.device)
        self.model.eval()
        self.preprocessor = PreprocessingPipeline(device=self.device)

    def predict(self, image: Image.Image) -> dict:
        tensor = self.preprocessor.process(image)
        with torch.no_grad():
            output = self.model(tensor)

        spoilage_prob = torch.sigmoid(output["spoilage_logit"]).item()
        severity = output["severity"].item()

        is_spoiled = spoilage_prob > 0.5

        if severity < 0.3:
            severity_label = "low"
        elif severity < 0.6:
            severity_label = "medium"
        else:
            severity_label = "high"

        return {
            "is_spoiled": is_spoiled,
            "spoilage_score": round(spoilage_prob, 4),
            "freshness_score": round(1.0 - spoilage_prob, 4),
            "severity": round(severity, 4),
            "severity_label": severity_label,
        }
