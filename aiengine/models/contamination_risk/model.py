import torch
import torch.nn as nn
from torchvision import models


RISK_CATEGORIES = [
    "biological_mold", "biological_bacteria", "biological_yeast",
    "chemical_pesticide", "chemical_cleaning_agent", "chemical_preservative",
    "physical_glass", "physical_metal", "physical_plastic",
    "physical_stone",
]


class ContaminationRiskAssessor(nn.Module):
    def __init__(self, num_risk_categories: int = 10, backbone: str = "resnet50"):
        super().__init__()
        self.num_categories = num_risk_categories

        if backbone == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == "efficientnet_b0":
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_risk_categories),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class ContaminationRiskInference:
    def __init__(
        self,
        model_path: str,
        backbone: str = "resnet50",
        device: str = "cpu",
        threshold: float = 0.3,
        risk_categories: list[str] | None = None,
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.threshold = threshold
        self.risk_categories = risk_categories or RISK_CATEGORIES
        self.model = ContaminationRiskAssessor(
            num_risk_categories=len(self.risk_categories),
            backbone=backbone,
        )
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
        self.model.to(self.device)
        self.model.eval()

        from ai_engine.preprocessing.pipeline import PreprocessingPipeline
        self.preprocessor = PreprocessingPipeline(device=self.device)

    def predict(self, image) -> dict:
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")
        tensor = self.preprocessor.process(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze(0)

        risk_scores = {}
        detected_risks = []
        max_score = 0.0
        overall_risk = "low"

        for i, category in enumerate(self.risk_categories):
            score = round(probs[i].item(), 4)
            risk_scores[category] = score
            if score > self.threshold:
                detected_risks.append({
                    "category": category,
                    "confidence": score,
                    "severity": "high" if score > 0.7 else "medium" if score > 0.5 else "low",
                })
            max_score = max(max_score, score)

        if max_score > 0.7:
            overall_risk = "high"
        elif max_score > 0.4:
            overall_risk = "medium"

        return {
            "risk_scores": risk_scores,
            "detected_risks": detected_risks,
            "overall_risk_level": overall_risk,
            "max_risk_score": round(max_score, 4),
        }
