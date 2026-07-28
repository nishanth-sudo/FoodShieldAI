import torch
import torch.nn as nn
from torchvision import models


class ShelfLifePredictor(nn.Module):
    def __init__(self, label_feature_dim: int = 64):
        super().__init__()
        self.visual_backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        visual_dim = self.visual_backbone.classifier[1].in_features
        self.visual_backbone.classifier = nn.Identity()

        self.label_encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, label_feature_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + label_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, label_features: torch.Tensor) -> torch.Tensor:
        visual_feats = self.visual_backbone(image)
        encoded_label = self.label_encoder(label_features)
        fused = self.fusion(torch.cat([visual_feats, encoded_label], dim=1))
        return self.regressor(fused)


class ShelfLifeInference:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = ShelfLifePredictor()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint)
        self.model.to(self.device)
        self.model.eval()

        from ai_engine.preprocessing.pipeline import PreprocessingPipeline
        self.preprocessor = PreprocessingPipeline(device=self.device)

    def predict(self, image, expiry_date: str | None = None) -> dict:
        from PIL import Image
        import datetime

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image).convert("RGB")

        tensor = self.preprocessor.process(image)

        label_feats = self._encode_label_features(expiry_date)
        label_tensor = label_feats.unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred_days = self.model(tensor, label_tensor).item()

        estimated_days = max(0, round(pred_days))
        today = datetime.date.today()
        estimated_expiry = (today + datetime.timedelta(days=estimated_days)).isoformat()

        if estimated_days <= 0:
            freshness = "expired"
        elif estimated_days <= 2:
            freshness = "critical"
        elif estimated_days <= 7:
            freshness = "short"
        elif estimated_days <= 30:
            freshness = "moderate"
        else:
            freshness = "long"

        return {
            "estimated_days_remaining": estimated_days,
            "estimated_expiry_date": estimated_expiry,
            "freshness_category": freshness,
            "confidence": 0.85,
        }

    def _encode_label_features(self, expiry_date: str | None = None) -> torch.Tensor:
        import datetime

        feats = torch.zeros(10)
        if expiry_date:
            try:
                expiry = datetime.date.fromisoformat(expiry_date)
                today = datetime.date.today()
                days_until_expiry = (expiry - today).days
                feats[0] = min(days_until_expiry / 365.0, 1.0)
                feats[1] = 1.0 if days_until_expiry < 0 else 0.0
            except (ValueError, TypeError):
                pass
        feats[2] = 0.5
        return feats
