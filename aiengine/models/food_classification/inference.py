import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from aiengine.models.food_classification.model import FoodClassifier
from aiengine.preprocessing.pipeline import PreprocessingPipeline

FOOD_CATEGORIES = [
    "apple",
    "banana",
    "beef",
    "bread",
    "butter",
    "cabbage",
    "carrot",
    "cheese",
    "chicken",
    "chocolate",
    "corn",
    "cucumber",
    "dairy",
    "egg",
    "fish",
    "flour",
    "garlic",
    "grape",
    "green_bean",
    "lettuce",
    "milk",
    "mushroom",
    "onion",
    "orange",
    "pasta",
    "pepper",
    "pork",
    "potato",
    "rice",
    "salad",
    "seafood",
    "spinach",
    "strawberry",
    "tomato",
    "water",
    "yogurt",
    "zucchini",
]


class FoodClassificationInference:
    def __init__(
        self,
        model_path: str,
        num_classes: int = 100,
        backbone: str = "efficientnet_b0",
        device: str = "cpu",
        class_names: list[str] | None = None,
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = FoodClassifier(num_classes=num_classes, backbone=backbone)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        self.model.to(self.device)
        self.model.eval()
        self.preprocessor = PreprocessingPipeline(device=self.device)
        self.class_names = class_names or FOOD_CATEGORIES

    def predict(self, image: Image.Image, top_k: int = 5) -> dict:
        tensor = self.preprocessor.process(image)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)

        top_probs, top_indices = torch.topk(probs, k=min(top_k, probs.size(1)))
        top_probs = top_probs.squeeze(0).tolist()
        top_indices = top_indices.squeeze(0).tolist()

        predictions = [
            {
                "food_type": self.class_names[idx]
                if idx < len(self.class_names)
                else f"class_{idx}",
                "confidence": round(prob, 4),
                "class_id": idx,
            }
            for idx, prob in zip(top_indices, top_probs, strict=True)
        ]

        return {
            "food_type": predictions[0]["food_type"],
            "confidence_scores": predictions,
            "all_confidences": probs.squeeze(0).tolist(),
        }
