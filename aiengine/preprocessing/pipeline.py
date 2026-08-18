import cv2
import numpy as np
import torch
import torchvision.transforms as T  # noqa: N812
from PIL import Image


class PreprocessingPipeline:
    def __init__(
        self,
        target_size: tuple[int, int] = (224, 224),
        mean: tuple[float, ...] = (0.485, 0.456, 0.406),
        std: tuple[float, ...] = (0.229, 0.224, 0.225),
        device: str = "cpu",
    ) -> None:
        self.target_size = target_size
        self.device = device
        self.transforms = T.Compose(
            [
                T.Resize(target_size),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def process(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transforms(image).unsqueeze(0).to(self.device)

    def process_batch(self, images: list[Image.Image]) -> torch.Tensor:
        return torch.cat([self.process(img) for img in images], dim=0)

    def detect_label_region(self, image: Image.Image) -> Image.Image | None:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        if w * h < 0.01 * image.width * image.height:
            return None
        return image.crop((x, y, x + w, y + h))

    def check_quality(self, image: Image.Image) -> dict:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        return {
            "is_blurry": laplacian_var < 100,
            "blur_score": round(laplacian_var, 2),
            "brightness": round(brightness, 2),
            "is_too_dark": brightness < 40,
            "is_too_bright": brightness > 215,
            "width": image.width,
            "height": image.height,
        }

    def augment(self, image: Image.Image) -> list[Image.Image]:
        augs = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )
        return [augs(image) for _ in range(4)]
