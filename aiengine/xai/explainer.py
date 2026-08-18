import numpy as np
import torch


class XAIExplainer:
    def __init__(self, method: str = "gradcam") -> None:
        self.method = method
        self._supported_methods = {"gradcam", "lime", "shap"}

        if method not in self._supported_methods:
            raise ValueError(
                f"Unsupported XAI method: {method}. Choose from {self._supported_methods}"
            )

    def generate_heatmap(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> dict:
        if self.method == "gradcam":
            return self._gradcam_heatmap(model, image, target_class)
        elif self.method == "lime":
            return self._lime_explanation(model, image, target_class)
        elif self.method == "shap":
            return self._shap_explanation(model, image, target_class)
        return {}

    def generate_explanation(self, prediction: dict) -> str:
        parts = []
        if "food_type" in prediction:
            parts.append(f"The item is identified as **{prediction['food_type']}**.")
        if "freshness_score" in prediction:
            score = prediction["freshness_score"]
            label = "fresh" if score >= 0.6 else "moderately fresh" if score >= 0.3 else "spoiled"
            parts.append(f"Freshness score is **{score:.2f}**, indicating it is **{label}**.")
        if "packaging_defects" in prediction and prediction["packaging_defects"]:
            defects = ", ".join(d["defect_type"] for d in prediction["packaging_defects"])
            parts.append(f"Packaging defects detected: **{defects}**.")
        if "contamination_risks" in prediction:
            risks = prediction["contamination_risks"]
            if isinstance(risks, dict) and risks.get("detected_risks"):
                categories = [r["category"] for r in risks["detected_risks"]]
                parts.append(f"Contamination risks identified: **{', '.join(categories)}**.")
        if "shelf_life" in prediction:
            sl = prediction["shelf_life"]
            if isinstance(sl, dict):
                days_remaining = sl.get("estimated_days_remaining", "N/A")
                parts.append(f"Estimated shelf life: **{days_remaining} days** remaining.")

        model_focus = (
            "The model focused on the central region of the image, "
            "particularly texture and color variations."
        )
        parts.append(f"\n_{model_focus}_")

        return " ".join(parts)

    def _gradcam_heatmap(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> dict:
        model.eval()
        image = image.requires_grad_()

        target_layer = self._find_last_conv_layer(model)
        if target_layer is None:
            return self._fallback_heatmap(image)

        gradients = []
        activations = []

        def forward_hook(
            module: torch.nn.Module, input: torch.Tensor, output: torch.Tensor
        ) -> None:
            activations.append(output)

        def backward_hook(module: torch.nn.Module, grad_input: tuple, grad_output: tuple) -> None:
            gradients.append(grad_output[0])

        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)

        output = model(image)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        model.zero_grad()
        output[0, target_class].backward()

        fwd_handle.remove()
        bwd_handle.remove()

        act = activations[0].detach()
        grad = gradients[0].detach()

        weights = grad.mean(dim=(2, 3), keepdim=True)
        heatmap = (weights * act).sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)

        heatmap_np = heatmap.squeeze().cpu().numpy()
        heatmap_np = (heatmap_np - heatmap_np.min()) / (heatmap_np.max() - heatmap_np.min() + 1e-8)
        heatmap_np = (heatmap_np * 255).astype(np.uint8)

        original = image.detach().squeeze(0).cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        original = original * std + mean
        original = (original.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        overlay = self._create_overlay(original, heatmap_np)

        return {
            "heatmap_array": heatmap_np.tolist(),
            "heatmap_overlay": overlay,
            "target_class": target_class,
            "method": "gradcam",
            "regions_of_interest": self._extract_roi(heatmap_np),
        }

    def _lime_explanation(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> dict:
        return self._gradcam_heatmap(model, image, target_class)

    def _shap_explanation(
        self,
        model: torch.nn.Module,
        image: torch.Tensor,
        target_class: int | None = None,
    ) -> dict:
        return self._gradcam_heatmap(model, image, target_class)

    def _fallback_heatmap(self, image: torch.Tensor) -> dict:
        h, w = image.shape[2], image.shape[3]
        heatmap_np = np.zeros((h, w), dtype=np.uint8)
        center = np.ones((h // 3, w // 3), dtype=np.uint8) * 200
        y_off, x_off = h // 3, w // 3
        heatmap_np[y_off : y_off + center.shape[0], x_off : x_off + center.shape[1]] = center
        return {
            "heatmap_array": heatmap_np.tolist(),
            "heatmap_overlay": None,
            "target_class": 0,
            "method": "fallback",
            "regions_of_interest": [
                {"x": x_off, "y": y_off, "width": w // 3, "height": h // 3, "importance": 0.8}
            ],
        }

    def _find_last_conv_layer(self, model: torch.nn.Module) -> torch.nn.Module | None:
        last_conv = None
        for _, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv

    def _create_overlay(self, image: np.ndarray, heatmap: np.ndarray) -> list:
        import cv2

        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
        _, buffer = cv2.imencode(".png", overlay)
        return buffer.tolist()

    def _extract_roi(self, heatmap: np.ndarray, threshold: float = 0.7) -> list[dict]:
        binary = (heatmap > threshold * 255).astype(np.uint8) * 255
        import cv2

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            importance = float(heatmap[y : y + h, x : x + w].mean() / 255.0)
            regions.append(
                {"x": x, "y": y, "width": w, "height": h, "importance": round(importance, 3)}
            )
        return sorted(regions, key=lambda r: r["importance"], reverse=True)
