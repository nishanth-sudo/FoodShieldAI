import numpy as np
import torch


# ---------------------------------------------------------------------------
# Ollama-powered explanation prompt
# ---------------------------------------------------------------------------

_EXPLANATION_PROMPT = """You are a food safety AI assistant. Based on the following inspection data,
generate a clear and concise 2-3 sentence explanation of the food safety findings.
Focus on what was found, why it matters, and what action should be taken.
Do NOT include JSON — respond with plain text only.

Inspection data:
- Food type: {food_type}
- Freshness score: {freshness_score}/1.0 ({freshness_label})
- Spoiled: {is_spoiled}
- Packaging defects: {defects}
- Contamination risks: {risks}
- Estimated shelf life: {shelf_life} days
- XAI method: {xai_method} (model focused on {focus_region})
"""

_EXPLANATION_SYSTEM = (
    "You are a concise food safety expert. Write plain text explanations. "
    "Do not use markdown, bullet points, or JSON."
)


class XAIExplainer:
    """
    Explainability module for food inspection CV models.

    Supports:
    - Grad-CAM heatmap generation (``method="gradcam"``)
    - LIME explanation (``method="lime"``, falls back to Grad-CAM)
    - SHAP explanation (``method="shap"``, falls back to Grad-CAM)

    Natural-language explanations:
        When Ollama is available, ``generate_explanation()`` produces a
        contextually-aware, human-readable summary using a local LLM.
        Falls back to deterministic template text when Ollama is unavailable.

    Args:
        method:        XAI heatmap method (``"gradcam"``, ``"lime"``, ``"shap"``).
        ollama_model:  Ollama text model for explanation generation
                       (default: ``"llama3.2:3b"``). Pass ``None`` to disable.
    """

    def __init__(
        self,
        method: str = "gradcam",
        ollama_model: str | None = "llama3.2:3b",
    ) -> None:
        self.method = method
        self._supported_methods = {"gradcam", "lime", "shap"}
        self._ollama_client = None

        if method not in self._supported_methods:
            raise ValueError(
                f"Unsupported XAI method: {method}. Choose from {self._supported_methods}"
            )

        if ollama_model:
            self._init_ollama(ollama_model)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        """
        Generate a natural-language explanation of the inspection results.

        Uses Ollama when available; falls back to template text otherwise.
        """
        if self._ollama_client is not None:
            explanation = self._ollama_explanation(prediction)
            if explanation:
                return explanation

        # Deterministic template fallback
        return self._template_explanation(prediction)

    # ------------------------------------------------------------------
    # Ollama explanation
    # ------------------------------------------------------------------

    def _init_ollama(self, model: str) -> None:
        try:
            from aiengine.ollama_client import OllamaClient
            import logging

            client = OllamaClient(model=model, temperature=0.4, max_tokens=300)
            if client.is_available:
                self._ollama_client = client
            else:
                logging.getLogger(__name__).info(
                    "XAIExplainer: Ollama not available — using template explanations. "
                    "Pull model with: ollama pull %s",
                    model,
                )
        except Exception:
            pass

    def _ollama_explanation(self, prediction: dict) -> str:
        """Generate a natural-language explanation via Ollama."""
        spoilage = prediction.get("spoilage", prediction)
        freshness_score = spoilage.get("freshness_score", prediction.get("freshness_score", 0))
        is_spoiled = spoilage.get("is_spoiled", False)
        shelf = prediction.get("shelf_life", {})
        contamination = prediction.get("contamination_risks", {})

        if freshness_score >= 0.6:
            freshness_label = "fresh"
        elif freshness_score >= 0.3:
            freshness_label = "moderately fresh"
        else:
            freshness_label = "spoiled"

        defects = prediction.get("packaging_defects", [])
        defect_str = (
            ", ".join(d["defect_type"] for d in defects) if isinstance(defects, list) and defects
            else "none"
        )

        risk_str = "none"
        if isinstance(contamination, dict) and contamination.get("detected_risks"):
            risk_str = ", ".join(
                r["category"] for r in contamination["detected_risks"]
            )

        shelf_life = "N/A"
        if isinstance(shelf, dict):
            shelf_life = str(shelf.get("estimated_days_remaining", "N/A"))

        prompt = _EXPLANATION_PROMPT.format(
            food_type=prediction.get("food_type", "Unknown"),
            freshness_score=f"{freshness_score:.2f}",
            freshness_label=freshness_label,
            is_spoiled=is_spoiled,
            defects=defect_str,
            risks=risk_str,
            shelf_life=shelf_life,
            xai_method=self.method,
            focus_region="texture and colour variations",
        )

        return self._ollama_client.chat(prompt, system=_EXPLANATION_SYSTEM)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Template explanation (no LLM)
    # ------------------------------------------------------------------

    @staticmethod
    def _template_explanation(prediction: dict) -> str:
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

    # ------------------------------------------------------------------
    # Heatmap methods (unchanged)
    # ------------------------------------------------------------------

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
    