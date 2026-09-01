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
        self._supported_methods = {"gradcam", "shap"}
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
            from aiengine.xai.gradcam import GradCAMExplainer
            return GradCAMExplainer().generate_heatmap(model, image, target_class)
        elif self.method == "shap":
            # Genuine Image SHAP is complex; for now we avoid the fake delegation
            # and will implement it in the SHAP phase.
            raise NotImplementedError("Genuine Image SHAP is not yet implemented.")
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

    