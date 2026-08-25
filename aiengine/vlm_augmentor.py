"""
Vision Language Model (VLM) augmentor using Ollama.

This module adds a second-opinion layer on top of the existing PyTorch CV models.
A locally-running vision-capable Ollama model (e.g. llava:13b, minicpm-v:8b)
analyses the food image and cross-validates the CNN predictions.

The augmentor does NOT replace the CV models — it enriches their output by:
  - Confirming or challenging the food classification
  - Providing a natural-language visual description
  - Flagging visible spoilage / contamination signs not caught by CV models
  - Returning a confidence adjustment factor

Supported Ollama vision models:
  - llava:13b      (best, needs ~10 GB VRAM)
  - llava:7b       (balanced)
  - llava-phi3:3.8b (lightweight)
  - minicpm-v:8b   (excellent food understanding)
  - moondream2     (ultra-light, basic QA)
"""

from __future__ import annotations

import base64
import io
import logging

from PIL import Image

from aiengine.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------

_FOOD_ANALYSIS_PROMPT = """Analyse this food image and answer the following questions.
Respond ONLY with a JSON object — no extra text.

Questions:
1. What food item is in the image? (be specific)
2. Does it look fresh, moderately fresh, or spoiled?
3. Are there any visible signs of mold, discolouration, or contamination?
4. Are there any visible packaging defects (tears, dents, leaks, bulges)?
5. On a scale of 0.0 to 1.0, what is the estimated freshness score?

Required JSON format:
{{
  "food_type": "<identified food>",
  "freshness_assessment": "fresh | moderately_fresh | spoiled",
  "freshness_score_estimate": <float 0.0-1.0>,
  "visible_spoilage_signs": ["<sign1>", "<sign2>"],
  "visible_contamination": ["<item1>"],
  "visible_packaging_defects": ["<defect1>"],
  "visual_description": "<1-2 sentence description>",
  "confidence": <float 0.0-1.0>
}}"""

_CROSS_VALIDATE_PROMPT = """A CNN model predicted the following about a food image:
- Food type: {food_type} (confidence: {confidence:.2f})
- Freshness score: {freshness_score:.2f}/1.0
- Spoiled: {is_spoiled}

Look at the image and tell me if you AGREE or DISAGREE with these predictions.
Respond ONLY with a JSON object:
{{
  "agree_food_type": true | false,
  "agree_freshness": true | false,
  "suggested_food_type": "<your suggestion or same as above>",
  "suggested_freshness_score": <float 0.0-1.0>,
  "reasoning": "<brief explanation>",
  "confidence": <float 0.0-1.0>
}}"""


class VisionLLMAugmentor:
    """
    Augments CV model predictions with a Ollama vision LLM second opinion.

    Args:
        model: Ollama vision model name (default: ``llava:7b``).
        base_url: Ollama server URL (default: ``http://localhost:11434/v1``).
        enabled: Set False to completely skip VLM augmentation.

    Example::

        augmentor = VisionLLMAugmentor(model="llava:13b")
        if augmentor.is_available:
            vlm_result = augmentor.analyse_image(pil_image)
            cross = augmentor.cross_validate(pil_image, cv_result)
    """

    def __init__(
        self,
        model: str = "llava:7b",
        base_url: str | None = None,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.enabled = enabled
        self._client: OllamaClient | None = None

        if enabled:
            self._client = OllamaClient(
                model=model,
                temperature=0.2,
                max_tokens=1000,
                base_url=base_url,
            )
            if not self._client.is_available:
                logger.info(
                    "VisionLLMAugmentor: Ollama server not reachable "
                    "or model '%s' not pulled. VLM augmentation disabled. "
                    "Pull with: ollama pull %s",
                    model,
                    model,
                )
                self._client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Return True when a reachable Ollama server with a vision model is ready."""
        return self._client is not None and self._client.is_available

    def analyse_image(self, image: Image.Image) -> dict:
        """
        Run a standalone visual analysis of the food image.

        Returns a dict with food type, freshness estimate, visible defects, etc.
        Returns an empty dict when VLM augmentation is unavailable.
        """
        if not self.is_available:
            return {}

        b64 = self._encode_image(image)
        result = self._client.chat_json(_FOOD_ANALYSIS_PROMPT, system=None)  # type: ignore[union-attr]
        # vision requests need different path — use chat_vision
        raw = self._client.chat_vision(_FOOD_ANALYSIS_PROMPT, b64)  # type: ignore[union-attr]
        if not raw:
            return {}
        try:
            import json

            start = raw.index("{")
            end = raw.rindex("}") + 1
            result = json.loads(raw[start:end])
            result["vlm_model"] = self.model
            return result
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("VLM analyse_image JSON parse failed: %s", exc)
            return {}

    def cross_validate(self, image: Image.Image, cv_prediction: dict) -> dict:
        """
        Cross-validate CV model predictions against VLM visual inspection.

        Args:
            image:         The food image.
            cv_prediction: Dict from CNN models containing ``food_type``,
                           ``freshness_score``, ``is_spoiled``.

        Returns:
            Cross-validation result dict with agreement flags and suggestions.
            Returns empty dict when VLM is unavailable.
        """
        if not self.is_available:
            return {}

        b64 = self._encode_image(image)
        spoilage = cv_prediction.get("spoilage", {})
        prompt = _CROSS_VALIDATE_PROMPT.format(
            food_type=cv_prediction.get("food_type", "Unknown"),
            confidence=cv_prediction.get("confidence_scores", [{"confidence": 0}])[0].get(
                "confidence", 0
            )
            if cv_prediction.get("confidence_scores")
            else 0,
            freshness_score=spoilage.get(
                "freshness_score", cv_prediction.get("freshness_score", 0)
            ),
            is_spoiled=spoilage.get("is_spoiled", False),
        )

        raw = self._client.chat_vision(prompt, b64)  # type: ignore[union-attr]
        if not raw:
            return {}
        try:
            import json

            start = raw.index("{")
            end = raw.rindex("}") + 1
            result = json.loads(raw[start:end])
            result["vlm_model"] = self.model
            return result
        except (ValueError, json.JSONDecodeError) as exc:
            logger.warning("VLM cross_validate JSON parse failed: %s", exc)
            return {}

    def enrich_results(self, cv_results: dict, image: Image.Image) -> dict:
        """
        Convenience method — runs both analyse + cross_validate and merges
        into ``cv_results`` under the key ``vlm_augmentation``.

        The original CV results are never overwritten; VLM data is additive.
        """
        if not self.is_available:
            cv_results["vlm_augmentation"] = {"enabled": False, "reason": "VLM not available"}
            return cv_results

        vlm_analysis = self.analyse_image(image)
        vlm_cross = self.cross_validate(image, cv_results)

        cv_results["vlm_augmentation"] = {
            "enabled": True,
            "model": self.model,
            "visual_analysis": vlm_analysis,
            "cross_validation": vlm_cross,
        }
        return cv_results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image: Image.Image, max_size: tuple[int, int] = (640, 640)) -> str:
        """Resize and base64-encode a PIL image for API submission."""
        img = image.copy()
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        if img.mode != "RGB":
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
