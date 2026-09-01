from __future__ import annotations

import logging
from typing import Callable

import torch

from aiengine.domain.exceptions import ExplanationError
from aiengine.xai.explainer import XAIExplainer
from aiengine.xai.gradcam import GradCAMExplainer

logger = logging.getLogger(__name__)


class ExplanationService:
    """
    Application service that encapsulates all XAI (explainability) logic.

    Supports:

    - **Grad-CAM** heatmap generation against the actual CV model responsible
      for the food-classification prediction (Spec §2, §5).
    - **Genuine SHAP** values via
      :class:`~aiengine.xai.shap_explainer.SHAPExplainer` — optional; returns
      ``None`` when the ``shap`` library is absent or ``shap_explainer`` is
      not injected (Spec §4).
    - **Counterfactual** scenarios via
      :class:`~aiengine.xai.counterfactual.CounterfactualExplainer` — optional;
      returns ``[]`` when not injected (Spec §6).
    - **Natural-language** explanation via
      :class:`~aiengine.xai.explainer.XAIExplainer` with Ollama/template
      fallback.

    LLM report logic is intentionally excluded — see
    :class:`~aiengine.application.report_service.ReportService`.
    """

    def __init__(
        self,
        xai_explainer: XAIExplainer,
        food_classifier,            # FoodClassificationInference — avoid circular import
        shap_explainer=None,        # SHAPExplainer | None
        counterfactual_explainer=None,  # CounterfactualExplainer | None
    ) -> None:
        self._xai_explainer = xai_explainer
        self._food_classifier = food_classifier
        self._shap_explainer = shap_explainer
        self._counterfactual_explainer = counterfactual_explainer
        self._gradcam = GradCAMExplainer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_gradcam(
        self,
        image_tensor: torch.Tensor,
        model: torch.nn.Module,
        target_class: int | None = None,
    ) -> dict:
        """
        Generate a Grad-CAM heatmap for the given model and image tensor.

        The heatmap explains the *actual* model that produced the prediction,
        not a surrogate.

        Args:
            image_tensor:  Pre-processed image tensor (1 × C × H × W).
            model:         The PyTorch model whose activations to visualise.
            target_class:  Optional class index to back-propagate from.
                           Defaults to the argmax of the model output.

        Returns:
            A dict with keys ``heatmap_array``, ``heatmap_overlay``,
            ``target_class``, ``method`` and ``regions_of_interest``.

        Raises:
            ExplanationError: If Grad-CAM computation fails.
        """
        try:
            return self._gradcam.generate_heatmap(model, image_tensor, target_class)
        except Exception as exc:
            raise ExplanationError(
                f"Grad-CAM generation failed: {exc}"
            ) from exc

    def generate_shap(
        self,
        features: dict,
        predict_func: Callable[[list[dict]], list[float]],
    ) -> dict | None:
        """
        Compute genuine SHAP values for tabular/structured features.

        Returns ``None`` when the ``shap_explainer`` was not injected or the
        ``shap`` library is unavailable.

        Args:
            features:      Feature dict (e.g. environmental conditions).
            predict_func:  Callable accepting ``list[dict]`` and returning
                           ``list[float]`` risk scores — must be the *actual*
                           risk model's predict method, not a surrogate.

        Returns:
            SHAP result dict or ``None``.

        Raises:
            ExplanationError: If SHAP raises an unexpected error.
        """
        if self._shap_explainer is None:
            logger.debug("SHAPExplainer not injected — skipping SHAP computation.")
            return None

        try:
            return self._shap_explainer.explain(features)
        except ImportError:
            logger.warning("SHAP library unavailable — skipping SHAP computation.")
            return None
        except Exception as exc:
            raise ExplanationError(
                f"SHAP computation failed: {exc}"
            ) from exc

    def generate_counterfactuals(
        self,
        features: dict,
        predict_func: Callable[[dict], float],
        target_threshold: float = 0.3,
    ) -> list[dict]:
        """
        Generate counterfactual scenarios showing minimal feature changes
        needed to bring the risk score below ``target_threshold``.

        Returns an empty list when ``counterfactual_explainer`` was not
        injected.

        Args:
            features:         Current feature dict.
            predict_func:     Callable accepting a feature dict and returning
                              a risk score (0.0 – 1.0).
            target_threshold: Risk score to optimise towards.

        Returns:
            List of counterfactual dicts (may be empty).

        Raises:
            ExplanationError: If counterfactual generation fails unexpectedly.
        """
        if self._counterfactual_explainer is None:
            logger.debug(
                "CounterfactualExplainer not injected — skipping counterfactual generation."
            )
            return []

        try:
            return self._counterfactual_explainer.generate_counterfactuals(
                features, target_threshold
            )
        except Exception as exc:
            raise ExplanationError(
                f"Counterfactual generation failed: {exc}"
            ) from exc

    def generate_natural_language_explanation(self, prediction: dict) -> str:
        """
        Generate a concise natural-language explanation of the inspection result.

        Delegates to :class:`~aiengine.xai.explainer.XAIExplainer` which uses
        Ollama when available and falls back to a deterministic template.

        Args:
            prediction: Merged CV model result dict.

        Returns:
            A plain-text explanation string.

        Raises:
            ExplanationError: If explanation generation fails completely.
        """
        try:
            return self._xai_explainer.generate_explanation(prediction)
        except Exception as exc:
            raise ExplanationError(
                f"Natural-language explanation generation failed: {exc}"
            ) from exc
