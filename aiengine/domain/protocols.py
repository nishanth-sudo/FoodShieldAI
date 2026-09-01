from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class RiskModel(Protocol):
    """
    Protocol for any model used in XAI risk prediction.

    Both :class:`~aiengine.xai.shap_explainer.SHAPExplainer` and
    :class:`~aiengine.xai.counterfactual.CounterfactualExplainer` must
    receive a callable that satisfies this protocol so that XAI always
    explains the *actual* production model, never a surrogate equation.
    """

    def predict(self, features: dict) -> float:
        """Predict risk score (0.0 – 1.0) from a feature dict."""
        ...

    def predict_batch(self, features_list: list[dict]) -> list[float]:
        """Batch prediction for SHAP/counterfactual explainers."""
        ...
