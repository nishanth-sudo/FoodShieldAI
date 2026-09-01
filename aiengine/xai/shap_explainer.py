from __future__ import annotations

import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """
    Genuine SHAP explainer for tabular/structured data models.
    Verified by the additivity relationship: base_value + sum(shap_values) ≈ model_prediction.
    """
    def __init__(self, feature_names: list[str], predict_func: Callable[[list[dict[str, Any]]], list[float]]):
        self.feature_names = feature_names
        self.predict_func = predict_func
        self._shap_lib = None

        try:
            import shap
            self._shap_lib = shap
            logger.info("SHAP library loaded successfully.")
        except ImportError:
            logger.warning("SHAP library not found. Genuine SHAP values cannot be computed.")

    def explain(self, features_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Compute SHAP values for the given features.
        """
        if self._shap_lib is None:
            raise ImportError("The 'shap' library is required for genuine SHAP explanations.")

        # Convert features_dict to a format SHAP expects (e.g., numpy array)
        # This requires a consistent feature order
        input_data = np.array([[features_dict.get(f, 0.0) for f in self.feature_names]])

        # Define a wrapper for the predict_func that takes numpy arrays
        def model_wrapper(X):
            # Convert numpy array back to list of dicts for the original predict_func
            data_list = []
            for row in X:
                data_list.append({f: val for f, val in zip(self.feature_names, row)})
            return np.array(self.predict_func(data_list))

        # Use KernelExplainer as a general-purpose explainer for the predict_func
        # We use a small background dataset for the baseline (e.g., zeros)
        background = np.zeros((1, len(self.feature_names)))
        explainer = self._shap_lib.KernelExplainer(model_wrapper, background)

        shap_values = explainer.shap_values(input_data, nsamples=100)[0]
        base_value = explainer.expected_value

        # Verify additivity: base_value + sum(shap_values) ≈ model_prediction
        actual_prediction = model_wrapper(input_data)[0]
        sum_shap = base_value + np.sum(shap_values)

        tolerance = 1e-3
        is_additive = abs(actual_prediction - sum_shap) < tolerance

        if not is_additive:
            logger.warning(
                f"SHAP additivity check failed: {sum_shap:.4f} vs {actual_prediction:.4f}"
            )

        # Create a mapping of feature name to SHAP value
        shap_map = {f: float(v) for f, v in zip(self.feature_names, shap_values)}

        feature_importance = sorted(
            [(f, float(v)) for f, v in shap_map.items()],
            key=lambda x: abs(x[1]),
            reverse=True
        )

        top_factors = []
        for feat, val in feature_importance[:4]:
            if val == 0: continue
            sign = "+" if val >= 0 else "-"
            top_factors.append(f"{feat.capitalize()} {sign}{abs(val):.2f}")

        return {
            "shap_values": shap_map,
            "base_value": float(base_value),
            "feature_importance": feature_importance,
            "top_factors": top_factors,
            "is_additive": bool(is_additive),
            "prediction": float(actual_prediction)
        }
