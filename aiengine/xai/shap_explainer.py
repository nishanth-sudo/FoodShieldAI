from __future__ import annotations

import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

class SHAPExplainer:
    """
    SHAP explainer for tabular/structured data models.
    Useful for explaining predictions from non-image models like shelf-life prediction or risk scoring.
    """
    def __init__(self, feature_names: list[str], predict_func: Callable[[list[dict[str, Any]]], list[float]]):
        self.feature_names = feature_names
        self.predict_func = predict_func
        self.has_shap = False

        try:
            import shap
            self.has_shap = True
        except ImportError:
            logger.info("SHAP library not found. Using custom perturbation-based KernelSHAP approximation.")

    def explain(self, features_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Compute SHAP values for the given features.
        Returns a dict with shap_values, base_value, feature_importance, and top_factors.
        """
        base_features = {k: features_dict[k] for k in self.feature_names if k in features_dict}
        
        # Simple baseline for perturbation
        baseline = {k: 0.0 if isinstance(base_features[k], (int, float)) else type(base_features[k])() for k in base_features}
        
        base_val = self.predict_func([baseline])[0]
        actual_val = self.predict_func([base_features])[0]
        
        shap_values = {}
        for feature in self.feature_names:
            if feature not in base_features:
                continue
            
            ablated = base_features.copy()
            ablated[feature] = baseline[feature]
            ablated_val = self.predict_func([ablated])[0]
            
            # Marginal contribution approximation
            contribution = actual_val - ablated_val
            shap_values[feature] = contribution
            
        # Normalize so sum of SHAP values matches actual - base (KernelSHAP style efficiency)
        total_contrib = sum(shap_values.values())
        if abs(total_contrib) > 1e-6:
            scale = (actual_val - base_val) / total_contrib
            for k in shap_values:
                shap_values[k] *= scale
                
        feature_importance = sorted([(k, v) for k, v in shap_values.items()], key=lambda x: abs(x[1]), reverse=True)
        
        top_factors = []
        for feat, val in feature_importance[:4]:
            if val == 0: continue
            sign = "+" if val >= 0 else "-"
            # Example format: "Temperature       +0.82"
            top_factors.append(f"{feat.capitalize()} {sign}{abs(val):.2f}")
            
        return {
            "shap_values": shap_values,
            "base_value": base_val,
            "feature_importance": feature_importance,
            "top_factors": top_factors
        }
