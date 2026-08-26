from __future__ import annotations

import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

class CounterfactualExplainer:
    """
    Generates counterfactual explanations for tabular data models.
    Helps answer "What minimal changes need to be made to achieve a different outcome?"
    """
    def __init__(self, predict_func: Callable[[dict[str, Any]], float], feature_constraints: dict[str, tuple[float, float]]):
        self.predict_func = predict_func
        self.feature_constraints = feature_constraints

    def generate_counterfactuals(
        self, current_features: dict[str, Any], target_threshold: float, n_counterfactuals: int = 3
    ) -> list[dict[str, Any]]:
        """
        Generate counterfactual scenarios to achieve a risk below the target_threshold.
        Uses greedy optimization to find minimal feature changes.
        """
        initial_risk = self.predict_func(current_features)
        
        if initial_risk <= target_threshold:
            return []
            
        counterfactuals = []
        features_to_try = [f for f in current_features if f in self.feature_constraints]
        
        # 1-feature changes
        for feature in features_to_try:
            min_val, max_val = self.feature_constraints[feature]
            current_val = current_features[feature]
            
            steps = 10
            best_risk = initial_risk
            best_val = current_val
            
            for i in range(steps + 1):
                test_val = min_val + (max_val - min_val) * (i / steps)
                if abs(test_val - current_val) < 1e-6:
                    continue
                    
                test_features = current_features.copy()
                test_features[feature] = test_val
                
                risk = self.predict_func(test_features)
                if risk < best_risk:
                    best_risk = risk
                    best_val = test_val
                    
            if best_risk < initial_risk:
                test_features = current_features.copy()
                test_features[feature] = best_val
                
                changes = {
                    feature: {
                        "from": current_val,
                        "to": best_val,
                        "delta": best_val - current_val
                    }
                }
                
                exp = f"If {feature} → {best_val:.1f}, Predicted risk → {best_risk*100:.0f}%"
                
                counterfactuals.append({
                    "modified_features": test_features,
                    "changes": changes,
                    "predicted_risk": best_risk,
                    "risk_reduction": initial_risk - best_risk,
                    "explanation": exp
                })
                
        # 2-feature changes (simple approach)
        if len(counterfactuals) < n_counterfactuals and len(features_to_try) >= 2:
            for i in range(len(features_to_try)):
                for j in range(i + 1, len(features_to_try)):
                    f1 = features_to_try[i]
                    f2 = features_to_try[j]
                    
                    test_features = current_features.copy()
                    
                    # Heuristically push both features to their midpoint between current and best
                    val1 = (current_features[f1] + sum(self.feature_constraints[f1])) / 3
                    val2 = (current_features[f2] + sum(self.feature_constraints[f2])) / 3
                    
                    test_features[f1] = val1
                    test_features[f2] = val2
                    
                    risk = self.predict_func(test_features)
                    if risk < initial_risk:
                        changes = {
                            f1: {"from": current_features[f1], "to": val1, "delta": val1 - current_features[f1]},
                            f2: {"from": current_features[f2], "to": val2, "delta": val2 - current_features[f2]},
                        }
                        exp = f"If {f1} → {val1:.1f} and {f2} → {val2:.1f}, Predicted risk → {risk*100:.0f}%"
                        counterfactuals.append({
                            "modified_features": test_features,
                            "changes": changes,
                            "predicted_risk": risk,
                            "risk_reduction": initial_risk - risk,
                            "explanation": exp
                        })

        # Sort and return
        counterfactuals.sort(key=lambda x: x["risk_reduction"], reverse=True)
        return counterfactuals[:n_counterfactuals]
