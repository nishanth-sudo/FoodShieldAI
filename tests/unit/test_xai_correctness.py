"""
XAI Correctness Tests — Spec §46

Tests for:
- GradCAMExplainer: output shape, value range, target-class handling, fallback
- SHAPExplainer: genuine shap library, additivity, required keys
- CounterfactualExplainer: risk reduction, constraint satisfaction, keys
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest
import torch
import torch.nn as nn

# Check shap availability at collection time (used for class-level skip).
_shap_available = importlib.util.find_spec("shap") is not None


# ---------------------------------------------------------------------------
# Helpers — tiny models
# ---------------------------------------------------------------------------

def _make_conv_model(in_channels: int = 3, num_classes: int = 5) -> nn.Module:
    """A minimal model with Conv2d so Grad-CAM has a target layer."""
    return nn.Sequential(
        nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, num_classes),
    )


def _make_linear_model(in_features: int = 8, num_classes: int = 5) -> nn.Module:
    """A model with NO Conv2d layers — triggers Grad-CAM fallback."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 16),
        nn.ReLU(),
        nn.Linear(16, num_classes),
    )


def _dummy_image_tensor(h: int = 32, w: int = 32) -> torch.Tensor:
    """A (1, 3, H, W) random image tensor."""
    return torch.rand(1, 3, h, w)


# ---------------------------------------------------------------------------
# GradCAM tests
# ---------------------------------------------------------------------------

class TestGradCAMExplainer:
    def setup_method(self) -> None:
        from aiengine.xai.gradcam import GradCAMExplainer
        self.explainer = GradCAMExplainer()

    def test_output_keys_present(self) -> None:
        model = _make_conv_model()
        tensor = _dummy_image_tensor()
        result = self.explainer.generate_heatmap(model, tensor)
        assert "heatmap_array" in result
        assert "heatmap_overlay" in result
        assert "target_class" in result
        assert "method" in result
        assert "regions_of_interest" in result

    def test_heatmap_values_in_valid_range(self) -> None:
        model = _make_conv_model()
        tensor = _dummy_image_tensor()
        result = self.explainer.generate_heatmap(model, tensor)
        arr = np.array(result["heatmap_array"])
        assert arr.min() >= 0, "Heatmap values must be >= 0"
        assert arr.max() <= 255, "Heatmap values must be <= 255"

    def test_target_class_defaults_to_argmax(self) -> None:
        model = _make_conv_model(num_classes=5)
        tensor = _dummy_image_tensor()
        with torch.no_grad():
            logits = model(tensor)
        expected_class = int(logits.argmax(dim=1).item())

        result = self.explainer.generate_heatmap(model, tensor, target_class=None)
        assert result["target_class"] == expected_class

    def test_explicit_target_class_is_respected(self) -> None:
        model = _make_conv_model(num_classes=5)
        tensor = _dummy_image_tensor()
        result = self.explainer.generate_heatmap(model, tensor, target_class=2)
        assert result["target_class"] == 2

    def test_fallback_heatmap_when_no_conv_layers(self) -> None:
        """Linear-only model has no Conv2d → should return fallback heatmap."""
        model = _make_linear_model()
        tensor = _dummy_image_tensor()
        result = self.explainer.generate_heatmap(model, tensor)
        assert result["method"] == "fallback"

    def test_fallback_heatmap_values_in_valid_range(self) -> None:
        model = _make_linear_model()
        tensor = _dummy_image_tensor()
        result = self.explainer.generate_heatmap(model, tensor)
        arr = np.array(result["heatmap_array"])
        assert arr.min() >= 0
        assert arr.max() <= 255

    def test_cpu_model_execution(self) -> None:
        model = _make_conv_model().cpu()
        tensor = _dummy_image_tensor().cpu()
        result = self.explainer.generate_heatmap(model, tensor)
        assert result["target_class"] is not None

    def test_regions_of_interest_is_list(self) -> None:
        model = _make_conv_model()
        tensor = _dummy_image_tensor()
        result = self.explainer.generate_heatmap(model, tensor)
        assert isinstance(result["regions_of_interest"], list)

    def test_roi_entries_have_required_keys(self) -> None:
        model = _make_conv_model()
        # Use a larger tensor to ensure some ROI regions are detected
        tensor = _dummy_image_tensor(64, 64)
        result = self.explainer.generate_heatmap(model, tensor)
        for roi in result["regions_of_interest"]:
            assert "x" in roi
            assert "y" in roi
            assert "width" in roi
            assert "height" in roi
            assert "importance" in roi

    def test_method_is_gradcam_for_conv_model(self) -> None:
        model = _make_conv_model()
        tensor = _dummy_image_tensor()
        result = self.explainer.generate_heatmap(model, tensor)
        assert result["method"] == "gradcam"


# ---------------------------------------------------------------------------
# SHAP tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _shap_available, reason="shap library not installed")
class TestSHAPExplainer:
    FEATURE_NAMES = ["temperature", "humidity", "storage_duration"]

    def _make_predict_func(self, coeffs: list[float] | None = None):
        """Simple linear risk function for deterministic SHAP testing."""
        if coeffs is None:
            coeffs = [0.5, 0.3, 0.2]

        def predict(samples: list[dict]) -> list[float]:
            results = []
            for sample in samples:
                score = sum(
                    sample.get(f, 0.0) * c
                    for f, c in zip(self.FEATURE_NAMES, coeffs)
                )
                results.append(float(min(max(score, 0.0), 1.0)))
            return results

        return predict

    def _make_explainer(self, predict_func=None):
        from aiengine.xai.shap_explainer import SHAPExplainer
        if predict_func is None:
            predict_func = self._make_predict_func()
        return SHAPExplainer(
            feature_names=self.FEATURE_NAMES,
            predict_func=predict_func,
        )

    def test_genuine_shap_library_is_used(self) -> None:
        """Verify the explainer actually loaded the shap library."""
        explainer = self._make_explainer()
        assert explainer._shap_lib is not None
        import shap as shap_lib
        assert explainer._shap_lib is shap_lib

    def test_output_has_required_keys(self) -> None:
        explainer = self._make_explainer()
        features = {"temperature": 0.5, "humidity": 0.6, "storage_duration": 0.3}
        result = explainer.explain(features)
        assert "shap_values" in result
        assert "base_value" in result
        assert "feature_importance" in result
        assert "is_additive" in result
        assert "prediction" in result

    def test_additivity_property(self) -> None:
        """base_value + sum(shap_values) ≈ model_prediction (tolerance 0.05)."""
        explainer = self._make_explainer()
        features = {"temperature": 0.4, "humidity": 0.5, "storage_duration": 0.2}
        result = explainer.explain(features)
        total = result["base_value"] + sum(result["shap_values"].values())
        assert abs(total - result["prediction"]) < 0.05, (
            f"Additivity failed: base + sum(shap) = {total:.4f}, "
            f"prediction = {result['prediction']:.4f}"
        )

    def test_is_additive_flag_is_bool(self) -> None:
        explainer = self._make_explainer()
        features = {"temperature": 0.3, "humidity": 0.4, "storage_duration": 0.1}
        result = explainer.explain(features)
        assert isinstance(result["is_additive"], bool)

    def test_feature_importance_sorted_by_magnitude(self) -> None:
        explainer = self._make_explainer()
        features = {"temperature": 0.6, "humidity": 0.7, "storage_duration": 0.4}
        result = explainer.explain(features)
        importances = [abs(v) for _, v in result["feature_importance"]]
        assert importances == sorted(importances, reverse=True), (
            "feature_importance should be sorted by |shap_value| descending"
        )

    def test_handles_missing_shap_library(self) -> None:
        from aiengine.xai.shap_explainer import SHAPExplainer
        explainer = SHAPExplainer(
            feature_names=self.FEATURE_NAMES,
            predict_func=self._make_predict_func(),
        )
        # Simulate missing shap library
        explainer._shap_lib = None
        with pytest.raises(ImportError):
            explainer.explain({"temperature": 0.5, "humidity": 0.5, "storage_duration": 0.2})

    def test_shap_values_dict_has_all_features(self) -> None:
        explainer = self._make_explainer()
        features = {"temperature": 0.5, "humidity": 0.6, "storage_duration": 0.3}
        result = explainer.explain(features)
        for feat in self.FEATURE_NAMES:
            assert feat in result["shap_values"]


# ---------------------------------------------------------------------------
# Counterfactual tests
# ---------------------------------------------------------------------------

class TestCounterfactualExplainer:
    CONSTRAINTS = {
        "temperature": (0.0, 1.0),
        "humidity": (0.0, 1.0),
        "storage_duration": (0.0, 1.0),
    }

    def _make_high_risk_features(self) -> dict:
        return {"temperature": 0.9, "humidity": 0.9, "storage_duration": 0.9}

    def _make_predict_func(self):
        """Linear risk function: higher feature values → higher risk."""
        def predict(features: dict) -> float:
            return min(
                (features.get("temperature", 0) * 0.4
                 + features.get("humidity", 0) * 0.3
                 + features.get("storage_duration", 0) * 0.3),
                1.0,
            )
        return predict

    def _make_explainer(self, predict_func=None):
        from aiengine.xai.counterfactual import CounterfactualExplainer
        if predict_func is None:
            predict_func = self._make_predict_func()
        return CounterfactualExplainer(
            predict_func=predict_func,
            feature_constraints=self.CONSTRAINTS,
        )

    def test_counterfactuals_reduce_predicted_risk(self) -> None:
        explainer = self._make_explainer()
        features = self._make_high_risk_features()
        predict_func = self._make_predict_func()
        initial_risk = predict_func(features)
        results = explainer.generate_counterfactuals(features, target_threshold=0.3)
        assert len(results) > 0, "Expected at least one counterfactual"
        for cf in results:
            assert cf["predicted_risk"] < initial_risk

    def test_counterfactual_output_keys(self) -> None:
        explainer = self._make_explainer()
        features = self._make_high_risk_features()
        results = explainer.generate_counterfactuals(features, target_threshold=0.3)
        assert len(results) > 0
        for cf in results:
            assert "changes" in cf
            assert "predicted_risk" in cf
            assert "risk_reduction" in cf

    def test_no_counterfactuals_when_risk_below_threshold(self) -> None:
        """If initial risk is already below threshold, return empty list."""
        explainer = self._make_explainer()
        low_risk_features = {"temperature": 0.1, "humidity": 0.1, "storage_duration": 0.1}
        results = explainer.generate_counterfactuals(
            low_risk_features, target_threshold=0.9
        )
        assert results == []

    def test_predict_func_is_called(self) -> None:
        """Counterfactuals must use the injected predict_func, not a hardcoded equation."""
        call_count = [0]

        def counting_predict(features: dict) -> float:
            call_count[0] += 1
            return (
                features.get("temperature", 0) * 0.5
                + features.get("humidity", 0) * 0.3
                + features.get("storage_duration", 0) * 0.2
            )

        from aiengine.xai.counterfactual import CounterfactualExplainer
        explainer = CounterfactualExplainer(
            predict_func=counting_predict,
            feature_constraints=self.CONSTRAINTS,
        )
        features = self._make_high_risk_features()
        explainer.generate_counterfactuals(features, target_threshold=0.3)
        assert call_count[0] > 1, "predict_func must be called during search"

    def test_feature_constraints_are_respected(self) -> None:
        """Modified feature values must stay within [lo, hi] bounds."""
        explainer = self._make_explainer()
        features = self._make_high_risk_features()
        results = explainer.generate_counterfactuals(features, target_threshold=0.3)
        for cf in results:
            for feat, change in cf["changes"].items():
                lo, hi = self.CONSTRAINTS[feat]
                assert lo <= change["to"] <= hi, (
                    f"Feature '{feat}' value {change['to']} violates constraint [{lo}, {hi}]"
                )

    def test_risk_reduction_is_positive(self) -> None:
        explainer = self._make_explainer()
        features = self._make_high_risk_features()
        results = explainer.generate_counterfactuals(features, target_threshold=0.3)
        for cf in results:
            assert cf["risk_reduction"] > 0

    def test_n_counterfactuals_limit(self) -> None:
        explainer = self._make_explainer()
        features = self._make_high_risk_features()
        results = explainer.generate_counterfactuals(
            features, target_threshold=0.3, n_counterfactuals=2
        )
        assert len(results) <= 2
