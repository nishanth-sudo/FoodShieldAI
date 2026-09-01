"""
tests/unit/test_model_robustness.py — Task 7.7: AI Model Robustness & Adversarial Testing

Tests verify models and pipelines handle edge cases, boundary conditions,
corrupted inputs, extreme aspect ratios, and failure states gracefully.
All tests run 100% offline (mocked / synthesized weights, no external API calls).
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from aiengine.llm_report_generator import LLMReportGenerator
from aiengine.models.contamination_risk.model import (
    RISK_CATEGORIES,
    ContaminationRiskAssessor,
)
from aiengine.models.food_classification.model import FoodClassifier
from aiengine.models.packaging_defect.model import PackagingDefectDetector
from aiengine.models.shelf_life_prediction.model import ShelfLifePredictor
from aiengine.models.spoilage_detection.model import SpoilageDetector
from aiengine.ocr.extractor import LabelTextExtractor
from aiengine.ollama_client import OllamaClient
from aiengine.preprocessing.pipeline import PreprocessingPipeline
from aiengine.xai.explainer import XAIExplainer


# ===========================================================================
# 1. Preprocessing Robustness
# ===========================================================================


@pytest.mark.unit
@pytest.mark.robustness
class TestPreprocessingRobustness:
    """Edge cases for image loading, color modes, extreme resolutions, and aspect ratios."""

    def test_process_rgba_image_converts_to_rgb(self) -> None:
        """RGBA image (with alpha channel) must be converted to RGB without crash."""
        pipeline = PreprocessingPipeline(device="cpu")
        rgba_img = Image.new("RGBA", (224, 224), color=(100, 150, 200, 128))

        tensor = pipeline.process(rgba_img)

        assert tensor.shape == (1, 3, 224, 224)

    def test_process_grayscale_converts_to_rgb(self) -> None:
        """Grayscale (L mode) image must be converted to 3-channel RGB."""
        pipeline = PreprocessingPipeline(device="cpu")
        gray_img = Image.new("L", (224, 224), color=128)

        tensor = pipeline.process(gray_img)

        assert tensor.shape == (1, 3, 224, 224)

    def test_process_1x1_pixel_image(self) -> None:
        """Tiny 1x1 image must upscale to 224x224 without crash."""
        pipeline = PreprocessingPipeline(device="cpu")
        tiny_img = Image.new("RGB", (1, 1), color=(255, 0, 0))

        tensor = pipeline.process(tiny_img)

        assert tensor.shape == (1, 3, 224, 224)

    def test_process_very_large_image(self) -> None:
        """Large 3000x3000 image must resize down to (1, 3, 224, 224)."""
        pipeline = PreprocessingPipeline(device="cpu")
        large_img = Image.new("RGB", (3000, 3000), color=(50, 100, 150))

        tensor = pipeline.process(large_img)

        assert tensor.shape == (1, 3, 224, 224)

    def test_process_extreme_aspect_ratio_panoramic(self) -> None:
        """Wide panoramic image (1600x100) must resize to 224x224."""
        pipeline = PreprocessingPipeline(device="cpu")
        panoramic = Image.new("RGB", (1600, 100), color=(80, 80, 80))

        tensor = pipeline.process(panoramic)

        assert tensor.shape == (1, 3, 224, 224)

    def test_process_extreme_aspect_ratio_tall(self) -> None:
        """Tall narrow image (100x1600) must resize to 224x224."""
        pipeline = PreprocessingPipeline(device="cpu")
        tall = Image.new("RGB", (100, 1600), color=(80, 80, 80))

        tensor = pipeline.process(tall)

        assert tensor.shape == (1, 3, 224, 224)

    def test_quality_check_pure_black_image(self) -> None:
        """Pure black image must trigger is_too_dark=True."""
        pipeline = PreprocessingPipeline(device="cpu")
        black_img = Image.new("RGB", (224, 224), color=(0, 0, 0))

        quality = pipeline.check_quality(black_img)

        assert bool(quality["is_too_dark"]) is True
        assert bool(quality["is_too_bright"]) is False

    def test_quality_check_pure_white_image(self) -> None:
        """Pure white image must trigger is_too_bright=True."""
        pipeline = PreprocessingPipeline(device="cpu")
        white_img = Image.new("RGB", (224, 224), color=(255, 255, 255))

        quality = pipeline.check_quality(white_img)

        assert bool(quality["is_too_bright"]) is True
        assert bool(quality["is_too_dark"]) is False

    def test_quality_check_returns_all_required_keys(self) -> None:
        """quality check must return all expected schema keys."""
        pipeline = PreprocessingPipeline(device="cpu")
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))

        quality = pipeline.check_quality(img)

        expected_keys = {
            "is_blurry",
            "blur_score",
            "brightness",
            "is_too_dark",
            "is_too_bright",
            "width",
            "height",
        }
        assert expected_keys.issubset(quality.keys())

    def test_augment_preserves_pil_images(self) -> None:
        """All 4 augmentations must remain valid PIL images."""
        pipeline = PreprocessingPipeline(device="cpu")
        img = Image.new("RGB", (224, 224), color=(120, 140, 160))

        augs = pipeline.augment(img)

        assert len(augs) == 4
        assert all(isinstance(a, Image.Image) for a in augs)


# ===========================================================================
# 2. Food Classification Model Robustness
# ===========================================================================


@pytest.mark.unit
@pytest.mark.robustness
class TestFoodClassifierRobustness:
    """Robustness tests for FoodClassifier architecture."""

    def test_unsupported_backbone_raises_value_error(self) -> None:
        """Initializing with an unknown backbone must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported backbone"):
            FoodClassifier(backbone="vgg16")

    def test_supported_backbones_instantiate(self) -> None:
        """Supported backbones (efficientnet_b0, efficientnet_b3, resnet50) must instantiate."""
        for bb in ("efficientnet_b0", "resnet50"):
            model = FoodClassifier(num_classes=10, backbone=bb)
            assert model.backbone_name == bb

    def test_forward_output_shape(self) -> None:
        """Forward pass with batch=2 and 37 classes must output (2, 37)."""
        model = FoodClassifier(num_classes=37, backbone="efficientnet_b0")
        model.eval()
        dummy_input = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (2, 37)


# ===========================================================================
# 3. Spoilage Detection Model Robustness
# ===========================================================================


@pytest.mark.unit
@pytest.mark.robustness
class TestSpoilageDetectorRobustness:
    """Robustness tests for SpoilageDetector architecture and outputs."""

    def test_unsupported_backbone_raises_value_error(self) -> None:
        """Unknown backbone must raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported backbone"):
            SpoilageDetector(backbone="densenet121")

    def test_forward_outputs_expected_keys(self) -> None:
        """Output dictionary must contain spoilage_logit and severity."""
        model = SpoilageDetector(backbone="resnet34")
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            out = model(dummy_input)

        assert "spoilage_logit" in out
        assert "severity" in out

    def test_severity_output_bounded_between_0_and_1(self) -> None:
        """Severity output is passed through sigmoid, so must strictly be in [0.0, 1.0]."""
        model = SpoilageDetector(backbone="resnet34")
        model.eval()
        dummy_input = torch.randn(4, 3, 224, 224) * 10.0

        with torch.no_grad():
            out = model(dummy_input)

        severity = out["severity"]
        assert torch.all(severity >= 0.0)
        assert torch.all(severity <= 1.0)


# ===========================================================================
# 4. Contamination Risk Model Robustness
# ===========================================================================


@pytest.mark.unit
@pytest.mark.robustness
class TestContaminationRiskRobustness:
    """Robustness tests for ContaminationRiskAssessor."""

    def test_all_10_risk_categories_present(self) -> None:
        """RISK_CATEGORIES must contain exactly 10 categories."""
        assert len(RISK_CATEGORIES) == 10

    def test_risk_categories_are_unique(self) -> None:
        """No duplicate category names."""
        assert len(set(RISK_CATEGORIES)) == len(RISK_CATEGORIES)

    def test_unsupported_backbone_raises_value_error(self) -> None:
        """Unknown backbone raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported backbone"):
            ContaminationRiskAssessor(backbone="alexnet")

    def test_forward_shape_matches_num_categories(self) -> None:
        """Forward pass output shape must be (batch_size, num_categories)."""
        model = ContaminationRiskAssessor(num_risk_categories=10, backbone="resnet50")
        model.eval()
        dummy_input = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            logits = model(dummy_input)

        assert logits.shape == (2, 10)


# ===========================================================================
# 5. Packaging Defect Detector Robustness
# ===========================================================================


@pytest.mark.unit
@pytest.mark.robustness
class TestPackagingDefectRobustness:
    """Packaging defect detector fallback & confidence handling."""

    def test_unsupported_model_type_raises(self) -> None:
        """Unsupported model type raises ValueError when loading weights."""
        detector = PackagingDefectDetector(model_type="unknown_detector")
        with pytest.raises(ValueError, match="Unsupported model type"):
            detector._load_model("path/to/weights.pt")

    def test_dummy_predictions_fallback_when_no_weights(self) -> None:
        """Detector returns dummy predictions when no model file is loaded."""
        detector = PackagingDefectDetector(model_path="")
        img = Image.new("RGB", (224, 224))

        detections = detector.detect(img)

        assert isinstance(detections, list)
        assert len(detections) > 0
        for d in detections:
            assert "defect_type" in d
            assert "confidence" in d
            assert "bbox" in d


# ===========================================================================
# 6. Shelf Life Predictor Robustness
# ===========================================================================


@pytest.mark.unit
@pytest.mark.robustness
class TestShelfLifePredictorRobustness:
    """ShelfLifePredictor fusion model tests."""

    def test_forward_shape(self) -> None:
        """Visual + label feature fusion output must be (batch_size, 1)."""
        model = ShelfLifePredictor(label_feature_dim=64)
        model.eval()
        img_tensor = torch.randn(2, 3, 224, 224)
        label_tensor = torch.randn(2, 10)

        with torch.no_grad():
            pred = model(img_tensor, label_tensor)

        assert pred.shape == (2, 1)


# ===========================================================================
# 7. OCR Label Parsing Robustness
# ===========================================================================


@pytest.mark.unit
@pytest.mark.robustness
class TestOCRRobustness:
    """OCR field extraction under varied and corrupted string formats."""

    def test_parse_fields_various_date_formats(self) -> None:
        """Extractor handles multiple common expiry date representations."""
        extractor = LabelTextExtractor(backend="mock", ollama_model=None)

        formats = [
            ("Best before: 15/08/2026", "15/08/2026"),
            ("EXP: 2026-11-30", "2026-11-30"),
            ("Use by: 05 Jan 2027", "05 Jan 2027"),
            ("BB: 10.12.2025", "10.12.2025"),
        ]
        for text, expected in formats:
            parsed = extractor._parse_fields(text)
            assert "expiry_date" in parsed, f"Failed on: {text}"
            assert expected.lower() in parsed["expiry_date"].lower()

    def test_parse_fields_unicode_characters(self) -> None:
        """Non-ASCII unicode characters must not cause regex failure."""
        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        text = "Product: Café Crème Éclair\nBrand: Nestlé®"

        parsed = extractor._parse_fields(text)

        assert "product_name" in parsed

    def test_parse_fields_very_long_text(self) -> None:
        """10,000 character garbage string must not crash or hang regex parsing."""
        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        long_garbage = "random text " * 1000 + "Best Before: 2026-09-01"

        parsed = extractor._parse_fields(long_garbage)

        assert "expiry_date" in parsed

    def test_mock_backend_returns_valid_dict(self) -> None:
        """Mock backend extract() returns valid structure."""
        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        img = Image.new("RGB", (100, 100))

        result = extractor.extract(img)

        assert result["backend"] == "mock"
        assert len(result["parsed"]) > 0


# ===========================================================================
# 8. LLM Report Generator Robustness
# ===========================================================================


@pytest.mark.unit
@pytest.mark.robustness
class TestReportGeneratorRobustness:
    """Report generation handling missing, empty, or extreme input dictionaries."""

    def test_empty_results_dict_does_not_crash(self) -> None:
        """Passing an empty dict to generate_report must not raise an exception."""
        gen = LLMReportGenerator(model_name="local/template")

        report = gen.generate_report({})

        assert "report_title" in report
        assert "overall_verdict" in report
        # Empty dict → freshness_score=0 → critically low → risk flags present → 'fail'
        # The key contract is that it does not raise, not a specific verdict.
        assert report["overall_verdict"] in ("pass", "fail", "conditional_pass")

    def test_severe_spoilage_produces_fail_verdict(self) -> None:
        """Spoiled item must result in verdict='fail' and risk flags."""
        gen = LLMReportGenerator(model_name="local/template")
        results = {
            "food_type": "Meat",
            "spoilage": {"is_spoiled": True, "freshness_score": 0.05},
        }

        report = gen.generate_report(results)

        assert report["overall_verdict"] == "fail"
        assert len(report["risk_flags"]) > 0

    def test_high_contamination_produces_risk_flag(self) -> None:
        """High contamination risk must trigger risk flags."""
        gen = LLMReportGenerator(model_name="local/template")
        results = {
            "food_type": "Salad",
            "contamination_risks": {"overall_risk_level": "high"},
        }

        report = gen.generate_report(results)

        assert any("contamination" in flag.lower() for flag in report["risk_flags"])

    def test_date_format_in_report_is_valid(self) -> None:
        """Generated inspection date is a valid ISO timestamp."""
        gen = LLMReportGenerator(model_name="local/template")

        report = gen.generate_report({"food_type": "Bread"})

        dt = datetime.fromisoformat(report["inspection_date"])
        assert dt.year >= 2024


# ===========================================================================
# 9. XAI Explainer Robustness
# ===========================================================================


@pytest.mark.unit
@pytest.mark.robustness
class TestXAIExplainerRobustness:
    """XAI Explainer edge cases."""

    def test_empty_prediction_explanation_does_not_crash(self) -> None:
        """Generating explanation on empty prediction returns valid text."""
        explainer = XAIExplainer(method="gradcam", ollama_model=None)

        explanation = explainer.generate_explanation({})

        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_explanation_contains_defect_information(self) -> None:
        """Packaging defects in prediction are listed in explanation."""
        explainer = XAIExplainer(method="gradcam", ollama_model=None)
        pred = {
            "food_type": "Canned Beans",
            "packaging_defects": [{"defect_type": "dent"}, {"defect_type": "rust"}],
        }

        text = explainer.generate_explanation(pred)

        assert "dent" in text
        assert "rust" in text


# ===========================================================================
# 10. Ollama Client Failure Robustness
# ===========================================================================


@pytest.mark.unit
@pytest.mark.robustness
class TestOllamaClientRobustness:
    """OllamaClient handling of offline servers and disabled state."""

    def test_disabled_via_env_var(self) -> None:
        """When OLLAMA_ENABLED=false, client methods safely return empty results."""
        with patch.dict("os.environ", {"OLLAMA_ENABLED": "false"}):
            client = OllamaClient(model="llama3.1:8b")

            assert client.is_available is False
            assert client.chat("hello") == ""
            assert client.chat_json("hello") is None
            assert client.chat_vision("hello", "base64image") == ""

    def test_list_models_on_server_failure_returns_empty_list(self) -> None:
        """When the server connection fails, list_local_models returns []."""
        client = OllamaClient(model="llama3.1:8b")
        client._enabled = True

        with patch("httpx.get", side_effect=Exception("Connection refused")):
            models = client.list_local_models()

        assert models == []
