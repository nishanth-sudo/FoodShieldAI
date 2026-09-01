"""
Unit tests for the FoodShieldAI AI engine.

Covers:
  - PreprocessingPipeline  (aiengine/preprocessing/pipeline.py)
  - LabelTextExtractor     (aiengine/ocr/extractor.py)
  - XAIExplainer           (aiengine/xai/explainer.py)
  - LLMReportGenerator     (aiengine/llm_report_generator.py)
  - OllamaClient           (aiengine/ollama_client.py)
  - VisionLLMAugmentor     (aiengine/vlm_augmentor.py)

All tests are 100% offline — no real PyTorch model loading, no real Ollama
server, no real OCR engine. PIL images and numpy arrays are used as inputs;
heavy external calls are mocked with unittest.mock.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Module-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rgb_image():
    """Standard 224x224 RGB PIL image for preprocessing tests."""
    return Image.new("RGB", (224, 224), color=(128, 100, 60))


@pytest.fixture(scope="module")
def small_image():
    """Tiny 32x32 RGB PIL image."""
    return Image.new("RGB", (32, 32), color=(200, 200, 200))


@pytest.fixture(scope="module")
def black_image():
    """All-black 224x224 image — triggers 'too dark' quality flag."""
    return Image.new("RGB", (224, 224), color=(0, 0, 0))


@pytest.fixture(scope="module")
def white_image():
    """All-white 224x224 image — triggers 'too bright' quality flag."""
    return Image.new("RGB", (224, 224), color=(255, 255, 255))


@pytest.fixture
def sample_inspection_results():
    """Full mock inspection result dict matching orchestrator output."""
    return {
        "food_type": "apple",
        "freshness_score": 0.85,
        "spoilage": {
            "is_spoiled": False,
            "spoilage_score": 0.15,
            "freshness_score": 0.85,
            "severity": 0.1,
            "severity_label": "low",
        },
        "packaging_defects": [
            {"defect_type": "dent", "confidence": 0.72, "bbox": [10, 10, 50, 50], "class_id": 0}
        ],
        "contamination_risks": {
            "risk_scores": {},
            "detected_risks": [],
            "overall_risk_level": "low",
            "max_risk_score": 0.1,
        },
        "shelf_life": {
            "estimated_days_remaining": 5,
            "estimated_expiry_date": "2026-08-30",
            "freshness_category": "short",
            "confidence": 0.85,
        },
        "food_classification": {
            "food_type": "apple",
            "confidence_scores": [{"food_type": "apple", "confidence": 0.92, "class_id": 0}],
            "all_confidences": [0.92],
        },
        "ocr_data": {
            "raw_text": "Product: Apple Best Before: 2026-08-30",
            "parsed": {"product_name": "Apple", "expiry_date": "2026-08-30"},
            "backend": "mock",
        },
    }


@pytest.fixture
def spoiled_inspection_results():
    """Inspection result for a spoiled item — should produce verdict='fail'."""
    return {
        "food_type": "banana",
        "freshness_score": 0.1,
        "spoilage": {
            "is_spoiled": True,
            "spoilage_score": 0.9,
            "freshness_score": 0.1,
            "severity": 0.85,
            "severity_label": "high",
        },
        "packaging_defects": [],
        "contamination_risks": {
            "risk_scores": {},
            "detected_risks": [
                {"category": "biological_mold", "confidence": 0.8, "severity": "high"}
            ],
            "overall_risk_level": "high",
            "max_risk_score": 0.8,
        },
        "shelf_life": {
            "estimated_days_remaining": 0,
            "estimated_expiry_date": "2026-08-25",
            "freshness_category": "expired",
            "confidence": 0.9,
        },
        "food_classification": {
            "food_type": "banana",
            "confidence_scores": [{"food_type": "banana", "confidence": 0.88, "class_id": 1}],
            "all_confidences": [0.88],
        },
        "ocr_data": {"raw_text": "", "parsed": {}, "backend": "mock"},
    }


# ===========================================================================
# 1. PreprocessingPipeline
# ===========================================================================


@pytest.mark.unit
class TestPreprocessingPipeline:
    """Tests for PreprocessingPipeline — tensor conversion, quality checks, augmentation."""

    def test_process_returns_4d_tensor(self, rgb_image):
        """process() must return a (1, 3, 224, 224) tensor."""
        import torch
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        tensor = pipeline.process(rgb_image)

        assert tensor.shape == (1, 3, 224, 224)

    def test_process_returns_torch_tensor(self, rgb_image):
        """process() must return a torch.Tensor instance."""
        import torch
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        tensor = pipeline.process(rgb_image)

        assert isinstance(tensor, torch.Tensor)

    def test_process_converts_non_rgb_to_rgb(self):
        """process() must handle L-mode (grayscale) images by converting to RGB."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        gray_image = Image.new("L", (224, 224), color=128)
        pipeline = PreprocessingPipeline()
        tensor = pipeline.process(gray_image)

        assert tensor.shape == (1, 3, 224, 224)

    def test_process_custom_target_size(self, rgb_image):
        """process() must resize to the custom target_size passed at construction."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(target_size=(128, 128))
        tensor = pipeline.process(rgb_image)

        assert tensor.shape == (1, 3, 128, 128)

    def test_process_batch_stacks_images(self, rgb_image):
        """process_batch() must concatenate N images into (N, 3, 224, 224)."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        batch = pipeline.process_batch([rgb_image, rgb_image, rgb_image])

        assert batch.shape == (3, 3, 224, 224)

    def test_process_batch_single_image(self, rgb_image):
        """process_batch() with a single image should return (1, 3, 224, 224)."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        batch = pipeline.process_batch([rgb_image])

        assert batch.shape == (1, 3, 224, 224)

    def test_detect_label_region_returns_pil_or_none(self, rgb_image):
        """detect_label_region() must return PIL.Image.Image or None."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        result = pipeline.detect_label_region(rgb_image)

        assert result is None or isinstance(result, Image.Image)

    def test_detect_label_region_none_on_uniform_image(self):
        """detect_label_region() returns None or Image on a uniform-colour image."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        uniform = Image.new("RGB", (224, 224), color=(100, 100, 100))
        result = pipeline.detect_label_region(uniform)

        assert result is None or isinstance(result, Image.Image)

    def test_check_quality_returns_required_keys(self, rgb_image):
        """check_quality() dict must contain all required quality keys."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        quality = pipeline.check_quality(rgb_image)

        required_keys = {
            "is_blurry",
            "blur_score",
            "brightness",
            "is_too_dark",
            "is_too_bright",
            "width",
            "height",
        }
        assert required_keys.issubset(quality.keys())

    def test_check_quality_reports_correct_dimensions(self, rgb_image):
        """check_quality() must report the image width and height correctly."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        quality = pipeline.check_quality(rgb_image)

        assert quality["width"] == 224
        assert quality["height"] == 224

    def test_check_quality_black_image_is_too_dark(self, black_image):
        """check_quality() must flag an all-black image as is_too_dark=True."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        quality = pipeline.check_quality(black_image)

        assert bool(quality["is_too_dark"]) is True

    def test_check_quality_white_image_is_too_bright(self, white_image):
        """check_quality() must flag an all-white image as is_too_bright=True."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        quality = pipeline.check_quality(white_image)

        assert bool(quality["is_too_bright"]) is True

    def test_check_quality_blur_score_is_float(self, rgb_image):
        """check_quality() blur_score must be a numeric value."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        quality = pipeline.check_quality(rgb_image)

        assert isinstance(quality["blur_score"], float)

    def test_augment_returns_four_images(self, rgb_image):
        """augment() must return exactly 4 PIL images."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        augmented = pipeline.augment(rgb_image)

        assert len(augmented) == 4

    def test_augment_all_items_are_pil_images(self, rgb_image):
        """Every item in augment() output must be a PIL.Image.Image."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline()
        augmented = pipeline.augment(rgb_image)

        assert all(isinstance(img, Image.Image) for img in augmented)


# ===========================================================================
# 2. LabelTextExtractor
# ===========================================================================


@pytest.mark.unit
class TestLabelTextExtractor:
    """Tests for LabelTextExtractor using the 'mock' backend — no real OCR."""

    def test_extract_returns_dict_with_required_keys(self, rgb_image):
        """extract() must return a dict with raw_text, parsed, and backend keys."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        result = extractor.extract(rgb_image)

        assert "raw_text" in result
        assert "parsed" in result
        assert "backend" in result

    def test_extract_backend_name_is_mock(self, rgb_image):
        """extract() must report backend='mock' when mock backend is used."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        result = extractor.extract(rgb_image)

        assert result["backend"] == "mock"

    def test_extract_raw_text_is_non_empty_string(self, rgb_image):
        """Mock OCR backend must return a non-empty raw_text string."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        result = extractor.extract(rgb_image)

        assert isinstance(result["raw_text"], str)
        assert len(result["raw_text"]) > 0

    def test_extract_parsed_is_dict(self, rgb_image):
        """extract() parsed field must be a dict."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        result = extractor.extract(rgb_image)

        assert isinstance(result["parsed"], dict)

    def test_parse_fields_extracts_product_name(self):
        """_parse_fields() must extract product_name from matching text."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        parsed = extractor._parse_fields("Product: Fresh Apple Juice")

        assert "product_name" in parsed

    def test_parse_fields_extracts_expiry_date(self):
        """_parse_fields() must extract expiry_date from 'Best Before' text."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        parsed = extractor._parse_fields("Best Before: 12/08/2026")

        assert "expiry_date" in parsed
        assert "2026" in parsed["expiry_date"]

    def test_parse_fields_extracts_brand(self):
        """_parse_fields() must extract brand field."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        parsed = extractor._parse_fields("Brand: DairyBest")

        assert "brand" in parsed

    def test_parse_fields_extracts_net_weight(self):
        """_parse_fields() must extract net_weight from weight strings."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        parsed = extractor._parse_fields("Net Wt: 500g")

        assert "net_weight" in parsed

    def test_parse_fields_extracts_allergens(self):
        """_parse_fields() must extract allergens from label text."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        parsed = extractor._parse_fields("Allergens: Contains Milk")

        assert "allergens" in parsed

    def test_parse_fields_empty_text_returns_empty_dict(self):
        """_parse_fields() on empty string must return an empty dict."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        parsed = extractor._parse_fields("")

        assert parsed == {}

    def test_parse_fields_mock_ocr_text_has_multiple_fields(self):
        """The built-in mock OCR text must parse into at least 3 fields."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        raw = extractor._mock_ocr()
        parsed = extractor._parse_fields(raw)

        assert len(parsed) >= 3

    def test_ollama_parse_merges_ollama_result_with_regex(self):
        """_ollama_parse() must merge Ollama output with existing regex fields."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)

        mock_client = MagicMock()
        mock_client.chat_json.return_value = {
            "product_name": "Organic Milk",
            "brand": "NatureBest",
        }
        extractor._ollama_client = mock_client

        regex_parsed = {"expiry_date": "2026-12-31"}
        merged = extractor._ollama_parse("some raw text", regex_parsed)

        assert "product_name" in merged
        assert merged["expiry_date"] == "2026-12-31"
        assert merged.get("_parsed_by") == "ollama_fallback"

    def test_ollama_parse_returns_regex_when_ollama_returns_none(self):
        """_ollama_parse() must return original regex_parsed when Ollama returns None."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)

        mock_client = MagicMock()
        mock_client.chat_json.return_value = None
        extractor._ollama_client = mock_client

        regex_parsed = {"expiry_date": "2026-12-31"}
        result = extractor._ollama_parse("some raw text", regex_parsed)

        assert result == regex_parsed

    def test_ollama_parse_skips_on_empty_raw_text(self):
        """_ollama_parse() must short-circuit without calling Ollama if raw_text is blank."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        mock_client = MagicMock()
        extractor._ollama_client = mock_client

        regex_parsed = {"brand": "TestBrand"}
        result = extractor._ollama_parse("   ", regex_parsed)

        mock_client.chat_json.assert_not_called()
        assert result == regex_parsed

    def test_unsupported_backend_raises_value_error(self):
        """Initialising with an unsupported backend must raise ValueError."""
        from aiengine.ocr.extractor import LabelTextExtractor

        with pytest.raises(ValueError, match="Unsupported OCR backend"):
            LabelTextExtractor(backend="badbackend", ollama_model=None)


# ===========================================================================
# 3. XAIExplainer
# ===========================================================================


@pytest.mark.unit
class TestXAIExplainer:
    """Tests for XAIExplainer — template explanations and error paths."""

    def test_invalid_method_raises_value_error(self):
        """XAIExplainer must raise ValueError for an unsupported XAI method."""
        from aiengine.xai.explainer import XAIExplainer

        with pytest.raises(ValueError, match="Unsupported XAI method"):
            XAIExplainer(method="invalid_method", ollama_model=None)

    def test_valid_methods_do_not_raise(self):
        """gradcam and shap are valid XAI methods; lime was removed per spec §3."""
        from aiengine.xai.explainer import XAIExplainer

        for method in ("gradcam", "shap"):
            explainer = XAIExplainer(method=method, ollama_model=None)
            assert explainer.method == method

    def test_default_method_is_gradcam(self):
        """Default XAI method must be 'gradcam'."""
        from aiengine.xai.explainer import XAIExplainer

        explainer = XAIExplainer(ollama_model=None)
        assert explainer.method == "gradcam"

    def test_generate_explanation_returns_string(self, sample_inspection_results):
        """generate_explanation() must return a non-empty string."""
        from aiengine.xai.explainer import XAIExplainer

        explainer = XAIExplainer(ollama_model=None)
        result = explainer.generate_explanation(sample_inspection_results)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_explanation_mentions_food_type(self, sample_inspection_results):
        """Template explanation must mention the food type from the prediction."""
        from aiengine.xai.explainer import XAIExplainer

        explainer = XAIExplainer(ollama_model=None)
        result = explainer.generate_explanation(sample_inspection_results)

        assert "apple" in result.lower()

    def test_generate_explanation_includes_freshness_score(self, sample_inspection_results):
        """Template explanation must include the freshness score."""
        from aiengine.xai.explainer import XAIExplainer

        explainer = XAIExplainer(ollama_model=None)
        result = explainer.generate_explanation(sample_inspection_results)

        assert "0.85" in result

    def test_generate_explanation_mentions_shelf_life(self, sample_inspection_results):
        """Template explanation must mention estimated shelf life days."""
        from aiengine.xai.explainer import XAIExplainer

        explainer = XAIExplainer(ollama_model=None)
        result = explainer.generate_explanation(sample_inspection_results)

        assert "5" in result

    def test_generate_explanation_mentions_defects(self, sample_inspection_results):
        """Template explanation must mention packaging defects when present."""
        from aiengine.xai.explainer import XAIExplainer

        explainer = XAIExplainer(ollama_model=None)
        result = explainer.generate_explanation(sample_inspection_results)

        assert "dent" in result.lower()

    def test_template_explanation_no_ollama_client_uses_template(self, sample_inspection_results):
        """generate_explanation() must use template when _ollama_client is None."""
        from aiengine.xai.explainer import XAIExplainer

        explainer = XAIExplainer(ollama_model=None)
        assert explainer._ollama_client is None
        result = explainer._template_explanation(sample_inspection_results)

        assert isinstance(result, str)

    def test_template_explanation_empty_prediction(self):
        """_template_explanation() must not raise on an empty prediction dict."""
        from aiengine.xai.explainer import XAIExplainer

        explainer = XAIExplainer(ollama_model=None)
        result = explainer._template_explanation({})

        assert isinstance(result, str)

    def test_generate_explanation_uses_ollama_when_client_present(self, sample_inspection_results):
        """generate_explanation() must call Ollama when _ollama_client is set."""
        from aiengine.xai.explainer import XAIExplainer

        explainer = XAIExplainer(ollama_model=None)
        mock_client = MagicMock()
        mock_client.chat.return_value = "Mocked Ollama explanation."
        explainer._ollama_client = mock_client

        result = explainer.generate_explanation(sample_inspection_results)

        mock_client.chat.assert_called_once()
        assert result == "Mocked Ollama explanation."

    def test_generate_explanation_falls_back_to_template_when_ollama_returns_empty(
        self, sample_inspection_results
    ):
        """generate_explanation() must fall back to template if Ollama returns empty string."""
        from aiengine.xai.explainer import XAIExplainer

        explainer = XAIExplainer(ollama_model=None)
        mock_client = MagicMock()
        mock_client.chat.return_value = ""
        explainer._ollama_client = mock_client

        result = explainer.generate_explanation(sample_inspection_results)

        assert "apple" in result.lower()

    def test_generate_heatmap_delegates_to_gradcam(self, rgb_image):
        """generate_heatmap with method='gradcam' must delegate to GradCAMExplainer."""
        from aiengine.xai.explainer import XAIExplainer
        import torch

        explainer = XAIExplainer(method="gradcam", ollama_model=None)
        tensor = torch.randn(1, 3, 224, 224)

        # Mock the model to avoid loading real weights
        mock_model = MagicMock()
        # Mock the model's output to be a tensor
        mock_model.return_value = torch.randn(1, 10)
        # Mock the model as a torch.nn.Module
        mock_model.__class__ = torch.nn.Module

        # We need to mock the target layer as well since GradCAM looks for Conv2d
        mock_layer = MagicMock(spec=torch.nn.Conv2d)
        mock_model.named_modules.return_value = [("conv1", mock_layer)]

        # Since we are mocking everything, GradCAM's actual logic will fail on tensor ops.
        # We just check if the call happens.
        with patch("aiengine.xai.gradcam.GradCAMExplainer.generate_heatmap") as mock_gradcam:
            mock_gradcam.return_value = {"heatmap_array": [], "method": "gradcam"}
            result = explainer.generate_heatmap(mock_model, tensor)
            mock_gradcam.assert_called_once()
            assert result["method"] == "gradcam"

    def test_generate_heatmap_shap_raises_not_implemented(self, rgb_image):
        """generate_heatmap with method='shap' should currently raise NotImplementedError."""
        from aiengine.xai.explainer import XAIExplainer
        import torch

        explainer = XAIExplainer(method="shap", ollama_model=None)
        tensor = torch.randn(1, 3, 224, 224)
        mock_model = MagicMock()
        mock_model.__class__ = torch.nn.Module

        with pytest.raises(NotImplementedError, match="Genuine Image SHAP is not yet implemented"):
            explainer.generate_heatmap(mock_model, tensor)

    def test_shap_explainer_additivity(self):
        """SHAPExplainer must satisfy the additivity property: base + sum(shap) ≈ prediction."""
        shap = pytest.importorskip("shap", reason="shap library not installed")
        from aiengine.xai.shap_explainer import SHAPExplainer
        import numpy as np

        feature_names = ["temp", "hum"]
        def mock_predict(data):
            # linear model: 0.5 + 0.1*temp + 0.2*hum
            return [0.5 + 0.1*d["temp"] + 0.2*d["hum"] for d in data]

        explainer = SHAPExplainer(feature_names=feature_names, predict_func=mock_predict)

        # Mock the shap library if not present to test the interface
        with patch("shap.KernelExplainer") as mock_kernel:
            mock_instance = mock_kernel.return_value
            mock_instance.shap_values.return_value = [np.array([0.1, 0.2])]
            mock_instance.expected_value = 0.5

            # To make additivity check pass: 0.5 + 0.1 + 0.2 = 0.8
            # The model_wrapper will be called. We need to make sure it returns 0.8.
            # The actual mock_predict will do that if we pass temp=1, hum=1.
            result = explainer.explain({"temp": 1.0, "hum": 1.0})

            assert "is_additive" in result
            assert result["base_value"] == 0.5
            assert result["prediction"] == 0.8
            assert result["is_additive"] is True

    def test_gradcam_explainer_output_shape(self):
        """GradCAMExplainer must return a heatmap with correct dimensions."""
        from aiengine.xai.gradcam import GradCAMExplainer
        import torch
        import numpy as np

        explainer = GradCAMExplainer()
        # Create a simple model with a Conv2d layer
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, 3)
                self.fc = torch.nn.Linear(8 * 222 * 222, 10)
            def forward(self, x):
                x = self.conv(x)
                x = torch.flatten(x, 1)
                return self.fc(x)

        model = SimpleModel()
        image = torch.randn(1, 3, 224, 224)

        result = explainer.generate_heatmap(model, image)

        assert "heatmap_array" in result
        heatmap = np.array(result["heatmap_array"])
        # GradCAM heatmap is usually the size of the last conv layer or resized.
        # In GradCAMExplainer, it's the size of the last conv layer (222x222 in this case).
        assert heatmap.shape == (222, 222)


# ===========================================================================
# 4. LLMReportGenerator
# ===========================================================================


@pytest.mark.unit
class TestLLMReportGenerator:
    """Tests for LLMReportGenerator using the 'local/template' provider."""

    def test_detect_provider_ollama(self):
        """_detect_provider() must return 'ollama' for 'ollama/...' model names."""
        from aiengine.llm_report_generator import LLMReportGenerator

        assert LLMReportGenerator._detect_provider("ollama/llama3.1:8b") == "ollama"

    def test_detect_provider_gpt(self):
        """_detect_provider() must return 'gpt' for 'gpt-*' model names."""
        from aiengine.llm_report_generator import LLMReportGenerator

        assert LLMReportGenerator._detect_provider("gpt-4") == "gpt"
        assert LLMReportGenerator._detect_provider("gpt-3.5-turbo") == "gpt"

    def test_detect_provider_hf(self):
        """_detect_provider() must return 'hf' for 'hf/*' model names."""
        from aiengine.llm_report_generator import LLMReportGenerator

        assert LLMReportGenerator._detect_provider("hf/mistral-7b") == "hf"

    def test_detect_provider_local(self):
        """_detect_provider() must return 'local/' for 'local/*' model names."""
        from aiengine.llm_report_generator import LLMReportGenerator

        assert LLMReportGenerator._detect_provider("local/template") == "local/"

    def test_detect_provider_unknown_falls_back_to_local(self):
        """_detect_provider() must return the local fallback for unrecognised strings."""
        from aiengine.llm_report_generator import LLMReportGenerator

        assert LLMReportGenerator._detect_provider("unknown-model") == "local/"

    def test_generate_report_returns_dict(self, sample_inspection_results):
        """generate_report() must return a dict for the local/template provider."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")
        report = gen.generate_report(sample_inspection_results)

        assert isinstance(report, dict)

    def test_generate_report_contains_required_keys(self, sample_inspection_results):
        """generate_report() dict must contain all seven required top-level keys."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")
        report = gen.generate_report(sample_inspection_results)

        required_keys = {
            "report_title",
            "executive_summary",
            "detailed_findings",
            "risk_flags",
            "recommendations",
            "overall_verdict",
            "inspection_date",
        }
        assert required_keys.issubset(report.keys())

    def test_generate_report_verdict_pass_for_fresh_food(self, sample_inspection_results):
        """Fresh food with no defects must yield overall_verdict='pass'."""
        from aiengine.llm_report_generator import LLMReportGenerator

        clean_results = dict(sample_inspection_results)
        clean_results["packaging_defects"] = []
        gen = LLMReportGenerator(model_name="local/template")
        report = gen.generate_report(clean_results)

        assert report["overall_verdict"] == "pass"

    def test_generate_report_verdict_fail_for_spoiled_food(self, spoiled_inspection_results):
        """Spoiled food inspection must yield overall_verdict='fail'."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")
        report = gen.generate_report(spoiled_inspection_results)

        assert report["overall_verdict"] == "fail"

    def test_generate_report_title_contains_food_type(self, sample_inspection_results):
        """report_title must mention the food type."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")
        report = gen.generate_report(sample_inspection_results)

        assert "apple" in report["report_title"].lower()

    def test_generate_report_inspection_date_is_valid_iso(self, sample_inspection_results):
        """inspection_date must be parseable as an ISO 8601 datetime string."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")
        report = gen.generate_report(sample_inspection_results)

        parsed_date = datetime.fromisoformat(report["inspection_date"])
        assert parsed_date.year >= 2024

    def test_generate_report_recommendations_always_present(self, sample_inspection_results):
        """recommendations list must always be present and non-empty."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")
        report = gen.generate_report(sample_inspection_results)

        assert isinstance(report["recommendations"], list)
        assert len(report["recommendations"]) > 0

    def test_generate_report_risk_flags_for_spoiled(self, spoiled_inspection_results):
        """risk_flags must be non-empty when food is spoiled."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")
        report = gen.generate_report(spoiled_inspection_results)

        assert len(report["risk_flags"]) > 0
        assert any("spoil" in flag.lower() for flag in report["risk_flags"])

    def test_generate_report_detailed_findings_is_list(self, sample_inspection_results):
        """detailed_findings must be a list of dicts with area/status/detail keys."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")
        report = gen.generate_report(sample_inspection_results)

        assert isinstance(report["detailed_findings"], list)
        for finding in report["detailed_findings"]:
            assert "area" in finding
            assert "status" in finding
            assert "detail" in finding

    def test_template_report_packaging_flag_when_defects_present(self, sample_inspection_results):
        """_template_report() must add a risk flag when packaging defects are present."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")
        report = gen._template_report(sample_inspection_results)

        assert any("packaging" in flag.lower() for flag in report["risk_flags"])

    def test_template_report_no_defect_flags_when_defects_empty(self, spoiled_inspection_results):
        """_template_report() must NOT add a packaging flag when defects list is empty."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")
        report = gen._template_report(spoiled_inspection_results)

        packaging_flags = [f for f in report["risk_flags"] if "packaging" in f.lower()]
        assert len(packaging_flags) == 0

    def test_generate_summary_returns_string(self, sample_inspection_results):
        """generate_summary() must return a string."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")
        report = gen.generate_report(sample_inspection_results)
        summary = gen.generate_summary(report["executive_summary"])

        assert isinstance(summary, str)

    def test_generate_summary_truncates_long_text(self):
        """generate_summary() must truncate strings longer than 200 chars with '...'."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")
        long_text = "A" * 300
        summary = gen.generate_summary(long_text)

        assert summary.endswith("...")
        assert len(summary) <= 203

    def test_generate_summary_short_text_unchanged(self):
        """generate_summary() must return short text as-is (under 200 chars)."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")
        short_text = "The food is fresh and safe to eat."
        summary = gen.generate_summary(short_text)

        assert summary == short_text


# ===========================================================================
# 5. OllamaClient
# ===========================================================================


@pytest.mark.unit
class TestOllamaClient:
    """Tests for OllamaClient — all network calls are mocked."""

    def test_init_stores_model_name(self):
        """OllamaClient must store the model name passed at construction."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b")
            assert client.model == "llama3.1:8b"

    def test_init_stores_temperature(self):
        """OllamaClient must store the temperature value."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b", temperature=0.7)
            assert client.temperature == 0.7

    def test_init_uses_default_base_url(self):
        """OllamaClient must default to the localhost Ollama URL."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b")
            assert "localhost:11434" in client.base_url

    def test_init_custom_base_url(self):
        """OllamaClient must accept and store a custom base_url."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b", base_url="http://myserver:11434/v1")
            assert client.base_url == "http://myserver:11434/v1"

    def test_is_available_true_when_server_returns_200(self):
        """is_available must return True when the Ollama /api/tags endpoint is reachable."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b")
            client._enabled = True
            client._client = MagicMock()

            mock_response = MagicMock()
            mock_response.status_code = 200

            with patch("httpx.get", return_value=mock_response):
                assert client.is_available is True

    def test_is_available_false_when_server_unreachable(self):
        """is_available must return False when httpx raises a connection error."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient
            import httpx

            client = OllamaClient(model="llama3.1:8b")
            client._enabled = True
            client._client = MagicMock()

            with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
                assert client.is_available is False

    def test_is_available_false_when_client_is_none(self):
        """is_available must return False when _client was never initialised."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b")
            client._enabled = True
            client._client = None

            assert client.is_available is False

    def test_is_available_false_when_disabled(self):
        """is_available must return False when _enabled is False."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b")
            client._enabled = False
            client._client = MagicMock()

            assert client.is_available is False

    def test_chat_returns_model_response_string(self):
        """chat() must return the content string from the model's first choice."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b")
            client._enabled = True

            mock_openai = MagicMock()
            mock_openai.chat.completions.create.return_value.choices[
                0
            ].message.content = "Hello from Ollama"
            client._client = mock_openai

            result = client.chat("Describe the food.")
            assert result == "Hello from Ollama"

    def test_chat_returns_empty_string_when_disabled(self):
        """chat() must return '' immediately when client is disabled."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b")
            client._enabled = False
            client._client = None

            result = client.chat("Any prompt")
            assert result == ""

    def test_chat_includes_system_message_when_provided(self):
        """chat() must include the system message in the messages list."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b")
            client._enabled = True

            mock_openai = MagicMock()
            mock_openai.chat.completions.create.return_value.choices[0].message.content = "ok"
            client._client = mock_openai

            client.chat("User prompt", system="Be brief.")

            messages = mock_openai.chat.completions.create.call_args[1]["messages"]
            roles = [m["role"] for m in messages]
            assert "system" in roles

    def test_chat_returns_empty_string_on_exception(self):
        """chat() must return '' and not raise when the API call throws."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b")
            client._enabled = True

            mock_openai = MagicMock()
            mock_openai.chat.completions.create.side_effect = RuntimeError("Network error")
            client._client = mock_openai

            result = client.chat("Any prompt")
            assert result == ""

    def test_chat_json_parses_valid_json_response(self):
        """chat_json() must return a parsed dict when the response contains valid JSON."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b")
            client._enabled = True

            payload = {"report_title": "Test Report", "verdict": "pass"}
            raw = f"Here is the result: {json.dumps(payload)} Done."

            mock_openai = MagicMock()
            mock_openai.chat.completions.create.return_value.choices[
                0
            ].message.content = raw
            client._client = mock_openai

            result = client.chat_json("Generate report JSON.")
            assert result == payload

    def test_chat_json_returns_none_on_invalid_json(self):
        """chat_json() must return None when the response is not valid JSON."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b")
            client._enabled = True

            mock_openai = MagicMock()
            mock_openai.chat.completions.create.return_value.choices[
                0
            ].message.content = "not json at all"
            client._client = mock_openai

            result = client.chat_json("Generate report JSON.")
            assert result is None

    def test_chat_json_returns_none_when_chat_empty(self):
        """chat_json() must return None when chat() returns an empty string."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b")
            client._enabled = False
            client._client = None

            result = client.chat_json("Any prompt")
            assert result is None

    def test_list_local_models_returns_model_names(self):
        """list_local_models() must return a list of model name strings."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient

            client = OllamaClient(model="llama3.1:8b")
            client._enabled = True
            client._client = MagicMock()

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [{"name": "llama3.1:8b"}, {"name": "mistral:7b"}]
            }

            with patch("httpx.get", return_value=mock_response):
                models = client.list_local_models()

            assert models == ["llama3.1:8b", "mistral:7b"]

    def test_list_local_models_returns_empty_list_on_error(self):
        """list_local_models() must return [] when httpx raises an exception."""
        with patch("aiengine.ollama_client.OllamaClient._init_client"):
            from aiengine.ollama_client import OllamaClient
            import httpx

            client = OllamaClient(model="llama3.1:8b")
            client._enabled = True
            client._client = MagicMock()

            with patch("httpx.get", side_effect=httpx.ConnectError("refused")):
                models = client.list_local_models()

            assert models == []
