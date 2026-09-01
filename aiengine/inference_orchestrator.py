import logging
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from aiengine.llm_report_generator import LLMReportGenerator
from aiengine.models.contamination_risk.model import ContaminationRiskInference
from aiengine.models.food_classification.inference import FoodClassificationInference
from aiengine.models.packaging_defect.model import PackagingDefectDetector
from aiengine.models.shelf_life_prediction.model import ShelfLifeInference
from aiengine.models.spoilage_detection.inference import SpoilageDetectionInference
from aiengine.ocr.extractor import LabelTextExtractor
from aiengine.preprocessing.pipeline import PreprocessingPipeline
from aiengine.vlm_augmentor import VisionLLMAugmentor
from aiengine.xai.counterfactual import CounterfactualExplainer
from aiengine.xai.explainer import XAIExplainer
from aiengine.xai.shap_explainer import SHAPExplainer

logger = logging.getLogger(__name__)


class AIInferenceOrchestrator:
    """
    End-to-end food inspection pipeline orchestrator.

    Wires together all AI/ML components:
      - CV models  : food classification, spoilage, defect, contamination, shelf-life
      - OCR        : label text extraction (PaddleOCR / EasyOCR / Tesseract)
      - XAI        : Grad-CAM heatmap + Ollama-powered natural-language explanation
      - VLM        : Ollama vision model cross-validation (optional)
      - LLM report : Ollama / OpenAI / HuggingFace inspection report generation

    Config keys (all optional):
    ┌─────────────────────────────┬──────────────────────────────────────────────────────┐
    │ Key                         │ Description / Default                                │
    ├─────────────────────────────┼──────────────────────────────────────────────────────┤
    │ device                      │ "cpu" or "cuda"                (default: "cpu")      │
    │ models_dir                  │ Directory containing .pt files (default: "")         │
    │ ocr_backend                 │ "paddleocr" | "easyocr" |                            │
    │                             │ "tesseract" | "mock"           (default: "mock")     │
    │ xai_method                  │ "gradcam" | "lime" | "shap"    (default: "gradcam")  │
    │ llm_model                   │ LLM backend + model name                             │
    │                             │   "ollama/llama3.1:8b"  ← default (recommended)      │
    │                             │   "gpt-4"               ← OpenAI                     │
    │                             │   "hf/<model>"          ← HuggingFace                │
    │                             │   "local/template"      ← offline template           │
    │ llm_temperature             │ LLM sampling temperature       (default: 0.3)        │
    │ ollama_xai_model            │ Ollama model for XAI text      (default: llama3.2:3b)│
    │ ollama_ocr_model            │ Ollama model for OCR fallback  (default: llama3.2:3b)│
    │ vlm_enabled                 │ Enable VLM cross-validation    (default: False)      │
    │ vlm_model                   │ Ollama vision model            (default: llava:7b)   │
    └─────────────────────────────┴──────────────────────────────────────────────────────┘
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._init_models()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_models(self) -> None:
        device = self.config.get("device", "cpu")
        models_dir = self.config.get("models_dir", "")

        self.preprocessor = PreprocessingPipeline(device=device)

        self.food_classifier = FoodClassificationInference(
            model_path=self.config.get(
                "food_classification_model", f"{models_dir}/food_classification.pt"
            ),
            device=device,
        )
        self.spoilage_detector = SpoilageDetectionInference(
            model_path=self.config.get(
                "spoilage_detection_model", f"{models_dir}/spoilage_detection.pt"
            ),
            device=device,
        )
        self.defect_detector = PackagingDefectDetector(
            model_type=self.config.get("defect_model_type", "yolov8"),
            model_path=self.config.get("packaging_defect_model", ""),
        )
        self.contamination_assessor = ContaminationRiskInference(
            model_path=self.config.get(
                "contamination_risk_model", f"{models_dir}/contamination_risk.pt"
            ),
            device=device,
        )
        self.shelf_life_predictor = ShelfLifeInference(
            model_path=self.config.get("shelf_life_model", f"{models_dir}/shelf_life.pt"),
            device=device,
        )

        # OCR — with Ollama fallback parsing
        self.ocr_extractor = LabelTextExtractor(
            backend=self.config.get("ocr_backend", "mock"),
            ollama_model=self.config.get("ollama_ocr_model", "llama3.2:3b"),
        )

        # XAI — with Ollama explanation generation
        self.xai_explainer = XAIExplainer(
            method=self.config.get("xai_method", "gradcam"),
            ollama_model=self.config.get("ollama_xai_model", "llama3.2:3b"),
        )

        # LLM report — provider-based
        llm_provider_type = self.config.get("llm_provider", "ollama")
        llm_model = self.config.get("llm_model", "llama3.1:8b")

        if llm_provider_type == "ollama":
            from aiengine.llm.ollama_provider import OllamaProvider
            provider = OllamaProvider(model=llm_model)
        elif llm_provider_type == "vllm":
            from aiengine.llm.vllm_provider import VLLMProvider
            provider = VLLMProvider(model=llm_model)
        elif llm_provider_type == "openai":
            from aiengine.llm.openai_provider import OpenAIProvider
            provider = OpenAIProvider(model=llm_model)
        else:
            provider = None

        if provider:
            from aiengine.llm.resilience import ResilientLLMProvider
            provider = ResilientLLMProvider(provider)

        self.report_generator = LLMReportGenerator(
            provider=provider,
            temperature=self.config.get("llm_temperature", 0.3),
        )


        # VLM augmentor — opt-in (disabled by default; requires vision model pull)
        self.vlm_augmentor = VisionLLMAugmentor(
            model=self.config.get("vlm_model", "llava:7b"),
            enabled=self.config.get("vlm_enabled", False),
        )

        # SHAP explainer — for tabular / shelf-life model explanations
        self._shap_feature_names = [
            "temperature", "humidity", "storage_duration", "packaging_type", "food_type",
        ]
        self.shap_explainer: SHAPExplainer | None = None  # lazily initialised

        # Counterfactual explainer — "what would need to change?"
        self._cf_feature_constraints = {
            "temperature": (0.0, 45.0),
            "humidity": (0.0, 100.0),
            "storage_duration": (0.0, 30.0),
        }
        self.counterfactual_explainer: CounterfactualExplainer | None = None  # lazily initialised

    # ------------------------------------------------------------------
    # Inspection pipelines
    # ------------------------------------------------------------------

    def run_full_inspection(
        self, image: Image.Image, environmental_data: dict | None = None,
    ) -> dict:
        """
        Run the complete multi-model inspection pipeline.

        Steps:
        1. Image quality check & label region detection
        2. Parallel CV inference (food class, spoilage, defects, contamination, shelf-life, OCR)
        3. XAI explanation generation (Grad-CAM + Ollama text)
        4. SHAP + Counterfactual XAI (when environmental_data provided)
        5. Optional VLM cross-validation (llava / minicpm-v)
        6. LLM report generation (Ollama / OpenAI / template)
        """
        quality = self.preprocessor.check_quality(image)
        label_region = self.preprocessor.detect_label_region(image)

        with ThreadPoolExecutor(max_workers=5) as pool:
            food_future = pool.submit(self.food_classifier.predict, image)
            spoilage_future = pool.submit(self.spoilage_detector.predict, image)
            defect_future = pool.submit(self.defect_detector.detect, image)
            contamination_future = pool.submit(self.contamination_assessor.predict, image)
            shelf_future = pool.submit(self.shelf_life_predictor.predict, image)
            ocr_future = pool.submit(
                self.ocr_extractor.extract_from_label_region, image, label_region
            )

            food_result = food_future.result()
            spoilage_result = spoilage_future.result()
            defect_result = defect_future.result()
            contamination_result = contamination_future.result()
            shelf_result = shelf_future.result()
            ocr_result = ocr_future.result()

        xai_result = self.xai_explainer.generate_explanation(food_result | spoilage_result)

        aggregated: dict = {
            "food_type": food_result["food_type"],
            "freshness_score": spoilage_result["freshness_score"],
            "spoilage": spoilage_result,
            "packaging_defects": defect_result,
            "contamination_risks": contamination_result,
            "shelf_life": shelf_result,
            "food_classification": food_result,
            "ocr_data": ocr_result,
            "xai_explanation": xai_result,
            "confidence_scores": {
                "food_classification": food_result["confidence_scores"][0]["confidence"],
                "spoilage_detection": spoilage_result["spoilage_score"],
                "shelf_life_prediction": shelf_result["confidence"],
            },
            "image_quality": quality,
        }

        # ── XAI: SHAP + Counterfactual (when environmental context provided) ──
        xai_evidence = {}
        # These are currently disabled as there is no trained environmental risk model.
        # Once a model is available, they will be re-enabled and connected to the actual model.
        if False and environmental_data:
            shap_result = self._compute_shap(environmental_data, spoilage_result)
            if shap_result:
                xai_evidence["shap"] = shap_result

            cf_result = self._compute_counterfactuals(
                environmental_data, spoilage_result,
            )
            if cf_result:
                xai_evidence["counterfactuals"] = cf_result

        if xai_evidence:
            aggregated["xai"] = xai_evidence

        # Optional VLM cross-validation (adds "vlm_augmentation" key)
        if self.vlm_augmentor.is_available:
            aggregated = self.vlm_augmentor.enrich_results(aggregated, image)
        else:
            aggregated["vlm_augmentation"] = {"enabled": False, "reason": "VLM not configured"}

        # LLM report
        aggregated["report"] = self.report_generator.generate_report(aggregated)

        return aggregated

    def run_fast_inspection(self, image: Image.Image) -> dict:
        """
        Lightweight inspection — food classification + spoilage only.
        No VLM, no LLM report, no OCR.
        """
        quality = self.preprocessor.check_quality(image)
        food_result = self.food_classifier.predict(image, top_k=3)
        spoilage_result = self.spoilage_detector.predict(image)
        return {
            "food_type": food_result["food_type"],
            "freshness_score": spoilage_result["freshness_score"],
            "spoilage": spoilage_result,
            "food_classification": food_result,
            "image_quality": quality,
        }

    def run_explanation(
        self, image: Image.Image, environmental_data: dict | None = None,
    ) -> dict:
        """
        Run the XAI explanation pipeline.

        Pipeline: Image → CV Models → Grad-CAM → SHAP → Counterfactuals → LLM Explanation

        Steps:
        1. Run food classification + spoilage detection
        2. Generate Grad-CAM heatmap
        3. If environmental_data provided, compute SHAP values and counterfactuals
        4. Generate natural-language explanation via LLM
        5. Return structured XAI results
        """
        food_result = self.food_classifier.predict(image, top_k=3)
        spoilage_result = self.spoilage_detector.predict(image)

        # Grad-CAM heatmap
        tensor = self.preprocessor.process(image)
        gradcam_result = self.xai_explainer.generate_heatmap(
            self.food_classifier.model, tensor,
        )

        # Natural-language explanation
        explanation = self.xai_explainer.generate_explanation(
            food_result | spoilage_result,
        )

        # Determine risk level
        spoilage_score = spoilage_result.get("spoilage_score", 0)
        if spoilage_score >= 0.7:
            risk_level = "HIGH"
        elif spoilage_score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        result: dict = {
            "gradcam": gradcam_result,
            "explanation": explanation,
            "risk_level": risk_level,
        }

        # SHAP + Counterfactuals (when environmental context provided)
        if False and environmental_data:
            shap_result = self._compute_shap(environmental_data, spoilage_result)
            if shap_result:
                result["shap_values"] = shap_result

                # Build risk factors table from SHAP values
                risk_factors = []
                for feat, val in shap_result.get("feature_importance", []):
                    contribution_pct = abs(val) * 100
                    risk_factors.append({
                        "factor": feat,
                        "contribution": round(contribution_pct, 1),
                        "direction": "increases risk" if val > 0 else "decreases risk",
                    })
                result["risk_factors"] = risk_factors

            cf_result = self._compute_counterfactuals(
                environmental_data, spoilage_result,
            )
            if cf_result:
                result["counterfactuals"] = cf_result

        return result

    def generate_report_from_evidence(self, evidence: dict) -> dict:
        """
        Generate an LLM inspection report from pre-computed structured evidence.

        This allows the /report endpoint to accept pre-computed CV + XAI results
        and produce a grounded LLM explanation without re-running inference.
        """
        # Build a synthetic inspection result from the evidence
        synthetic: dict = {
            "food_type": evidence.get("food_type", "Unknown"),
            "freshness_score": 1.0 - evidence.get("spoilage_probability", 0.5),
            "spoilage": {
                "is_spoiled": evidence.get("spoilage_probability", 0.5) > 0.5,
                "spoilage_score": evidence.get("spoilage_probability", 0.5),
                "freshness_score": 1.0 - evidence.get("spoilage_probability", 0.5),
                "severity_label": "high" if evidence.get("spoilage_probability", 0.5) > 0.7 else "medium",
            },
            "shelf_life": {
                "estimated_days_remaining": evidence.get("shelf_life_days", "N/A"),
                "freshness_category": "short" if evidence.get("shelf_life_days", 7) and evidence.get("shelf_life_days", 7) < 3 else "moderate",
                "confidence": 0.85,
            },
            "contamination_risks": {},
            "packaging_defects": [],
            "food_classification": {
                "food_type": evidence.get("food_type", "Unknown"),
                "confidence_scores": [{"confidence": 0.9}],
            },
            "ocr_data": {},
        }

        # Attach XAI evidence if available
        if evidence.get("xai_evidence"):
            synthetic["xai"] = evidence["xai_evidence"]
        if evidence.get("counterfactuals"):
            synthetic.setdefault("xai", {})["counterfactuals"] = evidence["counterfactuals"]

        return self.report_generator.generate_report(synthetic)

    # ------------------------------------------------------------------
    # Private XAI helpers
    # ------------------------------------------------------------------
    pass