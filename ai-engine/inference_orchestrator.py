import asyncio
import logging
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from ai_engine.preprocessing.pipeline import PreprocessingPipeline
from ai_engine.models.food_classification.inference import FoodClassificationInference
from ai_engine.models.spoilage_detection.inference import SpoilageDetectionInference
from ai_engine.models.packaging_defect.model import PackagingDefectDetector
from ai_engine.models.contamination_risk.model import ContaminationRiskInference
from ai_engine.models.shelf_life_prediction.model import ShelfLifeInference

logger = logging.getLogger(__name__)


class AIInferenceOrchestrator:
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._init_models()

    def _init_models(self):
        device = self.config.get("device", "cpu")
        self.preprocessor = PreprocessingPipeline(device=device)

        models_dir = self.config.get("models_dir", "")
        self.food_classifier = FoodClassificationInference(
            model_path=self.config.get("food_classification_model", f"{models_dir}/food_classification.pt"),
            device=device,
        )
        self.spoilage_detector = SpoilageDetectionInference(
            model_path=self.config.get("spoilage_detection_model", f"{models_dir}/spoilage_detection.pt"),
            device=device,
        )
        self.defect_detector = PackagingDefectDetector(
            model_type=self.config.get("defect_model_type", "yolov8"),
            model_path=self.config.get("packaging_defect_model", ""),
        )
        self.contamination_assessor = ContaminationRiskInference(
            model_path=self.config.get("contamination_risk_model", f"{models_dir}/contamination_risk.pt"),
            device=device,
        )
        self.shelf_life_predictor = ShelfLifeInference(
            model_path=self.config.get("shelf_life_model", f"{models_dir}/shelf_life.pt"),
            device=device,
        )

    def run_full_inspection(self, image) -> dict:
        quality = self.preprocessor.check_quality(image)
        tensor = self.preprocessor.process(image)

        with ThreadPoolExecutor(max_workers=5) as pool:
            food_future = pool.submit(self.food_classifier.predict, image)
            spoilage_future = pool.submit(self.spoilage_detector.predict, image)
            defect_future = pool.submit(self.defect_detector.detect, image)
            contamination_future = pool.submit(self.contamination_assessor.predict, image)
            shelf_future = pool.submit(self.shelf_life_predictor.predict, image)

            food_result = food_future.result()
            spoilage_result = spoilage_future.result()
            defect_result = defect_future.result()
            contamination_result = contamination_future.result()
            shelf_result = shelf_future.result()

        return {
            "food_type": food_result["food_type"],
            "freshness_score": spoilage_result["freshness_score"],
            "spoilage": spoilage_result,
            "packaging_defects": defect_result,
            "contamination_risks": contamination_result,
            "shelf_life": shelf_result,
            "food_classification": food_result,
            "confidence_scores": {
                "food_classification": food_result["confidence_scores"][0]["confidence"],
                "spoilage_detection": spoilage_result["spoilage_score"],
                "shelf_life_prediction": shelf_result["confidence"],
            },
            "image_quality": quality,
        }

    def run_fast_inspection(self, image) -> dict:
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
