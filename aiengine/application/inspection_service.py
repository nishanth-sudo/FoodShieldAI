from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor

from PIL import Image

from aiengine.domain.exceptions import ModelInferenceError

logger = logging.getLogger(__name__)


class InspectionService:
    """
    Application service that runs the core Computer Vision pipeline.

    Orchestrates food classification, spoilage detection, packaging defect
    detection, contamination assessment, shelf-life prediction and OCR in
    parallel using a :class:`~concurrent.futures.ThreadPoolExecutor`.

    XAI, LLM and VLM logic is intentionally excluded — see
    :class:`~aiengine.application.explanation_service.ExplanationService` and
    :class:`~aiengine.application.report_service.ReportService`.

    All dependencies are injected at construction time to keep this class
    testable without touching the file system or any ML framework.
    """

    def __init__(
        self,
        food_classifier,        # FoodClassificationInference — kept untyped to avoid circular import
        spoilage_detector,      # SpoilageDetectionInference
        defect_detector,        # PackagingDefectDetector
        contamination_assessor, # ContaminationRiskInference
        shelf_life_predictor,   # ShelfLifeInference
        preprocessor,           # PreprocessingPipeline
        ocr_extractor,          # LabelTextExtractor
    ) -> None:
        self._food_classifier = food_classifier
        self._spoilage_detector = spoilage_detector
        self._defect_detector = defect_detector
        self._contamination_assessor = contamination_assessor
        self._shelf_life_predictor = shelf_life_predictor
        self._preprocessor = preprocessor
        self._ocr_extractor = ocr_extractor

    # ------------------------------------------------------------------
    # Public pipeline methods
    # ------------------------------------------------------------------

    def run_full_cv_pipeline(self, image: Image.Image) -> dict:
        """
        Run all CV models in parallel and return aggregated results.

        Args:
            image: A PIL Image to inspect.

        Returns:
            A dict with keys: ``food_type``, ``freshness_score``, ``spoilage``,
            ``packaging_defects``, ``contamination_risks``, ``shelf_life``,
            ``food_classification``, ``ocr_data``, ``confidence_scores``,
            ``image_quality``.

        Raises:
            ModelInferenceError: If pre-processing or any CV model fails.
        """
        try:
            quality: dict = self._preprocessor.check_quality(image)
            label_region = self._preprocessor.detect_label_region(image)
        except Exception as exc:
            raise ModelInferenceError(
                f"Pre-processing failed before CV pipeline: {exc}"
            ) from exc

        with ThreadPoolExecutor(max_workers=6) as pool:
            food_future: Future[dict] = pool.submit(self._food_classifier.predict, image)
            spoilage_future: Future[dict] = pool.submit(self._spoilage_detector.predict, image)
            defect_future: Future[dict] = pool.submit(self._defect_detector.detect, image)
            contamination_future: Future[dict] = pool.submit(
                self._contamination_assessor.predict, image
            )
            shelf_future: Future[dict] = pool.submit(self._shelf_life_predictor.predict, image)
            ocr_future: Future[dict] = pool.submit(
                self._ocr_extractor.extract_from_label_region, image, label_region
            )

            try:
                food_result: dict = food_future.result()
                spoilage_result: dict = spoilage_future.result()
                defect_result: dict = defect_future.result()
                contamination_result: dict = contamination_future.result()
                shelf_result: dict = shelf_future.result()
                ocr_result: dict = ocr_future.result()
            except Exception as exc:
                raise ModelInferenceError(
                    f"CV model inference failed: {exc}"
                ) from exc

        return {
            "food_type": food_result["food_type"],
            "freshness_score": spoilage_result["freshness_score"],
            "spoilage": spoilage_result,
            "packaging_defects": defect_result,
            "contamination_risks": contamination_result,
            "shelf_life": shelf_result,
            "food_classification": food_result,
            "ocr_data": ocr_result,
            "confidence_scores": {
                "food_classification": food_result["confidence_scores"][0]["confidence"],
                "spoilage_detection": spoilage_result["spoilage_score"],
                "shelf_life_prediction": shelf_result["confidence"],
            },
            "image_quality": quality,
        }

    def run_fast_pipeline(self, image: Image.Image) -> dict:
        """
        Lightweight inspection — food classification + spoilage only.

        No defect detection, contamination assessment, OCR, XAI, VLM or
        LLM report. Suitable for real-time / low-latency use cases.

        Args:
            image: A PIL Image to inspect.

        Returns:
            A dict with keys: ``food_type``, ``freshness_score``, ``spoilage``,
            ``food_classification``, ``image_quality``.

        Raises:
            ModelInferenceError: If food classifier or spoilage detector fails.
        """
        try:
            quality: dict = self._preprocessor.check_quality(image)
        except Exception as exc:
            raise ModelInferenceError(
                f"Pre-processing failed before fast pipeline: {exc}"
            ) from exc

        try:
            food_result: dict = self._food_classifier.predict(image, top_k=3)
            spoilage_result: dict = self._spoilage_detector.predict(image)
        except Exception as exc:
            raise ModelInferenceError(
                f"Fast pipeline inference failed: {exc}"
            ) from exc

        return {
            "food_type": food_result["food_type"],
            "freshness_score": spoilage_result["freshness_score"],
            "spoilage": spoilage_result,
            "food_classification": food_result,
            "image_quality": quality,
        }
