"""
tests/unit/test_performance.py — Task 7.5: Performance / Load tests

All tests run entirely in-process with no running server.
Heavy GPU/network dependencies are mocked; timing is done with
time.perf_counter() so numbers are stable on CI runners.

Performance budgets (generous enough for CPU-only CI):
  - Single-image preprocessing  : < 100 ms
  - Batch of 10 images           : < 1 s
  - Quality check                : < 50 ms
  - Augmentation (4 variants)    : < 500 ms
  - Mock OCR extraction          : < 10 ms
  - Regex field parsing          : < 5 ms
  - Single template report       : < 50 ms
  - 100 template reports         : < 2 s
  - 10 concurrent mock OCRs      : < 2 s
  - 20 concurrent template rpts  : < 3 s
  - 2000x2000 image memory peak  : < 200 MB
"""

import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Reusable fixture — mirrors the orchestrator output shape
# ---------------------------------------------------------------------------

INSPECTION_RESULTS = {
    "food_type": "tomato",
    "freshness_score": 0.7,
    "spoilage": {
        "is_spoiled": False,
        "freshness_score": 0.7,
        "severity_label": "low",
    },
    "packaging_defects": [],
    "contamination_risks": {
        "overall_risk_level": "low",
        "detected_risks": [],
    },
    "shelf_life": {
        "estimated_days_remaining": 10,
        "freshness_category": "moderate",
        "confidence": 0.85,
    },
    "food_classification": {
        "food_type": "tomato",
        "confidence_scores": [{"confidence": 0.88}],
    },
    "ocr_data": {"parsed": {}, "raw_text": ""},
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgb_image(width: int = 640, height: int = 480) -> Image.Image:
    """Create a solid-colour PIL RGB image of the requested dimensions."""
    return Image.new("RGB", (width, height), color=(120, 80, 40))


# ---------------------------------------------------------------------------
# TestPreprocessingPerformance
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.performance
class TestPreprocessingPerformance:
    """Verify that the PreprocessingPipeline CPU operations complete within
    tight latency budgets suitable for real-time inspection pipelines."""

    def test_single_image_processing_under_100ms(self) -> None:
        """process() on a 640x480 RGB image must finish within 100 ms."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(device="cpu")
        image = _make_rgb_image(640, 480)

        start = time.perf_counter()
        tensor = pipeline.process(image)
        elapsed_ms = (time.perf_counter() - start) * 1_000

        assert tensor.shape == (1, 3, 224, 224), "Output tensor shape should be (1, 3, 224, 224)"
        assert elapsed_ms < 100, f"process() took {elapsed_ms:.1f} ms — expected < 100 ms"

    def test_batch_processing_10_images_under_1s(self) -> None:
        """process_batch() on 10 images (640x480 each) must finish within 1 s."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(device="cpu")
        images = [_make_rgb_image(640, 480) for _ in range(10)]

        start = time.perf_counter()
        batch = pipeline.process_batch(images)
        elapsed_s = time.perf_counter() - start

        assert batch.shape[0] == 10, "Batch tensor should contain 10 images"
        assert elapsed_s < 1.0, f"process_batch(10) took {elapsed_s:.3f} s — expected < 1 s"

    def test_quality_check_under_50ms(self) -> None:
        """check_quality() on a 640x480 image must finish within 50 ms."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(device="cpu")
        image = _make_rgb_image(640, 480)

        start = time.perf_counter()
        result = pipeline.check_quality(image)
        elapsed_ms = (time.perf_counter() - start) * 1_000

        assert isinstance(result, dict), "check_quality() should return a dict"
        assert elapsed_ms < 50, f"check_quality() took {elapsed_ms:.1f} ms — expected < 50 ms"

    def test_augmentation_under_500ms(self) -> None:
        """augment() producing 4 variants from a 640x480 image must finish within 500 ms."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(device="cpu")
        image = _make_rgb_image(640, 480)

        start = time.perf_counter()
        augmented = pipeline.augment(image)
        elapsed_ms = (time.perf_counter() - start) * 1_000

        assert len(augmented) == 4, "augment() should return 4 variants"
        assert elapsed_ms < 500, f"augment() took {elapsed_ms:.1f} ms — expected < 500 ms"


# ---------------------------------------------------------------------------
# TestOCRPerformance
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.performance
class TestOCRPerformance:
    """Verify OCR extraction and regex field-parsing complete within tight
    latency budgets when running against the in-process mock backend."""

    def test_mock_ocr_extraction_under_10ms(self) -> None:
        """LabelTextExtractor(backend='mock').extract() should complete under 10 ms."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)
        image = _make_rgb_image()

        start = time.perf_counter()
        result = extractor.extract(image)
        elapsed_ms = (time.perf_counter() - start) * 1_000

        assert "raw_text" in result, "extract() result should contain 'raw_text'"
        assert elapsed_ms < 10, f"mock OCR extract() took {elapsed_ms:.2f} ms — expected < 10 ms"

    def test_regex_field_parsing_under_5ms(self) -> None:
        """_parse_fields() on a long, realistic label string must finish within 5 ms."""
        from aiengine.ocr.extractor import LabelTextExtractor

        extractor = LabelTextExtractor(backend="mock", ollama_model=None)

        long_label = (
            "Product: Organic Whole Milk\n"
            "Brand: NatureFarm\n"
            "Ingredients: Pasteurized Whole Milk, Vitamin D3, Calcium\n"
            "Net Wt: 1 Liter (33.8 fl oz)\n"
            "Best Before: 12/08/2026\n"
            "Nutrition Facts: Calories 150, Total Fat 8g, Saturated Fat 5g, "
            "Trans Fat 0g, Cholesterol 35mg, Sodium 125mg, Total Carbohydrate 12g, "
            "Dietary Fiber 0g, Total Sugars 12g, Protein 8g\n"
            "Allergens: Contains Milk\n"
            "Manufactured by NatureFarm Dairies, 123 Farm Road, Springfield.\n"
        ) * 5

        start = time.perf_counter()
        parsed = extractor._parse_fields(long_label)
        elapsed_ms = (time.perf_counter() - start) * 1_000

        assert isinstance(parsed, dict), "_parse_fields() should return a dict"
        assert elapsed_ms < 5, f"_parse_fields() took {elapsed_ms:.2f} ms — expected < 5 ms"


# ---------------------------------------------------------------------------
# TestReportGeneratorPerformance
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.performance
class TestReportGeneratorPerformance:
    """Verify the template-based (no-LLM) report generator meets throughput
    targets suitable for batch inspection queues."""

    def test_template_report_under_50ms(self) -> None:
        """A single generate_report() in template mode must finish under 50 ms."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")

        start = time.perf_counter()
        report = gen.generate_report(INSPECTION_RESULTS)
        elapsed_ms = (time.perf_counter() - start) * 1_000

        assert "overall_verdict" in report, "Report must contain 'overall_verdict'"
        assert elapsed_ms < 50, (
            f"template generate_report() took {elapsed_ms:.1f} ms — expected < 50 ms"
        )

    def test_100_template_reports_under_2s(self) -> None:
        """Generating 100 consecutive template reports must complete under 2 s."""
        from aiengine.llm_report_generator import LLMReportGenerator

        gen = LLMReportGenerator(model_name="local/template")

        start = time.perf_counter()
        reports = [gen.generate_report(INSPECTION_RESULTS) for _ in range(100)]
        elapsed_s = time.perf_counter() - start

        assert len(reports) == 100, "Should have generated exactly 100 reports"
        assert all("overall_verdict" in r for r in reports), (
            "Every report must have 'overall_verdict'"
        )
        assert elapsed_s < 2.0, (
            f"100 template reports took {elapsed_s:.3f} s — expected < 2 s"
        )


# ---------------------------------------------------------------------------
# TestConcurrentSimulation
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.performance
class TestConcurrentSimulation:
    """Simulate concurrent in-process workloads using ThreadPoolExecutor to
    surface GIL-contention or shared-state issues without a real server."""

    def test_10_concurrent_mock_inspections(self) -> None:
        """10 concurrent mock OCR extractions must all finish within 2 s."""
        from aiengine.ocr.extractor import LabelTextExtractor

        def run_ocr(_: int) -> dict:
            extractor = LabelTextExtractor(backend="mock", ollama_model=None)
            image = _make_rgb_image()
            return extractor.extract(image)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(run_ocr, i) for i in range(10)]
            results = [f.result() for f in as_completed(futures)]
        elapsed_s = time.perf_counter() - start

        assert len(results) == 10, "All 10 concurrent tasks must return results"
        assert all("raw_text" in r for r in results), "Every result must have 'raw_text'"
        assert elapsed_s < 2.0, (
            f"10 concurrent OCR extractions took {elapsed_s:.3f} s — expected < 2 s"
        )

    def test_concurrent_report_generation(self) -> None:
        """20 concurrent template report generations must all finish within 3 s."""
        from aiengine.llm_report_generator import LLMReportGenerator

        def run_report(_: int) -> dict:
            gen = LLMReportGenerator(model_name="local/template")
            return gen.generate_report(INSPECTION_RESULTS)

        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=20) as pool:
            futures = [pool.submit(run_report, i) for i in range(20)]
            results = [f.result() for f in as_completed(futures)]
        elapsed_s = time.perf_counter() - start

        assert len(results) == 20, "All 20 concurrent tasks must return results"
        assert all("overall_verdict" in r for r in results), (
            "Every concurrent report must contain 'overall_verdict'"
        )
        assert elapsed_s < 3.0, (
            f"20 concurrent template reports took {elapsed_s:.3f} s — expected < 3 s"
        )


# ---------------------------------------------------------------------------
# TestMemoryUsage
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.performance
class TestMemoryUsage:
    """Verify that processing large images or image batches does not cause
    unbounded heap growth — guards against tensor/array retention bugs."""

    def test_processing_large_image_no_memory_explosion(self) -> None:
        """Processing a 2000x2000 image must keep peak allocation < 200 MB."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(device="cpu")
        large_image = _make_rgb_image(2000, 2000)

        tracemalloc.start()
        try:
            tensor = pipeline.process(large_image)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        peak_mb = peak / (1024 ** 2)
        assert tensor.shape == (1, 3, 224, 224), "Output shape must be (1, 3, 224, 224)"
        assert peak_mb < 200, (
            f"Peak memory for 2000x2000 process() was {peak_mb:.1f} MB — expected < 200 MB"
        )

    def test_batch_5_images_memory_bounded(self) -> None:
        """Processing a batch of 5x640x480 images must keep peak allocation < 200 MB."""
        from aiengine.preprocessing.pipeline import PreprocessingPipeline

        pipeline = PreprocessingPipeline(device="cpu")
        images = [_make_rgb_image(640, 480) for _ in range(5)]

        tracemalloc.start()
        try:
            batch = pipeline.process_batch(images)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        peak_mb = peak / (1024 ** 2)
        assert batch.shape[0] == 5, "Batch should contain 5 tensors"
        assert peak_mb < 200, (
            f"Peak memory for 5-image batch was {peak_mb:.1f} MB — expected < 200 MB"
        )
