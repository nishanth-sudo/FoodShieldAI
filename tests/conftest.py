"""
Root conftest.py — shared fixtures and environment setup for all tests.
Sets env vars before any backend module is imported.
"""
import os

import pytest

# ---------------------------------------------------------------------------
# Environment setup (must happen before backend imports)
# ---------------------------------------------------------------------------
os.environ["FOODSHIELD_DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/test"
os.environ["FOODSHIELD_REDIS_URL"] = "redis://localhost:6379/1"
os.environ["FOODSHIELD_JWT_SECRET_KEY"] = "test-secret-key-at-least-32-chars-long!"
os.environ["OLLAMA_ENABLED"] = "false"  # disable Ollama in all tests by default

from backend.config import settings  # noqa: E402

settings.database_url = "postgresql+asyncpg://test:test@localhost:5432/test"
settings.redis_url = "redis://localhost:6379/1"
settings.jwt_secret_key = "test-secret-key-at-least-32-chars-long!"
settings.storage_endpoint = "http://localhost:9000"
settings.debug = False


# ---------------------------------------------------------------------------
# Shared image fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rgb_image_224():
    """224x224 RGB PIL image — standard model input size."""
    from PIL import Image
    return Image.new("RGB", (224, 224), color=(128, 64, 32))


@pytest.fixture(scope="session")
def rgb_image_640():
    """640x480 RGB PIL image — realistic photo resolution."""
    from PIL import Image
    return Image.new("RGB", (640, 480), color=(100, 150, 80))


@pytest.fixture(scope="session")
def black_image():
    """All-black image for quality check tests."""
    from PIL import Image
    return Image.new("RGB", (224, 224), color=(0, 0, 0))


@pytest.fixture(scope="session")
def white_image():
    """All-white image for quality check tests."""
    from PIL import Image
    return Image.new("RGB", (224, 224), color=(255, 255, 255))


@pytest.fixture(scope="session")
def jpeg_bytes():
    """Minimal valid JPEG magic bytes for upload tests."""
    return b"\xff\xd8\xff\xe0" + b"\x00" * 100


@pytest.fixture(scope="session")
def png_bytes():
    """Minimal valid PNG magic bytes."""
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 100


# ---------------------------------------------------------------------------
# Shared inspection data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_inspection_results():
    """Full mock inspection result dict — mirrors orchestrator output."""
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
            "detected_risks": [{"category": "biological_mold", "confidence": 0.8, "severity": "high"}],
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
