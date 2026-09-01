# ruff: noqa: E501
"""
AI Engine API routes for FoodShieldAI.

Provides direct access to the AI pipeline components:
    /ai/predict  — Quick CV prediction (food type + spoilage)
    /ai/explain  — XAI explanations (Grad-CAM + SHAP + counterfactuals)
    /ai/inspect  — Full multi-model inspection pipeline
    /ai/report   — LLM-generated report from structured evidence
    /ai/health   — AI engine component health check
"""

import io
import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi import status as http_status
from PIL import Image
from pydantic import BaseModel, Field

from backend.core.rate_limit import limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["ai-engine"])

# ---------------------------------------------------------------------------
# Lazy orchestrator singleton
# ---------------------------------------------------------------------------

_orchestrator = None


def _get_orchestrator():
    """Lazily initialise the AI orchestrator on first request."""
    global _orchestrator  # noqa: PLW0603
    if _orchestrator is None:
        from aiengine.inference_orchestrator import AIInferenceOrchestrator

        _orchestrator = AIInferenceOrchestrator()
        logger.info("AIInferenceOrchestrator initialised for /ai routes")
    return _orchestrator


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class EnvironmentalData(BaseModel):
    """Optional environmental context for XAI and shelf-life analysis."""

    temperature: float | None = Field(None, description="Storage temperature in °C")
    humidity: float | None = Field(None, description="Relative humidity %")
    storage_duration: float | None = Field(None, description="Storage duration in days")
    packaging_type: str | None = Field(None, description="Packaging type (e.g. plastic, glass, vacuum)")


class PredictResponse(BaseModel):
    """Quick prediction result — food classification + spoilage."""

    food_type: str
    freshness_score: float
    spoilage: dict
    confidence_scores: dict
    image_quality: dict


class ExplainResponse(BaseModel):
    """XAI explanation result — Grad-CAM, SHAP values, counterfactuals."""

    gradcam: dict | None = None
    shap_values: dict | None = None
    counterfactuals: list[dict] | None = None
    explanation: str
    risk_level: str
    risk_factors: list[dict] | None = None


class InspectResponse(BaseModel):
    """Full inspection pipeline result."""

    food_type: str
    freshness_score: float
    spoilage: dict
    packaging_defects: list | None = None
    contamination_risks: dict | None = None
    shelf_life: dict | None = None
    ocr_data: dict | None = None
    xai: dict | None = None
    vlm_augmentation: dict | None = None
    report: dict | None = None
    confidence_scores: dict | None = None
    image_quality: dict | None = None


class ReportRequest(BaseModel):
    """Pre-computed evidence for LLM report generation."""

    food_type: str
    spoilage_probability: float
    temperature: float | None = None
    humidity: float | None = None
    shelf_life_days: float | None = None
    xai_evidence: dict | None = None
    counterfactuals: list[dict] | None = None


class ReportResponse(BaseModel):
    """LLM-generated inspection report."""

    report: dict
    summary: str | None = None


# ---------------------------------------------------------------------------
# Helper — open uploaded image
# ---------------------------------------------------------------------------


def _open_image(contents: bytes) -> Image.Image:
    """Open and validate uploaded image bytes."""
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        return img
    except Exception as exc:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Quick food prediction",
    description="Run fast food classification + spoilage detection. No XAI, no LLM report.",
)
@limiter.limit("30/minute")
async def predict(
    request: Request,
    file: UploadFile = File(..., description="Food image to analyse"),
) -> PredictResponse:
    contents = await file.read()
    image = _open_image(contents)

    orchestrator = _get_orchestrator()
    result = orchestrator.run_fast_inspection(image)

    food_cls = result.get("food_classification", {})
    confidence_scores = {}
    if food_cls.get("confidence_scores"):
        confidence_scores["food_classification"] = food_cls["confidence_scores"][0]["confidence"]
    if result.get("spoilage", {}).get("spoilage_score") is not None:
        confidence_scores["spoilage_detection"] = result["spoilage"]["spoilage_score"]

    return PredictResponse(
        food_type=result["food_type"],
        freshness_score=result["freshness_score"],
        spoilage=result["spoilage"],
        confidence_scores=confidence_scores,
        image_quality=result["image_quality"],
    )


@router.post(
    "/explain",
    response_model=ExplainResponse,
    summary="XAI explanation",
    description=(
        "Generate Grad-CAM heatmap, SHAP feature contributions, and "
        "counterfactual scenarios. Provide environmental data for SHAP/counterfactuals."
    ),
)
@limiter.limit("20/minute")
async def explain(
    request: Request,
    file: UploadFile = File(..., description="Food image to explain"),
    temperature: float = Form(default=None, description="Storage temperature °C"),
    humidity: float = Form(default=None, description="Relative humidity %"),
    storage_duration: float = Form(default=None, description="Storage duration in days"),
    packaging_type: str = Form(default=None, description="Packaging type"),
) -> ExplainResponse:
    contents = await file.read()
    image = _open_image(contents)

    env_data = None
    if any(v is not None for v in [temperature, humidity, storage_duration, packaging_type]):
        env_data = {
            "temperature": temperature,
            "humidity": humidity,
            "storage_duration": storage_duration,
            "packaging_type": packaging_type,
        }

    orchestrator = _get_orchestrator()
    result = orchestrator.run_explanation(image, environmental_data=env_data)

    return ExplainResponse(
        gradcam=result.get("gradcam"),
        shap_values=result.get("shap_values"),
        counterfactuals=result.get("counterfactuals"),
        explanation=result.get("explanation", ""),
        risk_level=result.get("risk_level", "unknown"),
        risk_factors=result.get("risk_factors"),
    )


@router.post(
    "/inspect",
    response_model=InspectResponse,
    summary="Full inspection",
    description="Run the complete multi-model inspection pipeline with CV, XAI, VLM, and LLM report.",
)
@limiter.limit("10/minute")
async def inspect(
    request: Request,
    file: UploadFile = File(..., description="Food image to inspect"),
    temperature: float = Form(default=None, description="Storage temperature °C"),
    humidity: float = Form(default=None, description="Relative humidity %"),
    storage_duration: float = Form(default=None, description="Storage duration in days"),
    packaging_type: str = Form(default=None, description="Packaging type"),
) -> InspectResponse:
    contents = await file.read()
    image = _open_image(contents)

    env_data = None
    if any(v is not None for v in [temperature, humidity, storage_duration, packaging_type]):
        env_data = {
            "temperature": temperature,
            "humidity": humidity,
            "storage_duration": storage_duration,
            "packaging_type": packaging_type,
        }

    orchestrator = _get_orchestrator()
    result = orchestrator.run_full_inspection(image, environmental_data=env_data)

    return InspectResponse(
        food_type=result["food_type"],
        freshness_score=result["freshness_score"],
        spoilage=result["spoilage"],
        packaging_defects=result.get("packaging_defects"),
        contamination_risks=result.get("contamination_risks"),
        shelf_life=result.get("shelf_life"),
        ocr_data=result.get("ocr_data"),
        xai=result.get("xai"),
        vlm_augmentation=result.get("vlm_augmentation"),
        report=result.get("report"),
        confidence_scores=result.get("confidence_scores"),
        image_quality=result.get("image_quality"),
    )


@router.post(
    "/report",
    response_model=ReportResponse,
    summary="Generate LLM report",
    description="Generate an LLM inspection report from pre-computed structured evidence.",
)
@limiter.limit("15/minute")
async def report(
    request: Request,
    payload: ReportRequest,
) -> ReportResponse:
    orchestrator = _get_orchestrator()
    evidence = payload.model_dump(exclude_none=True)
    result = orchestrator.generate_report_from_evidence(evidence)

    return ReportResponse(
        report=result,
        summary=result.get("executive_summary"),
    )


@router.get(
    "/health",
    summary="AI engine health",
    description="Check health status of all AI engine components.",
)
async def health() -> dict:
    orchestrator = _get_orchestrator()

    components = {
        "orchestrator": "healthy",
        "food_classifier": "loaded",
        "spoilage_detector": "loaded",
        "defect_detector": "loaded",
        "contamination_assessor": "loaded",
        "shelf_life_predictor": "loaded",
        "xai_explainer": "loaded",
        "shap_explainer": "loaded",
        "counterfactual_explainer": "loaded",
        "report_generator": "loaded",
    }

    # Check LLM availability
    if hasattr(orchestrator, "report_generator"):
        rg = orchestrator.report_generator
        components["llm_backend"] = rg._provider if hasattr(rg, "_provider") else "unknown"
        if hasattr(rg, "_client") and rg._client is not None:
            components["llm_status"] = "connected"
        else:
            components["llm_status"] = "fallback_template"

    # Check VLM availability
    if hasattr(orchestrator, "vlm_augmentor"):
        components["vlm_status"] = "available" if orchestrator.vlm_augmentor.is_available else "disabled"

    return {
        "status": "healthy",
        "service": "ai-engine",
        "components": components,
    }
