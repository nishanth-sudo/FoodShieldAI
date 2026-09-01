from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Literal

from pydantic import BaseModel, Field as PydanticField


class EnvironmentalData(BaseModel):
    """Environmental conditions at the time of inspection."""

    temperature: float = PydanticField(
        ..., ge=-50, le=100, description="Ambient temperature in °C"
    )
    humidity: float = PydanticField(
        ..., ge=0, le=100, description="Relative humidity percentage (0–100)"
    )
    storage_duration: float = PydanticField(
        ..., ge=0, description="Storage duration in days (must be non-negative)"
    )
    packaging_type: Optional[str] = None
    food_type: Optional[str] = None


@dataclass(frozen=True)
class ModelVersion:
    """Identifies a specific version of an ML model used in the pipeline."""

    model_name: str
    model_version: str
    model_hash: str | None
    model_framework: str
    artifact_uri: str | None


@dataclass(frozen=True)
class InspectionLineage:
    """Audit record linking an inspection result to the exact models and prompt used."""

    inspection_id: str
    image_hash: str
    model_versions: dict[str, ModelVersion]
    prompt_version: str | None
    llm_provider: str | None
    llm_model: str | None
    timestamp: str


@dataclass(frozen=True)
class FoodClassification:
    """Result of the food classification model."""
    food_type: str
    confidence: float
    class_id: int

@dataclass(frozen=True)
class SpoilageAssessment:
    """Assessment of food spoilage risk."""
    is_spoiled: bool
    spoilage_score: float
    freshness_score: float
    severity: float
    severity_label: Literal["low", "medium", "high"]

@dataclass(frozen=True)
class PackagingAssessment:
    """Analysis of packaging defects."""
    defects: List[PackagingDefect]

@dataclass(frozen=True)
class PackagingDefect:
    """A specific defect found in the packaging."""
    defect_type: str
    confidence: float
    bbox: List[float]
    class_id: int

@dataclass(frozen=True)
class ContaminationAssessment:
    """Analysis of potential contamination risks."""
    overall_risk_level: Literal["low", "medium", "high"]
    max_risk_score: float
    detected_risks: List[ContaminationRisk]

@dataclass(frozen=True)
class ContaminationRisk:
    """A specific contamination risk detected."""
    category: str
    confidence: float
    severity: str

@dataclass(frozen=True)
class ShelfLifePrediction:
    """Predicted remaining shelf life."""
    estimated_days_remaining: int
    estimated_expiry_date: str
    freshness_category: str
    confidence: float

@dataclass(frozen=True)
class OCRResult:
    """Extracted text from product labels."""
    raw_text: str
    parsed: dict[str, Any]
    backend: str

@dataclass(frozen=True)
class XAIExplanation:
    """Explainability results for a prediction."""
    method: str
    heatmap_array: List[List[int]]
    heatmap_overlay: Optional[List[int]]
    target_class: int
    regions_of_interest: List[dict]

@dataclass(frozen=True)
class CounterfactualResult:
    """A counterfactual scenario for risk reduction."""
    feature: str
    original: float
    counterfactual: float
    prediction_before: float
    prediction_after: float
    cost: float
    feasible: bool

@dataclass(frozen=True)
class InspectionResult:
    """Complete result of a food inspection pipeline."""
    food_type: str
    freshness_score: float
    spoilage: SpoilageAssessment
    packaging: PackagingAssessment
    contamination: ContaminationAssessment
    shelf_life: ShelfLifePrediction
    ocr: OCRResult
    xai: Optional[XAIExplanation] = None
    counterfactuals: Optional[List[CounterfactualResult]] = None
    confidence_scores: dict[str, float] = field(default_factory=dict)
    image_quality: dict = field(default_factory=dict)
