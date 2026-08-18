from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from backend.core.time import utc_now


class UserRole(StrEnum):
    CONSUMER = "consumer"
    QA_INSPECTOR = "qa_inspector"
    ADMIN = "admin"
    GOVERNMENT = "government"


class InspectionStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class User:
    id: str
    email: str
    name: str
    role: UserRole
    is_active: bool = True
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class FoodInspection:
    id: str
    user_id: str
    image_url: str
    image_thumbnail_url: str | None = None
    status: InspectionStatus = InspectionStatus.PENDING
    food_type: str | None = None
    freshness_score: float | None = None
    packaging_defects: list | None = None
    contamination_risks: dict | None = None
    shelf_life_days: int | None = None
    ocr_data: dict | None = None
    xai_heatmap_url: str | None = None
    confidence_scores: dict | None = None
    report: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None


@dataclass
class InspectionResult:
    food_type: str
    freshness_score: float
    shelf_life_days: int
    packaging_defects: list
    contamination_risks: dict
    ocr_data: dict
    xai_heatmap_url: str
    confidence_scores: dict
    report: str
