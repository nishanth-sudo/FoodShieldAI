from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class UserRole(str, Enum):
    CONSUMER = "consumer"
    QA_INSPECTOR = "qa_inspector"
    ADMIN = "admin"
    GOVERNMENT = "government"


class InspectionStatus(str, Enum):
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
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FoodInspection:
    id: str
    user_id: str
    image_url: str
    image_thumbnail_url: Optional[str] = None
    status: InspectionStatus = InspectionStatus.PENDING
    food_type: Optional[str] = None
    freshness_score: Optional[float] = None
    packaging_defects: Optional[list] = None
    contamination_risks: Optional[dict] = None
    shelf_life_days: Optional[int] = None
    ocr_data: Optional[dict] = None
    xai_heatmap_url: Optional[str] = None
    confidence_scores: Optional[dict] = None
    report: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


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
