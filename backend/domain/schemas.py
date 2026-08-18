from datetime import datetime

from pydantic import BaseModel, EmailStr, Field

from .entities import InspectionStatus, UserRole


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    name: str = Field(min_length=1, max_length=255)


class UserResponse(BaseModel):
    id: str | None = None
    email: str
    name: str
    role: UserRole
    created_at: datetime | None = None

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class InspectionResponse(BaseModel):
    id: str
    status: InspectionStatus
    image_url: str
    image_thumbnail_url: str | None = None
    food_type: str | None = None
    freshness_score: float | None = None
    shelf_life_days: int | None = None
    packaging_defects: list | None = None
    contamination_risks: dict | None = None
    ocr_data: dict | None = None
    xai_heatmap_url: str | None = None
    confidence_scores: dict | None = None
    report: str | None = None
    created_at: datetime | None = None
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class InspectionListResponse(BaseModel):
    items: list[InspectionResponse]
    total: int
    page: int
    limit: int


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
