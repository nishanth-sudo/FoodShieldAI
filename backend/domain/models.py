import uuid
from enum import Enum as PyEnum

from sqlalchemy import JSON, Column, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship

from backend.core.time import utc_now

from .entities import InspectionStatus, UserRole


class Base(DeclarativeBase):
    pass


def generate_uuid() -> str:
    return str(uuid.uuid4())


def _enum_values(enum_class: type[PyEnum]) -> list[str]:
    """Persist enum member values (e.g. "consumer") instead of member names
    ("CONSUMER"), matching the lowercase values of the DB enum types."""
    return [member.value for member in enum_class]


class UserModel(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    role = Column(
        Enum(UserRole, values_callable=_enum_values),
        default=UserRole.CONSUMER,
        nullable=False,
    )
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    inspections = relationship("InspectionModel", back_populates="user")


class InspectionModel(Base):
    __tablename__ = "inspections"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    image_url = Column(String(512), nullable=False)
    image_thumbnail_url = Column(String(512))
    status = Column(
        Enum(InspectionStatus, values_callable=_enum_values),
        default=InspectionStatus.PENDING,
        nullable=False,
    )

    food_type = Column(String(255))
    freshness_score = Column(Float)
    shelf_life_days = Column(Integer)
    packaging_defects = Column(JSON)
    contamination_risks = Column(JSON)
    ocr_data = Column(JSON)
    xai_heatmap_url = Column(String(512))
    confidence_scores = Column(JSON)
    report = Column(Text)

    created_at = Column(DateTime(timezone=True), default=utc_now)
    completed_at = Column(DateTime(timezone=True))

    user = relationship("UserModel", back_populates="inspections")
