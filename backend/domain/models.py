import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, Enum, ForeignKey, Text, JSON
from sqlalchemy.orm import DeclarativeBase, relationship
from .entities import UserRole, InspectionStatus


class Base(DeclarativeBase):
    pass


def generate_uuid() -> str:
    return str(uuid.uuid4())


class UserModel(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.CONSUMER, nullable=False)
    is_active = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    inspections = relationship("InspectionModel", back_populates="user")


class InspectionModel(Base):
    __tablename__ = "inspections"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    image_url = Column(String(512), nullable=False)
    image_thumbnail_url = Column(String(512))
    status = Column(Enum(InspectionStatus), default=InspectionStatus.PENDING, nullable=False)

    food_type = Column(String(255))
    freshness_score = Column(Float)
    shelf_life_days = Column(Integer)
    packaging_defects = Column(JSON)
    contamination_risks = Column(JSON)
    ocr_data = Column(JSON)
    xai_heatmap_url = Column(String(512))
    confidence_scores = Column(JSON)
    report = Column(Text)

    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

    user = relationship("UserModel", back_populates="inspections")
