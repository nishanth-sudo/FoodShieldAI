from datetime import datetime
from typing import Optional
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.domain.models import UserModel, InspectionModel
from backend.domain.entities import InspectionStatus


class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, email: str, hashed_password: str, name: str, role: str) -> UserModel:
        user = UserModel(
            email=email,
            hashed_password=hashed_password,
            name=name,
            role=role,
        )
        self.session.add(user)
        await self.session.flush()
        return user

    async def get_by_id(self, user_id: str) -> Optional[UserModel]:
        result = await self.session.execute(
            select(UserModel).where(UserModel.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[UserModel]:
        result = await self.session.execute(
            select(UserModel).where(UserModel.email == email)
        )
        return result.scalar_one_or_none()

    async def list_all(self, skip: int = 0, limit: int = 20) -> list[UserModel]:
        result = await self.session.execute(
            select(UserModel).offset(skip).limit(limit)
        )
        return list(result.scalars().all())


class InspectionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, user_id: str, image_url: str) -> InspectionModel:
        inspection = InspectionModel(
            user_id=user_id,
            image_url=image_url,
            status=InspectionStatus.PENDING,
        )
        self.session.add(inspection)
        await self.session.flush()
        return inspection

    async def get_by_id(self, inspection_id: str) -> Optional[InspectionModel]:
        result = await self.session.execute(
            select(InspectionModel).where(InspectionModel.id == inspection_id)
        )
        return result.scalar_one_or_none()

    async def list_by_user(
        self, user_id: str, page: int = 1, limit: int = 20
    ) -> tuple[list[InspectionModel], int]:
        offset = (page - 1) * limit
        count_result = await self.session.execute(
            select(func.count()).select_from(InspectionModel).where(
                InspectionModel.user_id == user_id
            )
        )
        total = count_result.scalar() or 0

        result = await self.session.execute(
            select(InspectionModel)
            .where(InspectionModel.user_id == user_id)
            .order_by(InspectionModel.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all()), total

    async def update_results(
        self,
        inspection_id: str,
        food_type: str = None,
        freshness_score: float = None,
        shelf_life_days: int = None,
        packaging_defects: list = None,
        contamination_risks: dict = None,
        ocr_data: dict = None,
        xai_heatmap_url: str = None,
        confidence_scores: dict = None,
        report: str = None,
    ) -> Optional[InspectionModel]:
        inspection = await self.get_by_id(inspection_id)
        if not inspection:
            return None

        if food_type is not None:
            inspection.food_type = food_type
        if freshness_score is not None:
            inspection.freshness_score = freshness_score
        if shelf_life_days is not None:
            inspection.shelf_life_days = shelf_life_days
        if packaging_defects is not None:
            inspection.packaging_defects = packaging_defects
        if contamination_risks is not None:
            inspection.contamination_risks = contamination_risks
        if ocr_data is not None:
            inspection.ocr_data = ocr_data
        if xai_heatmap_url is not None:
            inspection.xai_heatmap_url = xai_heatmap_url
        if confidence_scores is not None:
            inspection.confidence_scores = confidence_scores
        if report is not None:
            inspection.report = report

        await self.session.flush()
        return inspection

    async def update_status(
        self, inspection_id: str, status: InspectionStatus
    ) -> Optional[InspectionModel]:
        inspection = await self.get_by_id(inspection_id)
        if not inspection:
            return None
        inspection.status = status
        if status == InspectionStatus.COMPLETED:
            inspection.completed_at = datetime.utcnow()
        await self.session.flush()
        return inspection
