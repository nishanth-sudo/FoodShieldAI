import uuid
from datetime import datetime
from typing import Optional

from backend.domain.entities import InspectionStatus
from backend.domain.models import InspectionModel
from backend.infrastructure.repositories import InspectionRepository
from backend.infrastructure.storage import storage
from backend.infrastructure.cache import cache


class SubmitInspectionUseCase:
    def __init__(self, inspection_repo: InspectionRepository):
        self.repo = inspection_repo

    async def execute(self, user_id: str, image_bytes: bytes, filename: str) -> InspectionModel:
        key = f"inspections/{user_id}/{uuid.uuid4()}_{filename}"
        content_type = "image/jpeg"
        if filename.lower().endswith(".png"):
            content_type = "image/png"
        elif filename.lower().endswith(".webp"):
            content_type = "image/webp"

        image_url = await storage.upload_image(key, image_bytes, content_type)

        inspection = await self.repo.create(user_id=user_id, image_url=image_url)
        await self.repo.update_status(inspection.id, InspectionStatus.PROCESSING)

        await cache.set(f"inspection:{inspection.id}", {
            "status": InspectionStatus.PROCESSING.value,
        }, ttl=600)

        return inspection


class GetInspectionUseCase:
    def __init__(self, inspection_repo: InspectionRepository):
        self.repo = inspection_repo

    async def execute(self, inspection_id: str, user_id: str) -> Optional[InspectionModel]:
        cached = await cache.get(f"inspection:{inspection_id}")
        if cached:
            pass

        inspection = await self.repo.get_by_id(inspection_id)
        if inspection is None:
            return None
        if inspection.user_id != user_id:
            return None
        return inspection


class GetInspectionHistoryUseCase:
    def __init__(self, inspection_repo: InspectionRepository):
        self.repo = inspection_repo

    async def execute(
        self, user_id: str, page: int = 1, limit: int = 20
    ) -> dict:
        inspections, total = await self.repo.list_by_user(
            user_id=user_id, page=page, limit=limit
        )
        return {
            "items": inspections,
            "total": total,
            "page": page,
            "limit": limit,
        }


class UpdateInspectionResultsUseCase:
    def __init__(self, inspection_repo: InspectionRepository):
        self.repo = inspection_repo

    async def execute(
        self,
        inspection_id: str,
        food_type: str = None,
        freshness_score: float = None,
        shelf_life_days: int = None,
        packaging_defects: list = None,
        contamination_risks: dict = None,
        ocr_data: dict = None,
        xai_heatmap_bytes: bytes = None,
        confidence_scores: dict = None,
        report: str = None,
    ) -> Optional[InspectionModel]:
        xai_heatmap_url = None
        if xai_heatmap_bytes is not None:
            key = f"xai/{inspection_id}/heatmap.jpg"
            xai_heatmap_url = await storage.upload_image(key, xai_heatmap_bytes)

        inspection = await self.repo.update_results(
            inspection_id=inspection_id,
            food_type=food_type,
            freshness_score=freshness_score,
            shelf_life_days=shelf_life_days,
            packaging_defects=packaging_defects,
            contamination_risks=contamination_risks,
            ocr_data=ocr_data,
            xai_heatmap_url=xai_heatmap_url,
            confidence_scores=confidence_scores,
            report=report,
        )
        if inspection:
            await self.repo.update_status(inspection_id, InspectionStatus.COMPLETED)
            await cache.delete(f"inspection:{inspection_id}")
        return inspection
