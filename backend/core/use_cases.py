import uuid
from datetime import datetime

from backend.config import settings
from backend.core.logging_config import get_logger
from backend.domain.entities import InspectionStatus
from backend.domain.models import InspectionModel
from backend.infrastructure.cache import cache, compute_image_hash
from backend.infrastructure.repositories import InspectionRepository
from backend.infrastructure.storage import storage

logger = get_logger(__name__)

class SubmitInspectionUseCase:
    def __init__(self, inspection_repo: InspectionRepository) -> None:
        self.repo = inspection_repo

    async def execute(self, user_id: str, image_bytes: bytes, filename: str) -> InspectionModel:
        image_hash = compute_image_hash(image_bytes)
        cached_result = await cache.get_by_image_hash(image_hash)

        if cached_result:
            logger.info(f"Cache hit for image hash {image_hash}")
            # Create inspection with cached result and mark COMPLETED immediately
            inspection = await self.repo.create(
                user_id=user_id, image_url=cached_result.get('image_url', 'cached')
            )
            await self.repo.update_results(
                inspection_id=inspection.id,
                food_type=cached_result.get('food_type'),
                freshness_score=cached_result.get('freshness_score'),
                shelf_life_days=cached_result.get('shelf_life_days'),
                packaging_defects=cached_result.get('packaging_defects'),
                contamination_risks=cached_result.get('contamination_risks'),
                ocr_data=cached_result.get('ocr_data'),
                xai_heatmap_url=cached_result.get('xai_heatmap_url'),
                confidence_scores=cached_result.get('confidence_scores'),
                report=cached_result.get('report'),
            )
            await self.repo.update_status(inspection.id, InspectionStatus.COMPLETED)
            return await self.repo.get_by_id(inspection.id)

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
        }, ttl=settings.cache_inspection_ttl)

        return inspection

class BatchSubmitInspectionUseCase:
    def __init__(self, inspection_repo: InspectionRepository) -> None:
        self.repo = inspection_repo

    async def execute(
        self, user_id: str, image_files: list[tuple[bytes, str]]
    ) -> list[InspectionModel]:
        logger.info(f"Processing batch of {len(image_files)} images for user {user_id}")
        inspections = []
        submit_use_case = SubmitInspectionUseCase(self.repo)
        for image_bytes, filename in image_files:
            inspection = await submit_use_case.execute(user_id, image_bytes, filename)
            inspections.append(inspection)
        return inspections


class GetInspectionUseCase:
    def __init__(self, inspection_repo: InspectionRepository) -> None:
        self.repo = inspection_repo

    async def execute(self, inspection_id: str, user_id: str) -> InspectionModel | None:
        cached = await cache.get(f"inspection:{inspection_id}")
        if (
            cached
            and cached.get("status") == InspectionStatus.COMPLETED.value
            and 'id' in cached
            and 'user_id' in cached
        ):
            if cached['user_id'] == user_id:
                logger.info(f"Returning cached inspection {inspection_id}")
                return InspectionModel(**cached)
            return None

        inspection = await self.repo.get_by_id(inspection_id)
        if inspection is None:
            return None
        if inspection.user_id != user_id:
            return None
        return inspection


class GetInspectionHistoryUseCase:
    def __init__(self, inspection_repo: InspectionRepository) -> None:
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
    def __init__(self, inspection_repo: InspectionRepository) -> None:
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
    ) -> InspectionModel | None:
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

            # Cache the full inspection
            inspection_data = inspection.__dict__.copy()
            if '_sa_instance_state' in inspection_data:
                del inspection_data['_sa_instance_state']

            # Convert enums/dates to string if necessary
            for k, v in inspection_data.items():
                if isinstance(v, datetime):
                    inspection_data[k] = v.isoformat()
                elif hasattr(v, 'value'):
                    inspection_data[k] = v.value

            await cache.set(
                f"inspection:{inspection_id}", inspection_data, ttl=settings.cache_result_ttl
            )
            logger.info(f"Updated and cached results for inspection {inspection_id}")

        return inspection
