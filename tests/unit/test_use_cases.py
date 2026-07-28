import pytest
from unittest.mock import AsyncMock

from backend.domain.models import InspectionModel
from backend.domain.entities import InspectionStatus
from backend.core.use_cases import (
    SubmitInspectionUseCase,
    GetInspectionUseCase,
    GetInspectionHistoryUseCase,
    UpdateInspectionResultsUseCase,
)


class TestSubmitInspectionUseCase:
    async def test_submit_success(self, mock_inspection_repo, mock_storage, mock_cache):
        mock_inspection_repo.create = AsyncMock(return_value=InspectionModel(
            id="i1", user_id="u1", image_url="http://test.com/img.jpg",
            status=InspectionStatus.PENDING
        ))
        mock_inspection_repo.update_status = AsyncMock()

        use_case = SubmitInspectionUseCase(mock_inspection_repo)
        result = await use_case.execute(
            user_id="u1",
            image_bytes=b"fake-image-data",
            filename="apple.jpg",
        )

        assert result.id == "i1"
        mock_inspection_repo.create.assert_awaited_once()
        mock_inspection_repo.update_status.assert_awaited_with(
            "i1", InspectionStatus.PROCESSING
        )
        mock_storage.upload_image.assert_awaited_once()

    async def test_submit_png_image(self, mock_inspection_repo, mock_storage, mock_cache):
        mock_inspection_repo.create = AsyncMock(return_value=InspectionModel(
            id="i1", user_id="u1", image_url="http://test.com/img.png",
            status=InspectionStatus.PENDING
        ))
        mock_inspection_repo.update_status = AsyncMock()

        use_case = SubmitInspectionUseCase(mock_inspection_repo)
        result = await use_case.execute(
            user_id="u1", image_bytes=b"png-data", filename="photo.png"
        )

        assert result.id == "i1"
        mock_storage.upload_image.assert_awaited_once()


class TestGetInspectionUseCase:
    async def test_get_own_inspection_success(self, mock_inspection_repo, mock_cache):
        inspection = InspectionModel(
            id="i1", user_id="u1", image_url="http://test.com/img.jpg",
            status=InspectionStatus.COMPLETED, food_type="Apple",
        )
        mock_inspection_repo.get_by_id = AsyncMock(return_value=inspection)

        use_case = GetInspectionUseCase(mock_inspection_repo)
        result = await use_case.execute(inspection_id="i1", user_id="u1")
        assert result is not None
        assert result.food_type == "Apple"

    async def test_get_other_users_inspection_returns_none(self, mock_inspection_repo, mock_cache):
        inspection = InspectionModel(
            id="i1", user_id="u2", image_url="http://test.com/img.jpg",
            status=InspectionStatus.COMPLETED,
        )
        mock_inspection_repo.get_by_id = AsyncMock(return_value=inspection)

        use_case = GetInspectionUseCase(mock_inspection_repo)
        result = await use_case.execute(inspection_id="i1", user_id="u1")
        assert result is None

    async def test_get_nonexistent_inspection(self, mock_inspection_repo, mock_cache):
        mock_inspection_repo.get_by_id = AsyncMock(return_value=None)

        use_case = GetInspectionUseCase(mock_inspection_repo)
        result = await use_case.execute(inspection_id="invalid", user_id="u1")
        assert result is None


class TestGetInspectionHistoryUseCase:
    async def test_history_pagination(self, mock_inspection_repo):
        mock_inspection_repo.list_by_user = AsyncMock(
            return_value=([InspectionModel(id="i1", user_id="u1", image_url="http://test.com/img.jpg",
                                           status=InspectionStatus.COMPLETED)], 1)
        )

        use_case = GetInspectionHistoryUseCase(mock_inspection_repo)
        result = await use_case.execute(user_id="u1", page=1, limit=20)

        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert result["page"] == 1
        assert result["limit"] == 20

    async def test_history_empty(self, mock_inspection_repo):
        mock_inspection_repo.list_by_user = AsyncMock(return_value=([], 0))

        use_case = GetInspectionHistoryUseCase(mock_inspection_repo)
        result = await use_case.execute(user_id="u1", page=1, limit=20)

        assert result["total"] == 0
        assert len(result["items"]) == 0


class TestUpdateInspectionResultsUseCase:
    async def test_update_results_success(self, mock_inspection_repo, mock_cache):
        inspection = InspectionModel(
            id="i1", user_id="u1", image_url="http://test.com/img.jpg",
            status=InspectionStatus.PROCESSING,
        )
        mock_inspection_repo.update_results = AsyncMock(return_value=inspection)
        mock_inspection_repo.update_status = AsyncMock()

        use_case = UpdateInspectionResultsUseCase(mock_inspection_repo)
        result = await use_case.execute(
            inspection_id="i1",
            food_type="Banana",
            freshness_score=92.5,
            shelf_life_days=7,
            report="Fresh banana detected.",
        )

        assert result is not None
        mock_inspection_repo.update_results.assert_awaited_once()
        mock_inspection_repo.update_status.assert_awaited_with(
            "i1", InspectionStatus.COMPLETED
        )
