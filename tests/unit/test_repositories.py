from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import select

from backend.domain.models import UserModel, InspectionModel
from backend.domain.entities import UserRole, InspectionStatus
from backend.infrastructure.repositories import UserRepository, InspectionRepository


class TestUserRepository:
    async def test_create(self, mock_db_session):
        repo = UserRepository(mock_db_session)
        result = await repo.create("test@test.com", "hashed123", "Test User", "consumer")
        mock_db_session.add.assert_called_once()
        mock_db_session.flush.assert_awaited_once()

    async def test_get_by_id_found(self, mock_db_session):
        user = UserModel(id="u1", email="test@test.com", name="Test", role=UserRole.CONSUMER)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        repo = UserRepository(mock_db_session)
        result = await repo.get_by_id("u1")
        assert result is not None
        assert result.id == "u1"
        assert result.email == "test@test.com"

    async def test_get_by_id_not_found(self, mock_db_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        repo = UserRepository(mock_db_session)
        result = await repo.get_by_id("nonexistent")
        assert result is None

    async def test_get_by_email(self, mock_db_session):
        user = UserModel(id="u1", email="test@test.com", name="Test", role=UserRole.CONSUMER)
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        repo = UserRepository(mock_db_session)
        result = await repo.get_by_email("test@test.com")
        assert result is not None
        assert result.email == "test@test.com"


class TestInspectionRepository:
    async def test_create(self, mock_db_session):
        repo = InspectionRepository(mock_db_session)
        result = await repo.create(user_id="u1", image_url="http://test.com/img.jpg")
        mock_db_session.add.assert_called_once()
        mock_db_session.flush.assert_awaited_once()

    async def test_get_by_id(self, mock_db_session):
        inspection = InspectionModel(
            id="i1", user_id="u1", image_url="http://test.com/img.jpg",
            status=InspectionStatus.PENDING
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = inspection
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        repo = InspectionRepository(mock_db_session)
        result = await repo.get_by_id("i1")
        assert result is not None
        assert result.id == "i1"
        assert result.status == InspectionStatus.PENDING

    async def test_get_by_id_not_found(self, mock_db_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        repo = InspectionRepository(mock_db_session)
        result = await repo.get_by_id("nonexistent")
        assert result is None

    async def test_update_status(self, mock_db_session):
        inspection = InspectionModel(
            id="i1", user_id="u1", image_url="http://test.com/img.jpg",
            status=InspectionStatus.PENDING
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = inspection
        mock_db_session.execute = AsyncMock(return_value=mock_result)

        repo = InspectionRepository(mock_db_session)
        result = await repo.update_status("i1", InspectionStatus.COMPLETED)
        assert result is not None
        assert result.status == InspectionStatus.COMPLETED
