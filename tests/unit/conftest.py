from unittest.mock import AsyncMock, MagicMock, patch
import pytest


@pytest.fixture
def mock_db_session():
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def mock_user_repo(mock_db_session):
    with patch("backend.infrastructure.repositories.UserRepository") as mock:
        repo = mock.return_value
        repo.create = AsyncMock()
        repo.get_by_id = AsyncMock()
        repo.get_by_email = AsyncMock()
        repo.list_all = AsyncMock()
        yield repo


@pytest.fixture
def mock_inspection_repo(mock_db_session):
    with patch("backend.infrastructure.repositories.InspectionRepository") as mock:
        repo = mock.return_value
        repo.create = AsyncMock()
        repo.get_by_id = AsyncMock()
        repo.list_by_user = AsyncMock()
        repo.update_results = AsyncMock()
        repo.update_status = AsyncMock()
        yield repo


@pytest.fixture
def mock_storage():
    with patch("backend.core.use_cases.storage") as m:
        m.upload_image = AsyncMock(return_value="http://storage.test/image.jpg")
        m.get_image = AsyncMock(return_value=b"image-data")
        m.delete_image = AsyncMock()
        yield m


@pytest.fixture
def mock_cache():
    with patch("backend.core.use_cases.cache") as m:
        m.get = AsyncMock(return_value=None)
        m.set = AsyncMock(return_value=True)
        m.delete = AsyncMock(return_value=True)
        m.get_by_image_hash = AsyncMock(return_value=None)
        yield m
