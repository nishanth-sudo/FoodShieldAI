"""
Shared fixtures for API integration tests.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from backend.core.dependencies import (
    get_admin_user,
    get_current_user,
    get_db,
    get_inspection_repo,
    get_user_repo,
)
from backend.core.security import create_access_token, hash_password
from backend.domain.entities import InspectionStatus, UserRole
from backend.domain.models import InspectionModel, UserModel
from backend.infrastructure.database import get_session
from main import app


@pytest.fixture
def mock_db_session():
    """Mock async SQLAlchemy session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def integration_user():
    """Standard authenticated consumer user."""
    return UserModel(
        id="user-integration-123",
        email="consumer@example.com",
        name="Consumer User",
        role=UserRole.CONSUMER,
        hashed_password=hash_password("ValidPassword123!"),
    )


@pytest.fixture
def integration_admin():
    """Admin user."""
    return UserModel(
        id="admin-integration-456",
        email="admin@example.com",
        name="Admin User",
        role=UserRole.ADMIN,
        hashed_password=hash_password("AdminPassword123!"),
    )


@pytest.fixture
def auth_headers(integration_user):
    """Valid Bearer auth header for integration_user."""
    token = create_access_token(user_id=integration_user.id, role=integration_user.role.value)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers(integration_admin):
    """Valid Bearer auth header for integration_admin."""
    token = create_access_token(user_id=integration_admin.id, role=integration_admin.role.value)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def mock_user_repository():
    """Mock UserRepository."""
    repo = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.get_by_email = AsyncMock()
    repo.create = AsyncMock()
    repo.list_all = AsyncMock()
    return repo


@pytest.fixture
def mock_inspection_repository():
    """Mock InspectionRepository."""
    repo = AsyncMock()
    repo.create = AsyncMock()
    repo.get_by_id = AsyncMock()
    repo.list_by_user = AsyncMock()
    repo.update_results = AsyncMock()
    repo.update_status = AsyncMock()
    return repo


@pytest.fixture
async def async_client(
    integration_user,
    integration_admin,
    mock_db_session,
    mock_user_repository,
    mock_inspection_repository,
):
    """AsyncClient configured with dependency overrides."""
    app.dependency_overrides[get_db] = lambda: mock_db_session
    app.dependency_overrides[get_current_user] = lambda: integration_user
    app.dependency_overrides[get_admin_user] = lambda: integration_admin
    app.dependency_overrides[get_user_repo] = lambda: mock_user_repository
    app.dependency_overrides[get_inspection_repo] = lambda: mock_inspection_repository

    async def mock_session_gen():
        yield mock_db_session

    app.dependency_overrides[get_session] = mock_session_gen

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()
