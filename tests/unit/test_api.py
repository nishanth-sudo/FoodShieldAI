from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import status

from backend.main import app
from backend.domain.models import UserModel
from backend.domain.entities import UserRole, InspectionStatus
from backend.core.dependencies import get_current_user, get_admin_user, get_db
from backend.infrastructure.database import get_session


@pytest.fixture
def test_user():
    return UserModel(
        id="u1",
        email="test@test.com",
        name="Test",
        role=UserRole.CONSUMER,
        hashed_password="$2b$12$dummyhash",
    )


@pytest.fixture
def admin_user():
    return UserModel(
        id="admin1",
        email="admin@test.com",
        name="Admin",
        role=UserRole.ADMIN,
        hashed_password="$2b$12$dummyhash",
    )


@pytest.fixture
def mock_session():
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def override_deps(test_user, mock_session):
    app.dependency_overrides[get_current_user] = lambda: test_user
    app.dependency_overrides[get_db] = lambda: mock_session

    async def mock_session_gen():
        yield mock_session

    app.dependency_overrides[get_session] = mock_session_gen

    yield

    app.dependency_overrides.clear()


@pytest.fixture
def override_admin_deps(admin_user, mock_session):
    app.dependency_overrides[get_current_user] = lambda: admin_user
    app.dependency_overrides[get_admin_user] = lambda: admin_user
    app.dependency_overrides[get_db] = lambda: mock_session

    async def mock_session_gen():
        yield mock_session

    app.dependency_overrides[get_session] = mock_session_gen
    yield
    app.dependency_overrides.clear()


class TestHealthEndpoint:
    async def test_health_check(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/health")
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "foodshield-ai"


class TestAuthEndpoints:
    async def test_register_new_user(self, override_deps, mock_session):
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post("/auth/register", json={
                "email": "new@test.com",
                "password": "password123",
                "name": "New User",
            })
        assert response.status_code == status.HTTP_201_CREATED

    async def test_register_duplicate_email(self, override_deps, mock_session):
        existing_user = MagicMock()
        existing_user.scalar_one_or_none.return_value = UserModel(
            id="u1", email="dup@test.com", name="Dup", role=UserRole.CONSUMER
        )
        mock_session.execute = AsyncMock(return_value=existing_user)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post("/auth/register", json={
                "email": "dup@test.com",
                "password": "password123",
                "name": "Duplicate",
            })
        assert response.status_code == status.HTTP_409_CONFLICT

    async def test_login_success(self, override_deps, mock_session):
        from backend.core.security import hash_password

        user = UserModel(
            id="u1", email="test@test.com", name="Test",
            role=UserRole.CONSUMER,
            hashed_password=hash_password("password123"),
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute = AsyncMock(return_value=mock_result)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post("/auth/login", json={
                "email": "test@test.com",
                "password": "password123",
            })
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data

    async def test_login_wrong_password(self, override_deps, mock_session):
        from backend.core.security import hash_password

        user = UserModel(
            id="u1", email="test@test.com", name="Test",
            role=UserRole.CONSUMER,
            hashed_password=hash_password("correctpass"),
        )
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = user
        mock_session.execute = AsyncMock(return_value=mock_result)

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post("/auth/login", json={
                "email": "test@test.com",
                "password": "wrongpass",
            })
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_get_me(self, override_deps):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.get("/auth/me", headers={
                "Authorization": "Bearer test-token"
            })
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["email"] == "test@test.com"


class TestInspectionEndpoints:
    async def test_upload_inspection(self, override_deps, mock_session):
        with patch("backend.api.routes.SubmitInspectionUseCase.execute") as mock_execute:
            from backend.domain.models import InspectionModel
            mock_execute.return_value = InspectionModel(
                id="i1", user_id="u1", image_url="http://test.com/img.jpg",
                status=InspectionStatus.PROCESSING,
            )

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                response = await ac.post(
                    "/inspections/upload",
                    files={"file": ("apple.jpg", b"test-image-data", "image/jpeg")},
                    headers={"Authorization": "Bearer test-token"},
                )
            assert response.status_code == status.HTTP_201_CREATED
            data = response.json()
            assert data["status"] == "processing"

    async def test_upload_unsupported_type(self, override_deps):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/inspections/upload",
                files={"file": ("doc.pdf", b"pdf-data", "application/pdf")},
                headers={"Authorization": "Bearer test-token"},
            )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    async def test_get_inspection(self, override_deps, mock_session):
        with patch("backend.api.routes.GetInspectionUseCase.execute") as mock_execute:
            from backend.domain.models import InspectionModel
            mock_execute.return_value = InspectionModel(
                id="i1", user_id="u1", image_url="http://test.com/img.jpg",
                status=InspectionStatus.COMPLETED, food_type="Apple",
            )

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                response = await ac.get(
                    "/inspections/i1",
                    headers={"Authorization": "Bearer test-token"},
                )
            assert response.status_code == status.HTTP_200_OK
            assert response.json()["food_type"] == "Apple"

    async def test_get_inspection_not_found(self, override_deps):
        with patch("backend.api.routes.GetInspectionUseCase.execute") as mock_execute:
            mock_execute.return_value = None

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                response = await ac.get(
                    "/inspections/nonexistent",
                    headers={"Authorization": "Bearer test-token"},
                )
            assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_list_inspections(self, override_deps):
        with patch("backend.api.routes.GetInspectionHistoryUseCase.execute") as mock_execute:
            mock_execute.return_value = {
                "items": [],
                "total": 0,
                "page": 1,
                "limit": 20,
            }

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as ac:
                response = await ac.get(
                    "/inspections?page=1&limit=10",
                    headers={"Authorization": "Bearer test-token"},
                )
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["total"] == 0
            assert data["page"] == 1
