"""
tests/integration/test_api_integration.py — Task 7.3: API Integration Tests

Integration test suite testing full HTTP request/response cycles for
all auth and inspection endpoints using FastAPI's AsyncClient.
"""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.core.security import create_access_token, hash_password
from backend.domain.entities import InspectionStatus, UserRole
from backend.domain.models import InspectionModel, UserModel
from main import app

_JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 100


# ===========================================================================
# 1. Auth Endpoint Integration Tests
# ===========================================================================


@pytest.mark.integration
class TestAuthAPIIntegration:
    """Integration tests for user registration, login, refresh, and profile endpoints."""

    async def test_register_new_user_success(self, async_client, mock_user_repository) -> None:
        """POST /auth/register with valid body creates user and returns 201."""
        mock_user_repository.get_by_email.return_value = None
        mock_user_repository.create.return_value = UserModel(
            id="new-u1",
            email="newuser@example.com",
            name="New User",
            role=UserRole.CONSUMER,
            hashed_password=hash_password("Password123!"),
        )

        resp = await async_client.post(
            "/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "Password123!",
                "name": "New User",
            },
        )

        assert resp.status_code == 201
        data = resp.json()
        assert data["email"] == "newuser@example.com"
        assert data["name"] == "New User"
        assert "hashed_password" not in data

    async def test_register_duplicate_email_returns_409(
        self, async_client, mock_user_repository
    ) -> None:
        """POST /auth/register with existing email returns 409 Conflict."""
        mock_user_repository.get_by_email.return_value = UserModel(
            id="existing-u1",
            email="existing@example.com",
            name="Existing",
            role=UserRole.CONSUMER,
            hashed_password=hash_password("Pass123!"),
        )

        resp = await async_client.post(
            "/auth/register",
            json={
                "email": "existing@example.com",
                "password": "Password123!",
                "name": "Existing",
            },
        )

        assert resp.status_code == 409
        assert "already registered" in resp.json()["detail"]

    async def test_login_valid_credentials_returns_tokens(
        self, async_client, mock_user_repository
    ) -> None:
        """POST /auth/login returns access and refresh JWT tokens."""
        mock_user_repository.get_by_email.return_value = UserModel(
            id="u-login",
            email="login@example.com",
            name="Login User",
            role=UserRole.CONSUMER,
            hashed_password=hash_password("SecretPass123!"),
        )

        resp = await async_client.post(
            "/auth/login",
            json={
                "email": "login@example.com",
                "password": "SecretPass123!",
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    async def test_login_wrong_password_returns_401(
        self, async_client, mock_user_repository
    ) -> None:
        """POST /auth/login with wrong password returns 401 Unauthorized."""
        mock_user_repository.get_by_email.return_value = UserModel(
            id="u-login",
            email="login@example.com",
            name="Login User",
            role=UserRole.CONSUMER,
            hashed_password=hash_password("SecretPass123!"),
        )

        resp = await async_client.post(
            "/auth/login",
            json={
                "email": "login@example.com",
                "password": "WrongPassword!",
            },
        )

        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]

    async def test_get_me_returns_current_user_profile(
        self, async_client, integration_user
    ) -> None:
        """GET /auth/me returns current user info."""
        resp = await async_client.get("/auth/me")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == integration_user.id
        assert data["email"] == integration_user.email


# ===========================================================================
# 2. Inspection Endpoint Integration Tests
# ===========================================================================


@pytest.mark.integration
class TestInspectionAPIIntegration:
    """Integration tests for upload, get, and list inspection endpoints."""

    async def test_upload_inspection_valid_jpeg(
        self, async_client, mock_inspection_repository, integration_user
    ) -> None:
        """POST /inspections/upload with valid JPEG creates inspection and returns 201."""
        created_inspection = InspectionModel(
            id="insp-new-123",
            user_id=integration_user.id,
            image_url="http://storage.test/img.jpg",
            image_hash="hash123",
            status=InspectionStatus.COMPLETED,
            food_type="Apple",
            freshness_score=0.9,
            is_spoiled=False,
            spoilage_score=0.1,
            shelf_life_days=5,
            overall_verdict="pass",
        )
        mock_inspection_repository.create.return_value = created_inspection

        with (
            patch("backend.core.use_cases.storage.upload_image", new_callable=AsyncMock) as m_storage,
            patch("backend.core.use_cases.cache.get_by_image_hash", new_callable=AsyncMock) as m_cache,
        ):
            m_storage.return_value = "http://storage.test/img.jpg"
            m_cache.return_value = None

            resp = await async_client.post(
                "/inspections/upload",
                files={"file": ("test.jpg", _JPEG_BYTES, "image/jpeg")},
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "insp-new-123"
        assert data["food_type"] == "Apple"

    async def test_upload_inspection_unsupported_type_returns_400(
        self, async_client
    ) -> None:
        """POST /inspections/upload with unsupported file type returns 400."""
        resp = await async_client.post(
            "/inspections/upload",
            files={"file": ("test.txt", b"plain text", "text/plain")},
        )

        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    async def test_get_inspection_by_id_exists(
        self, async_client, mock_inspection_repository, integration_user
    ) -> None:
        """GET /inspections/{id} returns inspection details when found."""
        inspection = InspectionModel(
            id="insp-get-456",
            user_id=integration_user.id,
            image_url="http://storage.test/img.jpg",
            image_hash="hash456",
            status=InspectionStatus.COMPLETED,
            food_type="Tomato",
            freshness_score=0.8,
            is_spoiled=False,
            overall_verdict="pass",
        )
        mock_inspection_repository.get_by_id.return_value = inspection

        resp = await async_client.get("/inspections/insp-get-456")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "insp-get-456"
        assert data["food_type"] == "Tomato"

    async def test_get_inspection_by_id_not_found_returns_404(
        self, async_client, mock_inspection_repository
    ) -> None:
        """GET /inspections/{id} returns 404 when inspection does not exist."""
        mock_inspection_repository.get_by_id.return_value = None

        resp = await async_client.get("/inspections/insp-nonexistent")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    async def test_list_inspections_paginated(
        self, async_client, mock_inspection_repository, integration_user
    ) -> None:
        """GET /inspections returns paginated list of user's inspections."""
        items = [
            InspectionModel(
                id=f"insp-{i}",
                user_id=integration_user.id,
                image_url=f"http://storage.test/img{i}.jpg",
                image_hash=f"hash{i}",
                status=InspectionStatus.COMPLETED,
                food_type="Apple",
                overall_verdict="pass",
            )
            for i in range(3)
        ]
        mock_inspection_repository.list_by_user.return_value = (items, 3)

        resp = await async_client.get("/inspections?page=1&limit=10")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 3
        assert data["page"] == 1
        assert len(data["items"]) == 3


# ===========================================================================
# 3. Health & System Endpoint Integration Tests
# ===========================================================================


@pytest.mark.integration
class TestHealthAPIIntegration:
    """Integration test for /health service monitoring endpoint."""

    async def test_health_check_endpoint(self) -> None:
        """GET /health returns healthy status dict."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert data["service"] == "foodshield-ai"
        assert "version" in data
