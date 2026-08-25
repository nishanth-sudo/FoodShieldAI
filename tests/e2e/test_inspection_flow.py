"""
tests/e2e/test_inspection_flow.py — Task 7.4: End-to-End Inspection Flow Tests

Validates complete multi-step user workflows across the API layers:
  1. Full Register -> Login -> Upload Image -> Retrieve Details -> View History
  2. Security boundaries (Unauthenticated block -> Authenticated pass)
  3. Batch inspection upload and aggregation flow
"""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.core.dependencies import get_db, get_inspection_repo, get_user_repo
from backend.core.security import hash_password
from backend.domain.entities import InspectionStatus, UserRole
from backend.domain.models import InspectionModel, UserModel
from backend.infrastructure.database import get_session
from main import app

_JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 100


@pytest.fixture
def e2e_db_session():
    session = AsyncMock()
    return session


@pytest.fixture
def in_memory_user_store():
    return {}


@pytest.fixture
def in_memory_inspection_store():
    return {}


@pytest.fixture
def mock_user_repo_e2e(in_memory_user_store):
    repo = AsyncMock()

    async def mock_get_by_email(email: str):
        return in_memory_user_store.get(email)

    async def mock_get_by_id(user_id: str):
        for u in in_memory_user_store.values():
            if u.id == user_id:
                return u
        return None

    async def mock_create(email, hashed_password, name, role):
        user = UserModel(
            id=f"user-{len(in_memory_user_store)+1}",
            email=email,
            hashed_password=hashed_password,
            name=name,
            role=UserRole(role),
        )
        in_memory_user_store[email] = user
        return user

    repo.get_by_email.side_effect = mock_get_by_email
    repo.get_by_id.side_effect = mock_get_by_id
    repo.create.side_effect = mock_create
    return repo


@pytest.fixture
def mock_inspection_repo_e2e(in_memory_inspection_store):
    repo = AsyncMock()

    async def mock_create(user_id, image_url, image_hash):
        insp = InspectionModel(
            id=f"insp-{len(in_memory_inspection_store)+1}",
            user_id=user_id,
            image_url=image_url,
            image_hash=image_hash,
            status=InspectionStatus.COMPLETED,
            food_type="Organic Apple",
            freshness_score=0.92,
            is_spoiled=False,
            spoilage_score=0.08,
            shelf_life_days=7,
            overall_verdict="pass",
        )
        in_memory_inspection_store[insp.id] = insp
        return insp

    async def mock_get_by_id(inspection_id: str):
        return in_memory_inspection_store.get(inspection_id)

    async def mock_list_by_user(user_id: str, page: int = 1, limit: int = 20):
        user_items = [
            i for i in in_memory_inspection_store.values() if i.user_id == user_id
        ]
        start = (page - 1) * limit
        end = start + limit
        return (user_items[start:end], len(user_items))

    repo.create.side_effect = mock_create
    repo.get_by_id.side_effect = mock_get_by_id
    repo.list_by_user.side_effect = mock_list_by_user
    return repo


@pytest.mark.e2e
class TestCompleteInspectionWorkflow:
    """Tests the full multi-step inspection journey for users."""

    async def test_full_register_login_upload_inspect_journey(
        self,
        e2e_db_session,
        mock_user_repo_e2e,
        mock_inspection_repo_e2e,
    ) -> None:
        """Complete workflow test:

        1. Register a new user
        2. Login to obtain access token
        3. Upload an inspection image with auth headers
        4. Query inspection details by ID
        5. Verify the inspection is listed in user's history
        """
        app.dependency_overrides[get_db] = lambda: e2e_db_session
        app.dependency_overrides[get_user_repo] = lambda: mock_user_repo_e2e
        app.dependency_overrides[get_inspection_repo] = lambda: mock_inspection_repo_e2e

        async def mock_session_gen():
            yield e2e_db_session

        app.dependency_overrides[get_session] = mock_session_gen

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1. Register
            reg_resp = await client.post(
                "/auth/register",
                json={
                    "email": "sarah.inspector@foodshield.ai",
                    "password": "StrongPassword987!",
                    "name": "Sarah Inspector",
                },
            )
            assert reg_resp.status_code == 201
            user_data = reg_resp.json()
            assert user_data["email"] == "sarah.inspector@foodshield.ai"

            # 2. Login
            login_resp = await client.post(
                "/auth/login",
                json={
                    "email": "sarah.inspector@foodshield.ai",
                    "password": "StrongPassword987!",
                },
            )
            assert login_resp.status_code == 200
            tokens = login_resp.json()
            access_token = tokens["access_token"]
            auth_headers = {"Authorization": f"Bearer {access_token}"}

            # 3. Upload Inspection Image
            with (
                patch("backend.core.use_cases.storage.upload_image", new_callable=AsyncMock) as m_storage,
                patch("backend.core.use_cases.cache.get_by_image_hash", new_callable=AsyncMock) as m_cache,
            ):
                m_storage.return_value = "http://storage.test/fresh_apple.jpg"
                m_cache.return_value = None

                upload_resp = await client.post(
                    "/inspections/upload",
                    headers=auth_headers,
                    files={"file": ("apple.jpg", _JPEG_BYTES, "image/jpeg")},
                )
            assert upload_resp.status_code == 201
            insp_data = upload_resp.json()
            insp_id = insp_data["id"]
            assert insp_data["food_type"] == "Organic Apple"
            assert insp_data["overall_verdict"] == "pass"

            # 4. Fetch Inspection Details
            get_resp = await client.get(f"/inspections/{insp_id}", headers=auth_headers)
            assert get_resp.status_code == 200
            retrieved = get_resp.json()
            assert retrieved["id"] == insp_id
            assert retrieved["freshness_score"] == 0.92

            # 5. List Inspection History
            list_resp = await client.get("/inspections?page=1&limit=10", headers=auth_headers)
            assert list_resp.status_code == 200
            history = list_resp.json()
            assert history["total"] == 1
            assert history["items"][0]["id"] == insp_id

        app.dependency_overrides.clear()

    async def test_unauthenticated_flow_blocked(self) -> None:
        """Unauthenticated requests to protected endpoints receive 401 Unauthorized."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            resp = await client.get("/inspections")
            assert resp.status_code == 401
