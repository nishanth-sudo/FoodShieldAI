from datetime import UTC, datetime

from backend.domain.entities import FoodInspection, InspectionStatus, User, UserRole
from backend.domain.schemas import InspectionResponse, LoginRequest, UserCreate


class TestUserRole:
    def test_values(self):
        assert UserRole.CONSUMER.value == "consumer"
        assert UserRole.QA_INSPECTOR.value == "qa_inspector"
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.GOVERNMENT.value == "government"


class TestInspectionStatus:
    def test_values(self):
        assert InspectionStatus.PENDING.value == "pending"
        assert InspectionStatus.PROCESSING.value == "processing"
        assert InspectionStatus.COMPLETED.value == "completed"
        assert InspectionStatus.FAILED.value == "failed"


class TestUserEntity:
    def test_create_user(self):
        user = User(
            id="u1",
            email="test@example.com",
            name="Test User",
            role=UserRole.CONSUMER,
        )
        assert user.id == "u1"
        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.role == UserRole.CONSUMER
        assert user.is_active is True
        assert isinstance(user.created_at, datetime)

    def test_user_default_role(self):
        user = User(id="u2", email="qa@test.com", name="QA", role=UserRole.QA_INSPECTOR)
        assert user.role == UserRole.QA_INSPECTOR


class TestFoodInspectionEntity:
    def test_create_inspection(self):
        inspection = FoodInspection(
            id="i1",
            user_id="u1",
            image_url="http://storage.test/img.jpg",
        )
        assert inspection.id == "i1"
        assert inspection.user_id == "u1"
        assert inspection.status == InspectionStatus.PENDING
        assert inspection.food_type is None
        assert inspection.freshness_score is None

    def test_inspection_default_status(self):
        inspection = FoodInspection(
            id="i2", user_id="u2", image_url="http://test.com/img.jpg"
        )
        assert inspection.status == InspectionStatus.PENDING

    def test_inspection_with_all_fields(self):
        now = datetime.now(UTC)
        inspection = FoodInspection(
            id="i3",
            user_id="u1",
            image_url="http://test.com/img.jpg",
            status=InspectionStatus.COMPLETED,
            food_type="Apple",
            freshness_score=85.5,
            shelf_life_days=14,
            packaging_defects=[{"type": "dent", "confidence": 0.9}],
            contamination_risks={"biological": 0.1, "chemical": 0.0},
            ocr_data={"product": "Apple Juice", "expiry": "2026-12-31"},
            xai_heatmap_url="http://test.com/heatmap.jpg",
            confidence_scores={"food_type": 0.95},
            report="Fresh apple detected.",
            completed_at=now,
        )
        assert inspection.food_type == "Apple"
        assert inspection.freshness_score == 85.5
        assert len(inspection.packaging_defects) == 1
        assert inspection.ocr_data["product"] == "Apple Juice"


class TestSchemas:
    def test_user_create_valid(self):
        data = UserCreate(email="test@example.com", password="password123", name="Test")
        assert data.email == "test@example.com"

    def test_login_request(self):
        data = LoginRequest(email="a@b.com", password="pass")
        assert data.email == "a@b.com"

    def test_inspection_response(self):
        data = InspectionResponse(
            id="i1",
            status=InspectionStatus.COMPLETED,
            image_url="http://test.com/img.jpg",
            food_type="Banana",
            freshness_score=90.0,
            created_at=datetime.now(UTC),
        )
        assert data.food_type == "Banana"
        assert data.freshness_score == 90.0
