from fastapi import APIRouter, Depends, HTTPException, status

from backend.domain.models import UserModel
from backend.domain.schemas import UserResponse, InspectionResponse
from backend.core.dependencies import get_admin_user, get_user_repo, get_inspection_repo
from backend.infrastructure.repositories import UserRepository, InspectionRepository
from backend.infrastructure.database import check_db_health

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    admin: UserModel = Depends(get_admin_user),
    user_repo: UserRepository = Depends(get_user_repo),
):
    users = await user_repo.list_all()
    return users


@router.get("/inspections", response_model=list[InspectionResponse])
async def list_all_inspections(
    admin: UserModel = Depends(get_admin_user),
    inspection_repo: InspectionRepository = Depends(get_inspection_repo),
):
    inspections, _ = await inspection_repo.list_by_user(
        user_id=admin.id, page=1, limit=100
    )
    return inspections


@router.get("/health")
async def system_health():
    db_healthy = await check_db_health()
    return {
        "status": "healthy" if db_healthy else "degraded",
        "database": "connected" if db_healthy else "disconnected",
        "service": "foodshield-admin",
    }
