from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from backend.domain.models import UserModel
from backend.domain.schemas import UserResponse, InspectionResponse
from backend.core.dependencies import get_admin_user, get_user_repo, get_inspection_repo
from backend.infrastructure.repositories import UserRepository, InspectionRepository
from backend.infrastructure.database import check_db_health
from backend.services.monitoring import metrics
from backend.services.retraining import retraining_pipeline

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


@router.get("/metrics")
async def get_metrics(admin: UserModel = Depends(get_admin_user)):
    return metrics.get_stats()


class RetrainRequest(BaseModel):
    reason: str


@router.post("/retrain")
async def trigger_retrain(
    request: RetrainRequest,
    admin: UserModel = Depends(get_admin_user)
):
    job = retraining_pipeline.trigger_retraining(request.reason)
    return {"job_id": job.job_id, "status": job.status}


@router.get("/retrain/{job_id}")
async def get_retrain_status(
    job_id: str,
    admin: UserModel = Depends(get_admin_user)
):
    job = retraining_pipeline.get_job_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
