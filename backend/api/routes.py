from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Query, Request
from fastapi import status as http_status

from backend.domain.models import UserModel
from backend.domain.schemas import InspectionResponse, InspectionListResponse
from backend.core.use_cases import (
    SubmitInspectionUseCase,
    GetInspectionUseCase,
    GetInspectionHistoryUseCase,
    BatchSubmitInspectionUseCase,
)
from backend.core.dependencies import (
    get_current_user,
    get_inspection_repo,
)
from backend.infrastructure.repositories import InspectionRepository
from backend.config import settings
from backend.core.middleware import validate_image_bytes
from backend.core.logging_config import get_logger
from backend.core.rate_limit import limiter

logger = get_logger(__name__)
router = APIRouter(prefix="/inspections", tags=["inspections"])


@router.post(
    "/upload",
    response_model=InspectionResponse,
    status_code=http_status.HTTP_201_CREATED,
)
@limiter.limit("10/minute")
async def upload_inspection(
    request: Request,
    file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user),
    inspection_repo: InspectionRepository = Depends(get_inspection_repo),
):
    if file.content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type: {file.content_type}. "
                   f"Allowed: {', '.join(settings.allowed_image_types)}",
        )

    contents = await file.read()
    if len(contents) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=http_status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB",
        )

    sanitized_filename = validate_image_bytes(contents, file.filename or 'image.jpg')

    use_case = SubmitInspectionUseCase(inspection_repo)
    try:
        inspection = await use_case.execute(
            user_id=current_user.id,
            image_bytes=contents,
            filename=sanitized_filename,
        )
    except Exception as e:
        logger.error(f"Error submitting inspection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during upload")

    return inspection

@router.post(
    "/upload/batch",
    response_model=list[InspectionResponse],
    status_code=http_status.HTTP_201_CREATED,
)
@limiter.limit("10/minute")
async def batch_upload_inspections(
    request: Request,
    files: list[UploadFile] = File(...),
    current_user: UserModel = Depends(get_current_user),
    inspection_repo: InspectionRepository = Depends(get_inspection_repo),
):
    if len(files) > settings.batch_max_files:
        raise HTTPException(status_code=400, detail=f"Too many files. Max allowed: {settings.batch_max_files}")

    processed_files = []
    for file in files:
        if file.content_type not in settings.allowed_image_types:
            raise HTTPException(
                status_code=http_status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported image type: {file.content_type} in {file.filename}."
            )
        contents = await file.read()
        sanitized_filename = validate_image_bytes(contents, file.filename or 'image.jpg')
        processed_files.append((contents, sanitized_filename))

    use_case = BatchSubmitInspectionUseCase(inspection_repo)
    try:
        inspections = await use_case.execute(user_id=current_user.id, image_files=processed_files)
    except Exception as e:
        logger.error(f"Error submitting batch inspection: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during batch upload")

    return inspections


@router.get("/{inspection_id}", response_model=InspectionResponse)
async def get_inspection(
    inspection_id: str,
    current_user: UserModel = Depends(get_current_user),
    inspection_repo: InspectionRepository = Depends(get_inspection_repo),
):
    use_case = GetInspectionUseCase(inspection_repo)
    inspection = await use_case.execute(
        inspection_id=inspection_id,
        user_id=current_user.id,
    )
    if inspection is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail="Inspection not found",
        )
    return inspection


@router.get("", response_model=InspectionListResponse)
async def list_inspections(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    current_user: UserModel = Depends(get_current_user),
    inspection_repo: InspectionRepository = Depends(get_inspection_repo),
):
    use_case = GetInspectionHistoryUseCase(inspection_repo)
    result = await use_case.execute(
        user_id=current_user.id,
        page=page,
        limit=limit,
    )
    return result
