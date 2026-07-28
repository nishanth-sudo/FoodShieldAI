from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from backend.infrastructure.database import get_session
from backend.infrastructure.repositories import UserRepository, InspectionRepository
from backend.infrastructure.cache import cache
from backend.core.security import decode_token
from backend.domain.models import UserModel

bearer_scheme = HTTPBearer()


async def get_db() -> AsyncSession:
    async for session in get_session():
        yield session


def get_user_repo(session: AsyncSession = Depends(get_db)) -> UserRepository:
    return UserRepository(session)


def get_inspection_repo(session: AsyncSession = Depends(get_db)) -> InspectionRepository:
    return InspectionRepository(session)


def get_cache() -> type(cache):
    return cache


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    user_repo: UserRepository = Depends(get_user_repo),
) -> UserModel:
    payload = decode_token(credentials.credentials)
    if payload is None or payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token",
        )

    user = await user_repo.get_by_id(payload["sub"])
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


async def get_admin_user(
    current_user: UserModel = Depends(get_current_user),
) -> UserModel:
    from backend.domain.entities import UserRole
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user
