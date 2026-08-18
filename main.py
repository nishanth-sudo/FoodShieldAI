from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded as SlowApiRateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from backend.api.admin import router as admin_router
from backend.api.auth import router as auth_router
from backend.api.routes import router as inspection_router
from backend.config import settings
from backend.core.logging_config import setup_logging
from backend.core.middleware import LoggingMiddleware, RequestIDMiddleware
from backend.core.rate_limit import limiter, rate_limit_exceeded
from backend.infrastructure.cache import cache
from backend.infrastructure.database import close_db, init_db

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title=settings.app_name,
    description="AI-Powered Food Quality Inspection Platform",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

app.state.limiter = limiter
app.add_exception_handler(SlowApiRateLimitExceeded, rate_limit_exceeded)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(LoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(inspection_router)
app.include_router(admin_router)


@app.get("/health")
async def health_check() -> dict:
    redis_healthy = True
    try:
        await cache.redis.ping()
    except Exception:
        redis_healthy = False

    return {
        "status": "healthy" if redis_healthy else "degraded",
        "service": "foodshield-ai",
        "version": settings.app_version,
        "redis_connected": redis_healthy
    }
