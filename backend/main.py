from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.infrastructure.database import init_db, close_db
from backend.api.routes import router as inspection_router
from backend.api.auth import router as auth_router
from backend.api.admin import router as admin_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield
    await close_db()


app = FastAPI(
    title=settings.app_name,
    description="AI-Powered Food Quality Inspection Platform",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(inspection_router)
app.include_router(admin_router)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "foodshield-ai",
        "version": settings.app_version,
    }
