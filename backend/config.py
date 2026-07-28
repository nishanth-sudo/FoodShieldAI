from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    app_name: str = "FoodShield AI"
    app_version: str = "0.1.0"
    debug: bool = False

    database_url: str = "postgresql+asyncpg://neondb_owner:npg_G2kRAuW1hbxo@ep-empty-tree-ay8a8lnk-pooler.c-5.us-east-2.aws.neon.tech/neondb?sslmode=require"
    redis_url: str = "redis://localhost:6379/0"

    storage_endpoint: str = "http://localhost:9000"
    storage_access_key: str = "minioadmin"
    storage_secret_key: str = "minioadmin"
    storage_bucket: str = "foodshield-images"
    storage_region: str = "us-east-1"
    storage_secure: bool = False

    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    ai_engine_url: str = "http://ai-engine:8501"
    max_upload_size_mb: int = 10
    allowed_image_types: list[str] = ["image/jpeg", "image/png", "image/webp"]

    model_config = {"env_prefix": "FOODSHIELD_", "env_file": ".env"}


settings = Settings()
