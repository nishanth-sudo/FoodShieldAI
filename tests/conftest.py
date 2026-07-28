import pytest
from backend.config import settings

settings.database_url = "postgresql+asyncpg://test:test@localhost:5432/test"
settings.redis_url = "redis://localhost:6379/1"
settings.jwt_secret_key = "test-secret-key"
settings.storage_endpoint = "http://localhost:9000"
settings.debug = False
