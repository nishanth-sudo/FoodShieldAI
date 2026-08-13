import os
import pytest
from backend.config import settings

os.environ['FOODSHIELD_DATABASE_URL'] = 'postgresql+asyncpg://test:test@localhost:5432/test'
os.environ['FOODSHIELD_REDIS_URL'] = 'redis://localhost:6379/1'
os.environ['FOODSHIELD_JWT_SECRET_KEY'] = 'test-secret-key'

settings.database_url = "postgresql+asyncpg://test:test@localhost:5432/test"
settings.redis_url = "redis://localhost:6379/1"
settings.jwt_secret_key = "test-secret-key"
settings.storage_endpoint = "http://localhost:9000"
settings.debug = False
