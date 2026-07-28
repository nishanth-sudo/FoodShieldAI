import json
from typing import Optional, Any

from backend.config import settings


class CacheService:
    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import redis.asyncio as redis
            self._client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
        return self._client

    async def get(self, key: str) -> Optional[Any]:
        value = await self.client.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        serialized = json.dumps(value) if not isinstance(value, str) else value
        return await self.client.setex(key, ttl, serialized)

    async def delete(self, key: str) -> bool:
        return await self.client.delete(key) > 0

    async def exists(self, key: str) -> bool:
        return await self.client.exists(key) > 0

    async def clear_pattern(self, pattern: str):
        cursor = 0
        while True:
            cursor, keys = await self.client.scan(cursor=cursor, match=pattern)
            if keys:
                await self.client.delete(*keys)
            if cursor == 0:
                break


cache = CacheService()
