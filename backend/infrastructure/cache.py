import json
import hashlib
from typing import Optional, Any

from backend.config import settings

def compute_image_hash(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()

class CacheService:
    def __init__(self):
        self._client = None

    @property
    def redis(self):
        if self._client is None:
            import redis.asyncio as redis
            self._client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
        return self._client
        
    @property
    def client(self):
        return self.redis

    async def get(self, key: str) -> Optional[Any]:
        value = await self.redis.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    async def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        serialized = json.dumps(value) if not isinstance(value, str) else value
        return await self.redis.setex(key, ttl, serialized)

    async def delete(self, key: str) -> bool:
        return await self.redis.delete(key) > 0

    async def exists(self, key: str) -> bool:
        return await self.redis.exists(key) > 0

    async def clear_pattern(self, pattern: str):
        cursor = 0
        while True:
            cursor, keys = await self.redis.scan(cursor=cursor, match=pattern)
            if keys:
                await self.redis.delete(*keys)
            if cursor == 0:
                break
                
    async def get_by_image_hash(self, image_hash: str) -> Optional[Any]:
        return await self.get(f'img_hash:{image_hash}')
        
    async def set_image_hash_result(self, image_hash: str, result: dict, ttl: int = 3600):
        await self.set(f'img_hash:{image_hash}', result, ttl=ttl)
        
    async def close(self):
        if self._client:
            await self._client.aclose()


cache = CacheService()
