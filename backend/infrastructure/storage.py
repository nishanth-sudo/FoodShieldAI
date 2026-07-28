import io
from typing import Optional
from minio import Minio

from backend.config import settings


class ObjectStorage:
    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = Minio(
                endpoint=settings.storage_endpoint.replace("http://", ""),
                access_key=settings.storage_access_key,
                secret_key=settings.storage_secret_key,
                secure=settings.storage_secure,
            )
            self._ensure_bucket()
        return self._client

    def _ensure_bucket(self):
        if not self.client.bucket_exists(settings.storage_bucket):
            self.client.make_bucket(settings.storage_bucket)

    async def upload_image(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        data_stream = io.BytesIO(data)
        self.client.put_object(
            bucket_name=settings.storage_bucket,
            object_name=key,
            data=data_stream,
            length=len(data),
            content_type=content_type,
        )
        return f"{settings.storage_endpoint}/{settings.storage_bucket}/{key}"

    async def get_image(self, key: str) -> Optional[bytes]:
        try:
            response = self.client.get_object(settings.storage_bucket, key)
            return response.read()
        except Exception:
            return None

    async def delete_image(self, key: str):
        self.client.remove_object(settings.storage_bucket, key)

    def generate_presigned_url(self, key: str, expires: int = 3600) -> str:
        return self.client.presigned_get_object(
            settings.storage_bucket, key, expires=expires
        )


storage = ObjectStorage()
