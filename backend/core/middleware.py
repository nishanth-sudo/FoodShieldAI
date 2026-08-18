import os
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from backend.core.logging_config import get_logger

logger = get_logger(__name__)

class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID")
        if not request_id:
            request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start_time = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000
        request_id = getattr(request.state, "request_id", "")
        logger.info(
            f"Request {request.method} {request.url.path}",
            extra={
                "request_method": request.method,
                "request_path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
                "request_id": request_id,
            }
        )
        return response

def validate_image_bytes(data: bytes, filename: str = "image.jpg") -> str:
    # 0. Guard against empty content
    if not data or len(data) < 12:
        raise HTTPException(
            status_code=400, detail="File is empty or too small to be a valid image."
        )

    # 1. Checks magic bytes
    if data.startswith(b'\xFF\xD8\xFF'):
        ext = '.jpg'
    elif data.startswith(b'\x89\x50\x4E\x47'):
        ext = '.png'
    elif data.startswith(b'\x52\x49\x46\x46') and data[8:12] == b'\x57\x45\x42\x50':
        ext = '.webp'
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Supported formats: JPEG, PNG, WebP.",
        )

    # 2. Sanitizes filename
    base_name = os.path.basename(filename)
    sanitized = re.sub(r'[^\w\-_.]', '', base_name)
    if not sanitized:
        sanitized = 'image' + ext

    # 3. Validates extension is in [.jpg, .jpeg, .png, .webp]
    _, file_ext = os.path.splitext(sanitized)
    if file_ext.lower() not in ['.jpg', '.jpeg', '.png', '.webp']:
        raise HTTPException(status_code=400, detail="Invalid file extension.")

    return sanitized
