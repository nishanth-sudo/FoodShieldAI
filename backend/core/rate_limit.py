import time
from collections import deque

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

# SlowAPI limiter
# Example: @limiter.limit('10/minute')
limiter = Limiter(key_func=get_remote_address)

def rate_limit_exceeded(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"}
    )

class InMemoryRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.history = {}

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        if key not in self.history:
            self.history[key] = deque()

        # Clean expired timestamps
        while self.history[key] and self.history[key][0] < now - self.window_seconds:
            self.history[key].popleft()

        if len(self.history[key]) >= self.max_requests:
            return False

        self.history[key].append(now)
        return True
