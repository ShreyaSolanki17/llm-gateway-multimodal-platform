import time
from collections import defaultdict, deque
from typing import Deque, Dict
from fastapi import Request
from app.config import settings
from app.core.exceptions import RateLimitExceededError


class RateLimiter:
    """Sliding-window in-memory rate limiter keyed by client identity.

    ponytail: per-process in-memory counters -- resets on restart and isn't
    shared across processes/replicas. Fine at prototype scale; swap for a
    shared store (e.g. Redis) if this ever runs with multiple workers.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        window_start = now - self._window_seconds
        hits = self._hits[key]
        while hits and hits[0] < window_start:
            hits.popleft()
        if len(hits) >= self._max_requests:
            return False
        hits.append(now)
        return True


rate_limiter = RateLimiter(settings.RATE_LIMIT_REQUESTS, settings.RATE_LIMIT_WINDOW_SECONDS)


async def enforce_rate_limit(request: Request) -> None:
    """Reject requests once a client exceeds the configured rate limit.

    Runs before auth so repeated invalid-key attempts get throttled too, not
    just successful requests.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    client_key = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_key):
        raise RateLimitExceededError(
            f"Rate limit of {settings.RATE_LIMIT_REQUESTS} requests per {settings.RATE_LIMIT_WINDOW_SECONDS}s exceeded"
        )
