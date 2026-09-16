import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import logger


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware that assigns a unique X-Request-ID header to every request."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        logger.info(f"Incoming Request [{request.method} {request.url.path}] | Request-ID: {request_id}")

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        logger.info(f"Completed Request [{request.method} {request.url.path}] | Status: {response.status_code} | Request-ID: {request_id}")

        return response
