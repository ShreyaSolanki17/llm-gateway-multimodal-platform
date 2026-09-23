from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.core.logging import logger


class GatewayException(Exception):
    """Base exception class for Gateway errors."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR, error_type: str = "gateway_error"):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(message)


class AuthenticationError(GatewayException):
    """Raised when a request is missing or has an invalid API key."""

    def __init__(self, message: str = "Invalid or missing API key"):
        super().__init__(message=message, status_code=status.HTTP_401_UNAUTHORIZED, error_type="authentication_error")


class RateLimitExceededError(GatewayException):
    """Raised when a client exceeds the configured request rate limit."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message=message, status_code=status.HTTP_429_TOO_MANY_REQUESTS, error_type="rate_limit_exceeded")


class EvaluationError(GatewayException):
    """Raised when the LLM-as-a-Judge evaluator fails to produce a usable result."""

    def __init__(self, message: str):
        super().__init__(message=message, status_code=status.HTTP_502_BAD_GATEWAY, error_type="evaluation_error")


async def gateway_exception_handler(request: Request, exc: GatewayException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "N/A")
    logger.error(f"GatewayException [{exc.error_type}]: {exc.message} | Request-ID: {request_id}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.error_type,
                "request_id": request_id,
            }
        },
    )


def _json_safe_errors(errors: list) -> list:
    """Pydantic error dicts can carry a raw exception object in ctx.error (from custom
    field_validators), which the default JSON encoder can't serialize. Stringify it."""
    cleaned = []
    for err in errors:
        err = dict(err)
        ctx = err.get("ctx")
        if isinstance(ctx, dict) and "error" in ctx:
            err["ctx"] = {**ctx, "error": str(ctx["error"])}
        cleaned.append(err)
    return cleaned


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "N/A")
    logger.warning(f"ValidationError: {exc.errors()} | Request-ID: {request_id}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "message": "Invalid request payload",
                "type": "validation_error",
                "details": _json_safe_errors(exc.errors()),
                "request_id": request_id,
            }
        },
    )
