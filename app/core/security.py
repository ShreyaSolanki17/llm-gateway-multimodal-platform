import secrets
from fastapi import Request
from app.config import settings
from app.core.exceptions import AuthenticationError

_BEARER_PREFIX = "Bearer "


async def verify_api_key(request: Request) -> None:
    """Require a valid bearer token on every request.

    ponytail: a no-op if GATEWAY_API_KEY isn't configured (dev convenience --
    main.py logs a startup warning when that's the case). Set the key before
    exposing this gateway beyond local development.
    """
    if not settings.GATEWAY_API_KEY:
        return

    header = request.headers.get("Authorization", "")
    if not header.startswith(_BEARER_PREFIX):
        raise AuthenticationError("Missing or malformed Authorization header")

    provided_key = header[len(_BEARER_PREFIX):]
    if not secrets.compare_digest(provided_key, settings.GATEWAY_API_KEY):
        raise AuthenticationError("Invalid API key")
