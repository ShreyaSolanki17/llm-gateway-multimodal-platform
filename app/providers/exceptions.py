from app.core.exceptions import GatewayException


class ProviderException(GatewayException):
    """Base exception for provider errors."""

    def __init__(self, message: str, status_code: int = 502, error_type: str = "provider_error"):
        super().__init__(message=message, status_code=status_code, error_type=error_type)


class ProviderNotFoundError(ProviderException):
    """Raised when no provider is registered for a requested model."""

    def __init__(self, model: str):
        super().__init__(
            message=f"No LLM provider found capable of serving model '{model}'",
            status_code=404,
            error_type="provider_not_found",
        )


class ProviderTimeoutError(ProviderException):
    """Raised when an upstream provider request times out."""

    def __init__(self, provider_name: str, timeout_seconds: float):
        super().__init__(
            message=f"Provider '{provider_name}' request timed out after {timeout_seconds}s",
            status_code=504,
            error_type="provider_timeout",
        )


class ProviderAPIError(ProviderException):
    """Raised when an upstream provider returns an error response."""

    def __init__(self, provider_name: str, details: str):
        super().__init__(
            message=f"Provider '{provider_name}' returned error: {details}",
            status_code=502,
            error_type="provider_api_error",
        )
