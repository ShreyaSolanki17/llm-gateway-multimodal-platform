from fastapi import APIRouter, Request, status
from app.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from app.providers.registry import provider_registry
from app.core.logging import logger

router = APIRouter(prefix="/v1", tags=["Chat"])


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    status_code=status.HTTP_200_OK,
)
@router.post(
    "/chat",
    response_model=ChatCompletionResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def create_chat_completion(
    request: Request, payload: ChatCompletionRequest
) -> ChatCompletionResponse:
    """Create a chat completion response by routing the request to the appropriate LLM provider."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(
        f"Processing chat request for model '{payload.model}' with {len(payload.messages)} message(s) | Request-ID: {request_id}"
    )

    # Resolve provider via Provider Registry abstraction
    provider = provider_registry.get_provider_for_model(payload.model)
    logger.info(f"Routed model '{payload.model}' to provider '{provider.name}' | Request-ID: {request_id}")

    # Generate completion through provider interface
    response = await provider.generate(payload)
    return response
