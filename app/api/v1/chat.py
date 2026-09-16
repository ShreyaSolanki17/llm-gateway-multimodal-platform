from fastapi import APIRouter, Request, status
from app.schemas.chat import ChatCompletionRequest, ChatCompletionResponse
from app.router.router import model_router
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
    """Create a chat completion response by passing request through the intelligent ModelRouter."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(
        f"Processing chat request for model='{payload.model}' with {len(payload.messages)} message(s) | Request-ID: {request_id}"
    )

    # Route and execute request through ModelRouter with automatic fallback
    response = await model_router.execute_with_fallback(payload)
    return response
