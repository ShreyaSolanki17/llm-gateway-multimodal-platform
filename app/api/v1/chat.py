from fastapi import APIRouter, Request, status
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    UsageInfo,
)
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
    """Create a chat completion response using a baseline mock provider.

    (In Milestone 2 & 3, this will route through pluggable model providers and routers).
    """
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(
        f"Processing chat request for model '{payload.model}' with {len(payload.messages)} message(s) | Request-ID: {request_id}"
    )

    last_user_msg = next(
        (m.content for m in reversed(payload.messages) if m.role == "user"),
        "Hello!",
    )

    # Simple mock response logic for Milestone 1
    mock_reply = (
        f"[Gateway Mock Response] Received your prompt: '{last_user_msg}'. "
        f"Model '{payload.model}' processed this request successfully."
    )

    # Simple token estimation
    prompt_tokens = sum(len(m.content.split()) for m in payload.messages)
    completion_tokens = len(mock_reply.split())
    total_tokens = prompt_tokens + completion_tokens

    return ChatCompletionResponse(
        model=payload.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=mock_reply),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        ),
    )
