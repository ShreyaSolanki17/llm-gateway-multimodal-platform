import time
from fastapi import APIRouter, Request, Response, status
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
    request: Request, payload: ChatCompletionRequest, response: Response
) -> ChatCompletionResponse:
    """Create a chat completion response by passing request through the intelligent ModelRouter."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.info(
        f"Processing chat request for model='{payload.model}' with {len(payload.messages)} message(s) | Request-ID: {request_id}"
    )

    # Route and execute request through ModelRouter with automatic fallback
    start_time = time.perf_counter()
    result = await model_router.execute_with_fallback(payload)
    latency_ms = (time.perf_counter() - start_time) * 1000
    cost_usd = model_router.calculate_cost(result.model, result.usage)

    response.headers["X-Response-Latency-Ms"] = f"{latency_ms:.2f}"
    response.headers["X-Estimated-Cost-USD"] = f"{cost_usd:.6f}"

    logger.info(
        f"Chat completion served | model='{result.model}' | "
        f"tokens(prompt={result.usage.prompt_tokens}, completion={result.usage.completion_tokens}, total={result.usage.total_tokens}) | "
        f"cost_usd={cost_usd:.6f} | latency_ms={latency_ms:.2f} | Request-ID: {request_id}"
    )

    return result
