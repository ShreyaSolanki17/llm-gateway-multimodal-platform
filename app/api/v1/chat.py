import time
from fastapi import APIRouter, Request, Response, status
from app.cache.semantic_cache import semantic_cache
from app.config import settings
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

    start_time = time.perf_counter()

    # Check the semantic cache before routing to a real model
    if settings.SEMANTIC_CACHE_ENABLED:
        cached_result = await semantic_cache.lookup(payload)
        if cached_result is not None:
            latency_ms = (time.perf_counter() - start_time) * 1000
            response.headers["X-Cache-Hit"] = "true"
            response.headers["X-Response-Latency-Ms"] = f"{latency_ms:.2f}"
            response.headers["X-Estimated-Cost-USD"] = "0.000000"
            logger.info(f"Chat completion served from semantic cache | latency_ms={latency_ms:.2f} | Request-ID: {request_id}")
            return cached_result

    # Route and execute request through ModelRouter with automatic fallback
    result = await model_router.execute_with_fallback(payload)
    latency_ms = (time.perf_counter() - start_time) * 1000
    cost_usd = model_router.calculate_cost(result.model, result.usage)

    response.headers["X-Cache-Hit"] = "false"
    response.headers["X-Response-Latency-Ms"] = f"{latency_ms:.2f}"
    response.headers["X-Estimated-Cost-USD"] = f"{cost_usd:.6f}"

    logger.info(
        f"Chat completion served | model='{result.model}' | "
        f"tokens(prompt={result.usage.prompt_tokens}, completion={result.usage.completion_tokens}, total={result.usage.total_tokens}) | "
        f"cost_usd={cost_usd:.6f} | latency_ms={latency_ms:.2f} | Request-ID: {request_id}"
    )

    if settings.SEMANTIC_CACHE_ENABLED:
        await semantic_cache.store(payload, result)

    return result
