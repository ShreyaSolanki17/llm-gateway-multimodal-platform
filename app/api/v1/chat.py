import time
from typing import AsyncIterator
from fastapi import APIRouter, Depends, Request, Response, status
from fastapi.responses import StreamingResponse
from app.cache.semantic_cache import semantic_cache
from app.config import settings
from app.core.metrics import metrics_registry
from app.core.rate_limit import enforce_rate_limit
from app.core.security import verify_api_key
from app.rag.augment import augment_with_context
from app.schemas.chat import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse
from app.router.router import model_router
from app.core.logging import logger

router = APIRouter(prefix="/v1", tags=["Chat"], dependencies=[Depends(enforce_rate_limit), Depends(verify_api_key)])


async def _sse_event_stream(chunks: AsyncIterator[ChatCompletionChunk], request_id: str, start_time: float):
    """Format chunks as OpenAI-style SSE and record metrics once the stream completes.

    ponytail: cost/latency can't be exposed as response headers here (the HTTP
    headers are already sent before the body starts streaming) -- they're
    logged and recorded in metrics instead, same data, different surface.
    """
    model = "unknown"
    usage = None
    async for chunk in chunks:
        model = chunk.model
        if chunk.usage is not None:
            usage = chunk.usage
        yield f"data: {chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"

    latency_ms = (time.perf_counter() - start_time) * 1000
    cost_usd = model_router.calculate_cost(model, usage) if usage else 0.0
    logger.info(
        f"Chat completion stream served | model='{model}' | cost_usd={cost_usd:.6f} | "
        f"latency_ms={latency_ms:.2f} | Request-ID: {request_id}"
    )
    metrics_registry.record_request(model, latency_ms, cost_usd, cache_hit=False)


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

    if payload.stream:
        routed_payload = payload
        if payload.use_rag:
            routed_payload = await augment_with_context(payload)
        chunks = model_router.stream_with_fallback(routed_payload)
        return StreamingResponse(
            _sse_event_stream(chunks, request_id, start_time),
            media_type="text/event-stream",
            headers={"X-Cache-Hit": "false"},
        )

    # Semantic cache is bypassed for RAG requests: a cached answer may have been
    # grounded in document context that no longer reflects the current knowledge base.
    # It's also bypassed for streaming requests -- see the stream branch above.
    use_cache = settings.SEMANTIC_CACHE_ENABLED and not payload.use_rag

    if use_cache:
        cached_result = await semantic_cache.lookup(payload)
        if cached_result is not None:
            latency_ms = (time.perf_counter() - start_time) * 1000
            response.headers["X-Cache-Hit"] = "true"
            response.headers["X-Response-Latency-Ms"] = f"{latency_ms:.2f}"
            response.headers["X-Estimated-Cost-USD"] = "0.000000"
            logger.info(f"Chat completion served from semantic cache | latency_ms={latency_ms:.2f} | Request-ID: {request_id}")
            metrics_registry.record_request(cached_result.model, latency_ms, 0.0, cache_hit=True)
            return cached_result

    routed_payload = payload
    if payload.use_rag:
        routed_payload = await augment_with_context(payload)

    # Route and execute request through ModelRouter with automatic fallback
    result = await model_router.execute_with_fallback(routed_payload)
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
    metrics_registry.record_request(result.model, latency_ms, cost_usd, cache_hit=False)

    if use_cache:
        await semantic_cache.store(payload, result)

    return result
