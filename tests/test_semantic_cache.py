import pytest
from app.cache.semantic_cache import SemanticCache
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatMessage,
    UsageInfo,
)


class FakeEmbeddingClient:
    """Deterministic stand-in for EmbeddingClient, avoids real network calls in tests."""

    def __init__(self, vectors_by_text=None, should_fail: bool = False):
        self._vectors_by_text = vectors_by_text or {}
        self._should_fail = should_fail

    async def embed(self, text: str):
        if self._should_fail:
            raise RuntimeError("simulated embedding failure")
        return self._vectors_by_text.get(text, [1.0, 0.0, 0.0])


def _request(text: str) -> ChatCompletionRequest:
    return ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content=text)])


def _response(text: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        model="mock-gpt-4o",
        choices=[ChatCompletionChoice(index=0, message=ChatMessage(role="assistant", content=text))],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


@pytest.mark.asyncio
async def test_semantic_cache_hit_on_similar_embeddings():
    fake_client = FakeEmbeddingClient(
        vectors_by_text={
            "What's the capital of France?": [1.0, 0.0, 0.0],
            "What is the capital of France": [0.99, 0.01, 0.0],
        }
    )
    cache = SemanticCache(embedding_client=fake_client, threshold=0.95)

    original_response = _response("Paris is the capital of France.")
    await cache.store(_request("What's the capital of France?"), original_response)

    hit = await cache.lookup(_request("What is the capital of France"))
    assert hit is not None
    assert hit.choices[0].message.content == "Paris is the capital of France."


@pytest.mark.asyncio
async def test_semantic_cache_miss_below_threshold():
    fake_client = FakeEmbeddingClient(
        vectors_by_text={
            "What's the capital of France?": [1.0, 0.0, 0.0],
            "Write me a poem about the ocean": [0.0, 1.0, 0.0],
        }
    )
    cache = SemanticCache(embedding_client=fake_client, threshold=0.95)

    await cache.store(_request("What's the capital of France?"), _response("Paris."))
    miss = await cache.lookup(_request("Write me a poem about the ocean"))
    assert miss is None


@pytest.mark.asyncio
async def test_semantic_cache_lookup_empty_cache_returns_none():
    cache = SemanticCache(embedding_client=FakeEmbeddingClient(), threshold=0.95)
    assert await cache.lookup(_request("Anything")) is None


@pytest.mark.asyncio
async def test_semantic_cache_embedding_failure_degrades_gracefully():
    cache = SemanticCache(embedding_client=FakeEmbeddingClient(should_fail=True), threshold=0.95)

    # Store should not raise even though embedding fails
    await cache.store(_request("Hello"), _response("Hi there"))
    # Lookup should not raise either, and reports a miss
    assert await cache.lookup(_request("Hello")) is None
