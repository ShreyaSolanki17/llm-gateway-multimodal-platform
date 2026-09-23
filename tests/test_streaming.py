import json
import pytest
from fastapi.testclient import TestClient
from app.providers.base import BaseLLMProvider, ModelMetadata
from app.providers.exceptions import ProviderAPIError
from app.providers.registry import ProviderRegistry
from app.router.router import ModelRouter
from app.schemas.chat import ChatCompletionRequest, ChatMessage
from tests.test_semantic_cache import FakeEmbeddingClient


class FakeFailingStreamProvider(BaseLLMProvider):
    """Fails before yielding a single chunk, to test router fallback-before-first-chunk."""

    def __init__(self, model_name: str):
        self._model_name = model_name

    @property
    def name(self) -> str:
        return "failing-stream"

    def get_supported_models(self):
        return {self._model_name: ModelMetadata(model_name=self._model_name, provider_name=self.name)}

    async def generate(self, request: ChatCompletionRequest):
        raise NotImplementedError

    async def stream_generate(self, request: ChatCompletionRequest):
        raise ProviderAPIError(self.name, "stream failed to start")
        yield  # pragma: no cover -- makes this an async generator

    async def health_check(self) -> bool:
        return True


def _parse_sse(text: str):
    events = []
    for block in text.strip().split("\n\n"):
        if block.startswith("data: "):
            events.append(block[len("data: "):])
    return events


@pytest.mark.asyncio
async def test_router_stream_falls_back_before_first_chunk():
    registry = ProviderRegistry()
    registry.register_provider(FakeFailingStreamProvider("primary-model"))
    router = ModelRouter(registry=registry)

    req = ChatCompletionRequest(model="primary-model", messages=[ChatMessage(role="user", content="Hello")], stream=True)
    chunks = [chunk async for chunk in router.stream_with_fallback(req)]

    assert len(chunks) > 0
    assert chunks[0].model == "default-model"


def test_chat_endpoint_streaming_returns_sse_chunks(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock-gpt-4o", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers.get("X-Cache-Hit") == "false"

    events = _parse_sse(response.text)
    assert events[-1] == "[DONE]"

    content_pieces = []
    final_usage = None
    for raw in events[:-1]:
        chunk = json.loads(raw)
        delta = chunk["choices"][0]["delta"]
        if delta.get("content"):
            content_pieces.append(delta["content"])
        if chunk.get("usage"):
            final_usage = chunk["usage"]

    full_text = "".join(content_pieces)
    assert "MockProvider (mock-gpt-4o)" in full_text
    assert final_usage is not None
    assert final_usage["total_tokens"] > 0


def test_chat_endpoint_streaming_bypasses_cache(client: TestClient, monkeypatch):
    import app.api.v1.chat as chat_module
    from app.cache.semantic_cache import SemanticCache
    from app.config import settings

    monkeypatch.setattr(settings, "SEMANTIC_CACHE_ENABLED", True)
    fake_cache = SemanticCache(embedding_client=FakeEmbeddingClient(), threshold=0.5)
    monkeypatch.setattr(chat_module, "semantic_cache", fake_cache)

    client.post(
        "/v1/chat/completions",
        json={"model": "mock-gpt-4o", "messages": [{"role": "user", "content": "Hello"}], "stream": True},
    )

    assert fake_cache._entries == []
