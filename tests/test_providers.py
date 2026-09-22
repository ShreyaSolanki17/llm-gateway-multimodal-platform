import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.providers.mock import MockLLMProvider
from app.providers.openai_provider import OpenAICompatibleProvider
from app.providers.registry import ProviderRegistry
from app.providers.vllm_provider import VLLMProvider
from app.providers.exceptions import ProviderAPIError, ProviderTimeoutError
from app.schemas.chat import ChatCompletionRequest, ChatMessage

UNREACHABLE_URL = "http://10.255.255.1:81/v1"


@pytest.mark.asyncio
async def test_mock_provider_generate():
    provider = MockLLMProvider()
    request = ChatCompletionRequest(
        model="mock-gpt-4o",
        messages=[ChatMessage(role="user", content="Hello test")],
    )

    response = await provider.generate(request)
    assert response.model == "mock-gpt-4o"
    assert len(response.choices) == 1
    assert "MockProvider (mock-gpt-4o)" in response.choices[0].message.content
    assert response.usage.total_tokens > 0


def test_provider_registry_resolution():
    registry = ProviderRegistry()
    mock_prov = MockLLMProvider()
    registry.register_provider(mock_prov)

    # Test registered model lookup
    p1 = registry.get_provider_for_model("mock-gpt-4o")
    assert p1.name == "mock"

    # Test unregistered model fallback lookup
    p2 = registry.get_provider_for_model("unknown-model-xyz")
    assert p2.name == "mock"


@pytest.mark.asyncio
async def test_openai_provider_timeout(monkeypatch):
    provider = OpenAICompatibleProvider(api_base="http://10.255.255.1:81/v1", timeout=0.01)
    request = ChatCompletionRequest(
        model="gpt-4o",
        messages=[ChatMessage(role="user", content="Test timeout")],
    )

    with pytest.raises((ProviderTimeoutError, ProviderAPIError)):
        await provider.generate(request)


@pytest.mark.asyncio
async def test_vllm_provider_simulates_response_when_unreachable():
    provider = VLLMProvider(api_base=UNREACHABLE_URL, timeout=0.5)
    request = ChatCompletionRequest(
        model="vllm-local",
        messages=[ChatMessage(role="user", content="Hello vLLM")],
    )

    response = await provider.generate(request)
    assert response.model == "vllm-local"
    assert "Simulated local inference" in response.choices[0].message.content
    assert response.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_vllm_provider_raises_when_simulation_disabled(monkeypatch):
    monkeypatch.setattr(settings, "VLLM_SIMULATE_LOCAL", False)
    provider = VLLMProvider(api_base=UNREACHABLE_URL, timeout=0.5)
    request = ChatCompletionRequest(
        model="vllm-local",
        messages=[ChatMessage(role="user", content="Hello vLLM")],
    )

    with pytest.raises((ProviderTimeoutError, ProviderAPIError)):
        await provider.generate(request)


@pytest.mark.asyncio
async def test_vllm_provider_health_check_false_when_unreachable():
    provider = VLLMProvider(api_base=UNREACHABLE_URL, timeout=0.5)
    assert await provider.health_check() is False


def test_chat_endpoint_with_provider_registry(client: TestClient):
    payload = {
        "model": "mock-claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Explain pluggable architecture."}],
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "mock-claude-3-5-sonnet"
    assert "MockProvider (mock-claude-3-5-sonnet)" in data["choices"][0]["message"]["content"]
