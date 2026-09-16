import pytest
from unittest.mock import AsyncMock
from app.router.analyzer import RequestAnalyzer
from app.router.schemas import ComplexityLevel, RequestType
from app.router.router import ModelRouter
from app.schemas.chat import ChatCompletionRequest, ChatMessage
from app.providers.registry import ProviderRegistry
from app.providers.mock import MockLLMProvider
from app.providers.exceptions import ProviderAPIError


def test_analyzer_simple_prompt():
    analyzer = RequestAnalyzer()
    req = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Hi there")],
    )
    complexity, req_type = analyzer.analyze(req)
    assert complexity == ComplexityLevel.SIMPLE
    assert req_type == RequestType.TEXT


def test_analyzer_complex_prompt():
    analyzer = RequestAnalyzer()
    complex_text = "Please analyze and refactor the architecture of this system step by step. " + ("word " * 60)
    req = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content=complex_text)],
    )
    complexity, req_type = analyzer.analyze(req)
    assert complexity == ComplexityLevel.COMPLEX
    assert req_type == RequestType.TEXT


def test_analyzer_vision_prompt():
    analyzer = RequestAnalyzer()
    req = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Please describe this image: [image] data:image/png...")],
    )
    complexity, req_type = analyzer.analyze(req)
    assert req_type == RequestType.VISION


def test_router_routing_decisions():
    registry = ProviderRegistry()
    mock_prov = MockLLMProvider()
    registry.register_provider(mock_prov)
    router = ModelRouter(registry=registry)

    # 1. Simple request -> simple model
    simple_req = ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content="Hello")])
    d1 = router.determine_route(simple_req)
    assert d1.complexity == ComplexityLevel.SIMPLE
    assert d1.selected_model == "default-model"

    # 2. Complex request -> complex model
    complex_text = "analyze step by step " + ("token " * 100)
    complex_req = ChatCompletionRequest(model="auto", messages=[ChatMessage(role="user", content=complex_text)])
    d2 = router.determine_route(complex_req)
    assert d2.complexity == ComplexityLevel.COMPLEX
    assert d2.selected_model in ["mock-claude-3-5-sonnet", "mock-gpt-4o"]

    # 3. Explicit model override
    explicit_req = ChatCompletionRequest(model="mock-gpt-4o", messages=[ChatMessage(role="user", content="Hello")])
    d3 = router.determine_route(explicit_req)
    assert d3.selected_model == "mock-gpt-4o"


@pytest.mark.asyncio
async def test_router_fallback_execution():
    registry = ProviderRegistry()
    failing_provider = MockLLMProvider()
    # Mock generate to raise ProviderAPIError on first call
    failing_provider.generate = AsyncMock(side_effect=ProviderAPIError("mock", "Service Unavailable"))
    registry.register_provider(failing_provider)

    router = ModelRouter(registry=registry)
    req = ChatCompletionRequest(model="mock-gpt-4o", messages=[ChatMessage(role="user", content="Hello")])

    # Expect router fallback to succeed using fallback model
    # (Here since both models point to failing_provider mock, fallback will also invoke generate, but we verify fallback mechanism was triggered)
    with pytest.raises(ProviderAPIError):
        await router.execute_with_fallback(req)
    assert failing_provider.generate.call_count == 2


def test_router_vision_model_selection():
    registry = ProviderRegistry()
    mock_prov = MockLLMProvider()
    registry.register_provider(mock_prov)
    router = ModelRouter(registry=registry)

    vision_req = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content="Analyze image: [image] data:image/png...")],
    )
    decision = router.determine_route(vision_req)
    assert decision.request_type == RequestType.VISION
    assert decision.selected_model in router.vision_tier_models


@pytest.mark.asyncio
async def test_router_successful_fallback():
    registry = ProviderRegistry()
    failing_primary = MockLLMProvider()
    failing_primary.generate = AsyncMock(side_effect=ProviderAPIError("primary", "Primary Down"))

    successful_fallback = MockLLMProvider()

    # Route primary model to failing_primary, fallback model to successful_fallback
    def mock_get_provider(model_name: str):
        if model_name == "mock-gpt-4o":
            return failing_primary
        return successful_fallback

    registry.get_provider_for_model = mock_get_provider

    router = ModelRouter(registry=registry)
    req = ChatCompletionRequest(model="mock-gpt-4o", messages=[ChatMessage(role="user", content="Test fallback")])

    response = await router.execute_with_fallback(req)
    assert response is not None
    assert response.model == "default-model"
    assert "MockProvider" in response.choices[0].message.content
    assert failing_primary.generate.call_count == 1

