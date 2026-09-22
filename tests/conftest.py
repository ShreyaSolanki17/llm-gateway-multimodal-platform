import pytest
from fastapi.testclient import TestClient
from app.config import settings
from app.main import app
from app.providers.registry import provider_registry


@pytest.fixture(autouse=True)
def disable_semantic_cache(monkeypatch):
    """Prevent tests from making real, billed OpenAI embedding calls by default."""
    monkeypatch.setattr(settings, "SEMANTIC_CACHE_ENABLED", False)


@pytest.fixture(autouse=True)
def isolate_real_network_providers(monkeypatch):
    """Keep the real OpenAI-compatible provider from making a live network call in tests,
    even when a real API key is configured in the environment."""
    openai_provider = provider_registry.get_provider("openai-compatible")
    if openai_provider is not None:
        monkeypatch.setattr(openai_provider, "_api_base", "http://10.255.255.1:81/v1")
        monkeypatch.setattr(openai_provider, "_timeout", 0.5)


@pytest.fixture
def client() -> TestClient:
    """Fixture providing a FastAPI test client."""
    return TestClient(app)
