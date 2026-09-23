import pytest
from fastapi.testclient import TestClient
from app.config import settings
from app.core.rate_limit import RateLimiter
from app.main import app


def test_chat_endpoint_open_when_no_api_key_configured(client: TestClient):
    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock-gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
    )
    assert response.status_code == 200


def test_chat_endpoint_rejects_missing_auth_header_when_key_configured(monkeypatch):
    monkeypatch.setattr(settings, "GATEWAY_API_KEY", "secret-key")
    with TestClient(app) as authed_client:
        response = authed_client.post(
            "/v1/chat/completions",
            json={"model": "mock-gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
        )
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"


def test_chat_endpoint_rejects_wrong_api_key(monkeypatch):
    monkeypatch.setattr(settings, "GATEWAY_API_KEY", "secret-key")
    with TestClient(app) as authed_client:
        response = authed_client.post(
            "/v1/chat/completions",
            json={"model": "mock-gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            headers={"Authorization": "Bearer wrong-key"},
        )
    assert response.status_code == 401


def test_chat_endpoint_accepts_correct_api_key(monkeypatch):
    monkeypatch.setattr(settings, "GATEWAY_API_KEY", "secret-key")
    with TestClient(app) as authed_client:
        response = authed_client.post(
            "/v1/chat/completions",
            json={"model": "mock-gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
            headers={"Authorization": "Bearer secret-key"},
        )
    assert response.status_code == 200


def test_health_endpoint_stays_open_when_key_configured(monkeypatch):
    monkeypatch.setattr(settings, "GATEWAY_API_KEY", "secret-key")
    with TestClient(app) as authed_client:
        response = authed_client.get("/health")
    assert response.status_code == 200


def test_document_ingest_rejects_oversized_text():
    oversized_text = "x" * (settings.MAX_DOCUMENT_CHARS + 1)
    client = TestClient(app)
    response = client.post("/v1/documents", json={"text": oversized_text})
    assert response.status_code == 422


def test_chat_request_rejects_too_many_messages():
    messages = [{"role": "user", "content": "hi"} for _ in range(settings.MAX_MESSAGES_PER_REQUEST + 1)]
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={"model": "mock-gpt-4o", "messages": messages})
    assert response.status_code == 422


def test_chat_request_rejects_oversized_combined_content():
    oversized_message = "x" * (settings.MAX_TOTAL_CONTENT_CHARS + 1)
    client = TestClient(app)
    response = client.post(
        "/v1/chat/completions",
        json={"model": "mock-gpt-4o", "messages": [{"role": "user", "content": oversized_message}]},
    )
    assert response.status_code == 422


def test_rate_limiter_allows_up_to_max_then_blocks():
    limiter = RateLimiter(max_requests=3, window_seconds=60)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-a") is False


def test_rate_limiter_tracks_clients_independently():
    limiter = RateLimiter(max_requests=1, window_seconds=60)
    assert limiter.allow("client-a") is True
    assert limiter.allow("client-b") is True
    assert limiter.allow("client-a") is False


def test_chat_endpoint_returns_429_when_rate_limited(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_REQUESTS", 1)
    monkeypatch.setattr("app.core.rate_limit.rate_limiter", RateLimiter(max_requests=1, window_seconds=60))

    with TestClient(app) as limited_client:
        payload = {"model": "mock-gpt-4o", "messages": [{"role": "user", "content": "Hello"}]}
        first = limited_client.post("/v1/chat/completions", json=payload)
        second = limited_client.post("/v1/chat/completions", json=payload)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["type"] == "rate_limit_exceeded"
