import uuid
from fastapi.testclient import TestClient


def test_chat_completion_success(client: TestClient):
    payload = {
        "model": "mock-gpt-4o",
        "messages": [
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "temperature": 0.5,
    }
    custom_request_id = f"test-req-{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Request-ID": custom_request_id},
    )

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_request_id

    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "mock-gpt-4o"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert "France" in data["choices"][0]["message"]["content"]
    assert "usage" in data
    assert data["usage"]["total_tokens"] > 0


def test_chat_completion_validation_error(client: TestClient):
    # Empty messages list should fail validation
    payload = {
        "model": "gpt-4o-mini",
        "messages": [],
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["type"] == "validation_error"
    assert "request_id" in data["error"]


def test_chat_endpoint_invalid_model_fallback(client: TestClient):
    payload = {
        "model": "google",
        "messages": [{"role": "user", "content": "Test fallback for unknown model"}],
    }
    custom_request_id = f"test-req-{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Request-ID": custom_request_id},
    )

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_request_id

    data = response.json()
    assert data["object"] == "chat.completion"
    assert "MockProvider" in data["choices"][0]["message"]["content"]


def test_chat_endpoint_invalid_parameters(client: TestClient):
    # Invalid temperature > 2.0
    payload_temp = {
        "model": "mock-gpt-4o",
        "messages": [{"role": "user", "content": "Test temp"}],
        "temperature": 3.5,
    }
    res_temp = client.post("/v1/chat/completions", json=payload_temp)
    assert res_temp.status_code == 422

    # Invalid max_tokens <= 0
    payload_tokens = {
        "model": "mock-gpt-4o",
        "messages": [{"role": "user", "content": "Test tokens"}],
        "max_tokens": -10,
    }
    res_tokens = client.post("/v1/chat/completions", json=payload_tokens)
    assert res_tokens.status_code == 422


def test_chat_endpoint_failing_provider_fallback(client: TestClient):
    # Model gpt-4o-mini is registered under OpenAICompatibleProvider, which fails without API key.
    # Gateway router should intercept failure and return 200 via fallback to default-model.
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Test automatic fallback from unauthenticated provider"}],
    }
    custom_request_id = f"test-req-{uuid.uuid4().hex[:8]}"

    response = client.post(
        "/v1/chat/completions",
        json=payload,
        headers={"X-Request-ID": custom_request_id},
    )

    assert response.status_code == 200
    assert response.headers.get("X-Request-ID") == custom_request_id

    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["model"] == "default-model"
    assert "MockProvider" in data["choices"][0]["message"]["content"]

