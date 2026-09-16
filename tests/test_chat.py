import uuid
from fastapi.testclient import TestClient


def test_chat_completion_success(client: TestClient):
    payload = {
        "model": "gpt-4o-mini",
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
    assert data["model"] == "gpt-4o-mini"
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
