import base64
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from app.schemas.chat import ChatMessage, ContentPart, ImageURL, MAX_IMAGE_BYTES

TINY_PNG_DATA_URI = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def test_image_url_accepts_http_link():
    img = ImageURL(url="https://example.com/photo.jpg")
    assert img.url == "https://example.com/photo.jpg"


def test_image_url_accepts_valid_base64_data_uri():
    img = ImageURL(url=TINY_PNG_DATA_URI)
    assert img.url == TINY_PNG_DATA_URI


def test_image_url_rejects_malformed_scheme():
    with pytest.raises(ValidationError):
        ImageURL(url="ftp://example.com/photo.jpg")


def test_image_url_rejects_invalid_base64():
    with pytest.raises(ValidationError):
        ImageURL(url="data:image/png;base64,not-valid-base64!!!")


def test_image_url_rejects_oversized_image():
    oversized = base64.b64encode(b"0" * (MAX_IMAGE_BYTES + 1)).decode()
    with pytest.raises(ValidationError):
        ImageURL(url=f"data:image/png;base64,{oversized}")


def test_chat_message_get_text_with_mixed_content():
    message = ChatMessage(
        role="user",
        content=[
            ContentPart(type="text", text="What is in this image?"),
            ContentPart(type="image_url", image_url=ImageURL(url=TINY_PNG_DATA_URI)),
        ],
    )
    assert message.get_text() == "What is in this image?"
    assert message.has_image() is True


def test_chat_message_plain_string_content():
    message = ChatMessage(role="user", content="Hello there")
    assert message.get_text() == "Hello there"
    assert message.has_image() is False


def test_chat_endpoint_routes_multimodal_request_to_vision_model(client: TestClient):
    payload = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this picture?"},
                    {"type": "image_url", "image_url": {"url": TINY_PNG_DATA_URI}},
                ],
            }
        ],
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model"] in ["mock-gpt-4o", "gpt-4o"]


def test_chat_endpoint_rejects_invalid_image_payload(client: TestClient):
    payload = {
        "model": "auto",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "not-a-valid-url-or-data-uri"}},
                ],
            }
        ],
    }

    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["type"] == "validation_error"
