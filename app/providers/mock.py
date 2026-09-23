from typing import AsyncIterator, Dict
from app.providers.base import BaseLLMProvider, ModelMetadata
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChunkDelta,
    ChatMessage,
    UsageInfo,
)


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for testing, fallbacks, and local development."""

    def __init__(self):
        self._models: Dict[str, ModelMetadata] = {
            "default-model": ModelMetadata(
                model_name="default-model",
                provider_name="mock",
                max_context_length=4096,
                cost_per_1k_input_tokens=0.0005,
                cost_per_1k_output_tokens=0.0015,
                description="Default mock text LLM",
            ),
            "mock-gpt-4o": ModelMetadata(
                model_name="mock-gpt-4o",
                provider_name="mock",
                max_context_length=8192,
                cost_per_1k_input_tokens=0.0025,
                cost_per_1k_output_tokens=0.0100,
                description="High-tier mock model",
            ),
            "mock-claude-3-5-sonnet": ModelMetadata(
                model_name="mock-claude-3-5-sonnet",
                provider_name="mock",
                max_context_length=8192,
                cost_per_1k_input_tokens=0.0030,
                cost_per_1k_output_tokens=0.0150,
                description="Complex reasoning mock model",
            ),
        }

    @property
    def name(self) -> str:
        return "mock"

    def get_supported_models(self) -> Dict[str, ModelMetadata]:
        return self._models

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        last_user_msg = next(
            (m.get_text() for m in reversed(request.messages) if m.role == "user"),
            "Hello",
        )
        content = f"[MockProvider ({request.model})] Responded to: '{last_user_msg}'"

        prompt_tokens = sum(len(m.get_text().split()) for m in request.messages)
        completion_tokens = len(content.split())
        total_tokens = prompt_tokens + completion_tokens

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

    async def stream_generate(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        last_user_msg = next(
            (m.get_text() for m in reversed(request.messages) if m.role == "user"),
            "Hello",
        )
        content = f"[MockProvider ({request.model})] Responded to: '{last_user_msg}'"
        words = content.split(" ")

        for i, word in enumerate(words):
            piece = word if i == 0 else f" {word}"
            yield ChatCompletionChunk(
                model=request.model,
                choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(content=piece))],
            )

        prompt_tokens = sum(len(m.get_text().split()) for m in request.messages)
        completion_tokens = len(words)
        yield ChatCompletionChunk(
            model=request.model,
            choices=[ChatCompletionChunkChoice(delta=ChatCompletionChunkDelta(), finish_reason="stop")],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    async def health_check(self) -> bool:
        return True
