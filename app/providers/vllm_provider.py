import json
from typing import AsyncIterator, Dict, Optional
import httpx
from app.config import settings
from app.providers.base import BaseLLMProvider, ModelMetadata
from app.providers.exceptions import ProviderAPIError, ProviderTimeoutError
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
from app.core.logging import logger


class VLLMProvider(BaseLLMProvider):
    """Self-Hosted vLLM Inference Engine Provider.

    Connects to vLLM OpenAI-compatible endpoints with health probing,
    timeout handling, and fallback capabilities.
    """

    def __init__(
        self,
        api_base: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self._api_base = (api_base or settings.VLLM_SERVER_URL).rstrip("/")
        self._model_name = model_name or settings.VLLM_MODEL_NAME
        self._timeout = timeout or settings.VLLM_TIMEOUT

        self._models: Dict[str, ModelMetadata] = {
            self._model_name: ModelMetadata(
                model_name=self._model_name,
                provider_name="vllm",
                max_context_length=32768,
                supports_vision=False,
                supports_streaming=True,
                cost_per_1k_input_tokens=0.0000,  # Self-hosted open source model
                cost_per_1k_output_tokens=0.0000,
                description="Self-hosted open-source model served via vLLM engine",
            ),
            "vllm-local": ModelMetadata(
                model_name="vllm-local",
                provider_name="vllm",
                max_context_length=4096,
                cost_per_1k_input_tokens=0.0000,
                cost_per_1k_output_tokens=0.0000,
                description="Local light model endpoint for limited VRAM hardware",
            ),
        }

    @property
    def name(self) -> str:
        return "vllm"

    def get_supported_models(self) -> Dict[str, ModelMetadata]:
        return self._models

    async def health_check(self) -> bool:
        """Probe vLLM server health endpoint."""
        url = f"{self._api_base}/models"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(url)
                return response.status_code == 200
        except Exception:
            return False

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Send chat completion request to vLLM server."""
        url = f"{self._api_base}/chat/completions"
        payload = request.model_dump(exclude_none=True)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    raise ProviderAPIError("vllm", f"vLLM server error HTTP {response.status_code}: {response.text}")
                data = response.json()
                return ChatCompletionResponse(**data)
        except (httpx.TimeoutException, ProviderTimeoutError):
            logger.warning(f"vLLM server at '{self._api_base}' timed out after {self._timeout}s")
            if settings.VLLM_SIMULATE_LOCAL:
                return self._generate_simulated_response(request)
            raise ProviderTimeoutError("vllm", self._timeout)
        except (httpx.RequestError, ProviderAPIError) as exc:
            logger.warning(f"vLLM server connection failed at '{self._api_base}': {str(exc)}")
            if settings.VLLM_SIMULATE_LOCAL:
                return self._generate_simulated_response(request)
            raise ProviderAPIError("vllm", f"vLLM inference server unreachable: {str(exc)}")

    async def stream_generate(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        """Stream chat completion chunks from the vLLM server (SSE passthrough).

        ponytail: falls back to a fully-simulated stream only on a failure
        before any real chunk arrives (connect/timeout/HTTP error) -- a drop
        mid-stream after real chunks have already been yielded is not
        recovered, same as any other partial-stream failure would be.
        """
        url = f"{self._api_base}/chat/completions"
        payload = request.model_dump(exclude_none=True)
        payload["stream"] = True

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream("POST", url, json=payload) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise ProviderAPIError(
                            "vllm", f"vLLM server error HTTP {response.status_code}: {body.decode(errors='replace')}"
                        )
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[len("data: "):].strip()
                        if data == "[DONE]":
                            break
                        yield ChatCompletionChunk(**json.loads(data))
        except (httpx.TimeoutException, ProviderTimeoutError):
            logger.warning(f"vLLM server at '{self._api_base}' timed out after {self._timeout}s")
            if settings.VLLM_SIMULATE_LOCAL:
                async for chunk in self._stream_simulated_response(request):
                    yield chunk
                return
            raise ProviderTimeoutError("vllm", self._timeout)
        except (httpx.RequestError, ProviderAPIError) as exc:
            logger.warning(f"vLLM server connection failed at '{self._api_base}': {str(exc)}")
            if settings.VLLM_SIMULATE_LOCAL:
                async for chunk in self._stream_simulated_response(request):
                    yield chunk
                return
            raise ProviderAPIError("vllm", f"vLLM inference server unreachable: {str(exc)}")

    async def _stream_simulated_response(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        """Streaming counterpart to _generate_simulated_response, for GPU-less local dev."""
        last_user_msg = next(
            (m.get_text() for m in reversed(request.messages) if m.role == "user"),
            "Hello",
        )
        content = (
            f"[vLLM Engine ({request.model})] Simulated local inference output for prompt: '{last_user_msg}'. "
            f"Hardware profile: GTX 1650 4GB VRAM compatible mode."
        )
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

    def _generate_simulated_response(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generates simulated vLLM response when local hardware (GTX 1650 4GB VRAM) operates without live GPU daemon."""
        last_user_msg = next(
            (m.get_text() for m in reversed(request.messages) if m.role == "user"),
            "Hello",
        )
        content = (
            f"[vLLM Engine ({request.model})] Simulated local inference output for prompt: '{last_user_msg}'. "
            f"Hardware profile: GTX 1650 4GB VRAM compatible mode."
        )

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
