import json
from typing import AsyncIterator, Dict, Optional
import httpx
from app.providers.base import BaseLLMProvider, ModelMetadata
from app.providers.exceptions import ProviderAPIError, ProviderTimeoutError
from app.schemas.chat import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse


class OpenAICompatibleProvider(BaseLLMProvider):
    """Provider targeting any OpenAI-compatible API endpoint (OpenAI, vLLM, Ollama, LM Studio)."""

    def __init__(
        self,
        api_base: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        name: str = "openai-compatible",
    ):
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._name = name
        self._models: Dict[str, ModelMetadata] = {
            "gpt-4o": ModelMetadata(
                model_name="gpt-4o",
                provider_name=self._name,
                max_context_length=128000,
                supports_vision=True,
                cost_per_1k_input_tokens=0.0025,
                cost_per_1k_output_tokens=0.0100,
            ),
            "gpt-4o-mini": ModelMetadata(
                model_name="gpt-4o-mini",
                provider_name=self._name,
                max_context_length=128000,
                supports_vision=True,
                cost_per_1k_input_tokens=0.00015,
                cost_per_1k_output_tokens=0.0006,
            ),
        }

    @property
    def name(self) -> str:
        return self._name

    def get_supported_models(self) -> Dict[str, ModelMetadata]:
        return self._models

    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._api_base}/chat/completions"
        payload = request.model_dump(exclude_none=True)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    raise ProviderAPIError(self._name, f"HTTP {response.status_code}: {response.text}")
                data = response.json()
                return ChatCompletionResponse(**data)
            except httpx.TimeoutException:
                raise ProviderTimeoutError(self._name, self._timeout)
            except httpx.RequestError as exc:
                raise ProviderAPIError(self._name, f"Network error: {str(exc)}")

    async def stream_generate(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        """Stream chat completion chunks from the OpenAI-compatible endpoint (SSE passthrough)."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = f"{self._api_base}/chat/completions"
        payload = request.model_dump(exclude_none=True)
        payload["stream"] = True

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        body = await response.aread()
                        raise ProviderAPIError(self._name, f"HTTP {response.status_code}: {body.decode(errors='replace')}")
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[len("data: "):].strip()
                        if data == "[DONE]":
                            break
                        yield ChatCompletionChunk(**json.loads(data))
        except httpx.TimeoutException:
            raise ProviderTimeoutError(self._name, self._timeout)
        except httpx.RequestError as exc:
            raise ProviderAPIError(self._name, f"Network error: {str(exc)}")

    async def health_check(self) -> bool:
        url = f"{self._api_base}/models"
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(url, headers=headers)
                return response.status_code == 200
        except Exception:
            return False
