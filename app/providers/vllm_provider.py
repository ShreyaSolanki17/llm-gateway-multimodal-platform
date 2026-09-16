from typing import Dict, Optional
import httpx
from app.config import settings
from app.providers.base import BaseLLMProvider, ModelMetadata
from app.providers.exceptions import ProviderAPIError, ProviderTimeoutError
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
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

    def _generate_simulated_response(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generates simulated vLLM response when local hardware (GTX 1650 4GB VRAM) operates without live GPU daemon."""
        last_user_msg = next(
            (m.content for m in reversed(request.messages) if m.role == "user"),
            "Hello",
        )
        content = (
            f"[vLLM Engine ({request.model})] Simulated local inference output for prompt: '{last_user_msg}'. "
            f"Hardware profile: GTX 1650 4GB VRAM compatible mode."
        )

        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
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
