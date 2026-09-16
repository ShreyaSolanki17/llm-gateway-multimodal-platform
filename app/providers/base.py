from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.chat import ChatCompletionRequest, ChatCompletionResponse


class ModelMetadata(BaseModel):
    """Metadata describing a model served by a provider."""

    model_name: str
    provider_name: str
    max_context_length: int = 4096
    supports_vision: bool = False
    supports_streaming: bool = False
    cost_per_1k_input_tokens: float = 0.0015
    cost_per_1k_output_tokens: float = 0.0020
    description: Optional[str] = None


class BaseLLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the provider (e.g., 'mock', 'openai', 'vllm')."""
        pass

    @abstractmethod
    def get_supported_models(self) -> Dict[str, ModelMetadata]:
        """Return a mapping of model names to ModelMetadata supported by this provider."""
        pass

    @abstractmethod
    async def generate(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Generate a completion for the incoming request."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider endpoint/service is healthy."""
        pass
