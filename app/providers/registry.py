from typing import Dict, List, Optional
from app.providers.base import BaseLLMProvider, ModelMetadata
from app.providers.exceptions import ProviderNotFoundError
from app.providers.mock import MockLLMProvider
from app.core.logging import logger


from app.providers.openai_provider import OpenAICompatibleProvider
from app.providers.vllm_provider import VLLMProvider


class ProviderRegistry:
    """Central registry mapping models to registered LLM Providers."""

    def __init__(self):
        self._providers: Dict[str, BaseLLMProvider] = {}
        self._model_to_provider: Dict[str, BaseLLMProvider] = {}
        self._fallback_provider: BaseLLMProvider = MockLLMProvider()
        # Register default fallback & standard providers
        self.register_provider(self._fallback_provider)
        self.register_provider(OpenAICompatibleProvider())
        self.register_provider(VLLMProvider())

    def register_provider(self, provider: BaseLLMProvider) -> None:
        """Register a provider and index all models supported by it."""
        self._providers[provider.name] = provider
        models = provider.get_supported_models()
        for model_name in models.keys():
            self._model_to_provider[model_name] = provider
            logger.info(f"Registered model '{model_name}' under provider '{provider.name}'")

    def get_provider(self, provider_name: str) -> Optional[BaseLLMProvider]:
        """Get provider by its unique provider name."""
        return self._providers.get(provider_name)

    def get_provider_for_model(self, model_name: str) -> BaseLLMProvider:
        """Find the registered provider for a requested model, or use fallback if configured."""
        if model_name in self._model_to_provider:
            return self._model_to_provider[model_name]

        # If model is unknown, fall back to mock provider for development, or raise error
        logger.warning(
            f"Requested model '{model_name}' not explicitly registered. Routing to default fallback provider ('{self._fallback_provider.name}')"
        )
        return self._fallback_provider

    def list_all_models(self) -> Dict[str, ModelMetadata]:
        """Return a combined dictionary of all available models across registered providers."""
        all_models: Dict[str, ModelMetadata] = {}
        for provider in self._providers.values():
            all_models.update(provider.get_supported_models())
        return all_models


# Global Singleton Registry Instance
provider_registry = ProviderRegistry()
