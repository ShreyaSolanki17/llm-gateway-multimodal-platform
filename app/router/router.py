from typing import AsyncIterator, Dict, List, Optional
from app.config import settings
from app.router.schemas import ComplexityLevel, RequestType, RoutingDecision
from app.router.analyzer import RequestAnalyzer
from app.providers.registry import ProviderRegistry, provider_registry
from app.providers.base import BaseLLMProvider, ModelMetadata
from app.providers.exceptions import ProviderException
from app.schemas.chat import ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, UsageInfo
from app.core.logging import logger


class ModelRouter:
    """Complexity and cost-aware model router with automatic fallback execution."""

    def __init__(self, registry: Optional[ProviderRegistry] = None):
        self.registry = registry or provider_registry
        self.analyzer = RequestAnalyzer()

        # Model tiers
        self.simple_tier_models = ["vllm-local", settings.VLLM_MODEL_NAME, "default-model", "gpt-4o-mini", "mock-gpt-4o"]
        self.complex_tier_models = [settings.VLLM_MODEL_NAME, "mock-claude-3-5-sonnet", "mock-gpt-4o", "gpt-4o"]
        self.vision_tier_models = ["mock-gpt-4o", "gpt-4o"]
        self.fallback_tier_models = ["default-model"]

    def determine_route(self, request: ChatCompletionRequest) -> RoutingDecision:
        """Select best model based on explicit request override or automatic complexity/cost analysis."""
        all_models = self.registry.list_all_models()

        # 1. If explicit model requested by client (not generic 'auto' or default), respect it
        if request.model not in ["auto", "default-model", ""] and request.model in all_models:
            provider = self.registry.get_provider_for_model(request.model)
            meta = all_models[request.model]
            return RoutingDecision(
                selected_model=request.model,
                selected_provider_name=provider.name,
                complexity=ComplexityLevel.MODERATE,
                request_type=RequestType.TEXT,
                reason=f"Explicit client request for model '{request.model}'",
                fallback_model=self.fallback_tier_models[0],
                estimated_cost_usd=meta.cost_per_1k_input_tokens * 0.1,
            )

        # 2. Perform rule-based analysis
        complexity, req_type = self.analyzer.analyze(request)

        # 3. Select candidate model pool
        if req_type == RequestType.VISION:
            candidates = self.vision_tier_models
            reason = "Routed to Vision model due to image input"
        elif complexity == ComplexityLevel.COMPLEX:
            candidates = self.complex_tier_models
            reason = f"Routed to high-reasoning model due to [{complexity.value}] prompt"
        else:
            candidates = self.simple_tier_models
            reason = f"Routed to fast/cheap model due to [{complexity.value}] prompt"

        # 4. Find first available model in registry
        selected_model = None
        for cand in candidates:
            if cand in all_models:
                selected_model = cand
                break

        if not selected_model:
            selected_model = "default-model"

        provider = self.registry.get_provider_for_model(selected_model)
        meta = all_models.get(selected_model, ModelMetadata(model_name=selected_model, provider_name=provider.name))

        # Determine fallback model distinct from primary
        fallback = next((m for m in self.fallback_tier_models if m != selected_model), "default-model")

        return RoutingDecision(
            selected_model=selected_model,
            selected_provider_name=provider.name,
            complexity=complexity,
            request_type=req_type,
            reason=reason,
            fallback_model=fallback,
            estimated_cost_usd=meta.cost_per_1k_input_tokens * 0.1,
        )

    def calculate_cost(self, model_name: str, usage: UsageInfo) -> float:
        """Compute actual cost in USD from real token usage and the model's configured rate."""
        meta = self.registry.list_all_models().get(model_name)
        if not meta:
            return 0.0
        input_cost = (usage.prompt_tokens / 1000) * meta.cost_per_1k_input_tokens
        output_cost = (usage.completion_tokens / 1000) * meta.cost_per_1k_output_tokens
        return input_cost + output_cost

    async def execute_with_fallback(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Route and execute request with automatic model fallback on provider failures."""
        decision = self.determine_route(request)
        logger.info(
            f"ModelRouter Decision: selected='{decision.selected_model}' ({decision.selected_provider_name}) | "
            f"complexity={decision.complexity.value} | reason='{decision.reason}'"
        )

        primary_model = decision.selected_model
        # Temporarily set requested model in request payload to match decision
        effective_request = request.model_copy(update={"model": primary_model})
        provider = self.registry.get_provider_for_model(primary_model)

        try:
            return await provider.generate(effective_request)
        except (ProviderException, Exception) as exc:
            logger.warning(
                f"Primary provider '{provider.name}' failed for model '{primary_model}': {str(exc)}. Initiating fallback routing..."
            )
            fallback_model = decision.fallback_model or "default-model"
            fallback_request = request.model_copy(update={"model": fallback_model})
            fallback_provider = self.registry.get_provider_for_model(fallback_model)
            logger.info(f"Fallback routing executing with model '{fallback_model}' via provider '{fallback_provider.name}'")
            return await fallback_provider.generate(fallback_request)

    async def stream_with_fallback(self, request: ChatCompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        """Route and stream chunks, with fallback only if the primary fails before
        yielding its first chunk. Once real chunks have reached the client, the
        response can't be un-sent, so there's no mid-stream fallback."""
        decision = self.determine_route(request)
        logger.info(
            f"ModelRouter Decision (stream): selected='{decision.selected_model}' ({decision.selected_provider_name}) | "
            f"complexity={decision.complexity.value} | reason='{decision.reason}'"
        )

        primary_model = decision.selected_model
        effective_request = request.model_copy(update={"model": primary_model})
        provider = self.registry.get_provider_for_model(primary_model)

        primary_stream = provider.stream_generate(effective_request)
        try:
            first_chunk = await primary_stream.__anext__()
        except StopAsyncIteration:
            return
        except (ProviderException, Exception) as exc:
            logger.warning(
                f"Primary provider '{provider.name}' failed to start stream for model '{primary_model}': {str(exc)}. "
                f"Initiating fallback routing..."
            )
            fallback_model = decision.fallback_model or "default-model"
            fallback_request = request.model_copy(update={"model": fallback_model})
            fallback_provider = self.registry.get_provider_for_model(fallback_model)
            logger.info(f"Fallback routing executing with model '{fallback_model}' via provider '{fallback_provider.name}'")
            async for chunk in fallback_provider.stream_generate(fallback_request):
                yield chunk
            return

        yield first_chunk
        async for chunk in primary_stream:
            yield chunk


# Global Singleton Router Instance
model_router = ModelRouter()
