import math
from dataclasses import dataclass
from typing import List, Optional
from app.cache.embeddings import EmbeddingClient
from app.config import settings
from app.core.logging import logger
from app.schemas.chat import ChatCompletionRequest, ChatCompletionResponse


@dataclass
class CacheEntry:
    prompt: str
    embedding: List[float]
    response: ChatCompletionResponse


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """In-memory semantic cache keyed by embedding similarity of the last user message.

    ponytail: in-process list, O(n) scan per lookup. Fine at prototype scale;
    swap for pgvector (Milestone 17) once entries outgrow a linear scan.
    Embedding failures (missing key, network issue) degrade to a cache miss
    rather than breaking the chat request.
    """

    def __init__(self, embedding_client: Optional[EmbeddingClient] = None, threshold: Optional[float] = None):
        self._embedding_client = embedding_client or EmbeddingClient()
        self._threshold = threshold if threshold is not None else settings.SEMANTIC_CACHE_SIMILARITY_THRESHOLD
        self._entries: List[CacheEntry] = []

    @staticmethod
    def _extract_prompt(request: ChatCompletionRequest) -> str:
        return next((m.content for m in reversed(request.messages) if m.role == "user"), "")

    async def lookup(self, request: ChatCompletionRequest) -> Optional[ChatCompletionResponse]:
        prompt = self._extract_prompt(request)
        if not prompt or not self._entries:
            return None

        try:
            query_embedding = await self._embedding_client.embed(prompt)
        except Exception as exc:
            logger.warning(f"Semantic cache lookup skipped, embedding failed: {str(exc)}")
            return None

        best_score = 0.0
        best_entry: Optional[CacheEntry] = None
        for entry in self._entries:
            score = _cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= self._threshold:
            logger.info(f"Semantic cache HIT (similarity={best_score:.4f}) for prompt: '{prompt[:50]}'")
            return best_entry.response

        return None

    async def store(self, request: ChatCompletionRequest, response: ChatCompletionResponse) -> None:
        prompt = self._extract_prompt(request)
        if not prompt:
            return

        try:
            embedding = await self._embedding_client.embed(prompt)
        except Exception as exc:
            logger.warning(f"Semantic cache store skipped, embedding failed: {str(exc)}")
            return

        self._entries.append(CacheEntry(prompt=prompt, embedding=embedding, response=response))


# Global Singleton Semantic Cache Instance
semantic_cache = SemanticCache()
