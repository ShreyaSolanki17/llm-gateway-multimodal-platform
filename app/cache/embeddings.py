from typing import List, Optional
import httpx
from app.config import settings
from app.providers.exceptions import ProviderAPIError, ProviderTimeoutError


class EmbeddingClient:
    """Thin client for OpenAI's embeddings endpoint, used to compare prompt similarity for the semantic cache."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        timeout: float = 10.0,
    ):
        self._api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self._model = model or settings.EMBEDDING_MODEL
        self._api_base = api_base.rstrip("/")
        self._timeout = timeout

    async def embed(self, text: str) -> List[float]:
        """Return the embedding vector for a piece of text."""
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._api_base}/embeddings",
                    json={"model": self._model, "input": text},
                    headers=headers,
                )
                if response.status_code != 200:
                    raise ProviderAPIError("openai-embeddings", f"HTTP {response.status_code}: {response.text}")
                data = response.json()
                return data["data"][0]["embedding"]
        except httpx.TimeoutException:
            raise ProviderTimeoutError("openai-embeddings", self._timeout)
        except httpx.RequestError as exc:
            raise ProviderAPIError("openai-embeddings", f"Network error: {str(exc)}")
