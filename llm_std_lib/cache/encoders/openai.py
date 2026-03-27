"""
OpenAI embeddings encoder.

Converts prompt text into dense vectors using the OpenAI Embeddings API
(``text-embedding-3-small`` or ``text-embedding-3-large``).

Requires no extra dependency — uses the same ``httpx`` client as the rest of
the library.
"""

from __future__ import annotations

import httpx
import numpy as np

from llm_std_lib.cache.encoders.base import BaseEncoder
from llm_std_lib.exceptions import LLMCacheError

_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
_DEFAULT_MODEL = "text-embedding-3-small"
_BASE_URL = "https://api.openai.com/v1"


class OpenAIEncoder(BaseEncoder):
    """Encoder backed by the OpenAI Embeddings API.

    Args:
        api_key: OpenAI API key. Reads from ``OPENAI_API_KEY`` if not provided.
        model: Embedding model name (default: ``text-embedding-3-small``).
        timeout_ms: Request timeout in milliseconds.

    Raises:
        LLMCacheError: If the API call fails.

    Example::

        encoder = OpenAIEncoder(api_key="sk-...")
        vector = await encoder.encode("What is semantic caching?")
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        timeout_ms: int = 10_000,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._timeout = timeout_ms / 1000.0
        self._dim = _DIMENSIONS.get(model, 1536)

    @property
    def dimension(self) -> int:
        return self._dim

    async def encode(self, text: str) -> np.ndarray:
        """Encode *text* via the OpenAI Embeddings API.

        Args:
            text: Input string to embed.

        Returns:
            L2-normalised float32 vector of shape ``(dimension,)``.

        Raises:
            LLMCacheError: On API error or network failure.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{_BASE_URL}/embeddings",
                    json={"input": text, "model": self._model},
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                )
        except httpx.RequestError as exc:
            raise LLMCacheError(f"OpenAI embedding request failed: {exc}") from exc

        if response.status_code != 200:
            raise LLMCacheError(
                f"OpenAI embedding API error ({response.status_code}): {response.text}"
            )

        data = response.json()
        vector = np.array(data["data"][0]["embedding"], dtype=np.float32)
        return self.normalise(vector)
