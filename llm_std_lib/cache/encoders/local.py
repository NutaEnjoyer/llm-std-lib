"""
Local (on-device) embeddings encoder.

Converts prompt text into dense vectors using a locally hosted
sentence-transformers model — no external API calls required.

Requires the ``local-embeddings`` optional dependency::

    pip install llm-std-lib[local-embeddings]
"""

from __future__ import annotations

import asyncio
from functools import partial

import numpy as np

from llm_std_lib.cache.encoders.base import BaseEncoder
from llm_std_lib.exceptions import LLMCacheError

_DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast and accurate


class LocalEncoder(BaseEncoder):
    """Encoder backed by a local sentence-transformers model.

    The model is loaded lazily on the first call to :meth:`encode`.
    Encoding runs in a thread pool to avoid blocking the event loop.

    Args:
        model_name: Any sentence-transformers model name or local path.
            Default: ``all-MiniLM-L6-v2`` (384 dimensions, ~80MB).

    Raises:
        LLMCacheError: If ``sentence-transformers`` is not installed or the
            model cannot be loaded.

    Example::

        encoder = LocalEncoder()  # downloads all-MiniLM-L6-v2 on first use
        vector = await encoder.encode("What is semantic caching?")
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: object | None = None
        self._dim: int | None = None

    @property
    def dimension(self) -> int:
        """Vector dimension. Available only after the first :meth:`encode` call."""
        if self._dim is None:
            raise LLMCacheError(
                "dimension is not known before the first encode() call. "
                "Call await encoder.encode('') once to initialise the model."
            )
        return self._dim

    def _load_model(self) -> None:
        """Load the sentence-transformers model (blocking — runs in thread pool)."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise LLMCacheError(
                "sentence-transformers is not installed. "
                "Run: pip install llm-std-lib[local-embeddings]"
            ) from exc
        self._model = SentenceTransformer(self._model_name)
        # Determine dimension by encoding an empty string
        sample = self._model.encode("")  # type: ignore[union-attr]
        self._dim = int(sample.shape[0])

    def _encode_sync(self, text: str) -> np.ndarray:
        """Synchronous encode (runs inside a thread pool executor)."""
        if self._model is None:
            self._load_model()
        vector = self._model.encode(text, normalize_embeddings=True)  # type: ignore[union-attr]
        return np.array(vector, dtype=np.float32)

    async def encode(self, text: str) -> np.ndarray:
        """Encode *text* using the local sentence-transformers model.

        Runs the CPU-bound model inference in a thread pool so it does not
        block the asyncio event loop.

        Args:
            text: Input string to embed.

        Returns:
            L2-normalised float32 vector.

        Raises:
            LLMCacheError: If the model cannot be loaded.
        """
        loop = asyncio.get_event_loop()
        vector = await loop.run_in_executor(None, partial(self._encode_sync, text))
        return self.normalise(vector)
