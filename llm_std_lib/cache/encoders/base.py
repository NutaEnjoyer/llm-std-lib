"""
Abstract encoder interface.

All embedding encoder implementations must subclass BaseEncoder and implement
encode(). The returned vectors are used by cache backends for cosine-similarity
search.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEncoder(ABC):
    """Abstract base class for text-to-vector encoders.

    Subclasses wrap a specific embedding model (OpenAI API, sentence-transformers,
    or any other provider) and expose a unified encode() interface.

    Example::

        class MyEncoder(BaseEncoder):
            @property
            def dimension(self) -> int:
                return 768

            async def encode(self, text: str) -> np.ndarray:
                ...  # call your model
    """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the vectors produced by this encoder."""

    @abstractmethod
    async def encode(self, text: str) -> np.ndarray:
        """Encode *text* into a unit-normalised dense vector.

        The returned array must have shape ``(dimension,)`` and dtype
        ``float32``. Implementations are responsible for L2-normalising
        the output so that cosine similarity equals the dot product.

        Args:
            text: The prompt or query string to encode.

        Returns:
            A 1-D float32 numpy array of length ``dimension``.
        """

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Assumes both vectors are already L2-normalised (unit vectors), in which
        case the cosine similarity equals the dot product.

        Args:
            a: First vector, shape ``(d,)``.
            b: Second vector, shape ``(d,)``.

        Returns:
            Similarity score in ``[-1.0, 1.0]``.
        """
        return float(np.dot(a, b))

    @staticmethod
    def normalise(v: np.ndarray) -> np.ndarray:
        """L2-normalise *v* to a unit vector.

        Args:
            v: Input vector, shape ``(d,)``.

        Returns:
            Unit vector with the same direction as *v*, or a zero vector if
            the input norm is zero.
        """
        norm = np.linalg.norm(v)
        if norm == 0.0:
            return v
        return (v / norm).astype(np.float32)  # type: ignore[no-any-return]
