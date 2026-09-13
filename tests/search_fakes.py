"""Offline stand-ins for the embedding model and the network calls in search.py."""

from collections.abc import Iterable, Iterator

import numpy as np

VOCABULARY = ["cuda", "memory", "download", "toolkit", "openacc", "robotics", "jetson", "training", "gpu"]


class FakeEmbeddingModel:
    """Bag-of-vocabulary vectors: deterministic, instant, and good enough to order results."""

    def _vector(self, text: str) -> np.ndarray:
        lowered = text.lower()
        vector = np.array([float(word in lowered) for word in VOCABULARY])
        return vector if vector.any() else np.full(len(VOCABULARY), 1e-3)

    def query_embed(self, texts: Iterable[str]) -> Iterator[np.ndarray]:
        return (self._vector(text) for text in texts)

    def embed(self, texts: Iterable[str]) -> Iterator[np.ndarray]:
        return (self._vector(text) for text in texts)
