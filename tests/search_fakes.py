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


# Keyed by the clean domain that search_nvidia_domain puts after "site:".
FAKE_PAGES: dict[str, list[dict[str, str]]] = {
    "developer.nvidia.com": [
        {
            "title": "CUDA Toolkit Downloads",
            "href": "https://developer.nvidia.com/cuda-downloads",
            "body": "Download the CUDA Toolkit for GPU-accelerated applications.",
        },
        {
            "title": "CUDA Memory Management Guide",
            "href": "https://developer.nvidia.com/blog/cuda-memory",
            "body": "Unified memory, pinned allocations and memory pools in CUDA.",
        },
        {
            "title": "Jetson Community Projects",
            "href": "https://developer.nvidia.com/embedded/community",
            "body": "Robotics projects built by the Jetson community.",
        },
    ],
    # Overlapping subdomain searches return the same page, as measured on real queries.
    "catalog.ngc.nvidia.com": [
        {
            "title": "OpenACC Training Materials",
            "href": "https://catalog.ngc.nvidia.com/orgs/hpc/containers/openacc",
            "body": "Container with OpenACC training notebooks.",
        },
    ],
    "ngc.nvidia.com": [
        {
            "title": "OpenACC Training Materials",
            "href": "https://catalog.ngc.nvidia.com/orgs/hpc/containers/openacc",
            "body": "Container with OpenACC training notebooks.",
        },
    ],
}

FAILING_DOMAIN = "forums.nvidia.com"


def install_fake_search(monkeypatch, pages: dict[str, list[dict[str, str]]] = FAKE_PAGES) -> None:
    """Replace ddgs and page fetching inside mcp_nvidia.lib.search with offline fakes."""
    from mcp_nvidia.lib import search

    async def fake_fetch_ddgs_results(search_query: str, max_results: int) -> list[dict[str, str]]:
        domain = search_query.split()[0].removeprefix("site:")
        if domain == FAILING_DOMAIN:
            raise RuntimeError("simulated search outage")
        return [dict(page) for page in pages.get(domain, [])][:max_results]

    async def fake_fetch_url_context(client, url, snippet, context_chars=200):
        return snippet, None, {}

    monkeypatch.setattr(search, "_fetch_ddgs_results", fake_fetch_ddgs_results)
    monkeypatch.setattr(search, "fetch_url_context", fake_fetch_url_context)
