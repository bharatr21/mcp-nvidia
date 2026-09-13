"""Optional semantic ranking with a small local embedding model (fastembed, no torch).

The model is loaded lazily on first use, exactly once per process: a lock ensures that
threads arriving during a load wait for it instead of loading again. A failed load is
remembered until restart, so an offline deployment does not retry a slow download on
every query. Loading and encoding run in worker threads so they never block the event
loop, and callers receive an explicit reason whenever semantic ranking is unavailable.
"""

import asyncio
import logging
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from mcp_nvidia.lib.fusion import rank_by_score

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_THREADS = 4
ENCODE_TIMEOUT_SECONDS = 10.0

NOT_INSTALLED = "not_installed"
MODEL_LOAD_FAILED = "model_load_failed"
ENCODE_FAILED = "encode_failed"


class EmbeddingsNotInstalledError(Exception):
    """The optional [embeddings] extra (fastembed) is not installed."""


@dataclass(frozen=True)
class SemanticRanking:
    order: list[int] | None  # candidate indices, most similar first
    similarities: list[float] | None  # cosine similarity per candidate, aligned to the input
    unavailable: str | None  # reason when order is None


@dataclass
class _ModelState:
    model: Any = None
    failure: str | None = None
    loader: Callable[[], Any] | None = None


_state = _ModelState()
_lock = threading.Lock()


def _default_loader() -> Any:
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise EmbeddingsNotInstalledError("install mcp-nvidia[embeddings] to enable semantic ranking") from exc

    model_name = os.getenv("MCP_NVIDIA_EMBEDDING_MODEL", DEFAULT_MODEL)
    threads = int(os.getenv("MCP_NVIDIA_EMBEDDING_THREADS", str(DEFAULT_THREADS)))
    return TextEmbedding(model_name=model_name, threads=threads)


def _get_model() -> tuple[Any | None, str | None]:
    """Return (model, None) or (None, reason). Loads at most once per process."""
    # Fast path without the lock: once set, these only change through _reset_for_tests.
    if _state.model is not None:
        return _state.model, None
    if _state.failure is not None:
        return None, _state.failure

    with _lock:
        # Re-check: another thread may have finished loading while this one waited.
        if _state.model is not None:
            return _state.model, None
        if _state.failure is not None:
            return None, _state.failure

        loader = _state.loader or _default_loader
        try:
            _state.model = loader()
        except EmbeddingsNotInstalledError:
            _state.failure = NOT_INSTALLED
            logger.info("Semantic ranking disabled: fastembed is not installed (pip install 'mcp-nvidia[embeddings]')")
        except Exception as exc:
            _state.failure = MODEL_LOAD_FAILED
            logger.warning("Embedding model failed to load; semantic ranking disabled until restart: %r", exc)
        else:
            logger.info("Embedding model loaded")
        return _state.model, _state.failure


def _reset_for_tests(loader: Callable[[], Any] | None = None) -> None:
    """Forget any loaded model or failure. Tests only; pass a loader to swap in a fake."""
    with _lock:
        _state.model = None
        _state.failure = None
        _state.loader = loader


def build_document_text(result: Mapping[str, Any]) -> str:
    """Text embedded for one result: title plus plain snippet. The seam for future chunking."""
    title = str(result.get("title", ""))
    snippet = str(result.get("snippet_plain") or result.get("snippet", "")).replace("**", "")
    return f"{title}. {snippet}".strip()


def _cosine_similarities(model: Any, query: str, documents: Sequence[str]) -> list[float]:
    query_vector = np.asarray(next(iter(model.query_embed([query]))), dtype=np.float64)
    document_matrix = np.asarray(list(model.embed(list(documents))), dtype=np.float64)
    denominators = np.maximum(np.linalg.norm(document_matrix, axis=1) * np.linalg.norm(query_vector), 1e-12)
    return ((document_matrix @ query_vector) / denominators).tolist()


async def semantic_rank(query: str, results: Sequence[Mapping[str, Any]]) -> SemanticRanking:
    """Rank results by semantic similarity to the query, or say why that is unavailable."""
    if not results:
        return SemanticRanking(order=[], similarities=[], unavailable=None)

    model, failure = await asyncio.to_thread(_get_model)
    if model is None:
        return SemanticRanking(order=None, similarities=None, unavailable=failure)

    documents = [build_document_text(result) for result in results]
    try:
        # On timeout the worker thread finishes in the background; its result is discarded.
        similarities = await asyncio.wait_for(
            asyncio.to_thread(_cosine_similarities, model, query, documents),
            timeout=ENCODE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.warning("Semantic encoding failed; ranking without it: %r", exc)
        return SemanticRanking(order=None, similarities=None, unavailable=ENCODE_FAILED)

    return SemanticRanking(order=rank_by_score(similarities), similarities=similarities, unavailable=None)
