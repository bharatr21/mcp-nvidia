"""Unit tests for the load-once embedding model and semantic ranking."""

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from mcp_nvidia.lib import embeddings
from tests.search_fakes import FakeEmbeddingModel


@pytest.fixture(autouse=True)
def _fresh_model_state():
    embeddings._reset_for_tests()
    yield
    embeddings._reset_for_tests()


def test_concurrent_first_calls_load_the_model_exactly_once():
    load_count = 0
    count_lock = threading.Lock()

    def slow_loader():
        nonlocal load_count
        with count_lock:
            load_count += 1
        time.sleep(0.2)  # long enough that every thread arrives mid-load
        return FakeEmbeddingModel()

    embeddings._reset_for_tests(loader=slow_loader)
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda _: embeddings._get_model(), range(8)))

    assert load_count == 1
    first_model = outcomes[0][0]
    assert first_model is not None
    assert all(model is first_model and failure is None for model, failure in outcomes)


def test_load_failure_is_remembered_and_not_retried():
    calls = 0

    def failing_loader():
        nonlocal calls
        calls += 1
        raise RuntimeError("offline: cannot download weights")

    embeddings._reset_for_tests(loader=failing_loader)

    assert embeddings._get_model() == (None, embeddings.MODEL_LOAD_FAILED)
    assert embeddings._get_model() == (None, embeddings.MODEL_LOAD_FAILED)
    assert calls == 1


def test_missing_fastembed_is_reported_as_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)  # makes `import fastembed` raise ImportError
    embeddings._reset_for_tests()  # use the real default loader

    assert embeddings._get_model() == (None, embeddings.NOT_INSTALLED)


async def test_semantic_rank_orders_results_by_similarity():
    embeddings._reset_for_tests(loader=FakeEmbeddingModel)
    results = [
        {"title": "Jetson robotics", "snippet": "community builds"},
        {"title": "CUDA memory", "snippet": "memory pools in cuda"},
    ]

    ranking = await embeddings.semantic_rank("cuda memory", results)

    assert ranking.unavailable is None
    assert ranking.order == [1, 0]
    assert ranking.similarities is not None
    assert ranking.similarities[1] > ranking.similarities[0]


async def test_unavailable_model_returns_the_failure_reason():
    def failing_loader():
        raise RuntimeError("corrupt weights")

    embeddings._reset_for_tests(loader=failing_loader)

    ranking = await embeddings.semantic_rank("cuda", [{"title": "CUDA", "snippet": "cuda"}])

    assert ranking == embeddings.SemanticRanking(
        order=None, similarities=None, unavailable=embeddings.MODEL_LOAD_FAILED
    )


async def test_encode_exception_is_reported_as_encode_failed():
    class BrokenModel(FakeEmbeddingModel):
        def embed(self, _texts):
            raise RuntimeError("onnxruntime error")

    embeddings._reset_for_tests(loader=BrokenModel)

    ranking = await embeddings.semantic_rank("cuda", [{"title": "CUDA", "snippet": "cuda"}])

    assert ranking.unavailable == embeddings.ENCODE_FAILED
    assert ranking.order is None
    assert ranking.similarities is None


async def test_encode_timeout_is_reported_as_encode_failed(monkeypatch):
    class SlowModel(FakeEmbeddingModel):
        def embed(self, texts):
            time.sleep(0.5)
            return super().embed(texts)

    monkeypatch.setattr(embeddings, "ENCODE_TIMEOUT_SECONDS", 0.05)
    embeddings._reset_for_tests(loader=SlowModel)

    ranking = await embeddings.semantic_rank("cuda", [{"title": "CUDA", "snippet": "cuda"}])

    assert ranking.unavailable == embeddings.ENCODE_FAILED


def test_document_text_uses_title_and_plain_snippet():
    text = embeddings.build_document_text({"title": "CUDA", "snippet": "**bold** text", "snippet_plain": "bold text"})
    assert text == "CUDA. bold text"


def test_document_text_strips_highlight_markers_without_snippet_plain():
    assert embeddings.build_document_text({"title": "CUDA", "snippet": "**bold** text"}) == "CUDA. bold text"
