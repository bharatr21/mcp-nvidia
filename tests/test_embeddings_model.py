"""Smoke test with the real embedding model. Skipped unless the [embeddings] extra is installed."""

import pytest

pytest.importorskip("fastembed")

from mcp_nvidia.lib import embeddings  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_model_state():
    embeddings._reset_for_tests()
    yield
    embeddings._reset_for_tests()


async def test_real_model_ranks_the_relevant_result_first():
    results = [
        {"title": "Slow-cooker chili recipe", "snippet": "Brown the beef, add beans and simmer for eight hours."},
        {"title": "CUDA unified memory", "snippet": "Allocate managed memory that migrates between CPU and GPU."},
    ]

    ranking = await embeddings.semantic_rank("CUDA memory management", results)

    assert ranking.unavailable is None
    assert ranking.order == [1, 0]
    assert ranking.similarities[1] > ranking.similarities[0]
