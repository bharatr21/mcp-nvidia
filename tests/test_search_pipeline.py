"""Pipeline and protocol-level tests for hybrid ranking, fully offline."""

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mcp_nvidia.lib import embeddings, search
from mcp_nvidia.lib.fusion import relevance_from_position
from mcp_nvidia.server import app
from tests.search_fakes import FAILING_DOMAIN, FakeEmbeddingModel, install_fake_search

DOMAINS = ["https://developer.nvidia.com/", "https://catalog.ngc.nvidia.com/", "https://ngc.nvidia.com/"]
OPENACC_URL = "https://catalog.ngc.nvidia.com/orgs/hpc/containers/openacc"
JETSON_URL = "https://developer.nvidia.com/embedded/community"
MEMORY_GUIDE_URL = "https://developer.nvidia.com/blog/cuda-memory"


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    install_fake_search(monkeypatch)
    embeddings._reset_for_tests(loader=FakeEmbeddingModel)
    yield
    embeddings._reset_for_tests()


async def _search(query="cuda memory", domains=DOMAINS):
    return await search.search_all_domains(
        query=query, domains=domains, max_results_per_domain=3, min_relevance_score=0
    )


async def test_duplicates_are_removed_before_scoring(monkeypatch):
    scored_urls = []
    real_document_tokens = search.document_tokens

    def spy(result):
        scored_urls.append(result["url"])
        return real_document_tokens(result)

    monkeypatch.setattr(search, "document_tokens", spy)

    await _search()

    assert scored_urls.count(OPENACC_URL) == 1
    assert len(scored_urls) == len(set(scored_urls))


async def test_relevance_scores_are_derived_from_fused_position():
    results, _errors, _warnings, _timing = await _search()

    n = len(results)
    assert n > 0
    assert [r["relevance_score"] for r in results] == [relevance_from_position(p, n) for p in range(n)]


async def test_evidence_floor_drops_results_with_no_lexical_or_semantic_match():
    results, *_ = await _search()

    assert JETSON_URL not in [r["url"] for r in results]


async def test_bm25_alone_ranks_the_stronger_match_first():
    def not_installed_loader():
        raise embeddings.EmbeddingsNotInstalledError("no fastembed")

    embeddings._reset_for_tests(loader=not_installed_loader)

    results, *_ = await _search()

    assert results[0]["url"] == MEMORY_GUIDE_URL


async def test_semantic_ranking_embeds_the_original_query(monkeypatch):
    seen_queries = []
    real_semantic_rank = search.semantic_rank

    async def spy(query, results):
        seen_queries.append(query)
        return await real_semantic_rank(query, results)

    monkeypatch.setattr(search, "semantic_rank", spy)

    await _search(query="TensorRT inference")

    assert seen_queries == ["TensorRT inference"]


async def test_error_results_score_zero_and_are_not_ranked():
    results, *_ = await _search(domains=[*DOMAINS, f"https://{FAILING_DOMAIN}/"])

    ranked = [r for r in results if not r.get("is_error")]
    errored = [r for r in results if r.get("is_error")]
    assert errored, "the failing domain should produce an error result"
    assert all(r["relevance_score"] == 0 for r in errored)
    assert [r["relevance_score"] for r in ranked] == [
        relevance_from_position(p, len(ranked)) for p in range(len(ranked))
    ]


async def test_installed_but_failing_semantics_warns_and_still_returns_results():
    def failing_loader():
        raise RuntimeError("offline: cannot download weights")

    embeddings._reset_for_tests(loader=failing_loader)

    results, _errors, warnings, _timing = await _search()

    matching = [w for w in warnings if w["code"] == search.SEMANTIC_UNAVAILABLE_WARNING]
    assert matching == [
        {
            "code": "SEMANTIC_RANKING_UNAVAILABLE",
            "message": "Semantic ranking unavailable; results ranked by BM25 only",
            "reason": "model_load_failed",
        }
    ]
    assert results


async def test_search_tool_reports_semantic_failure_over_the_protocol():
    def failing_loader():
        raise RuntimeError("corrupt weights")

    embeddings._reset_for_tests(loader=failing_loader)

    # Session entered inside the test body, never in a fixture (see Global Constraints).
    async with create_connected_server_and_client_session(app) as session:
        result = await session.call_tool(
            "search_nvidia",
            {"query": "cuda memory", "domains": ["developer.nvidia.com"], "min_relevance_score": 0},
        )

    assert result.isError is False
    payload = result.structuredContent
    assert "SEMANTIC_RANKING_UNAVAILABLE" in [w["code"] for w in payload["warnings"]]
    assert payload["results"], "BM25 ranking should still return results"
