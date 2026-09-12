<!-- markdownlint-disable MD013 -->

# Hybrid Search (Keyword + Semantic) — Design

- **Date:** 2026-09-12
- **Status:** Approved 2026-09-12; both open questions resolved (§11)
- **Target release:** `mcp-nvidia` 0.9.0, on the `mcp` 1.x SDK
- **Branch:** `chore/pin-mcp-v1` (PR #8)
- **Inherited by:** 1.0.0, when `mcp2` merges `main`
- **Related:** `docs/decisions/2026-09-12-no-page-fetch-cache.md`

## 1. Context

`search_nvidia` ranks live results; there is no index. `search_all_domains` (`src/mcp_nvidia/lib/search.py:211`) runs one `ddgs` query per domain, fetches each result's page, and then scores the combined list in a single loop:

1. `calculate_tfidf_scores(all_results, query)` — sklearn TF-IDF cosine similarity over the result set.
2. For each result, `get_domain_boost(domain, query)`, then `calculate_search_relevance(result, query, domain_boost)`, a 0–100 keyword heuristic built from exact, fuzzy and phrase matches.
3. A hardcoded blend, `int(keyword_score * 0.7 + tfidf_score * 100 * 0.3)`, capped at 100.
4. Filter on `min_relevance_score` (default 17), then sort.

Step 3 already combines two signals measured in incomparable units, using a hand-picked weight. Adding a third signal to it would mean re-tuning those weights by hand, so this design replaces it.

### Measured baseline

| Quantity | Value |
| --- | --- |
| Candidates per default query | 15 domains × 3 = 45 |
| Wall time, default query | 30.3 s (23 of 45 results kept) |
| Page fetch, per result | ~0.75–0.83 s |
| URL recurrence across queries | 0% (3 queries, 30 URL slots) |
| Duplicate URLs within one query | 1 in 85 slots, all from subdomain overlap |
| Encode, query + 45 docs | ~0.6–3 s on a dev box under load |
| Production resources | 8 vCPU, 8 GB; 7-day average 0.0017 vCPU, 0.26 GB |

## 2. Goals and non-goals

### Goals

- Add semantic similarity as a ranking signal for `search_nvidia`.
- Replace the fixed 70/30 blend with rank fusion, which takes a third signal without any re-tuning.
- Leave the base install unchanged: embeddings are opt-in, and without them search ranks on keyword + TF-IDF.
- Make degraded ranking visible to clients, not only in server logs.
- Make ranking regressions detectable before users notice them.

### Non-goals

- `discover_nvidia_content`. It is an explicit follow-up, to be done once fusion is understood on the simpler tool; it currently has no relevance score to fuse against.
- A vector index (pgvector or otherwise) and a page-fetch cache. See the decision record.
- Chunked, page-level embeddings. The design leaves a seam for them and builds nothing.
- GPU inference.

## 3. Decisions

| # | Decision | Rationale |
| --- | --- | --- |
| 1 | Reciprocal rank fusion over keyword, TF-IDF and semantic rankings; the 70/30 blend is removed | It compares positions rather than scores, so signals in different units need no rescaling, and adding or losing a signal needs no re-tuning. |
| 2 | `k = 10`, not the textbook 60 | 60 was tuned on TREC lists of ~1000 documents. At 45 it leaves only a 1.72× spread between first and last place. 10 gives a 5× spread and sits just on the side where one signal's strong top hit can win, which is what known-item queries need. Tunable. |
| 3 | `relevance_score` stays a required integer from 0–100, now derived from fused rank | The field and its type are part of the declared output schema and the generated SDKs. Its meaning was a baseline, not a contract. |
| 4 | Embeddings via `fastembed`, shipped as the optional extra `[embeddings]` | Runs on onnxruntime with no torch; 178 MB installed. |
| 5 | Model `BAAI/bge-small-en-v1.5` | Pads only to the longest input in the batch (a title is ~16 tokens). fastembed's `all-MiniLM-L6-v2` pads every input to 128 tokens, so short inputs cost as much as long ones. 67 MB, 384 dimensions. |
| 6 | Embed the title plus `snippet_plain` | The same text keyword scoring reads, without the `**` highlight markers. The one function that builds this input is the seam for chunking. |
| 7 | Load lazily, exactly once, under a lock; encode in a worker thread | Import and stdio startup are unaffected, and concurrent first queries still trigger only one load. Encoding is 0.6–3 s of synchronous CPU work that would otherwise freeze the event loop for every client. |
| 8 | Prefetch weights in the Docker build; download lazily for pip installs | Avoids a ~10 s download on every container cold start. |
| 9 | On failure, rank on the remaining signals and report it in `warnings` | Degraded ranking shows up where clients already look. |
| 10 | Deduplicate URLs within a request | A correctness fix: overlapping subdomains return the same page twice. |
| 11 | Evaluate with a comparison script first, then with labeled fixtures | Labels come from judging real disagreements between rankings, not from an imagined ideal order. |
| 12 | Evidence floor before fusion: drop candidates with no keyword match *and* semantic similarity below τ | Rank-derived scores are relative; without a floor, an all-junk query still returns ~83% of its candidates. τ comes from measurement, not a guess. |
| 13 | "Extra not installed" is a supported mode with no per-query warning | Warnings are reserved for an installed extra that fails, so they keep their meaning. |

## 4. Scoring pipeline

### Fusion

```text
fused(d) = Σ over available signals s of  1 / (k + rank_s(d))      k = 10, ranks start at 1
```

- All signals rank the same candidate set, since this is pure reranking. No result can be missing from a signal's list, so fusion behaves like a smoothed Borda count rather than classic multi-system RRF.
- Error results (`is_error`) are excluded from fusion and scored 0.
- Ties within a signal are broken by original candidate order, so output is deterministic and fixtures stay stable.

A new pure module, `src/mcp_nvidia/lib/fusion.py`, with no I/O:

```python
RRF_K = 10  # Not the textbook 60, which was tuned on ~1000-document lists; ours hold ~45.

def rank_by_score(scores: Sequence[float]) -> list[int]:
    """Candidate indices ordered best-first; ties keep original order."""

def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    """Fuse rankings of the same candidates into one best-first order."""
```

The result at fused position `p` (counting from 0), out of `n` candidates, gets:

```text
relevance_score = round(100 * (n - p) / n)
```

The top result scores 100; with `n = 45`, the last scores 2.

### Where it slots in

`search_all_domains` keeps its signature and its return shape. After results are gathered:

1. Deduplicate (§7).
2. Keyword scores from `calculate_search_relevance`, unchanged and still domain-boosted, turned into a ranking.
3. TF-IDF scores from `calculate_tfidf_scores`, unchanged, turned into a ranking.
4. `semantic_rank` (§5), which returns a ranking or reports itself unavailable.
5. **Evidence floor** (below): drop candidates with no keyword evidence and low semantic similarity.
6. `reciprocal_rank_fusion` over the surviving candidates, re-ranked on each available signal, giving the fused order and each `relevance_score`.
7. The `min_relevance_score` filter on the fused ranking, then `sort_by` as today. The `date` and `domain` sorts are unchanged.

`relevance.py` is not modified.

### Evidence floor

Rank-derived scores are relative. On their own, `min_relevance_score = 17` would keep the top ~83% of candidates on every query, even when all of them are poor. So an absolute gate runs before fusion. A candidate is **dropped** only when both of these hold:

- its keyword score is 0, meaning no exact, fuzzy or phrase match; and
- its semantic similarity to the query is below a threshold τ.

Edge cases:

- **Semantic ranking unavailable:** the floor uses keyword evidence alone.
- **The query yields no keywords** (for example, it is all stopwords): `extract_keywords` returns nothing and every keyword score is 0, so the keyword condition is skipped. Otherwise the floor would drop every candidate.

τ is not guessed. The eval script measures bge-small's similarity distribution on real results, and τ is set from that measurement before release. It is a named constant in `fusion.py`, next to `RRF_K`, and the script reports how many candidates the floor removes.

The floor is a pure function in `fusion.py`:

```python
EVIDENCE_FLOOR_TAU: float  # set from eval-script measurement before release

def apply_evidence_floor(
    keyword_scores: Sequence[int],
    similarities: Sequence[float] | None,
    tau: float = EVIDENCE_FLOOR_TAU,
    query_has_keywords: bool = True,
) -> list[int]:
    """Indices of candidates that pass the floor, in original order."""
```

### Behaviour change

Result order changes for every query, including on installs without the extra, because the 70/30 blend is gone from the keyword + TF-IDF path as well.

## 5. Embedding module

A new module, `src/mcp_nvidia/lib/embeddings.py`:

```python
@dataclass(frozen=True)
class SemanticRanking:
    order: list[int] | None       # candidate indices, most similar first
    unavailable: str | None       # reason when order is None

async def semantic_rank(query: str, results: Sequence[Mapping[str, Any]]) -> SemanticRanking: ...
```

### Loading

- Module-level state: the loaded model, a `threading.Lock`, and a remembered failure reason.
- Double-checked locking. If the model is loaded, return it. Otherwise take the lock, check again, and load. A thread that arrives while another is loading waits on the lock, then finds the model already loaded. **The model loads once per process.**
- A load failure is remembered for the lifetime of the process, so an offline deployment doesn't retry a ~10 s download on every query. Restarting the server retries.
- Loading happens on first use, never at import time.

### Encoding

- `query_embed` for the query, which adds bge's retrieval instruction prefix; `embed` for result text. Using plain `embed` for both would quietly weaken matching.
- Cosine similarity is computed explicitly, not assumed from normalized vectors.
- Encoding runs in `asyncio.to_thread` with a timeout (default 10 s). A timeout counts as an encode failure.

### Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `MCP_NVIDIA_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Model name; the eval script uses it to compare models |
| `MCP_NVIDIA_EMBEDDING_THREADS` | `4` | onnxruntime intra-op threads |
| `FASTEMBED_CACHE_PATH` | fastembed's default | Where weights live; set explicitly in the Docker image |

### Packaging and deployment

- `pyproject.toml`: `embeddings = ["fastembed>=0.8,<0.9"]` under `[project.optional-dependencies]`.
- `Dockerfile`: install `.[embeddings]`, set `ENV FASTEMBED_CACHE_PATH=/opt/models`, and prefetch the model with `RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"`. This adds about 250 MB to the image.
- Production appears to build from the Dockerfile. This is inferred: the running process uses the Dockerfile's uvicorn `CMD`, and the build logs of the May deployment have expired. If it is built with Railpack instead, the lazy download still works; only the first query after each start is slower.

## 6. Failure and degradation

| Situation | `unavailable` | Ranking signals | `warnings` entry | Server log |
| --- | --- | --- | --- | --- |
| Extra not installed | `not_installed` | keyword + TF-IDF | none: a supported mode, not a failure | once, INFO, on first query |
| Model fails to load | `model_load_failed` | keyword + TF-IDF | yes | once, WARNING, with the cause |
| Encode raises or times out | `encode_failed` | keyword + TF-IDF | yes | each time, WARNING |

The warning entry follows the shape of existing warnings. `outputSchema` types `warnings` items as plain objects, so no schema change is needed:

```json
{
  "code": "SEMANTIC_RANKING_UNAVAILABLE",
  "message": "Semantic ranking unavailable; results ranked by keyword and TF-IDF only",
  "reason": "model_load_failed"
}
```

### Adjacent known issue (not fixed here)

`ddgs` raises `DDGSException("No results found.")` when a domain has no matches. `search_nvidia_domain` handles it as a failure, so a domain that simply has nothing to return shows up as a fake "Search temporarily unavailable" result plus a `PARTIAL_FAILURE` warning. This design uses the same `warnings` channel, so the issue makes warnings noisier than they should be. It is recorded as a follow-up.

## 7. Deduplication

Runs after gathering and before scoring. The key is the URL with its scheme and host lowercased, its fragment removed and any trailing slash stripped. The first occurrence wins, in domain order.

Measured duplicate rate: 1 in 85 URL slots. Duplicates come from `site:` matching subdomains: `ngc.nvidia.com` also matches `catalog.ngc.nvidia.com`, and `docs.nvidia.com` also matches `docs.omniverse.nvidia.com`. This is a correctness fix, not a performance one.

## 8. Evaluation and testing

### Stage 1 — comparison script

`scripts/eval_ranking.py` runs real queries. For each one it prints the baseline ranking beside the new ranking, ordered by how far each result moved. The baseline is the removed 70/30 blend, reimplemented inside the script only.

Options: `--rrf-k` (sweeps 5, 10, 20, 30), `--model`, and `--no-embeddings`. The script also reports encode time, the duplicate count, the distribution of semantic similarities (which sets τ), and how many candidates the evidence floor removes. It uses the live network, so it does not run in CI.

### Stage 2 — labeled fixtures

- Record the raw search results (ddgs output plus page context) for the evaluated queries as JSON under `tests/fixtures/ranking/`.
- Every assertion is one of three kinds:
  - **known-item**: the named product's official page ranks in the top 3;
  - **pairwise**: `rank(A) < rank(B)`;
  - **negative**: a given URL is not in the top 5.
- The bar for adding an assertion is "I would call it a bug if this regressed". Ambiguous cases are left out.
- Labels are chosen from the script's disagreements, after it has been run.
- Fixtures that need semantic ranking skip cleanly when the extra isn't installed. Keyword and TF-IDF fixtures always run.

### Unit and protocol tests

- `fusion.py`: small known rankings, tie-breaking, the effect of `k`, one, two and three signals, and exclusion of error results.
- Evidence floor: candidates that survive on keyword evidence alone, candidates that survive on semantic similarity alone, candidates with neither are dropped, a keywordless query skips the keyword condition, and unavailable semantics fall back to keyword evidence.
- Deduplication key normalization.
- `embeddings.py`:
  - concurrent first calls load the model only once (several threads racing a deliberately slow fake loader);
  - a load failure is remembered;
  - `not_installed` is reported when the import fails;
  - `encode_failed` is reported on both an exception and a timeout.
- Protocol level, through `create_connected_server_and_client_session`: with the network patched out, `search_nvidia` returns a `SEMANTIC_RANKING_UNAVAILABLE` warning when model loading is forced to fail. This goes through the client, not the handler, per the lesson from the `AnyUrl` bug.
- The suite has no network mocking today. This adds one seam: `_fetch_ddgs_results` and `fetch_url_context` are patched to return fixture data.

## 9. Public surface changes

These all go into the 0.9.0 CHANGELOG entry:

- Result order changes for every `search_nvidia` query.
- `relevance_score` is now derived from fused rank. Its description in `outputSchema` is updated, and the generated SDKs pick up the new text.
- `min_relevance_score` now cuts the fused ranking, after an evidence floor removes candidates with no keyword match and low semantic similarity. Queries with only weak matches can return fewer results.
- The new optional extra `[embeddings]` and its environment variables.
- The new warning code `SEMANTIC_RANKING_UNAVAILABLE`.
- Duplicate URLs are no longer listed twice.

## 10. Risks

| Risk | Mitigation |
| --- | --- |
| PR #8 carries the protective `mcp<2` pin, and bundling this feature delays it | Trade-off accepted, 2026-09-12 |
| Merging PR #8 deploys production, because Railway builds from `main` | Docker prefetch plus the lazy fallback; smoke test before merging |
| Encode latency on production CPUs is unmeasured | Worker thread plus timeout; the eval script measures it |
| The ranking change surprises clients | CHANGELOG entry and labeled fixtures |
| On Python 3.10, fastembed caps onnxruntime below 1.24 | The CI matrix (3.10–3.12) installs the extra |

## 11. Resolved questions

Both were raised at spec review and resolved on 2026-09-12.

1. **The cutoff would lose its absolute quality floor.** With rank-derived scores alone, `min_relevance_score = 17` keeps the top ~83% of candidates on every query, even when every result is poor.
   **Resolved:** add an evidence floor before fusion, then apply `min_relevance_score` to the fused ranking. See "Evidence floor" in §4.
2. **Warning when the extra isn't installed.** A warning on every query for every install that never opted in would appear on 100% of those responses and teach clients to ignore warnings.
   **Resolved:** warn only when the extra is installed but failing. "Not installed" is a supported mode, logged once.

## 12. Relationship to 1.0.0

1.0.0 is a pure protocol and SDK migration and makes no ranking changes. `mcp2` merges `main` after this release ships, and the migration's tests must not assert result order or specific scores.
