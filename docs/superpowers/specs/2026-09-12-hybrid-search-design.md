<!-- markdownlint-disable MD013 -->

# Hybrid Search (BM25 + Semantic) — Design

- **Date:** 2026-09-12
- **Status:** Approved 2026-09-12. Revised the same day to use BM25 as the single lexical signal (§12).
- **Target release:** `mcp-nvidia` 0.9.0, on the `mcp` 1.x SDK
- **Branch:** `chore/pin-mcp-v1` (PR #8)
- **Inherited by:** 1.0.0, when `mcp2` merges `main`
- **Related:** `docs/decisions/2026-09-12-no-page-fetch-cache.md`

## 1. Context

`search_nvidia` ranks live results; there is no index. `search_all_domains` (`src/mcp_nvidia/lib/search.py:211`) runs one `ddgs` query per domain, fetches each result's page, and then scores the combined list in a single loop:

1. `calculate_tfidf_scores(all_results, query)`: sklearn TF-IDF cosine similarity over the result set.
2. For each result, `get_domain_boost(domain, query)`, then `calculate_search_relevance(result, query, domain_boost)`: a 0–100 keyword heuristic built from exact, fuzzy and phrase matches.
3. A hardcoded blend, `int(keyword_score * 0.7 + tfidf_score * 100 * 0.3)`, capped at 100.
4. Filter on `min_relevance_score` (default 17), then sort.

Steps 1 and 2 both measure word overlap. If a semantic signal were simply added as a third input, word overlap would get two votes to meaning's one. This design replaces both lexical signals with a single BM25 score and fuses it with semantic similarity.

### Measured baseline

| Quantity | Value |
| --- | --- |
| Candidates per default query | 15 domains × 3 = 45 |
| Wall time, default query | 30.3 s (23 of 45 results kept) |
| Page fetch, per result | ~0.75–0.83 s |
| URL recurrence across queries | 0% (3 queries, 30 URL slots) |
| Duplicate URLs within one query | 1 in 85 slots, all from subdomain overlap |
| Encode, query + 45 docs | ~0.6–3 s on a dev box under load |
| Core query terms across candidates | `cuda` in 34/43 and `memory` in 35/43 results for "CUDA memory management" |
| IDF of an expansion variant | `trt` 1.83 vs `tensorrt` 0.47 (Lucene IDF, "TensorRT inference optimization") |
| Production resources | 8 vCPU, 8 GB; 7-day average 0.0017 vCPU, 0.26 GB |

## 2. Goals and non-goals

### Goals

- Add semantic similarity as a ranking signal for `search_nvidia`.
- Replace the keyword heuristic and TF-IDF with one BM25 score, so word overlap and meaning carry equal weight in fusion.
- Replace the fixed 70/30 blend with rank fusion.
- Leave the base install unchanged: embeddings are opt-in, and without them search ranks on BM25 alone.
- Make degraded ranking visible to clients, not only in server logs.
- Make ranking regressions detectable before users notice them.

### Non-goals

- `discover_nvidia_content`. It keeps its current scoring, including `calculate_tfidf_scores`, so `relevance.py` is not modified. Extending hybrid ranking to it is an explicit follow-up.
- Typo-tolerant matching. BM25 matches stems exactly, and the old fuzzy matcher is not carried over (§4).
- A vector index (pgvector or otherwise) and a page-fetch cache. See the decision record.
- Chunked, page-level embeddings. The design leaves a seam for them and builds nothing.
- GPU inference.

## 3. Decisions

| # | Decision | Rationale |
| --- | --- | --- |
| 1 | BM25 over title + snippet replaces both the keyword heuristic and TF-IDF in search | Both measured word overlap, so fusing them with semantics would double-count it. BM25 covers what they did: term rarity (IDF), term-frequency saturation and length normalization. |
| 2 | BM25 is implemented in-house with Lucene's non-negative IDF, `k1 = 1.2`, `b = 0.75` | On these candidates the Okapi IDF goes negative for core query terms (`cuda` −1.29), which would rank pages mentioning CUDA *below* pages that don't. `rank-bm25` uses Okapi IDF and was last released in February 2022. The in-house version is ~40 lines with no new dependency. |
| 3 | Porter stemming, with stopwords removed | Recovers word-form matches the fuzzy matcher used to catch (`downloads`→`download`, `installation`→`instal`). nltk is already a dependency; the stemmer works offline. |
| 4 | Title terms count twice | Preserves the old scorer's title-over-snippet weighting. |
| 5 | The BM25 score is multiplied by `get_domain_boost` | Preserves the existing boost for docs, research and news sites, cheaply. |
| 6 | Terms that come only from query expansion count at half weight | Expansion variants are rare in results, so IDF weights them far above the real query terms: `trt` 1.83 vs `tensorrt` 0.47. |
| 7 | Reciprocal rank fusion over the BM25 and semantic rankings; the 70/30 blend is removed | Fusion compares positions rather than scores, so signals in different units need no rescaling. |
| 8 | `k = 30`, provisional until measured | With two signals, a result ranked first by one signal still beats one ranked 15th by both up to k≈35; from k≈40 agreement wins. 30 sits just on the side where a strong top hit can win. The textbook 60 was tuned on ~1000-document lists. |
| 9 | `relevance_score` stays a required integer from 0–100, now derived from fused rank | The field and its type are part of the declared output schema and the generated SDKs. Its meaning was a baseline, not a contract. |
| 10 | Embeddings via `fastembed`, shipped as the optional extra `[embeddings]` | Runs on onnxruntime with no torch; 178 MB installed. |
| 11 | Model `BAAI/bge-small-en-v1.5` | Pads only to the longest input in the batch (a title is ~16 tokens). fastembed's `all-MiniLM-L6-v2` pads every input to 128 tokens, so short inputs cost as much as long ones. 67 MB, 384 dimensions. |
| 12 | Embed the title plus `snippet_plain` | The same text BM25 reads, without the `**` highlight markers. The one function that builds this input is the seam for chunking. |
| 13 | Load lazily, exactly once, under a lock; encode in a worker thread | Import and stdio startup are unaffected, and concurrent first queries still trigger only one load. Encoding is 0.6–3 s of synchronous CPU work that would otherwise freeze the event loop for every client. |
| 14 | Prefetch weights in the Docker build; download lazily for pip installs | Avoids a ~10 s download on every container cold start. |
| 15 | On failure of an installed extra, rank on BM25 and report it in `warnings`; "not installed" is a supported mode with no warning | Degraded ranking shows up where clients look, and warnings keep their meaning. |
| 16 | Move the existing `deduplicate_results` call ahead of scoring | It already keeps duplicates out of client output but runs after scoring, so duplicates inflate document frequencies and each copy would take its own rank position. |
| 17 | An evidence floor before fusion drops candidates that match no query term *and* have semantic similarity below τ | Rank-derived scores are relative; without a floor, an all-junk query still returns most of its candidates. τ comes from measurement. |
| 18 | Evaluate with a comparison script first, then with labeled fixtures | Labels come from judging real disagreements between rankings, not from an imagined ideal order. |

## 4. Lexical scoring (BM25)

A new pure module, `src/mcp_nvidia/lib/lexical.py`, with no I/O.

### Tokenization

Query and documents share one tokenizer:

1. Lowercase, then split on every run of characters that are not ASCII letters or digits.
2. Drop tokens shorter than two characters, tokens with no letter, and stopwords (the same `STOPWORDS` set `relevance.py` uses).
3. Porter-stem what remains.

### Document text

A result's tokens are its title tokens repeated `TITLE_WEIGHT = 2` times, followed by the tokens of `snippet_plain` (or `snippet` with `**` markers removed). The URL is not indexed.

### Query terms

Each distinct stem in the expanded query gets a weight: **1.0** if it also occurs in the user's original query, **`EXPANSION_TERM_WEIGHT = 0.5`** if it comes only from expansion. `search_all_domains` keeps the original query for this purpose; its signature does not change.

### Formula

```text
idf(t)  = ln(1 + (N − df(t) + 0.5) / (df(t) + 0.5))       Lucene form; never negative
bm25(d) = Σ over query terms t in d of
          w(t) · idf(t) · tf(t,d) · (k1 + 1) / (tf(t,d) + k1 · (1 − b + b · |d| / avgdl))
k1 = 1.2,  b = 0.75
lexical_score(d) = bm25(d) × get_domain_boost(domain(d), expanded_query)
```

`N`, `df` and `avgdl` are computed over this query's candidates, after deduplication.

### IDF on a candidate set that was retrieved for the query

Every candidate was fetched *because* it matches the query, so the query's core terms appear in most of them. For "CUDA memory management", `cuda` is in 34 of 43 results and `memory` in 35. That has two consequences:

- The textbook Okapi IDF goes **negative** for those terms (−1.29, −1.43), so a page mentioning CUDA would score lower than one that doesn't. Lucene's form keeps them positive (0.243, 0.215).
- Even with Lucene IDF, core terms carry less weight than rarer ones. Within a set already retrieved for the query this is reasonable: terms every candidate shares can't separate them. It is also why expansion variants need down-weighting (decision 6); otherwise a rare variant dominates.

The eval script prints each term's document frequency, IDF and weight per query, so this behaviour is inspected on real data rather than assumed.

### What is not carried over

- Typo-tolerant fuzzy matching.
- The phrase bonus.
- Matching terms in the URL.

Stemming covers word forms, and semantic similarity covers some paraphrase and phrasing. `calculate_search_relevance` and `calculate_tfidf_scores` stay in `relevance.py`, unmodified: `discover_nvidia_content` still uses `calculate_tfidf_scores`. Search simply stops calling them.

### Interface

```python
TITLE_WEIGHT = 2
EXPANSION_TERM_WEIGHT = 0.5
BM25_K1 = 1.2
BM25_B = 0.75

def tokenize(text: str) -> list[str]: ...
def query_term_weights(original_query: str, expanded_query: str) -> dict[str, float]: ...
def document_tokens(result: Mapping[str, Any]) -> list[str]: ...
def bm25_scores(documents: Sequence[Sequence[str]], query_weights: Mapping[str, float]) -> list[float]: ...
```

## 5. Scoring pipeline

### Fusion

```text
fused(d) = Σ over available signals s of  1 / (k + rank_s(d))      k = 30, ranks start at 1
```

- Both signals rank the same candidate set, since this is pure reranking. No result can be missing from a signal's list, so fusion behaves like a smoothed Borda count rather than classic multi-system RRF.
- Error results (`is_error`) are excluded from fusion and scored 0.
- Ties within a signal are broken by original candidate order, so output is deterministic and fixtures stay stable.

How `k` trades a strong top hit against agreement, for two signals and 45 candidates. A is ranked first by one signal and 45th by the other; B is ranked 15th by both:

| k | A | B | Winner |
| --- | --- | --- | --- |
| 10 | 0.10909 | 0.08000 | A (strong top hit) |
| 20 | 0.06300 | 0.05714 | A |
| **30** | **0.04559** | **0.04444** | **A, narrowly** |
| 35 | 0.04028 | 0.04000 | A, barely |
| 40 | 0.03615 | 0.03636 | B (agreement) |
| 60 | 0.02592 | 0.02667 | B |

A new pure module, `src/mcp_nvidia/lib/fusion.py`, with no I/O:

```python
RRF_K = 30
EVIDENCE_FLOOR_TAU: float  # set from eval-script measurement before release

def rank_by_score(scores: Sequence[float]) -> list[int]: ...
def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]: ...
def relevance_from_position(position: int, n: int) -> int: ...
def apply_evidence_floor(
    lexical_scores: Sequence[float],
    similarities: Sequence[float] | None,
    tau: float = EVIDENCE_FLOOR_TAU,
    query_has_terms: bool = True,
) -> list[int]: ...
```

The result at fused position `p` (counting from 0), out of `n` surviving candidates, gets `relevance_score = round(100 * (n - p) / n)`. The top result scores 100; with `n = 45`, the last scores 2.

### Where it slots in

`search_all_domains` keeps its signature and its return shape. After results are gathered:

1. Deduplicate with the existing `deduplicate_results` (§9), moved here from after the cutoff.
2. Lexical scores: BM25 (§4) multiplied by the domain boost.
3. `semantic_rank` (§6) on the user's **original** query, which returns similarities and a ranking, or reports itself unavailable. Expansion variants such as `tensor rt trt` help lexical matching but only add noise to an embedding.
4. **Evidence floor** (below).
5. `reciprocal_rank_fusion` over the survivors' BM25 ranking, plus their semantic ranking when available, giving the fused order and each `relevance_score`.
6. The `min_relevance_score` filter on the fused ranking, then `sort_by` as today. The `date` and `domain` sorts are unchanged.

### Evidence floor

Rank-derived scores are relative. On their own, `min_relevance_score = 17` would keep the top ~83% of candidates on every query, even when all of them are poor. So an absolute gate runs before fusion. A candidate is **dropped** only when both of these hold:

- its lexical score is 0, meaning no query stem appears in its title or snippet; and
- its semantic similarity to the query is below a threshold τ.

Edge cases:

- **Semantic ranking unavailable:** the floor uses lexical evidence alone.
- **The query has no terms after tokenization** (for example, it is all stopwords): every lexical score is 0, so lexical evidence is ignored rather than counted as missing. With semantic ranking available the floor gates on similarity alone; without it, every candidate is kept. Treating those zeros as missing evidence would drop every candidate.

τ is not guessed. The eval script measures the similarity distribution of candidates that match no query term, and τ is set from that measurement before release. It is a named constant in `fusion.py`, next to `RRF_K`.

### Behaviour change

Result order changes for every query, including on installs without the extra, because BM25 replaces both previous lexical signals and the 70/30 blend is gone.

## 6. Embedding module

A new module, `src/mcp_nvidia/lib/embeddings.py`:

```python
@dataclass(frozen=True)
class SemanticRanking:
    order: list[int] | None              # candidate indices, most similar first
    similarities: list[float] | None     # cosine similarity per candidate, aligned to the input
    unavailable: str | None              # reason when order is None

async def semantic_rank(query: str, results: Sequence[Mapping[str, Any]]) -> SemanticRanking: ...
```

`semantic_rank` returns the similarity values as well as the order, because the evidence floor compares values against τ.

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

## 7. Failure and degradation

| Situation | `unavailable` | Ranking signals | `warnings` entry | Server log |
| --- | --- | --- | --- | --- |
| Extra not installed | `not_installed` | BM25 | none: a supported mode, not a failure | once, INFO, on first query |
| Model fails to load | `model_load_failed` | BM25 | yes | once, WARNING, with the cause |
| Encode raises or times out | `encode_failed` | BM25 | yes | each time, WARNING |

The warning entry follows the shape of existing warnings. `outputSchema` types `warnings` items as plain objects, so no schema change is needed:

```json
{
  "code": "SEMANTIC_RANKING_UNAVAILABLE",
  "message": "Semantic ranking unavailable; results ranked by BM25 only",
  "reason": "model_load_failed"
}
```

### Adjacent known issue (not fixed here)

`ddgs` raises `DDGSException("No results found.")` when a domain has no matches. `search_nvidia_domain` handles it as a failure, so a domain that simply has nothing to return shows up as a fake "Search temporarily unavailable" result plus a `PARTIAL_FAILURE` warning. This design uses the same `warnings` channel, so the issue makes warnings noisier than they should be. It is recorded as a follow-up.

## 8. Public surface changes

These all go into the 0.9.0 CHANGELOG entry:

- Result order changes for every `search_nvidia` query.
- `relevance_score` is now derived from fused rank. Its description in `outputSchema` is updated, and the generated SDKs pick up the new text.
- `min_relevance_score` now cuts the fused ranking, after an evidence floor removes candidates that match no query term and have low semantic similarity. Queries with only weak matches can return fewer results.
- Typo-tolerant fuzzy matching, the phrase bonus and URL term matching no longer affect search ranking.
- The new optional extra `[embeddings]` and its environment variables.
- The new warning code `SEMANTIC_RANKING_UNAVAILABLE`.

## 9. Deduplication

**Deduplication already exists; this design only moves it.** `deduplicate_results` (`src/mcp_nvidia/lib/deduplication.py`) drops exact-URL repeats, and drops results whose title (similarity ≥ 0.85) and snippet (≥ 0.90) are both near-identical, keeping the first occurrence. `search_all_domains` currently calls it *after* scoring and after the `min_relevance_score` cutoff, so client output is already free of duplicates.

The problem is ordering. Duplicates are scored before they are removed, which inflates the document frequencies BM25's IDF depends on, and under rank fusion each copy would take its own rank position, skewing every rank-derived score below it. The existing call moves to run immediately after gathering, before any scoring. The function itself is not changed.

Measured duplicate rate: 1 in 85 URL slots. Duplicates come from `site:` matching subdomains: `ngc.nvidia.com` also matches `catalog.ngc.nvidia.com`, and `docs.nvidia.com` also matches `docs.omniverse.nvidia.com`.

> **Correction.** An earlier draft of this spec proposed a new, URL-normalizing dedupe and said duplicates reached clients. Reading the code showed that an existing dedupe already removes them from output. The real fix is where that step sits in the pipeline.

## 10. Evaluation and testing

### Stage 1 — comparison script

`scripts/eval_ranking.py` runs real queries. For each one it prints the baseline ranking beside the new ranking, ordered by how far each result moved. The baseline is the removed 70/30 blend of the keyword heuristic and TF-IDF, reimplemented inside the script only.

Options: `--rrf-k` (sweeps 10, 20, 30, 40, 60), `--tau`, `--title-weight`, `--expansion-weight`, `--model` and `--no-embeddings`. For each query the script also reports:

- each query term's document frequency, IDF and weight;
- encode time and the duplicate count;
- the similarity distribution of candidates that match no query term, which is what sets τ;
- how many candidates the evidence floor removes.

It uses the live network, so it does not run in CI.

### Stage 2 — labeled fixtures

- Record the raw search results (ddgs output plus page context) for the evaluated queries as JSON under `tests/fixtures/ranking/`.
- Every assertion is one of three kinds:
  - **known-item**: the named product's official page ranks in the top 3;
  - **pairwise**: `rank(A) < rank(B)`;
  - **negative**: a given URL is not in the top 5.
- Every assertion runs in one mode: **lexical** (BM25 only; always runs) or **semantic** (with the real model; skips without the extra).
- The bar for adding an assertion is "I would call it a bug if this regressed". Ambiguous cases are left out.
- Labels are chosen from the script's disagreements, after it has been run.

### Unit and protocol tests

- `lexical.py`:
  - tokenization splits on punctuation, drops stopwords and short or letterless tokens, and stems;
  - title terms count twice;
  - expansion-only terms get half weight;
  - IDF stays positive when a term appears in every document;
  - a document containing no query term scores 0, and a query with no terms scores 0 everywhere.
- `fusion.py`: small known rankings, tie-breaking, the effect of `k` (a strong top hit wins at `k = 30` and loses at `k = 60` with two signals), and exclusion of error results.
- Evidence floor: candidates that survive on lexical evidence alone, candidates that survive on semantic similarity alone, candidates with neither are dropped, a query with no terms skips the lexical condition, and unavailable semantics fall back to lexical evidence.
- Dedupe ordering: a page returned by two domain searches is scored once, and fused positions count only unique candidates.
- `embeddings.py`:
  - concurrent first calls load the model only once (several threads racing a deliberately slow fake loader);
  - a load failure is remembered;
  - `not_installed` is reported when the import fails;
  - `encode_failed` is reported on both an exception and a timeout.
- Protocol level, through `create_connected_server_and_client_session`: with the network patched out, `search_nvidia` returns a `SEMANTIC_RANKING_UNAVAILABLE` warning when model loading is forced to fail. This goes through the client, not the handler, per the lesson from the `AnyUrl` bug.
- The suite has no network mocking today. This adds one seam: `_fetch_ddgs_results` and `fetch_url_context` are patched to return fixture data.

## 11. Risks

| Risk | Mitigation |
| --- | --- |
| PR #8 carries the protective `mcp<2` pin, and bundling this feature delays it | Trade-off accepted, 2026-09-12 |
| Merging PR #8 deploys production, because Railway builds from `main` | Docker prefetch plus the lazy fallback; smoke test before merging |
| Encode latency on production CPUs is unmeasured | Worker thread plus timeout; the eval script measures it |
| Losing fuzzy matching hurts typo'd queries | Stemming covers word forms; semantic similarity covers some misspellings; fixtures catch known-item regressions |
| IDF computed on ~45 query-biased candidates is noisy | Lucene IDF, expansion down-weighting, and per-term IDF printed by the eval script |
| The ranking change surprises clients | CHANGELOG entry and labeled fixtures |
| On Python 3.10, fastembed caps onnxruntime below 1.24 | The CI matrix (3.10 and 3.12) installs the extra |

## 12. Resolved questions and revisions

Raised at spec review and resolved on 2026-09-12:

1. **The cutoff would lose its absolute quality floor.** With rank-derived scores alone, `min_relevance_score = 17` keeps the top ~83% of candidates on every query, even when every result is poor.
   **Resolved:** add an evidence floor before fusion, then apply `min_relevance_score` to the fused ranking (§5).
2. **Warning when the extra isn't installed.** A warning on every query for every install that never opted in would appear on 100% of those responses and teach clients to ignore warnings.
   **Resolved:** warn only when the extra is installed but failing. "Not installed" is a supported mode, logged once.

Revised the same day, after the design was first approved:

1. **BM25 replaces the keyword heuristic and TF-IDF** as the single lexical signal, so word overlap no longer gets two votes in fusion. Measuring term frequencies on real results led to three follow-on decisions:
   - Lucene's non-negative IDF instead of Okapi's, which goes negative for core query terms (decision 2);
   - expansion-only terms at half weight, because rare variants otherwise dominate (decision 6);
   - `k` moved from 10 to a provisional 30, because with two signals instead of three the balance between a strong top hit and agreement shifts (decision 8).

   Kept from the old scorer: the domain boost, title weighting and (via stemming) word-form matching. Dropped: fuzzy typo matching, the phrase bonus and URL term matching.

## 13. Relationship to 1.0.0

1.0.0 is a pure protocol and SDK migration and makes no ranking changes. `mcp2` merges `main` after this release ships, and the migration's tests must not assert result order or specific scores.
