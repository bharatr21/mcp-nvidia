<!-- markdownlint-disable MD013 -->

# Hybrid Search (BM25 + Semantic) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rank `search_nvidia` results by fusing a BM25 lexical ranking with an optional local-embedding similarity ranking via reciprocal rank fusion, behind an evidence floor, for the 0.9.0 release.

**Architecture:** Three new modules:

- `lib/lexical.py` — BM25 over title and snippet, with stemming and Lucene IDF. Pure; no I/O.
- `lib/fusion.py` — rank fusion, the evidence floor and rank-derived scores. Pure; no I/O.
- `lib/embeddings.py` — a fastembed model loaded lazily and only once; encoding runs off the event loop and reports explicit failure reasons.

`search_all_domains` keeps its signature and return shape. It moves the existing dedupe ahead of scoring and replaces the keyword heuristic, TF-IDF and their fixed 70/30 blend with a call to a new `_rank_candidates`. Everything is testable offline through two patched network seams.

**Tech Stack:** Python ≥3.10; the `mcp` 1.x SDK (1.30.0 installed); fastembed 0.8 (onnxruntime, no torch) as an optional extra; nltk's Porter stemmer (nltk is already a dependency); numpy (already present via scikit-learn); pytest with pytest-asyncio (`asyncio_mode = "auto"`); hatchling; GitHub Actions; Docker.

**Spec:** `docs/superpowers/specs/2026-09-12-hybrid-search-design.md` (approved 2026-09-12, revised the same day for BM25). Background measurements: `docs/decisions/2026-09-12-no-page-fetch-cache.md`.

## Global Constraints

Every task's requirements implicitly include these.

- **Where:** branch `chore/pin-mcp-v1` (PR #8, draft). Run every command from the repo root of that checkout (locally, `.claude/worktrees/release-0.9.0`) with its venv: `.venv/bin/pytest` and `.venv/bin/python`. Ruff is not installed in the venv, so run the version pre-commit pins: `uv tool run ruff@0.8.4`.
- **The SDK stays on 1.x:** do not change `mcp>=1.28,<2.0.0`. Python floor is `>=3.10`.
- **Baseline before this plan:** `.venv/bin/pytest tests/ -q` → **48 passed, 11 skipped** (mcp 1.30.0). The 11 skips are in `tests/test_client.py`, which needs an installed `mcp-nvidia` binary.
- **Optional extra:** `embeddings = ["fastembed>=0.8,<0.9"]`. The base install must keep working without fastembed, so nothing may import it at module import time.
- **BM25 (exact):**
  - Tokenize: lowercase; split on every run of characters that are not `[0-9a-z]`; drop tokens shorter than 2 characters, tokens with no letter, and anything in `STOPWORDS` from `mcp_nvidia.lib.relevance`; Porter-stem the rest.
  - A document is its title tokens repeated `TITLE_WEIGHT = 2` times, then the tokens of `snippet_plain` (or `snippet` with `**` removed). URLs are not indexed.
  - Query terms: every distinct stem from the original query weighs `1.0`; stems that appear only in the expanded query weigh `EXPANSION_TERM_WEIGHT = 0.5`.
  - `idf(t) = ln(1 + (N − df + 0.5) / (df + 0.5))` (Lucene form, never negative). `bm25(d) = Σ w(t) · idf(t) · tf · (k1 + 1) / (tf + k1 · (1 − b + b · |d| / avgdl))`, with `k1 = 1.2` and `b = 0.75`. `N`, `df` and `avgdl` come from this query's deduplicated candidates.
  - `lexical_score = bm25 × get_domain_boost(domain, expanded_query)`.
- **Model:** `BAAI/bge-small-en-v1.5`. Env vars: `MCP_NVIDIA_EMBEDDING_MODEL` (defaults to that model), `MCP_NVIDIA_EMBEDDING_THREADS` (default `4`), and fastembed's own `FASTEMBED_CACHE_PATH`. Read them with `os.getenv`, like `MCP_NVIDIA_DOMAINS` and `MCP_NVIDIA_LOG_LEVEL`.
- **Semantic ranking embeds the user's original query**, not the expanded one: expansion variants such as `tensor rt trt` only add noise to an embedding. Encode the query with `query_embed` and documents with `embed`; bge is asymmetric.
- **The model loads lazily, exactly once per process, under a `threading.Lock`.** Threads that arrive during a load wait on the lock. A load failure is remembered for the lifetime of the process. Loading and encoding both run via `asyncio.to_thread`, and encoding has a 10 s timeout.
- **Fusion:** `RRF_K = 30`, provisional until Task 6 measures it. Ranks are 1-based inside `1 / (k + rank)`, and ties are broken by original candidate index.
- **Score:** `relevance_score = round(100 * (n - p) / n)` for 0-based fused position `p` among `n` surviving candidates. It remains a required `integer` in the output schema.
- **Evidence floor (exact semantics):**
  - With similarities available, keep a candidate if its lexical score is `> 0` (and the query has terms), **or** if its similarity is `>= τ`.
  - Without similarities, keep it if its lexical score is `> 0`. If the query has no terms, keep every candidate.
  - `τ` is `EVIDENCE_FLOOR_TAU` in `fusion.py`: a provisional `0.5` until Task 6 replaces it with a measured value.
- **Warning:** append this to `warnings` **only** when semantics are installed but fail (`model_load_failed` or `encode_failed`), never for `not_installed`:

  ```json
  {"code": "SEMANTIC_RANKING_UNAVAILABLE", "message": "Semantic ranking unavailable; results ranked by BM25 only", "reason": "model_load_failed"}
  ```

- **Do not modify** `lib/relevance.py`, `lib/deduplication.py`, `discover_nvidia_content`, or the signature and return shape of `search_all_domains`. `discover_nvidia_content` still uses `calculate_tfidf_scores`. Search stops importing `calculate_search_relevance` and `calculate_tfidf_scores`, but both stay in `relevance.py` and in `mcp_nvidia.lib`'s exports.
- **Tests never touch the network.** Patch `mcp_nvidia.lib.search._fetch_ddgs_results` and `mcp_nvidia.lib.search.fetch_url_context`, the names *as imported into* `search.py`.
- **Test request/response behaviour at the protocol layer**, through `mcp.shared.memory.create_connected_server_and_client_session`. On this very branch, a handler called directly with a Python string hid an `AnyUrl` crash through a fully green suite.
- **Never wrap an MCP client session in an async yield fixture.** pytest-asyncio tears fixtures down in a different task, which breaks anyio cancel scopes. Enter sessions with `async with` inside the test body. Plain sync fixtures such as `monkeypatch` are fine.
- **Do not undraft or merge PR #8.** Railway deploys production from `main`, so that decision belongs to the user, after Task 8.
- **Commit attribution:** every commit message ends with:

  ```text
  Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_013FriD63a2waebg8Qmtscj6
  ```

- Markdown outside `docs/superpowers/` is linted at 120 columns (markdownlint MD013, via pre-commit).

## File Structure

| File | Responsibility | Task |
| --- | --- | --- |
| `src/mcp_nvidia/lib/fusion.py` | **New.** Pure: `rank_by_score`, `reciprocal_rank_fusion`, `relevance_from_position`, `apply_evidence_floor`; constants `RRF_K`, `EVIDENCE_FLOOR_TAU` | 1, 6 |
| `src/mcp_nvidia/lib/lexical.py` | **New.** Pure: `tokenize`, `query_term_weights`, `document_tokens`, `bm25_scores`; constants `TITLE_WEIGHT`, `EXPANSION_TERM_WEIGHT`, `BM25_K1`, `BM25_B` | 2 |
| `src/mcp_nvidia/lib/embeddings.py` | **New.** Load-once model, `semantic_rank`, failure reasons, `build_document_text` | 3 |
| `src/mcp_nvidia/lib/search.py` | Add `_rank_candidates`; keep the original query; move dedupe ahead of scoring; replace the 70/30 blend | 4 |
| `src/mcp_nvidia/server.py` | Update two schema descriptions | 4 |
| `tests/search_fakes.py` | **New.** Offline fakes: `FakeEmbeddingModel`, `install_fake_search` | 3, 4 |
| `tests/test_fusion.py` | **New.** Fusion unit tests | 1 |
| `tests/test_lexical.py` | **New.** BM25 unit tests | 2 |
| `tests/test_embeddings.py` | **New.** Loader and encoder unit tests (no fastembed needed) | 3 |
| `tests/test_search_pipeline.py` | **New.** Pipeline and protocol-level tests with the network patched out | 4 |
| `tests/test_embeddings_model.py` | **New.** Real-model smoke test; skips without fastembed | 5 |
| `pyproject.toml` | `[embeddings]` extra; ruff per-file ignore for `scripts/` if needed | 5, 6 |
| `Dockerfile` | Install the extra; bake the model weights into the image | 5 |
| `.github/workflows/test.yml` | New `embeddings` job | 5, 7 |
| `scripts/eval_ranking.py` | **New.** Live comparison: baseline vs hybrid, per-term IDF, k sweep, τ measurement, fixture recording | 6 |
| `tests/test_ranking_fixtures.py` | **New.** Labeled ranking regressions | 7 |
| `tests/fixtures/ranking/*.json` | **New.** Recorded candidates plus human-judged labels | 6, 7 |
| `CHANGELOG.md`, spec status line | Final accuracy pass | 8 |

---

### Task 1: Rank fusion primitives

Pure functions with no I/O: the whole ranking contract, testable in milliseconds.

**Files:**

- Create: `src/mcp_nvidia/lib/fusion.py`
- Test: `tests/test_fusion.py`

**Interfaces:**

- Consumes: nothing.
- Produces (all in `mcp_nvidia.lib.fusion`):
  - `RRF_K: int = 30` (provisional; Task 6 confirms or changes it)
  - `EVIDENCE_FLOOR_TAU: float = 0.5` (provisional; Task 6 replaces the value)
  - `rank_by_score(scores: Sequence[float]) -> list[int]`
  - `reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]`
  - `relevance_from_position(position: int, n: int) -> int`
  - `apply_evidence_floor(lexical_scores: Sequence[float], similarities: Sequence[float] | None, tau: float = EVIDENCE_FLOOR_TAU, query_has_terms: bool = True) -> list[int]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fusion.py`:

```python
"""Unit tests for rank fusion, rank-derived scores and the evidence floor."""

import pytest

from mcp_nvidia.lib.fusion import (
    RRF_K,
    apply_evidence_floor,
    rank_by_score,
    reciprocal_rank_fusion,
    relevance_from_position,
)


def test_default_k_is_thirty_for_two_signals():
    assert RRF_K == 30


def test_rank_by_score_orders_best_first_and_keeps_ties_stable():
    assert rank_by_score([1, 3, 3, 0]) == [1, 2, 0, 3]


def test_single_ranking_passes_through_unchanged():
    assert reciprocal_rank_fusion([[2, 0, 1]]) == [2, 0, 1]


def test_fusion_ties_break_by_original_index():
    # Candidates 0 and 2 score the same (1/31 + 1/33); candidate 1 scores 2/32, which is lower.
    assert reciprocal_rank_fusion([[0, 1, 2], [2, 1, 0]]) == [0, 2, 1]


def test_k_decides_between_a_strong_top_hit_and_agreement():
    # Two signals over 45 candidates. A (id 0) is 1st in one signal and 45th in the other.
    # B (id 1) is 15th in both. A narrowly wins at k=30; B wins at k=60.
    others = list(range(2, 45))
    first_signal = [0, *others[:13], 1, *others[13:]]
    second_signal = [*others[:14], 1, *others[14:], 0]
    assert len(first_signal) == len(second_signal) == 45
    rankings = [first_signal, second_signal]

    with_k30 = reciprocal_rank_fusion(rankings, k=30)
    with_k60 = reciprocal_rank_fusion(rankings, k=60)

    assert with_k30.index(0) < with_k30.index(1), "k=30 should let the strong top hit win"
    assert with_k60.index(1) < with_k60.index(0), "k=60 should let agreement win"


def test_fusion_rejects_rankings_of_different_candidates():
    with pytest.raises(ValueError, match="permutation"):
        reciprocal_rank_fusion([[0, 1, 2], [0, 1]])


def test_fusion_of_nothing_is_empty():
    assert reciprocal_rank_fusion([]) == []


def test_relevance_from_position():
    assert relevance_from_position(0, 45) == 100
    assert relevance_from_position(44, 45) == 2
    assert relevance_from_position(0, 1) == 100


def test_floor_keeps_lexical_evidence_without_semantics():
    assert apply_evidence_floor([1.5, 0.0], similarities=None) == [0]


def test_floor_keeps_semantic_evidence_without_lexical_match():
    assert apply_evidence_floor([0.0, 0.0], similarities=[0.9, 0.1], tau=0.5) == [0]


def test_floor_drops_candidates_with_neither():
    assert apply_evidence_floor([2.0, 0.0, 0.0], similarities=[0.1, 0.8, 0.2], tau=0.5) == [0, 1]


def test_termless_query_without_semantics_keeps_everything():
    assert apply_evidence_floor([0.0, 0.0, 0.0], similarities=None, query_has_terms=False) == [0, 1, 2]


def test_termless_query_with_semantics_gates_on_similarity_alone():
    assert apply_evidence_floor([0.0, 0.0], similarities=[0.9, 0.2], tau=0.5, query_has_terms=False) == [0]


def test_floor_rejects_misaligned_similarities():
    with pytest.raises(ValueError, match="align"):
        apply_evidence_floor([1.0, 2.0], similarities=[0.5])
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `.venv/bin/pytest tests/test_fusion.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'mcp_nvidia.lib.fusion'`.

- [ ] **Step 3: Implement the module**

Create `src/mcp_nvidia/lib/fusion.py`:

```python
"""Rank fusion, rank-derived relevance scores and the evidence floor.

Pure functions with no I/O. Every signal ranks the *same* candidate set, so fusion is
reranking: there is no penalty for being absent from a list, and it behaves like a
smoothed Borda count rather than classic multi-system reciprocal rank fusion.
"""

from collections.abc import Sequence

# Not the textbook 60, which was tuned on TREC lists of ~1000 documents. With two signals
# (BM25 and semantic) over ~45 candidates, a result ranked first by one signal still beats
# one ranked 15th by both up to k~35; from k~40 agreement wins. 30 sits just on the side
# where a strong top hit can still win. Provisional: confirmed with scripts/eval_ranking.py.
RRF_K = 30

# Semantic similarity below which a candidate matching no query term is dropped.
# Provisional: replaced with a measured value from scripts/eval_ranking.py before release.
EVIDENCE_FLOOR_TAU = 0.5


def rank_by_score(scores: Sequence[float]) -> list[int]:
    """Candidate indices ordered best-first; equal scores keep their original order."""
    return sorted(range(len(scores)), key=lambda index: (-scores[index], index))


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = RRF_K) -> list[int]:
    """Fuse best-first rankings of the same candidates into one best-first order."""
    if not rankings:
        return []

    n = len(rankings[0])
    for ranking in rankings:
        if sorted(ranking) != list(range(n)):
            raise ValueError("every ranking must be a permutation of the same candidates")

    fused = [0.0] * n
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            fused[index] += 1.0 / (k + rank)

    return sorted(range(n), key=lambda index: (-fused[index], index))


def relevance_from_position(position: int, n: int) -> int:
    """0-100 score for a 0-based fused position among n candidates; the top result is 100."""
    if n <= 0:
        raise ValueError("n must be positive")
    return round(100 * (n - position) / n)


def apply_evidence_floor(
    lexical_scores: Sequence[float],
    similarities: Sequence[float] | None,
    tau: float = EVIDENCE_FLOOR_TAU,
    query_has_terms: bool = True,
) -> list[int]:
    """Indices of candidates with some evidence of relevance, in original order.

    Rank-derived scores are relative, so without this gate a query whose results are all
    poor would still return most of them. A candidate is dropped only when it matches no
    query term and (when semantics are available) its similarity is below tau.
    """
    if similarities is not None and len(similarities) != len(lexical_scores):
        raise ValueError("similarities must align with lexical_scores")

    kept = []
    for index, lexical_score in enumerate(lexical_scores):
        has_lexical_evidence = query_has_terms and lexical_score > 0
        if similarities is None:
            # A query with no terms offers no lexical evidence to gate on, so nothing is dropped.
            keep = has_lexical_evidence or not query_has_terms
        else:
            keep = has_lexical_evidence or similarities[index] >= tau
        if keep:
            kept.append(index)
    return kept
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `.venv/bin/pytest tests/test_fusion.py -q`
Expected: `14 passed`.

- [ ] **Step 5: Lint**

Run: `uv tool run ruff@0.8.4 check src/mcp_nvidia/lib/fusion.py tests/test_fusion.py && uv tool run ruff@0.8.4 format --check src/mcp_nvidia/lib/fusion.py tests/test_fusion.py`
Expected: no findings. If `ruff check` reports fixable findings, such as import order, run `uv tool run ruff@0.8.4 check --fix` on the two files. If `ruff format --check` reports a diff, run `uv tool run ruff@0.8.4 format` on them. Re-run the tests after either.

- [ ] **Step 6: Commit**

```bash
git add src/mcp_nvidia/lib/fusion.py tests/test_fusion.py
git commit -F - <<'EOF'
feat: Add rank fusion primitives for hybrid search

Pure functions for reciprocal rank fusion (provisional k=30 for two
signals), rank-derived relevance scores and the evidence floor, with
unit tests including the k=30 vs k=60 crossover.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013FriD63a2waebg8Qmtscj6
EOF
```

---

### Task 2: BM25 lexical scoring

Replaces both the keyword heuristic and TF-IDF with one BM25 score. Pure and offline.

**Files:**

- Create: `src/mcp_nvidia/lib/lexical.py`
- Test: `tests/test_lexical.py`

**Interfaces:**

- Consumes: `STOPWORDS` from `mcp_nvidia.lib.relevance` (read-only).
- Produces (all in `mcp_nvidia.lib.lexical`):
  - Constants: `TITLE_WEIGHT = 2`, `EXPANSION_TERM_WEIGHT = 0.5`, `BM25_K1 = 1.2`, `BM25_B = 0.75`. All are read at call time, so Task 6's script can override them.
  - `tokenize(text: str) -> list[str]`
  - `query_term_weights(original_query: str, expanded_query: str) -> dict[str, float]`
  - `document_tokens(result: Mapping[str, Any]) -> list[str]`
  - `bm25_scores(documents: Sequence[Sequence[str]], query_weights: Mapping[str, float]) -> list[float]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_lexical.py`:

```python
"""Unit tests for BM25 lexical scoring."""

import math

from mcp_nvidia.lib.lexical import bm25_scores, document_tokens, query_term_weights, tokenize


def test_tokenize_splits_punctuation_drops_noise_and_stems():
    assert tokenize("CUDA Toolkit Downloads, GPU-accelerated profiling in the SDK 2.0 x") == [
        "cuda",
        "toolkit",
        "download",
        "gpu",
        "acceler",
        "profil",
        "sdk",
    ]


def test_expansion_only_terms_weigh_half():
    weights = query_term_weights("TensorRT inference optimization", "TensorRT inference optimization tensor rt trt")
    assert weights == {"tensorrt": 1.0, "infer": 1.0, "optim": 1.0, "tensor": 0.5, "rt": 0.5, "trt": 0.5}


def test_stopword_only_query_has_no_terms():
    assert query_term_weights("how to do it", "how to do it") == {}


def test_title_terms_count_twice():
    result = {"title": "CUDA Guide", "snippet_plain": "memory pools"}
    assert document_tokens(result) == tokenize("CUDA Guide") * 2 + tokenize("memory pools")


def test_document_falls_back_to_snippet_without_highlight_markers():
    result = {"title": "CUDA", "snippet": "**bold** text"}
    assert document_tokens(result) == tokenize("CUDA") * 2 + tokenize("bold text")


def test_idf_stays_positive_when_every_document_has_the_term():
    scores = bm25_scores([["cuda"], ["cuda", "memori"]], {"cuda": 1.0})
    assert all(score > 0 for score in scores)
    # Lucene IDF for df = N = 2 is ln(1 + 0.5 / 2.5), which is positive.
    assert math.log(1 + 0.5 / 2.5) > 0


def test_document_without_query_terms_scores_zero():
    assert bm25_scores([["cuda"], ["jetson"]], {"cuda": 1.0})[1] == 0.0


def test_no_query_terms_scores_zero_everywhere():
    assert bm25_scores([["cuda"], ["memori"]], {}) == [0.0, 0.0]


def test_half_weight_lowers_a_rare_expansion_term_contribution():
    documents = [["cuda", "trt"], ["cuda"], ["cuda"]]
    full = bm25_scores(documents, {"cuda": 1.0, "trt": 1.0})[0]
    half = bm25_scores(documents, {"cuda": 1.0, "trt": 0.5})[0]
    assert half < full


def test_no_documents_is_empty():
    assert bm25_scores([], {"cuda": 1.0}) == []
```

- [ ] **Step 2: Run the tests to confirm they fail**

Run: `.venv/bin/pytest tests/test_lexical.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'mcp_nvidia.lib.lexical'`.

- [ ] **Step 3: Implement the module**

Create `src/mcp_nvidia/lib/lexical.py`:

```python
"""BM25 lexical scoring over result titles and snippets.

Pure functions with no I/O. IDF uses Lucene's non-negative form: every candidate was
retrieved *for* the query, so core query terms appear in most of them, and the textbook
Okapi IDF goes negative for exactly those terms (measured: `cuda` in 34 of 43 results).
"""

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from nltk.stem import PorterStemmer

from mcp_nvidia.lib.relevance import STOPWORDS

# Title terms count this many times, preserving the old scorer's title-over-snippet weighting.
TITLE_WEIGHT = 2
# Weight of stems that come only from query expansion. Expansion variants are rare in
# results, so IDF would otherwise weight them far above real terms (measured: trt 1.83 vs tensorrt 0.47).
EXPANSION_TERM_WEIGHT = 0.5
BM25_K1 = 1.2
BM25_B = 0.75

_TOKEN_SPLIT = re.compile(r"[^0-9a-z]+")
_stem = PorterStemmer().stem


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and noise, and stem."""
    tokens = []
    for raw in _TOKEN_SPLIT.split(text.lower()):
        if len(raw) >= 2 and any(char.isalpha() for char in raw) and raw not in STOPWORDS:
            tokens.append(_stem(raw))
    return tokens


def query_term_weights(original_query: str, expanded_query: str) -> dict[str, float]:
    """Weight per distinct query stem: 1.0 for the user's own terms, less for expansion-only ones."""
    original_terms = set(tokenize(original_query))
    return {
        term: (1.0 if term in original_terms else EXPANSION_TERM_WEIGHT)
        for term in dict.fromkeys([*tokenize(original_query), *tokenize(expanded_query)])
    }


def document_tokens(result: Mapping[str, Any]) -> list[str]:
    """Tokens for one result: the title repeated TITLE_WEIGHT times, then the plain snippet."""
    title = tokenize(str(result.get("title", "")))
    snippet = str(result.get("snippet_plain") or result.get("snippet", "")).replace("**", "")
    return title * TITLE_WEIGHT + tokenize(snippet)


def bm25_scores(documents: Sequence[Sequence[str]], query_weights: Mapping[str, float]) -> list[float]:
    """BM25 score per document, with IDF and average length computed over these documents."""
    n = len(documents)
    if n == 0:
        return []
    if not query_weights:
        return [0.0] * n

    average_length = sum(len(document) for document in documents) / n or 1.0
    term_counts = [Counter(document) for document in documents]
    idf = {}
    for term in query_weights:
        document_frequency = sum(1 for counts in term_counts if term in counts)
        idf[term] = math.log(1 + (n - document_frequency + 0.5) / (document_frequency + 0.5))

    scores = []
    for document, counts in zip(documents, term_counts, strict=True):
        length_norm = BM25_K1 * (1 - BM25_B + BM25_B * len(document) / average_length)
        score = 0.0
        for term, weight in query_weights.items():
            frequency = counts.get(term, 0)
            if frequency:
                score += weight * idf[term] * frequency * (BM25_K1 + 1) / (frequency + length_norm)
        scores.append(score)
    return scores
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `.venv/bin/pytest tests/test_lexical.py -q`
Expected: `10 passed`.

- [ ] **Step 5: Lint and run the whole suite**

Run: `uv tool run ruff@0.8.4 check src/mcp_nvidia/lib/lexical.py tests/test_lexical.py && .venv/bin/pytest tests/ -q`
Expected: no ruff findings; `72 passed, 11 skipped` (the 48 baseline tests + 14 from Task 1 + 10 here).

- [ ] **Step 6: Commit**

```bash
git add src/mcp_nvidia/lib/lexical.py tests/test_lexical.py
git commit -F - <<'EOF'
feat: Add BM25 lexical scoring

BM25 over title and snippet with Porter stemming, titles weighted twice,
Lucene's non-negative IDF (Okapi IDF goes negative for core query terms
on query-biased candidate sets) and expansion-only terms at half weight.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013FriD63a2waebg8Qmtscj6
EOF
```

---

### Task 3: Semantic ranking module

The load-once model, off-loop encoding and explicit failure reasons. Tested entirely with fakes: this task does **not** need fastembed.

**Files:**

- Create: `src/mcp_nvidia/lib/embeddings.py`
- Create: `tests/search_fakes.py`
- Test: `tests/test_embeddings.py`

**Interfaces:**

- Consumes: `rank_by_score` from Task 1.
- Produces (all in `mcp_nvidia.lib.embeddings`):
  - Constants: `DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"`, `DEFAULT_THREADS = 4`, `ENCODE_TIMEOUT_SECONDS = 10.0`
  - Reasons: `NOT_INSTALLED = "not_installed"`, `MODEL_LOAD_FAILED = "model_load_failed"`, `ENCODE_FAILED = "encode_failed"`
  - `class EmbeddingsNotInstalledError(Exception)`
  - `@dataclass(frozen=True) class SemanticRanking: order: list[int] | None; similarities: list[float] | None; unavailable: str | None`
  - `build_document_text(result: Mapping[str, Any]) -> str`
  - `async semantic_rank(query: str, results: Sequence[Mapping[str, Any]]) -> SemanticRanking`
  - Test seams: `_get_model() -> tuple[Any | None, str | None]` and `_reset_for_tests(loader: Callable[[], Any] | None = None) -> None`
- Produces (in `tests.search_fakes`): `FakeEmbeddingModel`, a class whose `query_embed(texts)` and `embed(texts)` yield numpy vectors over a fixed vocabulary.

- [ ] **Step 1: Write the fake model**

Create `tests/search_fakes.py`:

```python
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
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_embeddings.py`:

```python
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
```

- [ ] **Step 3: Run the tests to confirm they fail**

Run: `.venv/bin/pytest tests/test_embeddings.py -q`
Expected: collection error — `ImportError: cannot import name 'embeddings' from 'mcp_nvidia.lib'`.

- [ ] **Step 4: Implement the module**

Create `src/mcp_nvidia/lib/embeddings.py`:

```python
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
```

- [ ] **Step 5: Run the tests to confirm they pass**

Run: `.venv/bin/pytest tests/test_embeddings.py -q`
Expected: `9 passed`, **0 errors**.

- [ ] **Step 6: Confirm the base install is untouched**

Run: `.venv/bin/python -c "import mcp_nvidia.lib.embeddings, sys; print('fastembed imported at import time:', 'fastembed' in sys.modules)"`
Expected: `fastembed imported at import time: False`.

- [ ] **Step 7: Lint and run the whole suite**

Run: `uv tool run ruff@0.8.4 check src/mcp_nvidia/lib/embeddings.py tests/ && .venv/bin/pytest tests/ -q`
Expected: no ruff findings; `81 passed, 11 skipped`.

- [ ] **Step 8: Commit**

```bash
git add src/mcp_nvidia/lib/embeddings.py tests/search_fakes.py tests/test_embeddings.py
git commit -F - <<'EOF'
feat: Add load-once semantic ranking module

Lazily loads a fastembed model exactly once per process under a lock,
remembers load failures until restart, encodes in a worker thread with a
10s timeout, and reports why semantic ranking is unavailable
(not_installed, model_load_failed, encode_failed).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013FriD63a2waebg8Qmtscj6
EOF
```

---

### Task 4: Hybrid ranking in the search pipeline

This task:

- keeps the original query alongside the expanded one;
- moves the existing dedupe ahead of scoring;
- replaces the keyword heuristic, TF-IDF and their 70/30 blend with BM25, the evidence floor and fusion;
- reports installed-but-failing semantics in `warnings`;
- updates the two schema descriptions whose meaning changed.

**Files:**

- Modify: `src/mcp_nvidia/lib/search.py`: imports (~lines 19–25); the query-expansion block (~lines 276–280); a new function after `_search_domain_with_semaphore` (~line 209); the scoring block (~lines 327–369); the dedupe call (~lines 388–389)
- Modify: `src/mcp_nvidia/server.py` (descriptions at ~line 100 and ~line 220)
- Modify: `tests/search_fakes.py`
- Test: `tests/test_search_pipeline.py`

**Interfaces:**

- Consumes: everything Tasks 1–3 produce; `deduplicate_results`, `expand_query_with_product_variants` and `get_domain_boost` (existing).
- Produces:
  - `mcp_nvidia.lib.search.SEMANTIC_UNAVAILABLE_WARNING = "SEMANTIC_RANKING_UNAVAILABLE"`
  - `async mcp_nvidia.lib.search._rank_candidates(candidates: list[dict[str, Any]], query: str, warnings: list[dict[str, Any]], *, original_query: str | None = None, rrf_k: int = RRF_K, tau: float = EVIDENCE_FLOOR_TAU) -> list[dict[str, Any]]`. `query` is the expanded query, and `original_query` defaults to it. The function returns the surviving candidates best-first with `relevance_score` set, and mutates the candidate dicts. Tasks 6 and 7 call it directly.
  - `tests.search_fakes.install_fake_search(monkeypatch, pages=FAKE_PAGES) -> None`, `tests.search_fakes.FAKE_PAGES`, `tests.search_fakes.FAILING_DOMAIN = "forums.nvidia.com"`

- [ ] **Step 1: Add the network fakes**

Append to `tests/search_fakes.py`:

```python
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
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_search_pipeline.py`:

```python
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
```

- [ ] **Step 3: Run the tests to confirm they fail**

Run: `.venv/bin/pytest tests/test_search_pipeline.py -q`
Expected: failures — `AttributeError: module 'mcp_nvidia.lib.search' has no attribute 'document_tokens'` (and `SEMANTIC_UNAVAILABLE_WARNING`), and the scoring assertions fail against the current 70/30 blend.

- [ ] **Step 4: Update the imports in `search.py`**

Replace the existing relevance import block:

```python
from mcp_nvidia.lib.relevance import (
    calculate_search_relevance,
    calculate_tfidf_scores,
    expand_query_with_product_variants,
    get_domain_boost,
)
```

with:

```python
from mcp_nvidia.lib.embeddings import ENCODE_FAILED, MODEL_LOAD_FAILED, semantic_rank
from mcp_nvidia.lib.fusion import (
    EVIDENCE_FLOOR_TAU,
    RRF_K,
    apply_evidence_floor,
    rank_by_score,
    reciprocal_rank_fusion,
    relevance_from_position,
)
from mcp_nvidia.lib.lexical import bm25_scores, document_tokens, query_term_weights
from mcp_nvidia.lib.relevance import expand_query_with_product_variants, get_domain_boost
```

- [ ] **Step 5: Add `_rank_candidates`**

Insert immediately after the `_search_domain_with_semaphore` function, before `async def search_all_domains`:

```python
SEMANTIC_UNAVAILABLE_WARNING = "SEMANTIC_RANKING_UNAVAILABLE"


async def _rank_candidates(
    candidates: list[dict[str, Any]],
    query: str,
    warnings: list[dict[str, Any]],
    *,
    original_query: str | None = None,
    rrf_k: int = RRF_K,
    tau: float = EVIDENCE_FLOOR_TAU,
) -> list[dict[str, Any]]:
    """Score, gate and fuse candidates; return survivors best-first with relevance_score set.

    `query` is the expanded query and `original_query` the user's own (defaulting to
    `query`). BM25 always ranks. Semantic similarity to the original query ranks too
    when the [embeddings] extra is installed and working; when it is installed but
    fails, a warning is appended so degraded ranking is visible in the response.
    """
    if not candidates:
        return []

    user_query = original_query if original_query is not None else query
    query_weights = query_term_weights(user_query, query)
    bm25 = bm25_scores([document_tokens(result) for result in candidates], query_weights)
    domain_boosts = [get_domain_boost(result.get("domain", ""), query) for result in candidates]
    lexical_scores = [score * boost for score, boost in zip(bm25, domain_boosts, strict=True)]

    semantic = await semantic_rank(user_query, candidates)
    if semantic.unavailable in (MODEL_LOAD_FAILED, ENCODE_FAILED):
        warnings.append(
            {
                "code": SEMANTIC_UNAVAILABLE_WARNING,
                "message": "Semantic ranking unavailable; results ranked by BM25 only",
                "reason": semantic.unavailable,
            }
        )

    survivors = apply_evidence_floor(
        lexical_scores,
        semantic.similarities,
        tau=tau,
        query_has_terms=bool(query_weights),
    )
    if not survivors:
        return []

    rankings = [rank_by_score([lexical_scores[i] for i in survivors])]
    if semantic.similarities is not None:
        rankings.append(rank_by_score([semantic.similarities[i] for i in survivors]))

    ranked = []
    for position, local_index in enumerate(reciprocal_rank_fusion(rankings, k=rrf_k)):
        index = survivors[local_index]
        result = candidates[index]
        result["relevance_score"] = relevance_from_position(position, len(survivors))
        if logger.isEnabledFor(logging.DEBUG):
            result["_debug_scores"] = {
                "bm25": round(bm25[index], 4),
                "domain_boost": domain_boosts[index],
                "lexical_score": round(lexical_scores[index], 4),
                "semantic_similarity": (
                    None if semantic.similarities is None else round(semantic.similarities[index], 4)
                ),
                "fused_position": position,
                "signals": len(rankings),
            }
        ranked.append(result)
    return ranked
```

- [ ] **Step 6: Keep the original query**

In `search_all_domains`, replace:

```python
    # Expand query with product variants for better matching
    expanded_query = expand_query_with_product_variants(query)
```

with:

```python
    # Keep the user's own query: BM25 weighs expansion-only terms lower, and semantic
    # ranking embeds the original query without expansion variants.
    original_query = query

    # Expand query with product variants for better matching
    expanded_query = expand_query_with_product_variants(query)
```

- [ ] **Step 7: Replace the 70/30 scoring block**

In `search_all_domains`, replace this entire block, from the TF-IDF comment through the end of the per-result `_debug_scores` block, immediately before `# Add content type detection to each result`:

```python
    # Calculate TF-IDF scores for all results
    logger.debug("Calculating TF-IDF scores...")
    tfidf_scores = calculate_tfidf_scores(all_results, query)

    # Calculate relevance scores with domain boosts and TF-IDF
    for i, result in enumerate(all_results):
        # Error results get a score of 0 to prevent query text in error message from inflating scores
        if result.get("is_error", False):
            result["relevance_score"] = 0
            if logger.isEnabledFor(logging.DEBUG):
                result["_debug_scores"] = {
                    "keyword_score": 0,
                    "tfidf_score": 0,
                    "domain_boost": 0,
                    "combined_score": 0,
                    "reason": "error_result",
                }
            continue

        domain = result.get("domain", "")

        # Get domain-specific boost
        domain_boost = get_domain_boost(domain, query)

        # Calculate keyword-based score with fuzzy matching and phrase matching
        keyword_score = calculate_search_relevance(result, query, domain_boost)

        # Get TF-IDF score
        tfidf_score = tfidf_scores[i] if i < len(tfidf_scores) else 0.5

        # Combine scores: 70% keyword-based + 30% TF-IDF
        combined_score = int(keyword_score * 0.7 + tfidf_score * 100 * 0.3)

        result["relevance_score"] = min(combined_score, 100)

        # Store component scores for debugging
        if logger.isEnabledFor(logging.DEBUG):
            result["_debug_scores"] = {
                "keyword_score": keyword_score,
                "tfidf_score": int(tfidf_score * 100),
                "domain_boost": domain_boost,
                "combined_score": combined_score,
            }
```

with:

```python
    # Remove duplicates before scoring, so a page returned by two overlapping domains is
    # scored once and occupies one position in the fused ranking.
    all_results = deduplicate_results(all_results)

    # Error results are never ranked. They score 0 so the query text inside an error
    # message cannot inflate them.
    error_results = [result for result in all_results if result.get("is_error", False)]
    for result in error_results:
        result["relevance_score"] = 0
    candidates = [result for result in all_results if not result.get("is_error", False)]

    all_results = await _rank_candidates(candidates, query, warnings, original_query=original_query) + error_results
```

- [ ] **Step 8: Remove the old post-filter dedupe**

Further down in `search_all_domains`, delete these two lines and the blank line after them:

```python
    # Deduplicate results (v0.3.0 feature)
    filtered_results = deduplicate_results(filtered_results)
```

- [ ] **Step 9: Update the two schema descriptions in `server.py`**

Replace:

```python
                        "description": "Minimum relevance score threshold (0-100) to filter results (default: 17)",
```

with:

```python
                        "description": (
                            "Minimum relevance score (0-100). Scores come from each result's position in the fused "
                            "ranking, after results matching no query term and with low semantic similarity are "
                            "removed (default: 17)"
                        ),
```

and replace:

```python
                                    "description": "Relevance score from 0-100 based on keyword matching and TF-IDF",
```

with:

```python
                                    "description": (
                                        "Relevance score from 0-100, derived from the result's position after fusing "
                                        "BM25 and (when installed) semantic rankings; 100 is the top result"
                                    ),
```

- [ ] **Step 10: Run the pipeline tests**

Run: `.venv/bin/pytest tests/test_search_pipeline.py -q`
Expected: `8 passed`, **0 errors**.

If `test_evidence_floor_drops_results_with_no_lexical_or_semantic_match` fails, print the expanded query: `.venv/bin/python -c "from mcp_nvidia.lib.relevance import expand_query_with_product_variants as e; print(e('cuda memory'))"`. Expansion may have added a term that matches the Jetson page. If so, change the Jetson fake's title and body to words outside both the expanded query and `VOCABULARY`; don't change the assertion.

- [ ] **Step 11: Run the whole suite and lint**

Run: `uv tool run ruff@0.8.4 check src/ tests/ && .venv/bin/pytest tests/ -q`
Expected: no ruff findings (in particular, no unused imports left in `search.py`); `89 passed, 11 skipped`.

`tests/test_sdk_generation.py` regenerates the SDKs from the tool schemas. If it fails on the description change, read the assertion. It should not pin description text, so a failure means the assertion needs updating to the new wording; don't revert the wording.

- [ ] **Step 12: Commit**

```bash
git add src/mcp_nvidia/lib/search.py src/mcp_nvidia/server.py tests/search_fakes.py tests/test_search_pipeline.py
git commit -F - <<'EOF'
feat: Rank search results with BM25, an evidence floor and rank fusion

Replaces the keyword heuristic, TF-IDF and their fixed 70/30 blend with a
single BM25 score fused with semantic similarity (when installed) by
reciprocal rank fusion, gated by an evidence floor. relevance_score is
now derived from fused position. The existing dedupe moves ahead of
scoring. Semantic ranking embeds the user's original query. An
installed-but-failing embedding model adds a
SEMANTIC_RANKING_UNAVAILABLE warning.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013FriD63a2waebg8Qmtscj6
EOF
```

---

### Task 5: Optional extra, Docker image and CI

Makes the feature installable, bakes the model into the production image, and adds a CI job that exercises the real model.

**Files:**

- Modify: `pyproject.toml` (`[project.optional-dependencies]`)
- Modify: `Dockerfile`
- Modify: `.github/workflows/test.yml`
- Create: `tests/test_embeddings_model.py`

**Interfaces:**

- Consumes: `embeddings.semantic_rank` and `embeddings._reset_for_tests` (Task 3).
- Produces: the `embeddings` extra; a Docker image with weights at `/opt/models`; the CI job `embeddings`.

- [ ] **Step 1: Write the real-model smoke test**

Create `tests/test_embeddings_model.py`:

```python
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
```

- [ ] **Step 2: Confirm it skips without the extra**

Run: `.venv/bin/pytest tests/test_embeddings_model.py -q -rs`
Expected: `1 skipped`, with reason `could not import 'fastembed'`.

- [ ] **Step 3: Add the extra**

In `pyproject.toml`, under `[project.optional-dependencies]`, add after the `ui` entry:

```toml
embeddings = [
    "fastembed>=0.8,<0.9",  # Local ONNX embedding model for hybrid search (no torch)
]
```

- [ ] **Step 4: Install the extra and run the real model**

Run: `uv pip install -p .venv/bin/python -e ".[test,embeddings]" && .venv/bin/pytest tests/test_embeddings_model.py -q`
Expected: `1 passed`. The first run downloads ~65 MB of weights (~10 s).

- [ ] **Step 5: Bake the model into the Docker image**

In `Dockerfile`, replace:

```dockerfile
RUN pip install --no-cache-dir .
```

with:

```dockerfile
# Hybrid search: install the embeddings extra and bake the model weights into the image,
# so containers never download them on a cold start.
ENV FASTEMBED_CACHE_PATH=/opt/models
RUN pip install --no-cache-dir ".[embeddings]"
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"
```

- [ ] **Step 6: Verify the image (if Docker is available locally)**

Run: `docker version --format '{{.Server.Version}}'`. If that fails, Docker isn't available: note it and continue, since Task 8 verifies the image instead.

Otherwise run:

```bash
docker build -t mcp-nvidia:hybrid .
docker run --rm mcp-nvidia:hybrid python -c "import os; print(sorted(os.listdir('/opt/models')))"
```

Expected: the build succeeds and the listing is non-empty (a directory for `bge-small-en-v1.5`).

- [ ] **Step 7: Add the CI job**

Append this job to `.github/workflows/test.yml`, as a sibling of the existing `test` job:

```yaml
  embeddings:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        # 3.10 is where fastembed pins onnxruntime below 1.24; 3.12 matches the Docker image.
        python-version: ["3.10", "3.12"]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install uv
      run: pip install uv

    - name: Create virtual environment
      run: uv venv

    - name: Install dependencies with the embeddings extra
      run: |
        source .venv/bin/activate
        uv pip install -e ".[test,embeddings]"

    - name: Run ranking tests with the real model
      run: |
        source .venv/bin/activate
        pytest tests/test_embeddings_model.py tests/test_search_pipeline.py tests/test_embeddings.py -v --tb=short
```

- [ ] **Step 8: Validate the workflow file**

Run: `.venv/bin/python -c "import yaml; wf = yaml.safe_load(open('.github/workflows/test.yml')); print(list(wf['jobs']))"`
Expected: `['test', 'embeddings']`. If `yaml` is missing, run `uv pip install -p .venv/bin/python pyyaml` first.

- [ ] **Step 9: Run the whole suite with the extra installed**

Run: `.venv/bin/pytest tests/ -q`
Expected: `90 passed, 11 skipped`. The real-model test now runs.

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml Dockerfile .github/workflows/test.yml tests/test_embeddings_model.py
git commit -F - <<'EOF'
build: Ship hybrid search as an optional extra

Adds the [embeddings] extra (fastembed), bakes the bge-small weights into
the Docker image at /opt/models, and adds a CI job running the ranking
tests against the real model on Python 3.10 and 3.12.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013FriD63a2waebg8Qmtscj6
EOF
```

---

### Task 6: Ranking comparison script, k and τ

This is a live tool, not a test. It compares the removed 70/30 baseline with hybrid ranking, prints each query term's IDF, sweeps `k`, measures the similarity distribution that sets τ, and records candidates for Task 7.

**Files:**

- Create: `scripts/eval_ranking.py`
- Modify: `pyproject.toml` (ruff per-file ignore, only if needed)
- Modify: `src/mcp_nvidia/lib/fusion.py` (the `EVIDENCE_FLOOR_TAU` value, and `RRF_K` if the measurement says so)
- Modify: `tests/test_fusion.py` (only if `RRF_K` changes)
- Create: `tests/fixtures/ranking/*.json` (recorded, unlabeled)

**Interfaces:**

- Consumes: `search_nvidia_domain` and `_rank_candidates` (Task 4); `semantic_rank` (Task 3); `tokenize`, `document_tokens`, `query_term_weights` and `bm25_scores`, plus the module constants in `mcp_nvidia.lib.lexical` (Task 2); `deduplicate_results`, `calculate_search_relevance`, `calculate_tfidf_scores`, `get_domain_boost` and `expand_query_with_product_variants` (existing; the last three also form the baseline).
- Produces: `scripts/eval_ranking.py` with the options `--queries`, `--rrf-k`, `--tau`, `--title-weight`, `--expansion-weight`, `--model`, `--no-embeddings`, `--per-domain` and `--record DIR`; recorded fixture files of the form `{"query", "expanded_query", "candidates", "assertions": []}`.

- [ ] **Step 1: Write the script**

Create `scripts/eval_ranking.py`:

```python
#!/usr/bin/env python3
"""Compare the removed 70/30 keyword/TF-IDF blend with hybrid ranking on live queries.

Not a test: it hits DuckDuckGo and NVIDIA sites. Use it to choose k and tau, to inspect
BM25 term weights on real results, and to record candidates that
tests/test_ranking_fixtures.py turns into regression fixtures.

    .venv/bin/python scripts/eval_ranking.py
    .venv/bin/python scripts/eval_ranking.py --no-embeddings --rrf-k 30
    .venv/bin/python scripts/eval_ranking.py --record tests/fixtures/ranking
"""

import argparse
import asyncio
import copy
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import httpx

DEFAULT_QUERIES = [
    "CUDA Toolkit download",
    "TensorRT documentation",
    "Nsight Systems profiler",
    "cuDNN installation guide",
    "CUDA memory management",
    "TensorRT inference optimization",
    "Omniverse USD Composer tutorial",
    "Jetson Orin power modes",
]

RECORDED_FIELDS = ("title", "url", "snippet", "snippet_plain", "domain", "published_date", "metadata")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", nargs="+", default=DEFAULT_QUERIES)
    parser.add_argument("--rrf-k", nargs="+", type=int, default=[10, 20, 30, 40, 60])
    parser.add_argument("--tau", type=float, default=None, help="evidence-floor threshold (default: fusion.py)")
    parser.add_argument("--title-weight", type=int, default=None, help="override lexical.TITLE_WEIGHT")
    parser.add_argument("--expansion-weight", type=float, default=None, help="override lexical.EXPANSION_TERM_WEIGHT")
    parser.add_argument("--model", default=None, help="override MCP_NVIDIA_EMBEDDING_MODEL")
    parser.add_argument("--no-embeddings", action="store_true", help="rank on BM25 only")
    parser.add_argument("--per-domain", type=int, default=3)
    parser.add_argument("--record", type=Path, default=None, help="write candidates to DIR/<query>.json")
    return parser.parse_args()


def configure(args: argparse.Namespace) -> None:
    # Must run before anything loads the model.
    if args.no_embeddings:
        sys.modules["fastembed"] = None  # makes the default loader report not_installed
    if args.model:
        os.environ["MCP_NVIDIA_EMBEDDING_MODEL"] = args.model

    from mcp_nvidia.lib import lexical

    if args.title_weight is not None:
        lexical.TITLE_WEIGHT = args.title_weight
    if args.expansion_weight is not None:
        lexical.EXPANSION_TERM_WEIGHT = args.expansion_weight


async def gather_candidates(query: str, per_domain: int):
    from mcp_nvidia.lib.constants import DEFAULT_DOMAINS
    from mcp_nvidia.lib.deduplication import deduplicate_results
    from mcp_nvidia.lib.relevance import expand_query_with_product_variants
    from mcp_nvidia.lib.search import search_nvidia_domain

    expanded = expand_query_with_product_variants(query)
    async with httpx.AsyncClient(timeout=30.0) as client:
        batches = await asyncio.gather(
            *(search_nvidia_domain(client, domain, expanded, per_domain) for domain in DEFAULT_DOMAINS),
            return_exceptions=True,
        )
    raw = [r for batch in batches if isinstance(batch, list) for r in batch if not r.get("is_error")]
    return expanded, deduplicate_results(raw), len(raw)


def baseline_order(candidates: list[dict], query: str) -> list[int]:
    """The removed 70% keyword / 30% TF-IDF blend, reproduced here for comparison only."""
    from mcp_nvidia.lib.relevance import calculate_search_relevance, calculate_tfidf_scores, get_domain_boost

    tfidf = calculate_tfidf_scores(candidates, query)
    blended = [
        int(calculate_search_relevance(r, query, get_domain_boost(r.get("domain", ""), query)) * 0.7 + tfidf[i] * 30)
        for i, r in enumerate(candidates)
    ]
    return sorted(range(len(candidates)), key=lambda i: (-blended[i], i))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


async def evaluate(query: str, args: argparse.Namespace) -> None:
    from mcp_nvidia.lib.embeddings import semantic_rank
    from mcp_nvidia.lib.fusion import EVIDENCE_FLOOR_TAU
    from mcp_nvidia.lib.lexical import bm25_scores, document_tokens, query_term_weights
    from mcp_nvidia.lib.search import _rank_candidates

    tau = EVIDENCE_FLOOR_TAU if args.tau is None else args.tau
    expanded, candidates, raw_count = await gather_candidates(query, args.per_domain)
    print(f"\n{'=' * 100}\n{query!r} -> expanded {expanded!r}")
    print(f"candidates: {len(candidates)} after dedupe ({raw_count - len(candidates)} duplicates removed)")
    if not candidates:
        return

    weights = query_term_weights(query, expanded)
    tokens = [document_tokens(c) for c in candidates]
    n = len(tokens)
    print("BM25 query terms (df / Lucene idf / weight):")
    for term, weight in weights.items():
        df = sum(1 for document in tokens if term in document)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        print(f"  {term:16s} {df:>3}/{n:<3}  idf={idf:.3f}  weight={weight}")
    bm25 = bm25_scores(tokens, weights)

    started = time.perf_counter()
    semantic = await semantic_rank(query, candidates)
    encode_ms = (time.perf_counter() - started) * 1000
    print(f"semantic: {semantic.unavailable or 'available'} ({encode_ms:.0f} ms incl. any first load)")

    urls = [c["url"] for c in candidates]
    baseline_rank = {urls[i]: rank for rank, i in enumerate(baseline_order(candidates, expanded), 1)}

    hybrid_rank_by_k = {}
    for k in args.rrf_k:
        ranked = await _rank_candidates(
            copy.deepcopy(candidates), expanded, warnings=[], original_query=query, rrf_k=k, tau=tau
        )
        hybrid_rank_by_k[k] = {r["url"]: rank for rank, r in enumerate(ranked, 1)}
    kept = len(hybrid_rank_by_k[args.rrf_k[0]])
    print(f"evidence floor (tau={tau}): kept {kept}, removed {len(candidates) - kept}")

    main_k = 30 if 30 in args.rrf_k else args.rrf_k[0]
    header = "".join(f"  k={k:<3}" for k in args.rrf_k)
    print(f"\n  {'base':>4}{header}  {'move':>5}  title / url")
    for url in sorted(urls, key=lambda u: hybrid_rank_by_k[main_k].get(u, 10_000)):
        base = baseline_rank[url]
        cells = "".join(f"  {hybrid_rank_by_k[k].get(url, '-')!s:<5}" for k in args.rrf_k)
        new = hybrid_rank_by_k[main_k].get(url)
        move = "drop" if new is None else f"{base - new:+d}"
        title = next(c["title"] for c in candidates if c["url"] == url)
        print(f"  {base:>4}{cells}  {move:>5}  {title[:60]}  <{url[:70]}>")

    if semantic.similarities is not None:
        similarities = semantic.similarities
        print(
            "\nsimilarity p0/p25/p50/p75/p100: "
            + " / ".join(f"{percentile(similarities, f):.3f}" for f in (0.0, 0.25, 0.5, 0.75, 1.0))
        )
        if weights:
            unmatched = [(similarities[i], c["title"]) for i, c in enumerate(candidates) if bm25[i] == 0]
            print(f"results matching NO query term ({len(unmatched)}), by similarity — these are what tau gates:")
            for similarity, title in sorted(unmatched, reverse=True):
                print(f"  {similarity:.3f}  {title[:90]}")

    if args.record:
        args.record.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")
        path = args.record / f"{slug}.json"
        fixture = {
            "query": query,
            "expanded_query": expanded,
            "candidates": [{field: c.get(field) for field in RECORDED_FIELDS} for c in candidates],
            "assertions": [],
        }
        path.write_text(json.dumps(fixture, indent=2) + "\n")
        print(f"recorded -> {path}")


async def main() -> None:
    args = parse_args()
    configure(args)
    for query in args.queries:
        await evaluate(query, args)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Lint the script**

Run: `uv tool run ruff@0.8.4 check scripts/eval_ranking.py`
If it reports `T201` (`print` found), add this under `[tool.ruff.lint.per-file-ignores]` in `pyproject.toml` and re-run:

```toml
"scripts/*.py" = ["T201"]
```

Expected after any fix: no findings.

- [ ] **Step 3: Run the comparison with embeddings**

Run: `.venv/bin/python scripts/eval_ranking.py 2>/dev/null | tee /tmp/eval-embeddings.txt`
Expected, for each query:

- BM25 term weights with document frequencies;
- a table of baseline rank vs hybrid rank for k = 10, 20, 30, 40 and 60;
- the evidence floor's kept and removed counts;
- similarity percentiles, and the list of results matching no query term, ordered by similarity.

This takes several minutes: live search runs at roughly 30 s per query.

- [ ] **Step 4: Run the comparison without embeddings**

Run: `.venv/bin/python scripts/eval_ranking.py --no-embeddings --rrf-k 30 2>/dev/null | tee /tmp/eval-bm25.txt`
Expected: the same tables with `semantic: not_installed` and no similarity section.

- [ ] **Step 5: Choose τ and confirm k — requires the user's judgment**

This step can't be pre-filled: both values depend on what real results look like. Read `/tmp/eval-embeddings.txt` with the user.

**τ:** in the "results matching NO query term" lists, mark each result as relevant or irrelevant to its query. Then:

- If some are judged relevant: set τ just below the **lowest** similarity among the relevant ones, provided that is above the **highest** similarity among the irrelevant ones.
- If the two overlap (a relevant result scores below an irrelevant one): set τ at the lowest relevant similarity, and record the overlap in the spec's §5 "Evidence floor" section. The floor will then admit some irrelevant results, which the fused ranking pushes down.
- If none are judged relevant: set τ at the 75th percentile of those similarities.

**k:** compare the `k` columns. `k = 30` should keep exact product-name matches (the first four default queries) at or near the top, without letting one signal's odd pick jump to first place. If another `k` is clearly better on these queries, change `RRF_K` in `fusion.py`, and update `test_default_k_is_thirty_for_two_signals` (its name and value) to match.

**Term weights:** check the per-term IDF lines. If an expansion variant still dominates a query despite its half weight, rerun that query with `--expansion-weight 0.25` and compare the orderings with the user before changing `EXPANSION_TERM_WEIGHT`.

- [ ] **Step 6: Record τ (and k, if it changed)**

In `src/mcp_nvidia/lib/fusion.py`, replace:

```python
# Semantic similarity below which a candidate matching no query term is dropped.
# Provisional: replaced with a measured value from scripts/eval_ranking.py before release.
EVIDENCE_FLOOR_TAU = 0.5
```

with the following, substituting the chosen value and the date measured:

```python
# Semantic similarity below which a candidate matching no query term is dropped.
# Measured with scripts/eval_ranking.py on 8 live queries (YYYY-MM-DD); see Task 6 of
# docs/superpowers/plans/2026-09-12-hybrid-search.md for how it was chosen.
EVIDENCE_FLOOR_TAU = <chosen value, e.g. 0.62>
```

If `k` was confirmed at 30, change `Provisional: confirmed with scripts/eval_ranking.py.` in the `RRF_K` comment to `Confirmed with scripts/eval_ranking.py (YYYY-MM-DD).`

Then re-run `.venv/bin/pytest tests/ -q`. Task 1's floor tests pass `tau` explicitly, so they are unaffected. Expected: `90 passed, 11 skipped`.

- [ ] **Step 7: Record candidates for fixtures**

Run: `.venv/bin/python scripts/eval_ranking.py --rrf-k 30 --record tests/fixtures/ranking 2>/dev/null`
Expected: one `tests/fixtures/ranking/<slug>.json` per query, each with `"assertions": []`.

- [ ] **Step 8: Commit**

```bash
git add scripts/eval_ranking.py src/mcp_nvidia/lib/fusion.py tests/test_fusion.py pyproject.toml tests/fixtures/ranking/
git commit -F - <<'EOF'
feat: Add ranking comparison script and set k and the evidence floor

scripts/eval_ranking.py compares the removed 70/30 blend with hybrid
ranking on live queries, prints BM25 term weights, sweeps k, reports the
similarity distribution the evidence floor gates, and records candidates
for regression fixtures. EVIDENCE_FLOOR_TAU (and RRF_K) are set from its
measurements.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013FriD63a2waebg8Qmtscj6
EOF
```

---

### Task 7: Labeled ranking regression fixtures

Turns the recorded candidates into deterministic, offline regression tests. The harness is code; the labels are human judgment.

**Files:**

- Create: `tests/test_ranking_fixtures.py`
- Modify: `tests/fixtures/ranking/*.json` (add `assertions`)
- Modify: `.github/workflows/test.yml`

**Interfaces:**

- Consumes: `_rank_candidates` (Task 4); fixtures recorded in Task 6; `embeddings._reset_for_tests` (Task 3).
- Produces: the assertion format. Each entry has a `"kind"` (`known_item` | `pairwise` | `negative`), a `"mode"` (`lexical` | `semantic`), and fields specific to its kind:
  - `known_item`: `"url"`, `"top"` (int)
  - `pairwise`: `"better"`, `"worse"` (URLs)
  - `negative`: `"url"`, `"not_in_top"` (int)

- [ ] **Step 1: Write the harness**

Create `tests/test_ranking_fixtures.py`:

```python
"""Labeled ranking regressions, frozen from real queries by scripts/eval_ranking.py.

Each assertion runs in one mode:
- "lexical": fastembed is blocked, so ranking uses BM25 only. Always runs.
- "semantic": the real model participates. Skips unless the [embeddings] extra is installed.

Admission bar for a new assertion: "I would call it a bug if this regressed."
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from mcp_nvidia.lib import embeddings
from mcp_nvidia.lib.search import _rank_candidates

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "ranking"
HAS_FASTEMBED = importlib.util.find_spec("fastembed") is not None


def _cases():
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        fixture = json.loads(path.read_text())
        for number, assertion in enumerate(fixture["assertions"]):
            yield pytest.param(fixture, assertion, id=f"{path.stem}-{number}-{assertion['kind']}-{assertion['mode']}")


@pytest.fixture(autouse=True)
def _fresh_model_state():
    embeddings._reset_for_tests()
    yield
    embeddings._reset_for_tests()


@pytest.mark.parametrize(("fixture", "assertion"), list(_cases()))
async def test_ranking_regression(fixture, assertion, monkeypatch):
    if assertion["mode"] == "semantic":
        if not HAS_FASTEMBED:
            pytest.skip("needs mcp-nvidia[embeddings]")
    elif assertion["mode"] == "lexical":
        monkeypatch.setitem(sys.modules, "fastembed", None)
    else:
        pytest.fail(f"unknown mode {assertion['mode']!r}")

    ranked = await _rank_candidates(
        copy.deepcopy(fixture["candidates"]),
        fixture["expanded_query"],
        warnings=[],
        original_query=fixture["query"],
    )
    urls = [result["url"] for result in ranked]

    kind = assertion["kind"]
    if kind == "known_item":
        assert assertion["url"] in urls[: assertion["top"]], f"{assertion['url']} not in top {assertion['top']}: {urls}"
    elif kind == "pairwise":
        assert assertion["better"] in urls, f"{assertion['better']} was dropped"
        if assertion["worse"] in urls:
            assert urls.index(assertion["better"]) < urls.index(assertion["worse"]), urls
    elif kind == "negative":
        assert assertion["url"] not in urls[: assertion["not_in_top"]], urls
    else:
        pytest.fail(f"unknown assertion kind {kind!r}")
```

- [ ] **Step 2: Confirm it collects cleanly with no labels yet**

Run: `.venv/bin/pytest tests/test_ranking_fixtures.py -q -rs`
Expected: `1 skipped`. Pytest reports an empty parameter set, because every recorded fixture still has `"assertions": []`.

- [ ] **Step 3: Label — requires the user's judgment**

Work through each recorded fixture with the user, using `/tmp/eval-embeddings.txt` and `/tmp/eval-bm25.txt` from Task 6. Start from the rows where baseline and hybrid disagree most.

- Add only assertions that pass the admission bar. **Leave ambiguous cases out.**
- Copy URLs from that fixture's own `candidates`.

The format:

```json
"assertions": [
  {"kind": "known_item", "mode": "lexical", "url": "<official page URL from candidates>", "top": 3},
  {"kind": "pairwise", "mode": "semantic", "better": "<URL>", "worse": "<URL>"},
  {"kind": "negative", "mode": "lexical", "url": "<clearly irrelevant URL>", "not_in_top": 5}
]
```

Guidance:

- The four known-item queries ("CUDA Toolkit download", "TensorRT documentation", "Nsight Systems profiler", "cuDNN installation guide") should each get a `known_item` assertion for the official page, in **both** modes, provided that page was retrieved.
- Use `mode: "semantic"` only for judgments that depend on meaning rather than shared words.
- Delete any recorded fixture file that ends up with no assertions.

- [ ] **Step 4: Run the fixtures**

Run: `.venv/bin/pytest tests/test_ranking_fixtures.py -q`
Expected: every labeled assertion passes. The extra is installed from Task 5, so both modes run.

Lexical-mode assertions block fastembed inside the harness, so they already prove ranking works without the model. The base `test` CI job, which doesn't install the extra, runs them again.

If an assertion fails, do **not** loosen it to make it pass. Either the label was wrong, in which case re-judge it with the user and remove it; or ranking has a real problem, in which case fix `_rank_candidates`, `RRF_K`, τ or a lexical constant, and re-run Task 6's comparison.

- [ ] **Step 5: Add the fixtures to the CI embeddings job**

In `.github/workflows/test.yml`, in the `embeddings` job's final step, change:

```yaml
        pytest tests/test_embeddings_model.py tests/test_search_pipeline.py tests/test_embeddings.py -v --tb=short
```

to:

```yaml
        pytest tests/test_embeddings_model.py tests/test_search_pipeline.py tests/test_embeddings.py tests/test_ranking_fixtures.py -v --tb=short
```

- [ ] **Step 6: Run the whole suite**

Run: `.venv/bin/pytest tests/ -q`
Expected: everything passes, with `90 + <number of labeled assertions>` passed and `11 skipped`.

- [ ] **Step 7: Commit**

```bash
git add tests/test_ranking_fixtures.py tests/fixtures/ranking/ .github/workflows/test.yml
git commit -F - <<'EOF'
test: Add labeled ranking regression fixtures

Freezes real candidates recorded by scripts/eval_ranking.py and asserts
known-item, pairwise and negative judgments in BM25-only and semantic
modes, so a ranking regression fails CI instead of reaching users.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013FriD63a2waebg8Qmtscj6
EOF
```

---

### Task 8: Final verification and release notes

Proves the feature end to end in the real server, brings the docs in line with what was measured, and hands the release decision back to the user.

**Files:**

- Modify: `CHANGELOG.md` (only if something shipped differently from its 0.9.0 entry)
- Modify: `docs/superpowers/specs/2026-09-12-hybrid-search-design.md` (status line)

**Interfaces:**

- Consumes: everything above.
- Produces: a verified branch and an updated PR #8 description. No code interfaces.

- [ ] **Step 1: Full suite and lint**

Run: `uv tool run ruff@0.8.4 check src/ tests/ scripts/ && uv tool run ruff@0.8.4 format --check src/ tests/ scripts/ && .venv/bin/pytest tests/ -q`
Expected: no findings; all tests pass.

- [ ] **Step 2: Live stdio smoke test through a real MCP client**

Run:

```bash
.venv/bin/python - <<'PY'
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    params = StdioServerParameters(command=".venv/bin/mcp-nvidia", args=[], env=None)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("search_nvidia", {"query": "CUDA Toolkit download", "max_results_per_domain": 2})
            payload = result.structuredContent
            print("isError:", result.isError, "| results:", len(payload["results"]), "| warnings:", [w["code"] for w in payload["warnings"]])
            for r in payload["results"][:5]:
                print(f"  {r['relevance_score']:>3}  {r['title'][:70]}")

asyncio.run(main())
PY
```

Expected:

- `isError: False`.
- A non-empty result list, with scores descending from 100.
- **No** `SEMANTIC_RANKING_UNAVAILABLE` warning, since the extra is installed.
- The official CUDA downloads page at or near the top.

- [ ] **Step 3: Live HTTP smoke test**

Run:

```bash
.venv/bin/python -m uvicorn mcp_nvidia.http_server:http_app --host 127.0.0.1 --port 8903 --log-level error &
SERVER_PID=$!
sleep 5
curl -s http://127.0.0.1:8903/health
kill $SERVER_PID
```

Expected: `{"status":"healthy",...,"version":"0.9.0",...}`.

- [ ] **Step 4: Container smoke test (if Docker is available)**

If Task 5 Step 6 found Docker available, run:

```bash
docker build -t mcp-nvidia:hybrid .
docker run --rm -d --name mcp-nvidia-hybrid -e PORT=8000 -p 8904:8000 mcp-nvidia:hybrid
sleep 8
curl -s http://127.0.0.1:8904/health
docker logs mcp-nvidia-hybrid 2>&1 | grep -i "embedding" || echo "(model loads on first search, not at startup)"
docker rm -f mcp-nvidia-hybrid
```

Expected: healthy. No weight download appears in the logs at any point, because the weights are baked in.

- [ ] **Step 5: Bring the docs in line with what shipped**

- Read the 0.9.0 entry in `CHANGELOG.md`. It already describes BM25, the ranking change, the evidence floor, the dedupe move, the dropped fuzzy, phrase and URL matching, and the new warning. If Task 6 changed `RRF_K` or a lexical constant, or anything else shipped differently, correct the entry to match.
- In the spec, change the status line to `Implemented YYYY-MM-DD (τ = <value>, k = <value>)`.

Then run: `pre-commit run --files CHANGELOG.md docs/superpowers/specs/2026-09-12-hybrid-search-design.md`
Expected: all hooks pass.

- [ ] **Step 6: Commit and push**

```bash
git add CHANGELOG.md docs/superpowers/specs/2026-09-12-hybrid-search-design.md
git commit -F - <<'EOF'
docs: Record hybrid search as implemented

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_013FriD63a2waebg8Qmtscj6
EOF
git push origin chore/pin-mcp-v1
```

- [ ] **Step 7: Update PR #8's description**

Make these changes:

- Change section 3's heading from "design done, implementation in progress 🚧" to "✅".
- Replace the "Tests for hybrid search land with its implementation" line with the final test counts from Step 1.
- Add the chosen τ and k.
- Remove "implementation is in progress" from the "Draft on purpose" callout.

`gh pr edit` fails on this repository with a GraphQL `projectCards` error (seen 2026-09-12). Write the full body to a file and use the REST API instead:

```bash
gh api --method PATCH repos/bharatr21/mcp-nvidia/pulls/8 -F body=@/tmp/pr8-body.md --jq '{number, draft}'
```

The body must end with:

```text
🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_013FriD63a2waebg8Qmtscj6
```

- [ ] **Step 8: Stop — the release decision is the user's**

Report to the user: the final test counts, the τ and k chosen, the smoke-test results, and that PR #8 is still a draft. **Do not undraft or merge it.** Merging deploys production on Railway, and live clients are still connected over `/sse`.

---

## Self-Review

**Spec coverage** (spec section → task):

- §3 decisions 1–6 and §4 (BM25 replacing the keyword heuristic and TF-IDF; Lucene IDF; stemming; title weight; domain boost; expansion weight) → Task 2 (functions), Task 4 (applied, with the original query kept), Task 6 (per-term IDF inspected; weights tunable).
- §3 decisions 7–8 and §5 fusion (RRF, provisional `k = 30`, the two-signal crossover) → Task 1; confirmed against data in Task 6.
- §3 decision 9 and §5 score formula → Task 1 (`relevance_from_position`), Task 4 (applied, plus the schema descriptions).
- §3 decisions 10–11 and §6 packaging (fastembed extra, bge-small) → Task 5; the model default is in Task 3.
- §3 decision 12 and §6 input text → Task 3 (`build_document_text`); embedding the original query → Task 4.
- §3 decision 13 and §6 loading and encoding (lazy, once, under a lock, worker thread, timeout, `query_embed` vs `embed`) → Task 3.
- §3 decision 14 and §6 weights (Docker prefetch, lazy pip download) → Task 5.
- §3 decision 15 and §7 failure table and warning shape → Task 3 (reasons), Task 4 (warning only when installed but failing; BM25 message).
- §3 decision 16 and §9 dedupe moved ahead of scoring → Task 4.
- §3 decision 17 and §5 evidence floor, including queries with no terms → Task 1 (function), Task 4 (applied), Task 6 (τ measured).
- §3 decision 18 and §10 stage 1 script → Task 6; stage 2 fixtures in lexical and semantic modes → Task 7.
- §10 unit and protocol tests → Tasks 1–4; network seam → Task 4.
- §8 public surface (ordering, scores, cutoff, dropped fuzzy/phrase/URL matching, extra, env vars, warning) → Tasks 4 and 5; CHANGELOG checked in Task 8.
- §11 risks (deploy on merge, unmeasured production latency, lost typo tolerance, noisy IDF, Python 3.10 onnxruntime) → Task 5 CI matrix, Task 6 measurements, Task 7 fixtures, Task 8 smoke tests and the stop before merging.

**Deviations from the spec:** none remaining. `SemanticRanking.similarities` and embedding the original query are both written into the spec. `_rank_candidates` takes `original_query`, `rrf_k` and `tau` as keyword arguments so Tasks 6 and 7 can drive it directly; `search_all_domains`'s signature is unchanged.

**Human-judgment steps:** Task 6 Step 5 (τ, k, expansion weight) and Task 7 Step 3 (labels). Both are inherent to the design: the spec requires τ and k to come from measurement and labels from judging real disagreements. Each step gives an explicit decision rule and a format rather than a guess.
