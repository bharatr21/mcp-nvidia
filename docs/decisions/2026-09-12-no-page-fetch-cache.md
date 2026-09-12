<!-- markdownlint-disable MD013 -->

# Why mcp-nvidia does not cache page fetches

- **Date:** 2026-09-12
- **Status:** Decided — dedupe within a request; no cross-query cache
- **Context:** raised while scoping hybrid (keyword + embedding) search

## The question

A search query fetches dozens of NVIDIA pages live. That is the dominant cost of a request, so caching those fetches looks like the obvious optimisation. Is it?

The objection raised first: search results are dynamic — `ddgs` returns a different set of URLs each time — so what would a cache even hold?

## What a cache would actually key on

That objection turns out to be answerable. `fetch_url_context` (`src/mcp_nvidia/lib/snippet.py:77`) splits cleanly into two halves:

| Work | Depends on |
| --- | --- |
| HTTP GET, BeautifulSoup parse, date extraction, metadata extraction, whitespace normalisation | **URL only** |
| Locating the snippet in the page text and highlighting it | URL **and** query |

So a cache would not store result sets at all. It would store *parsed page content* keyed by URL, and the cheap query-specific highlighting would still run per request. Churn in the result set is irrelevant to that design.

So the idea is sound in principle. It was measured anyway — and the measurements killed it.

## Measurement 1: the cost is real

- A default query spans **15 domains × 3 results = 45 results**, with domain searches capped at 5 concurrent (`MAX_CONCURRENT_SEARCHES`).
- Each result triggers one live page fetch. Two timed samples: **0.75 s** (`developer.nvidia.com/cuda-toolkit`) and **0.83 s** (`blogs.nvidia.com`).
- There is no caching anywhere in `src/mcp_nvidia/lib/` today.

A full default query was timed end to end: **30.3 s** wall time, 23 of 45 results kept. Almost all of that is network I/O and page parsing.

For contrast, encoding the query plus 45 title+snippet strings with a small ONNX model (fastembed, no torch) took roughly **0.6–3 s** on a loaded 12-core dev machine, depending on model and input length. That is about **2–10% of a query**. It is small next to the network, but it is not negligible.

> **Correction.** An earlier draft of this record said embedding took "single-digit to low-tens of milliseconds". That figure was never measured, and measurement showed it was off by one to two orders of magnitude. The corrected figures above are measured. They come from a dev box with background load (load average 2.6 → 7.1 during the run), so treat them as a range, not a benchmark.

Why the range is so wide: `all-MiniLM-L6-v2` as shipped by fastembed pads every input to a fixed 128 tokens, so a 16-token title costs as much as a full snippet. `bge-small-en-v1.5` pads only to the longest input in the batch, so titles are cheap (~16 tokens) and snippets cost more (~102 tokens). Model choice is covered in the hybrid-search design.

## Measurement 2: URLs barely recur across queries

A cross-query cache only pays off if the same URLs come back. Three related queries were run across two domains, collecting the URLs `ddgs` returned:

| Query | URLs |
| --- | --- |
| CUDA memory management | 10 |
| TensorRT inference optimization | 10 |
| CUDA kernel profiling | 10 |

Pairwise overlap: **0, 0, and 0**. Thirty URL slots, thirty distinct URLs — a **0% repeat rate**. Even two queries sharing the word "CUDA" and the same domains returned entirely disjoint sets.

Caveat, stated honestly: this is a small sample of deliberately distinct queries. It says nothing about a real user population repeating or refining the *same* query, where overlap would be higher. But it removes the assumption that a cache would help "every query".

## Measurement 3: the origins refuse caching

Response headers from the three sampled pages:

| Page | Cache-Control |
| --- | --- |
| `developer.nvidia.com/cuda-toolkit` | `max-age=0, private, must-revalidate` (+ `ETag`) |
| `blogs.nvidia.com` | `no-cache, no-store, must-revalidate, max-age=0` |
| `docs.nvidia.com/cuda/` | `no-store, no-cache, must-revalidate, max-age=0` (+ `ETag`, `Last-Modified`) |

A standards-respecting HTTP cache would store **nothing**. Caching would mean deliberately overriding the origin's directives — a decision about serving stale NVIDIA content, not a free optimisation. The `ETag`/`Last-Modified` headers do permit revalidation, but a conditional request is still a network round trip: it saves parse time, not latency, which is not the cost that matters here.

## Decision

**Dedupe within a request. No cross-query cache.**

Dedupe was measured before committing to it too, and the result reframed why it's worth doing. Two default queries were run (15 domain searches × 3 results each, counted before any filtering). They produced **1 duplicate in 85 URL slots**: one `catalog.ngc.nvidia.com/...` page, returned by both the `catalog.ngc.nvidia.com` search and the `ngc.nvidia.com` search. The cause is subdomain overlap in the default domain list: `site:ngc.nvidia.com` also matches `catalog.ngc.nvidia.com`, just as `docs.nvidia.com` overlaps `docs.omniverse.nvidia.com` and `docs.api.nvidia.com`.

So dedupe is **not a performance optimisation**. It saves about one redundant fetch per query, and because domain searches run concurrently, that barely changes wall time. It is a **correctness fix**: without it, the same page appears twice in the results a client receives, and TF-IDF and rank fusion count it twice. It costs nothing, carries no staleness risk and overrides no origin directives, so it stays.

A short-TTL in-process cache keyed by URL was considered and rejected *for now*. It would require overriding `no-store`, and measurement 2 gives no evidence that it would hit.

## What would reverse this

Real usage logs showing URL recurrence across queries. If an MCP agent iterating on one topic reissues similar queries, overlap could be substantial — plausible, but currently unmeasured. That evidence, not intuition, is what should reopen it.

## Related: why not a vector index (e.g. pgvector)

The same measurements answer this. An index pays off when the *same corpus* is queried repeatedly, but the candidate set here is whatever `ddgs` returns for the current query, fetched fresh. There is nothing stable to precompute: with a 0% URL repeat rate, stored vectors would almost never be reused. Even if they were, an index could remove at most the embedding share of a query, about 2–10% of 30 s, while leaving the network I/O that makes up the rest untouched.

pgvector would matter for a *different* feature: maintaining an owned, crawled corpus of NVIDIA pages to surface results `ddgs` never returns. That is a recall improvement, not a latency one — and it would give a deliberately stateless server (the `2026-07-28` protocol is itself stateless) a stateful dependency.

## The short version

> Caching looked obvious because page fetches dominate query time. But the pages are fetched per-URL, and measurement showed near-zero URL recurrence across queries, while every sampled origin sends `no-store` or `must-revalidate`. So a cache would have had a near-zero hit rate *and* required overriding the sites' explicit caching directives. What survived was deduplicating repeat URLs within a single request. Measuring that as well showed it isn't a speedup at all: duplicates come from overlapping subdomains in the domain list, about one in 85 results, so dedupe is a correctness fix that stops the same page being listed and scored twice.
