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
