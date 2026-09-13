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
