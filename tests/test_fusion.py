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
