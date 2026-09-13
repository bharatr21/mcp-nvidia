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
