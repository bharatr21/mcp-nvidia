"""Deduplication functions for search results."""

import logging
from typing import Any

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two text strings using fuzzy matching.

    Args:
        text1: First text string
        text2: Second text string

    Returns:
        Similarity score from 0.0 to 1.0
    """
    if not text1 or not text2:
        return 0.0

    # Use token_set_ratio for better similarity detection
    return fuzz.token_set_ratio(text1.lower(), text2.lower()) / 100.0


def deduplicate_results(
    results: list[dict[str, Any]], title_threshold: float = 0.85, snippet_threshold: float = 0.90
) -> list[dict[str, Any]]:
    """
    Deduplicate search results based on title and snippet similarity.

    Args:
        results: List of search results
        title_threshold: Minimum title similarity to consider duplicates (0-1)
        snippet_threshold: Minimum snippet similarity to consider duplicates (0-1)

    Returns:
        Deduplicated list of results
    """
    if not results:
        return results

    deduplicated = []
    seen_urls = set()

    for result in results:
        url = result.get("url", "")

        # Skip if exact URL match (only for non-empty URLs)
        if url and url in seen_urls:
            logger.debug(f"Skipping duplicate URL: {url}")
            continue

        # Check similarity with existing results
        is_duplicate = False
        title = result.get("title", "")
        snippet = result.get("snippet_plain", result.get("snippet", ""))

        for existing in deduplicated:
            existing_title = existing.get("title", "")
            existing_snippet = existing.get("snippet_plain", existing.get("snippet", ""))

            # Calculate similarities
            title_similarity = calculate_text_similarity(title, existing_title)
            snippet_similarity = calculate_text_similarity(snippet, existing_snippet)

            # Consider duplicate if both title and snippet are very similar
            if title_similarity >= title_threshold and snippet_similarity >= snippet_threshold:
                logger.debug(
                    f"Skipping similar result: {title[:50]}... "
                    f"(title_sim={title_similarity:.2f}, snippet_sim={snippet_similarity:.2f})"
                )
                is_duplicate = True
                break

        if not is_duplicate:
            deduplicated.append(result)
            if url:
                seen_urls.add(url)

    logger.info(
        f"Deduplication: {len(results)} -> {len(deduplicated)} results ({len(results) - len(deduplicated)} duplicates removed)"
    )
    return deduplicated


def merge_and_deduplicate_results(
    *result_lists: list[dict[str, Any]],
    title_threshold: float = 0.85,
    snippet_threshold: float = 0.90,
) -> list[dict[str, Any]]:
    """
    Merge multiple result lists and deduplicate across all of them.

    Useful when combining results from search_nvidia and discover_nvidia_content
    to avoid showing the same result twice.

    Args:
        *result_lists: Variable number of result lists to merge
        title_threshold: Minimum title similarity to consider duplicates (0-1)
        snippet_threshold: Minimum snippet similarity to consider duplicates (0-1)

    Returns:
        Merged and deduplicated list, preserving relative order within each source
    """
    # Flatten all results while preserving order
    all_results = []
    for result_list in result_lists:
        if result_list:
            all_results.extend(result_list)

    if not all_results:
        return []

    # Deduplicate the combined results
    return deduplicate_results(all_results, title_threshold=title_threshold, snippet_threshold=snippet_threshold)
