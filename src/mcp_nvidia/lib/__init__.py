"""
MCP NVIDIA Server - Core Library

This package contains the core functionality for the MCP NVIDIA server,
organized into logical modules for better maintainability.
"""

from mcp_nvidia.lib.constants import (
    DDGS_MIN_INTERVAL,
    DEFAULT_DOMAINS,
    DOMAIN_CATEGORY_MAP,
    MAX_CONCURRENT_SEARCHES,
    MAX_QUERY_LENGTH,
    MAX_RESULTS_PER_DOMAIN,
    NVIDIA_PRODUCT_VARIANTS,
    validate_nvidia_domain,
)
from mcp_nvidia.lib.content_discovery import discover_content, format_content_results
from mcp_nvidia.lib.deduplication import deduplicate_results, merge_and_deduplicate_results
from mcp_nvidia.lib.relevance import (
    calculate_fuzzy_match_score,
    calculate_search_relevance,
    calculate_tfidf_scores,
    expand_query_with_product_variants,
    expand_topic_with_synonyms,
    extract_keywords,
    extract_matched_keywords,
    extract_phrases,
    get_domain_boost,
)
from mcp_nvidia.lib.response_builders import (
    build_content_response_json,
    build_error_response_json,
    build_search_response_json,
    build_tool_result,
    build_tool_result_with_ui,
    format_search_results,
)
from mcp_nvidia.lib.search import search_all_domains, search_nvidia_domain
from mcp_nvidia.lib.snippet import extract_sentence_snippet, fetch_url_context
from mcp_nvidia.lib.utils import (
    detect_content_type,
    extract_date_from_html,
    extract_date_from_text,
    extract_metadata_from_html,
    get_domain_category,
    is_ad_url,
)

__all__ = [
    "DDGS_MIN_INTERVAL",
    # Constants
    "DEFAULT_DOMAINS",
    "DOMAIN_CATEGORY_MAP",
    "MAX_CONCURRENT_SEARCHES",
    "MAX_QUERY_LENGTH",
    "MAX_RESULTS_PER_DOMAIN",
    "NVIDIA_PRODUCT_VARIANTS",
    # Response Builders
    "build_content_response_json",
    "build_error_response_json",
    "build_search_response_json",
    "build_tool_result",
    "build_tool_result_with_ui",
    # Relevance
    "calculate_fuzzy_match_score",
    "calculate_search_relevance",
    "calculate_tfidf_scores",
    # Deduplication
    "deduplicate_results",
    # Utils
    "detect_content_type",
    # Content Discovery
    "discover_content",
    "expand_query_with_product_variants",
    "expand_topic_with_synonyms",
    "extract_date_from_html",
    "extract_date_from_text",
    "extract_keywords",
    "extract_matched_keywords",
    "extract_metadata_from_html",
    "extract_phrases",
    # Snippet
    "extract_sentence_snippet",
    "fetch_url_context",
    "format_content_results",
    "format_search_results",
    "get_domain_boost",
    "get_domain_category",
    "is_ad_url",
    "merge_and_deduplicate_results",
    # Search
    "search_all_domains",
    "search_nvidia_domain",
    "validate_nvidia_domain",
]
