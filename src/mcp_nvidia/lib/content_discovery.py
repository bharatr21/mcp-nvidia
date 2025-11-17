"""Content discovery functions for finding specific types of NVIDIA content."""

import logging
import time
from datetime import date, datetime
from typing import Any

from mcp_nvidia.lib.constants import DEFAULT_DOMAINS, MAX_RESULTS_PER_DOMAIN
from mcp_nvidia.lib.relevance import calculate_fuzzy_match_score, calculate_tfidf_scores, expand_topic_with_synonyms
from mcp_nvidia.lib.search import search_all_domains

logger = logging.getLogger(__name__)


def _parse_date(date_str: str) -> date | None:
    """
    Parse a date string into a date object.

    Supports multiple common formats:
    - ISO: YYYY-MM-DD
    - US: MM/DD/YYYY
    - European: DD/MM/YYYY
    - With time: YYYY-MM-DD HH:MM:SS

    Args:
        date_str: Date string in various formats

    Returns:
        date object or None if parsing fails
    """
    if not date_str:
        return None

    # Try ISO format first (most common and fastest)
    try:
        return datetime.fromisoformat(date_str.split("T")[0]).date()
    except (ValueError, AttributeError, TypeError):
        pass

    # Try common date formats
    common_formats = [
        "%Y-%m-%d",  # 2024-01-15
        "%m/%d/%Y",  # 01/15/2024 (US)
        "%d/%m/%Y",  # 15/01/2024 (European)
        "%B %d, %Y",  # January 15, 2024
        "%b %d, %Y",  # Jan 15, 2024
        "%Y/%m/%d",  # 2024/01/15
        "%d-%m-%Y",  # 15-01-2024
        "%Y-%m-%d %H:%M:%S",  # With time
    ]

    for fmt in common_formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()  # noqa: DTZ007
        except (ValueError, AttributeError, TypeError):
            continue

    logger.debug(f"Could not parse date in any known format: {date_str}")
    return None


async def discover_content(
    content_type: str,
    topic: str,
    max_results: int = 5,
    date_from: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Discover specific types of NVIDIA content (videos, courses, tutorials, etc.) with improved semantic matching.

    Args:
        content_type: Type of content to find (video, course, tutorial, webinar, blog)
        topic: Topic or keyword to search for
        max_results: Maximum number of results to return
        date_from: Optional date filter (YYYY-MM-DD) to only include content from this date onwards

    Returns:
        Tuple of (results, errors, warnings, timing_info)
    """
    start_time = time.time()

    # Expand topic with semantic synonyms for better coverage
    expanded_topics = expand_topic_with_synonyms(topic)
    logger.info(f"Expanded topic '{topic}' to: {expanded_topics}")

    # Map content types to search strategies with enhanced domains
    content_strategies = {
        "video": {
            "query": f"{' '.join(expanded_topics[:3])} video tutorial youtube",  # Use top 3 expanded terms
            "domains": [
                "https://developer.nvidia.com/",
                "https://blogs.nvidia.com/",
                "https://resources.nvidia.com/",
                "https://forums.developer.nvidia.com/",
            ],
            "keywords": ["youtube", "video", "watch", "tutorial", "webinar", "livestream"],
            "required_keywords": ["video", "youtube", "watch", "webinar"],  # At least one must match
        },
        "course": {
            "query": f"{' '.join(expanded_topics[:3])} course training certification DLI",
            "domains": [
                "https://developer.nvidia.com/",
                "https://resources.nvidia.com/",
                "https://docs.nvidia.com/",
            ],
            "keywords": ["course", "training", "dli", "deep learning institute", "certification", "learn", "workshop"],
            "required_keywords": ["course", "training", "dli", "certification", "workshop"],
        },
        "tutorial": {
            "query": f"{' '.join(expanded_topics[:3])} tutorial guide how-to getting started",
            "domains": [
                "https://developer.nvidia.com/",
                "https://docs.nvidia.com/",
                "https://blogs.nvidia.com/",
                "https://nvidia.github.io/",
            ],
            "keywords": ["tutorial", "guide", "how-to", "how to", "getting started", "quickstart", "walkthrough"],
            "required_keywords": ["tutorial", "guide", "how-to", "how to", "getting started"],
        },
        "webinar": {
            "query": f"{' '.join(expanded_topics[:3])} webinar event session GTC",
            "domains": [
                "https://developer.nvidia.com/",
                "https://blogs.nvidia.com/",
                "https://resources.nvidia.com/",
            ],
            "keywords": ["webinar", "event", "session", "livestream", "gtc", "conference", "talk"],
            "required_keywords": ["webinar", "event", "session", "gtc", "conference"],
        },
        "blog": {
            "query": f"{' '.join(expanded_topics[:3])}",
            "domains": ["https://blogs.nvidia.com/", "https://nvidianews.nvidia.com/"],
            "keywords": ["blog", "article", "post"],
            "required_keywords": [],  # Blog is broad, no strict requirement
        },
    }

    strategy = content_strategies.get(
        content_type.lower(),
        {
            "query": f"{topic} {content_type}",
            "domains": DEFAULT_DOMAINS,
            "keywords": [content_type],
            "required_keywords": [content_type],
        },
    )

    # Search using the strategy with fuzzy matching
    capped_max_results = min(max_results * 2, MAX_RESULTS_PER_DOMAIN)
    results, errors, warnings, timing_info = await search_all_domains(
        query=strategy["query"],
        domains=strategy.get("domains"),
        max_results_per_domain=capped_max_results,  # Get more results for better filtering
        min_relevance_score=10,  # Lower threshold for content discovery
    )

    # Filter and rank results based on content type match
    filtered_results = []
    content_keywords = strategy.get("keywords", [])
    required_keywords = strategy.get("required_keywords", [])

    # Calculate TF-IDF scores for semantic relevance
    tfidf_scores = calculate_tfidf_scores(results, topic)

    for i, result in enumerate(results):
        title = result.get("title", "").lower()
        snippet = result.get("snippet_plain", result.get("snippet", "")).lower()
        url = result.get("url", "").lower()
        detected_content_type = result.get("content_type", "")

        # Check if required keywords are present (for strict content type filtering)
        has_required_keyword = False
        if not required_keywords:
            has_required_keyword = True  # No requirements, accept all
        else:
            for req_kw in required_keywords:
                if req_kw in title or req_kw in snippet or req_kw in url or req_kw in detected_content_type:
                    has_required_keyword = True
                    break

        # Skip if doesn't match required content type
        if not has_required_keyword:
            logger.debug(f"Skipping result (no required keyword): {title[:50]}...")
            continue

        # Calculate content type match score based on keyword presence
        content_type_score = 0
        matched_keywords = []

        for keyword in content_keywords:
            keyword_score = 0

            # Check for exact or fuzzy match
            if keyword in title:
                keyword_score += 3
                matched_keywords.append(keyword)
            elif calculate_fuzzy_match_score(keyword, title, threshold=75) > 0:
                keyword_score += 2

            if keyword in snippet:
                keyword_score += 2
                if keyword not in matched_keywords:
                    matched_keywords.append(keyword)
            elif calculate_fuzzy_match_score(keyword, snippet, threshold=75) > 0:
                keyword_score += 1

            if keyword in url:
                keyword_score += 1

            content_type_score += keyword_score

        # Combine content type score with TF-IDF semantic score and original relevance
        original_relevance = result.get("relevance_score", 0)
        tfidf_score = tfidf_scores[i] if i < len(tfidf_scores) else 0.5

        # Weighted combination: 40% content type match, 30% TF-IDF, 30% original relevance
        max_content_score = len(content_keywords) * 6
        normalized_content_score = int((content_type_score / max_content_score) * 100) if max_content_score > 0 else 50

        combined_score = int(normalized_content_score * 0.4 + tfidf_score * 100 * 0.3 + original_relevance * 0.3)

        result["relevance_score"] = min(combined_score, 100)
        result["_content_match_keywords"] = matched_keywords  # For debugging

        # Apply date filter if provided
        if date_from:
            result_date_str = result.get("published_date")
            if result_date_str:
                # Parse both dates into date objects for proper comparison
                filter_date = _parse_date(date_from)
                result_date = _parse_date(result_date_str)

                if filter_date and result_date:
                    # Both dates parsed successfully - compare as date objects
                    if result_date < filter_date:
                        logger.debug(f"Skipping result (too old): {title[:50]}... ({result_date} < {filter_date})")
                        continue
                elif filter_date and not result_date:
                    # Filter date is valid but result date failed to parse
                    # Log warning and skip date filter for this result (accept it)
                    logger.debug(
                        f"Could not parse result date '{result_date_str}', skipping date filter for this result"
                    )
                # If filter_date is None (invalid), we skip the date filter entirely

        filtered_results.append(result)

    # Sort by relevance score (highest first) and limit results
    filtered_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    # Take top results
    top_results = filtered_results[:max_results]

    # Generate suggestions if no results found
    suggestions = {}
    if len(top_results) == 0:
        suggestions = {
            "similar_topics": expanded_topics[1:4] if len(expanded_topics) > 1 else [],  # Suggest alternative terms
            "alternative_content_types": {},
            "recommendation": f"Try broader search terms or different content types. Related topics: {', '.join(expanded_topics[1:3]) if len(expanded_topics) > 1 else 'N/A'}",
        }

        # Check if other content types have results
        for alt_type in ["video", "tutorial", "blog", "course", "webinar"]:
            if alt_type != content_type:
                # Quick check: count results that match this alternative type
                alt_strategy = content_strategies.get(alt_type, {})
                alt_keywords = alt_strategy.get("keywords", [])
                count = 0
                for result in results[:20]:  # Check first 20 results
                    text = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
                    if any(kw in text for kw in alt_keywords):
                        count += 1
                if count > 0:
                    suggestions["alternative_content_types"][alt_type] = count

    total_time_ms = int((time.time() - start_time) * 1000)

    # Add suggestions to timing_info for return
    enhanced_timing_info = {
        "total_time_ms": total_time_ms,
        "suggestions": suggestions,
        "expanded_topics": expanded_topics,
        "debug_info": timing_info.get("debug_info", {}),
    }

    return top_results, errors, warnings, enhanced_timing_info


def format_content_results(results: list[dict[str, Any]], content_type: str, topic: str) -> str:
    """Format content discovery results."""
    if not results:
        return f"No {content_type} content found for topic: {topic}"

    output = [f"Recommended {content_type.upper()} content for: {topic}\n"]
    output.append("=" * 60)

    for i, result in enumerate(results, 1):
        score = result.get("relevance_score", 0)
        output.append(f"\n{i}. {result.get('title', 'Untitled')} (Score: {score}/100)")
        if url := result.get("url"):
            output.append(f"   URL: {url}")
        if snippet := result.get("snippet"):
            output.append(f"   {snippet}")
        if domain := result.get("domain"):
            output.append(f"   Source: {domain}")

    # Add citations
    output.append("\n" + "=" * 60)
    output.append("\nRESOURCE LINKS:")
    output.append("-" * 60)
    for i, result in enumerate(results, 1):
        if url := result.get("url"):
            title = result.get("title", "Untitled")
            output.append(f"[{i}] {title}")
            output.append(f"    {url}")

    return "\n".join(output)
