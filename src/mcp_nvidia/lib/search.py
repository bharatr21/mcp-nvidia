"""Search functions for NVIDIA domains using DuckDuckGo."""

import asyncio
import logging
import time
from typing import Any

import httpx
from ddgs import DDGS

from mcp_nvidia.lib.constants import (
    DDGS_MIN_INTERVAL,
    DDGS_RATE_LIMIT_LOCK,
    DEFAULT_DOMAINS,
    MAX_CONCURRENT_SEARCHES,
    MAX_RESULTS_PER_DOMAIN,
    validate_nvidia_domain,
)
from mcp_nvidia.lib.deduplication import deduplicate_results
from mcp_nvidia.lib.relevance import (
    calculate_search_relevance,
    calculate_tfidf_scores,
    expand_query_with_product_variants,
    get_domain_boost,
)
from mcp_nvidia.lib.snippet import fetch_url_context
from mcp_nvidia.lib.utils import detect_content_type, extract_date_from_text, get_domain_category, is_ad_url

logger = logging.getLogger(__name__)

# Rate limiting state for DDGS API
# Lock is imported from constants.py to ensure process-wide coordination
_last_ddgs_call_time = 0.0


def _fetch_ddgs_results_sync(search_query: str, max_results: int) -> list[dict[str, Any]]:
    """
    Synchronous helper to fetch DuckDuckGo search results.

    This function runs in a worker thread to avoid blocking the async event loop.

    Args:
        search_query: The search query with site: operator
        max_results: Maximum number of results to return

    Returns:
        List of raw search results from DDGS
    """
    with DDGS() as ddgs:
        return list(ddgs.text(search_query, max_results=max_results))


async def _fetch_ddgs_results(search_query: str, max_results: int) -> list[dict[str, Any]]:
    """
    Async wrapper for DDGS with rate limiting.

    SECURITY: Implements rate limiting to prevent exhausting DuckDuckGo's limits.
    Minimum 0.2 seconds between calls to avoid RatelimitException (HTTP 202).

    Args:
        search_query: The search query with site: operator
        max_results: Maximum number of results to return

    Returns:
        List of raw search results from DDGS

    Raises:
        Exception: If DDGS search fails (including rate limit errors)
    """
    global _last_ddgs_call_time

    async with DDGS_RATE_LIMIT_LOCK:
        # SECURITY: Enforce minimum interval between DDGS calls
        now = asyncio.get_event_loop().time()
        elapsed = now - _last_ddgs_call_time
        if elapsed < DDGS_MIN_INTERVAL:
            wait_time = DDGS_MIN_INTERVAL - elapsed
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s before DDGS call")
            await asyncio.sleep(wait_time)

        _last_ddgs_call_time = asyncio.get_event_loop().time()

    # Run DDGS in thread pool to avoid blocking event loop
    try:
        return await asyncio.to_thread(_fetch_ddgs_results_sync, search_query, max_results)
    except Exception as e:
        logger.exception(f"DDGS search failed: {e}")
        raise


async def search_nvidia_domain(
    client: httpx.AsyncClient, domain: str, query: str, max_results: int = 5
) -> list[dict[str, Any]]:
    """
    Search a specific NVIDIA domain using ddgs package.

    Args:
        client: HTTP client for making requests (used for context fetching)
        domain: Domain to search (e.g., "developer.nvidia.com")
        query: Search query
        max_results: Maximum number of results to return

    Returns:
        List of search results with title, url, snippet, and enhanced context
    """
    results = []

    try:
        # Clean domain for site: operator
        clean_domain = domain.replace("https://", "").replace("http://", "").rstrip("/")

        # Use ddgs package with site: operator for domain-specific search
        search_query = f"site:{clean_domain} {query}"

        # Perform search using ddgs with rate limiting
        search_results = await _fetch_ddgs_results(search_query, max_results)

        # Process each result and fetch enhanced context
        for result in search_results:
            try:
                title = result.get("title", "")
                url = result.get("href", "")
                snippet = result.get("body", "")

                if not title or not url:
                    continue

                # SECURITY: Block ad URLs and tracking URLs
                if is_ad_url(url):
                    logger.debug(f"Skipping ad URL: {url}")
                    continue

                # SECURITY: Re-validate that the result URL is from an NVIDIA domain
                if not validate_nvidia_domain(url):
                    logger.debug(f"Skipping non-NVIDIA URL: {url}")
                    continue

                # Fetch enhanced context with highlighted snippet, date, and metadata
                enhanced_snippet, published_date, page_metadata = await fetch_url_context(
                    client, url, snippet, context_chars=200
                )

                # Create plain version without bold markers
                snippet_plain = enhanced_snippet.replace("**", "")

                # If date not extracted from page, try from snippet
                if not published_date:
                    published_date = extract_date_from_text(f"{title} {snippet}")

                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": enhanced_snippet,
                        "snippet_plain": snippet_plain,
                        "domain": clean_domain,
                        "published_date": published_date,
                        "metadata": page_metadata,
                    }
                )

            except Exception as e:
                logger.debug(f"Error processing result item: {e!s}")
                continue

    except Exception as e:
        logger.exception(f"Error searching {domain}: {e!s}")
        # Add fallback message if search completely fails
        error_msg = f"Search temporarily unavailable. Error: {e!s}"
        results.append(
            {
                "title": f"Search error on {clean_domain}",
                "url": f"https://{clean_domain}",
                "snippet": error_msg,
                "snippet_plain": error_msg,
                "domain": clean_domain,
                "is_error": True,  # Mark as error result to prevent inflated relevance scores
            }
        )

    return results


async def _search_domain_with_semaphore(
    semaphore: asyncio.Semaphore,
    client: httpx.AsyncClient,
    domain: str,
    query: str,
    max_results: int,
) -> list[dict[str, Any]]:
    """
    Wrapper to search a domain with semaphore-based concurrency limiting.

    This ensures that at most MAX_CONCURRENT_SEARCHES domains are searched
    concurrently within a single request, preventing resource exhaustion.

    Args:
        semaphore: Asyncio semaphore to limit concurrency
        client: HTTP client for making requests
        domain: Domain to search
        query: Search query
        max_results: Maximum number of results

    Returns:
        List of search results
    """
    async with semaphore:
        return await search_nvidia_domain(client, domain, query, max_results)


async def search_all_domains(
    query: str,
    domains: list[str] | None = None,
    max_results_per_domain: int = 3,
    min_relevance_score: int = 17,
    sort_by: str = "relevance",
    date_from: str | None = None,
    date_to: str | None = None,
    max_total_results: int | None = None,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Search across all NVIDIA domains with controlled concurrency.

    Domain searches are parallelized but limited to MAX_CONCURRENT_SEARCHES
    concurrent operations to prevent resource exhaustion and ensure predictable
    performance under heavier configurations.

    Args:
        query: Search query
        domains: List of domains to search (uses DEFAULT_DOMAINS if None)
        max_results_per_domain: Maximum results per domain
        min_relevance_score: Minimum relevance score threshold (0-100, default 17)
        sort_by: Sort order - "relevance", "date", or "domain" (default: "relevance")
        date_from: Optional date filter (YYYY-MM-DD) - only include results from this date onwards
        date_to: Optional date filter (YYYY-MM-DD) - only include results up to this date
        max_total_results: Optional limit on total results across all domains
        allowed_domains: Optional list of domains to include (filter results to only these domains)
        blocked_domains: Optional list of domains to exclude (filter out results from these domains)

    Returns:
        Tuple of (results, errors, warnings, timing_info)
    """
    # SECURITY: Enforce MAX_RESULTS_PER_DOMAIN limit defensively
    # Prevent callers from bypassing the limit by requesting excessive results
    if max_results_per_domain > MAX_RESULTS_PER_DOMAIN:
        logger.warning(
            f"max_results_per_domain={max_results_per_domain} exceeds limit. "
            f"Capping to MAX_RESULTS_PER_DOMAIN={MAX_RESULTS_PER_DOMAIN}"
        )
    max_results_per_domain = min(max_results_per_domain, MAX_RESULTS_PER_DOMAIN)

    if domains is None:
        domains = DEFAULT_DOMAINS

    # Apply domain filtering if specified
    if allowed_domains:
        # Only search domains that are in the allowed list
        allowed_set = {d.lower().replace("https://", "").replace("http://", "").rstrip("/") for d in allowed_domains}
        domains = [
            d for d in domains if d.lower().replace("https://", "").replace("http://", "").rstrip("/") in allowed_set
        ]
        logger.info(f"Domain filtering: {len(domains)} domains after applying allowed_domains filter")

    if blocked_domains:
        # Exclude domains that are in the blocked list
        blocked_set = {d.lower().replace("https://", "").replace("http://", "").rstrip("/") for d in blocked_domains}
        domains = [
            d
            for d in domains
            if d.lower().replace("https://", "").replace("http://", "").rstrip("/") not in blocked_set
        ]
        logger.info(f"Domain filtering: {len(domains)} domains after applying blocked_domains filter")

    # Expand query with product variants for better matching
    expanded_query = expand_query_with_product_variants(query)
    if expanded_query != query:
        logger.info(f"Expanded query with product variants: '{query}' -> '{expanded_query}'")
        query = expanded_query

    all_results = []
    errors = []
    warnings = []
    timing_info = {}

    start_time = time.time()

    # Create semaphore to limit concurrent domain searches
    # This prevents resource exhaustion when searching many domains
    domain_search_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Search all domains with controlled concurrency
        domain_start_times = {domain: time.time() for domain in domains}

        tasks = [
            _search_domain_with_semaphore(domain_search_semaphore, client, domain, query, max_results_per_domain)
            for domain in domains
        ]

        domain_results = await asyncio.gather(*tasks, return_exceptions=True)

        for domain, results in zip(domains, domain_results, strict=False):
            # Calculate timing for this domain
            domain_time_ms = int((time.time() - domain_start_times[domain]) * 1000)
            clean_domain = domain.replace("https://", "").replace("http://", "").rstrip("/")
            timing_info[clean_domain] = domain_time_ms

            if isinstance(results, Exception):
                error_msg = str(results)
                logger.error(f"Domain search failed for {domain}: {error_msg}")
                errors.append({"domain": clean_domain, "code": "SEARCH_FAILED", "message": error_msg})
                warnings.append(
                    {
                        "code": "PARTIAL_FAILURE",
                        "message": f"Search failed for domain: {clean_domain}",
                        "affected_domains": [clean_domain],
                    }
                )
                continue

            all_results.extend(results)

    total_time_ms = int((time.time() - start_time) * 1000)

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

    # Add content type detection to each result
    for result in all_results:
        if result.get("is_error"):
            continue

        domain_category = get_domain_category(result.get("domain", ""))
        content_type = detect_content_type(
            result.get("title", ""),
            result.get("snippet_plain", result.get("snippet", "")),
            result.get("url", ""),
            domain_category,
        )
        result["content_type"] = content_type

    # Filter by minimum relevance score
    filtered_results = [r for r in all_results if r.get("relevance_score", 0) >= min_relevance_score]

    # Deduplicate results (v0.3.0 feature)
    filtered_results = deduplicate_results(filtered_results)

    # Apply date filtering if specified
    if date_from or date_to:
        date_filtered_results = []
        for result in filtered_results:
            result_date = result.get("published_date")
            if not result_date:
                # Include results without dates (can't filter them)
                date_filtered_results.append(result)
                continue

            # Apply date_from filter
            if date_from and result_date < date_from:
                logger.debug(
                    f"Filtering out result (before date_from): {result.get('title', '')[:50]}... ({result_date} < {date_from})"
                )
                continue

            # Apply date_to filter
            if date_to and result_date > date_to:
                logger.debug(
                    f"Filtering out result (after date_to): {result.get('title', '')[:50]}... ({result_date} > {date_to})"
                )
                continue

            date_filtered_results.append(result)

        logger.info(f"Date filtering: {len(filtered_results)} -> {len(date_filtered_results)} results")
        filtered_results = date_filtered_results

    # Apply domain filtering on results (in case some results have different domains)
    if allowed_domains or blocked_domains:
        domain_filtered_results = []
        allowed_set = {
            d.lower().replace("https://", "").replace("http://", "").rstrip("/") for d in (allowed_domains or [])
        }
        blocked_set = {
            d.lower().replace("https://", "").replace("http://", "").rstrip("/") for d in (blocked_domains or [])
        }

        for result in filtered_results:
            result_domain = result.get("domain", "").lower()

            # Check if domain is blocked
            if blocked_domains and result_domain in blocked_set:
                logger.debug(f"Filtering out result from blocked domain: {result_domain}")
                continue

            # Check if domain is in allowed list (if specified)
            if allowed_domains and result_domain not in allowed_set:
                logger.debug(f"Filtering out result not in allowed domains: {result_domain}")
                continue

            domain_filtered_results.append(result)

        if len(domain_filtered_results) != len(filtered_results):
            logger.info(f"Result domain filtering: {len(filtered_results)} -> {len(domain_filtered_results)} results")
            filtered_results = domain_filtered_results

    # Sort results based on sort_by parameter
    if sort_by == "date":
        # Sort by date (newest first), then by relevance
        filtered_results.sort(
            key=lambda x: (
                x.get("published_date") or "0000-00-00",  # Put undated results last
                x.get("relevance_score", 0),
            ),
            reverse=True,
        )
    elif sort_by == "domain":
        # Sort by domain, then by relevance
        filtered_results.sort(
            key=lambda x: (
                x.get("domain", ""),
                -x.get("relevance_score", 0),
            ),
            reverse=False,  # Alphabetical domain, highest relevance first within domain
        )
    else:
        # Default: sort by relevance score (highest first)
        filtered_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    # Apply max_total_results limit if specified
    if max_total_results is not None and max_total_results > 0 and len(filtered_results) > max_total_results:
        logger.info(f"Limiting results to max_total_results: {len(filtered_results)} -> {max_total_results}")
        filtered_results = filtered_results[:max_total_results]

    # Build debug info if debug logging is enabled
    debug_info = None
    if logger.isEnabledFor(logging.DEBUG):
        search_strategies = [
            f"site:{domain.replace('https://', '').replace('http://', '').rstrip('/')} {query}" for domain in domains
        ]
        debug_info = {
            "search_strategies": search_strategies,
            "timing_breakdown": timing_info,
            "sort_by": sort_by,
            "filters_applied": {
                "date_from": date_from,
                "date_to": date_to,
                "max_total_results": max_total_results,
                "allowed_domains": allowed_domains,
                "blocked_domains": blocked_domains,
            },
        }

    return filtered_results, errors, warnings, {"total_time_ms": total_time_ms, "debug_info": debug_info}
