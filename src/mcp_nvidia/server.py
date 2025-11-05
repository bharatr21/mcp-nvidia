"""MCP server for searching across NVIDIA domains."""

import asyncio
import json
import logging
import os
import re
import signal
import sys
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult

# Import NLTK stopwords
try:
    from nltk.corpus import stopwords
    import nltk
    try:
        STOPWORDS = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        STOPWORDS = set(stopwords.words('english'))
except ImportError:
    STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'this', 'but', 'or', 'not', 'can'
    }

# Configure logging
log_level = os.getenv("MCP_NVIDIA_LOG_LEVEL", "INFO")

# Set up logging to file (doesn't interfere with stdio)
log_dir = os.path.expanduser("~/.mcp-nvidia")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "server.log")

# Configure logging to file
logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also log to stderr for client capture
    ]
)
logger = logging.getLogger(__name__)

# Log startup
logger.info(f"MCP NVIDIA server starting (log level: {log_level})")
logger.info(f"Logs written to: {log_file}")

# Default NVIDIA domains to search
DEFAULT_DOMAINS = [
    "https://blogs.nvidia.com/",
    "https://build.nvidia.com/",
    "https://catalog.ngc.nvidia.com/",
    "https://developer.download.nvidia.com/",
    "https://developer.nvidia.com/",
    "https://docs.api.nvidia.com/",
    "https://docs.nvidia.com/",
    "https://docs.omniverse.nvidia.com/",
    "https://forums.developer.nvidia.com/",
    "https://forums.nvidia.com/",
    "https://gameworksdocs.nvidia.com/",
    "https://ngc.nvidia.com/",
    "https://nvidia.github.io/",
    "https://nvidianews.nvidia.com/",
    "https://research.nvidia.com/",
    "https://resources.nvidia.com/",
]


def validate_nvidia_domain(domain: str) -> bool:
    """
    Validate that a domain is a valid NVIDIA domain or subdomain.

    Args:
        domain: URL string to validate

    Returns:
        True if domain is nvidia.com, a subdomain, or nvidia.github.io, False otherwise
    """
    try:
        parsed = urlparse(domain)
        hostname = parsed.netloc or parsed.path.split('/')[0]
        hostname = hostname.lower()

        # Check if it's nvidia.com or a subdomain of nvidia.com
        if hostname == "nvidia.com" or hostname.endswith(".nvidia.com"):
            return True

        # Allow NVIDIA's official GitHub Pages (specifically nvidia.github.io only)
        if hostname == "nvidia.github.io":
            return True

        logger.warning(f"Domain validation failed for: {domain} (hostname: {hostname})")
        return False
    except Exception as e:
        logger.exception(f"Error validating domain {domain}: {e}")
        return False


def is_ad_url(url: str) -> bool:
    """
    Check if a URL is an advertisement or tracking URL.

    Args:
        url: URL string to check

    Returns:
        True if the URL is an ad/tracking URL, False otherwise
    """
    try:
        url_lower = url.lower()

        # Block DuckDuckGo ad URLs
        if "duckduckgo.com/y.js" in url_lower:
            return True

        # Block URLs with ad-related query parameters
        ad_patterns = [
            "ad_domain=",
            "ad_provider=",
            "ad_type=",
            "adurl=",
            "adclick=",
        ]

        for pattern in ad_patterns:
            if pattern in url_lower:
                return True

        return False
    except Exception as e:
        logger.debug(f"Error checking ad URL {url}: {e}")
        return False


# Allow override via environment variable (comma-separated list)
if custom_domains := os.getenv("MCP_NVIDIA_DOMAINS"):
    raw_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]
    validated_domains = []
    
    for domain in raw_domains:
        if validate_nvidia_domain(domain):
            validated_domains.append(domain)
        else:
            logger.warning(f"Skipping invalid domain (not nvidia.com): {domain}")
    
    if validated_domains:
        DEFAULT_DOMAINS = validated_domains
        logger.info(f"Using custom domains from environment: {DEFAULT_DOMAINS}")
    else:
        logger.warning("No valid NVIDIA domains found in MCP_NVIDIA_DOMAINS. Using defaults.")

# Create server instance
app = Server("mcp-nvidia")

# SECURITY: Rate limiting for DDGS calls
_last_ddgs_call_time = 0.0
_ddgs_call_lock = asyncio.Lock()
DDGS_MIN_INTERVAL = 0.2  # Minimum 0.2 second (200ms) between DDGS calls (~5 searches/sec, 25% of 20 req/sec limit)

# SECURITY: Concurrency limits
MAX_CONCURRENT_SEARCHES = 5
_search_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)

# SECURITY: Input validation limits
MAX_QUERY_LENGTH = 500
MAX_RESULTS_PER_DOMAIN = 10

# Domain category mapping - order matters! More specific patterns first
DOMAIN_CATEGORY_MAP = [
    # Forums (most specific first)
    ("forums.developer.nvidia.com", "forum"),
    ("forums.nvidia.com", "forum"),

    # Downloads
    ("developer.download.nvidia.com", "downloads"),

    # Documentation (specific subdomains first)
    ("nvidia.github.io", "documentation"),
    ("docs.api.nvidia.com", "documentation"),
    ("docs.omniverse.nvidia.com", "documentation"),
    ("gameworksdocs.nvidia.com", "documentation"),
    ("docs.nvidia.com", "documentation"),

    # Catalog
    ("catalog.ngc.nvidia.com", "catalog"),
    ("ngc.nvidia.com", "catalog"),

    # Resources
    ("resources.nvidia.com", "resources"),

    # Blog, News, Research
    ("blogs.nvidia.com", "blog"),
    ("nvidianews.nvidia.com", "news"),
    ("research.nvidia.com", "research"),

    # Build and Developer (broader matches last)
    ("build.nvidia.com", "build"),
    ("developer.nvidia.com", "developer"),
]


def extract_keywords(query: str) -> list[str]:
    """
    Extract meaningful keywords from a query string.

    Filters out stopwords and very short words to get actual keywords.

    Args:
        query: Search query string

    Returns:
        List of keywords (non-stopwords, meaningful words)
    """
    # Split and normalize
    words = query.lower().split()

    # Filter out stopwords and very short words
    keywords = []
    for word in words:
        # Remove common punctuation
        cleaned = word.strip('.,!?;:()"\'')

        # Keep if:
        # - Not a stopword
        # - Length >= 2 characters
        # - Contains at least one letter (to avoid pure numbers/symbols unless they're tech terms)
        if (cleaned and
            cleaned not in STOPWORDS and
            len(cleaned) >= 2 and
            any(c.isalpha() for c in cleaned)):
            keywords.append(cleaned)

    return keywords


def get_domain_category(domain: str) -> str:
    """
    Categorize an NVIDIA domain.

    Args:
        domain: Domain URL or hostname

    Returns:
        Category string: documentation, blog, news, developer, build, research, catalog, forum, downloads, resources, or other
    """
    domain_lower = domain.lower()

    # Check patterns in order (most specific first)
    for pattern, category in DOMAIN_CATEGORY_MAP:
        if pattern in domain_lower:
            return category

    return "other"


def extract_matched_keywords(query: str, result: dict[str, Any]) -> list[str]:
    """
    Extract which meaningful keywords from the query matched in the result.

    Only returns actual keywords (non-stopwords) that appear in the result.

    Args:
        query: Search query string
        result: Search result dictionary

    Returns:
        List of matched keywords from the query
    """
    # Extract only meaningful keywords from query
    keywords = extract_keywords(query)

    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()
    url = result.get("url", "").lower()

    matched = []
    for keyword in keywords:
        if keyword in title or keyword in snippet or keyword in url:
            matched.append(keyword)

    return matched


def build_search_response_json(
    results: list[dict[str, Any]],
    query: str,
    domains_searched: int,
    search_time_ms: int,
    errors: list[dict[str, Any]] = None,
    warnings: list[dict[str, Any]] = None,
    debug_info: dict[str, Any] = None
) -> dict[str, Any]:
    """
    Build structured JSON response for search_nvidia tool.

    Args:
        results: List of search results
        query: Original search query
        domains_searched: Number of domains searched
        search_time_ms: Total search time in milliseconds
        errors: List of error objects
        warnings: List of warning objects
        debug_info: Debug information (only if DEBUG logging enabled)

    Returns:
        Structured JSON response
    """
    if errors is None:
        errors = []
    if warnings is None:
        warnings = []

    # Build results with all fields
    structured_results = []
    for i, result in enumerate(results, 1):
        domain = result.get("domain", "")
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = result.get("snippet", "")
        snippet_plain = result.get("snippet_plain", snippet.replace("**", ""))
        relevance_score = result.get("relevance_score", 0)

        # Build formatted_text for this result (per-result markdown)
        formatted_text = f"**{title}**\n{snippet}\n🔗 {url}\n📍 {domain}"

        structured_results.append({
            "id": i,
            "title": title,
            "url": url,
            "snippet": snippet,
            "snippet_plain": snippet_plain,
            "domain": domain,
            "domain_category": get_domain_category(domain),
            "relevance_score": relevance_score,
            "matched_keywords": extract_matched_keywords(query, result),
            "metadata": {},
            "formatted_text": formatted_text
        })

    # Build citations
    citations = []
    for i, result in enumerate(results, 1):
        citations.append({
            "number": i,
            "url": result.get("url", ""),
            "title": result.get("title", "Untitled"),
            "domain": result.get("domain", "")
        })

    # Count domains with results
    domains_with_results = len(set(r.get("domain", "") for r in results if r.get("domain")))

    # Build summary
    summary = {
        "query": query,
        "total_results": len(results),
        "domains_searched": domains_searched,
        "domains_with_results": domains_with_results,
        "search_time_ms": search_time_ms
    }

    # Add debug info if provided
    if debug_info is not None:
        summary["debug_info"] = debug_info

    return {
        "success": len(errors) == 0 or len(results) > 0,
        "summary": summary,
        "results": structured_results,
        "citations": citations,
        "warnings": warnings,
        "errors": errors
    }


def build_content_response_json(
    results: list[dict[str, Any]],
    content_type: str,
    topic: str,
    search_time_ms: int,
    errors: list[dict[str, Any]] = None,
    warnings: list[dict[str, Any]] = None,
    debug_info: dict[str, Any] = None
) -> dict[str, Any]:
    """
    Build structured JSON response for discover_nvidia_content tool.

    Args:
        results: List of content results
        content_type: Type of content searched
        topic: Search topic
        search_time_ms: Total search time in milliseconds
        errors: List of error objects
        warnings: List of warning objects
        debug_info: Debug information (only if DEBUG logging enabled)

    Returns:
        Structured JSON response
    """
    if errors is None:
        errors = []
    if warnings is None:
        warnings = []

    # Build content array with all fields
    structured_content = []
    for i, result in enumerate(results, 1):
        domain = result.get("domain", "")
        title = result.get("title", "Untitled")
        url = result.get("url", "")
        snippet = result.get("snippet", "")
        snippet_plain = result.get("snippet_plain", snippet.replace("**", ""))
        relevance_score = result.get("relevance_score", 0)

        # Build formatted_text without stars (per-result markdown)
        formatted_text = f"**{title}** (Score: {relevance_score}/100)\n{snippet}\n🔗 {url}\n📍 {domain}"

        # Add content type emoji
        type_emoji = {
            "video": "🎥",
            "course": "📚",
            "tutorial": "📖",
            "webinar": "🎤",
            "blog": "📝"
        }.get(content_type.lower(), "📄")

        formatted_text += f"\n{type_emoji} {content_type.title()}"

        structured_content.append({
            "id": i,
            "title": title,
            "url": url,
            "content_type": content_type.lower(),
            "snippet": snippet,
            "snippet_plain": snippet_plain,
            "relevance_score": relevance_score,
            "domain": domain,
            "domain_category": get_domain_category(domain),
            "matched_keywords": extract_matched_keywords(topic, result),
            "metadata": {},
            "formatted_text": formatted_text
        })

    # Build resource links
    resource_links = []
    for i, result in enumerate(results, 1):
        resource_links.append({
            "number": i,
            "url": result.get("url", ""),
            "title": result.get("title", "Untitled"),
            "type": content_type.lower()
        })

    # Build summary
    summary = {
        "content_type": content_type.lower(),
        "topic": topic,
        "total_found": len(results),
        "search_time_ms": search_time_ms
    }

    # Add debug info if provided
    if debug_info is not None:
        summary["debug_info"] = debug_info

    return {
        "success": len(errors) == 0 or len(results) > 0,
        "summary": summary,
        "content": structured_content,
        "resource_links": resource_links,
        "warnings": warnings,
        "errors": errors
    }


def build_error_response_json(
    error_code: str,
    error_message: str,
    details: dict[str, Any] = None
) -> dict[str, Any]:
    """
    Build uniform error response structure.

    Args:
        error_code: Error code string
        error_message: Human-readable error message
        details: Additional error details

    Returns:
        Structured error response
    """
    error_response = {
        "success": False,
        "error": {
            "code": error_code,
            "message": error_message
        }
    }

    if details:
        error_response["error"]["details"] = details

    return error_response


def build_tool_result(response: dict[str, Any]) -> CallToolResult:
    """
    Build CallToolResult with both text content and structured data.

    Args:
        response: The JSON response dictionary

    Returns:
        CallToolResult with both content and structuredContent
    """
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(response, indent=2))],
        structuredContent=response,
        isError=not response.get("success", False)
    )


def format_search_results(results: list[dict[str, Any]], query: str) -> str:
    """Format search results into a readable string with citations."""
    if not results:
        return f"No results found for query: {query}"
    
    output = [f"Search results for: {query}\n"]
    output.append("=" * 60)
    
    # Format main results
    for i, result in enumerate(results, 1):
        score = result.get("relevance_score", 0)

        output.append(f"\n{i}. {result.get('title', 'Untitled')} (Score: {score}/100)")
        if url := result.get('url'):
            output.append(f"   URL: {url}")
        if snippet := result.get('snippet'):
            output.append(f"   {snippet}")
        if domain := result.get('domain'):
            output.append(f"   Source: {domain}")
    
    # Add citations section for easy reference
    output.append("\n" + "=" * 60)
    output.append("\nCITATIONS:")
    output.append("-" * 60)
    for i, result in enumerate(results, 1):
        if url := result.get('url'):
            title = result.get('title', 'Untitled')
            output.append(f"[{i}] {title}")
            output.append(f"    {url}")
    
    return "\n".join(output)


async def fetch_url_context(
    client: httpx.AsyncClient,
    url: str,
    snippet: str,
    context_chars: int = 200
) -> str:
    """
    Fetch the webpage and extract surrounding context for the snippet.

    Args:
        client: HTTP client for making requests
        url: URL to fetch
        snippet: Snippet text to find in the page
        context_chars: Number of characters to include on each side of snippet

    Returns:
        Extended snippet with surrounding context, or original snippet if fetch fails
    """
    try:
        # SECURITY: Re-validate URL before fetching to prevent SSRF
        if not validate_nvidia_domain(url):
            logger.warning(f"Skipping fetch for non-NVIDIA URL: {url}")
            return snippet

        # SECURITY: Validate URL scheme (only allow https)
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            logger.warning(f"Skipping fetch for non-HTTP(S) URL: {url}")
            return snippet

        # SECURITY: Disable redirects to prevent redirect-based SSRF
        response = await client.get(url, timeout=10.0, follow_redirects=False)
        if response.status_code != 200:
            return snippet
        
        # Parse HTML to get text content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Try to find the snippet or similar text in the page
        snippet_clean = re.sub(r'\s+', ' ', snippet).strip().lower()
        text_lower = text.lower()
        
        # Find position of snippet in text
        pos = text_lower.find(snippet_clean[:50])  # Use first 50 chars for matching
        
        if pos != -1:
            # Extract context around the snippet
            start = max(0, pos - context_chars)
            end = min(len(text), pos + len(snippet) + context_chars)
            
            context = text[start:end]
            
            # Add ellipsis if truncated
            if start > 0:
                context = "..." + context
            if end < len(text):
                context = context + "..."
            
            # Highlight the snippet portion
            snippet_start = context.lower().find(snippet_clean[:30])
            if snippet_start != -1:
                snippet_end = snippet_start + len(snippet)
                highlighted = (
                    context[:snippet_start] +
                    "**" + context[snippet_start:snippet_end] + "**" +
                    context[snippet_end:]
                )
                return highlighted
            
            return context
        
        # If snippet not found, return original
        return snippet
        
    except Exception as e:
        logger.debug(f"Error fetching context from {url}: {str(e)}")
        return snippet


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
    search_results = []
    with DDGS() as ddgs:
        search_results = list(ddgs.text(search_query, max_results=max_results))
    return search_results


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

    async with _ddgs_call_lock:
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
        logger.error(f"DDGS search failed: {e}")
        raise


async def search_nvidia_domain(
    client: httpx.AsyncClient,
    domain: str,
    query: str,
    max_results: int = 5
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
        clean_domain = domain.replace('https://', '').replace('http://', '').rstrip('/')

        # Use ddgs package with site: operator for domain-specific search
        search_query = f"site:{clean_domain} {query}"

        # Perform search using ddgs with rate limiting
        search_results = await _fetch_ddgs_results(search_query, max_results)

        # Process each result and fetch enhanced context
        for result in search_results:
            try:
                title = result.get('title', '')
                url = result.get('href', '')
                snippet = result.get('body', '')

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

                # Fetch enhanced context with highlighted snippet
                enhanced_snippet = await fetch_url_context(client, url, snippet, context_chars=200)

                # Create plain version without bold markers
                snippet_plain = enhanced_snippet.replace("**", "")

                results.append({
                    "title": title,
                    "url": url,
                    "snippet": enhanced_snippet,
                    "snippet_plain": snippet_plain,
                    "domain": clean_domain
                })

            except Exception as e:
                logger.debug(f"Error processing result item: {str(e)}")
                continue

    except Exception as e:
        logger.exception(f"Error searching {domain}: {str(e)}")
        # Add fallback message if search completely fails
        error_msg = f"Search temporarily unavailable. Error: {str(e)}"
        results.append({
            "title": f"Search error for '{query}' on {clean_domain}",
            "url": f"https://{clean_domain}",
            "snippet": error_msg,
            "snippet_plain": error_msg,
            "domain": clean_domain
        })

    return results


def calculate_search_relevance(result: dict[str, Any], query: str) -> int:
    """
    Calculate relevance score for a search result based on keyword matches.

    Uses only meaningful keywords (non-stopwords) from the query.
    Weighs title matches higher than snippet, and snippet higher than URL.

    Args:
        result: Search result dictionary
        query: Search query string

    Returns:
        Relevance score from 0-100
    """
    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()
    url = result.get("url", "").lower()

    # Extract meaningful keywords only (no stopwords)
    keywords = extract_keywords(query)

    if not keywords:
        return 0

    # Calculate raw score based on keyword matches
    raw_score = 0
    max_score_per_keyword = 6  # 3 + 2 + 1

    for keyword in keywords:
        keyword_score = 0

        # Title matches are most important (3 points)
        if keyword in title:
            keyword_score += 3

        # Snippet matches are moderately important (2 points)
        if keyword in snippet:
            keyword_score += 2

        # URL matches are least important (1 point)
        if keyword in url:
            keyword_score += 1

        raw_score += keyword_score

    # Normalize to 0-100 scale
    max_possible_score = len(keywords) * max_score_per_keyword
    normalized_score = int((raw_score / max_possible_score) * 100) if max_possible_score > 0 else 0

    return normalized_score


async def search_all_domains(
    query: str,
    domains: list[str] | None = None,
    max_results_per_domain: int = 3,
    min_relevance_score: int = 33,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Search across all NVIDIA domains.

    Args:
        query: Search query
        domains: List of domains to search (uses DEFAULT_DOMAINS if None)
        max_results_per_domain: Maximum results per domain
        min_relevance_score: Minimum relevance score threshold (0-100, default 50)

    Returns:
        Tuple of (results, errors, warnings, timing_info)
    """
    import time

    if domains is None:
        domains = DEFAULT_DOMAINS

    all_results = []
    errors = []
    warnings = []
    timing_info = {}

    start_time = time.time()

    async with httpx.AsyncClient() as client:
        # Search all domains concurrently with timing
        domain_start_times = {domain: time.time() for domain in domains}

        tasks = [
            search_nvidia_domain(client, domain, query, max_results_per_domain)
            for domain in domains
        ]

        domain_results = await asyncio.gather(*tasks, return_exceptions=True)

        for domain, results in zip(domains, domain_results):
            # Calculate timing for this domain
            domain_time_ms = int((time.time() - domain_start_times[domain]) * 1000)
            clean_domain = domain.replace('https://', '').replace('http://', '').rstrip('/')
            timing_info[clean_domain] = domain_time_ms

            if isinstance(results, Exception):
                error_msg = str(results)
                logger.error(f"Domain search failed for {domain}: {error_msg}")
                errors.append({
                    "domain": clean_domain,
                    "code": "SEARCH_FAILED",
                    "message": error_msg
                })
                warnings.append({
                    "code": "PARTIAL_FAILURE",
                    "message": f"Search failed for domain: {clean_domain}",
                    "affected_domains": [clean_domain]
                })
                continue

            all_results.extend(results)

    total_time_ms = int((time.time() - start_time) * 1000)

    # Calculate relevance scores for all results
    for result in all_results:
        result["relevance_score"] = calculate_search_relevance(result, query)

    # Filter by minimum relevance score
    filtered_results = [r for r in all_results if r.get("relevance_score", 0) >= min_relevance_score]

    # Sort by relevance score (highest first)
    filtered_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    # Build debug info if debug logging is enabled
    debug_info = None
    if logger.isEnabledFor(logging.DEBUG):
        search_strategies = [f"site:{domain.replace('https://', '').replace('http://', '').rstrip('/')} {query}" for domain in domains]
        debug_info = {
            "search_strategies": search_strategies,
            "timing_breakdown": timing_info
        }

    return filtered_results, errors, warnings, {"total_time_ms": total_time_ms, "debug_info": debug_info}


async def discover_content(
    content_type: str,
    topic: str,
    max_results: int = 5
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Discover specific types of NVIDIA content (videos, courses, tutorials, etc.).

    Args:
        content_type: Type of content to find (video, course, tutorial, webinar, blog)
        topic: Topic or keyword to search for
        max_results: Maximum number of results to return

    Returns:
        Tuple of (results, errors, warnings, timing_info)
    """
    # Map content types to search strategies
    content_strategies = {
        "video": {
            "query": f"{topic} video OR tutorial",
            "domains": ["https://developer.nvidia.com/", "https://blogs.nvidia.com/"],
            "keywords": ["youtube", "video", "watch", "tutorial"]
        },
        "course": {
            "query": f"{topic} course OR training OR certification",
            "domains": ["https://developer.nvidia.com/"],
            "keywords": ["course", "training", "dli", "certification", "learn"]
        },
        "tutorial": {
            "query": f"{topic} tutorial OR guide OR how-to",
            "domains": ["https://developer.nvidia.com/", "https://docs.nvidia.com/"],
            "keywords": ["tutorial", "guide", "how-to", "getting started"]
        },
        "webinar": {
            "query": f"{topic} webinar OR event OR session",
            "domains": ["https://developer.nvidia.com/", "https://blogs.nvidia.com/"],
            "keywords": ["webinar", "event", "session", "livestream"]
        },
        "blog": {
            "query": f"{topic}",
            "domains": ["https://blogs.nvidia.com/"],
            "keywords": ["blog", "article", "post"]
        }
    }

    strategy = content_strategies.get(content_type.lower(), {
        "query": f"{topic} {content_type}",
        "domains": DEFAULT_DOMAINS,
        "keywords": []
    })

    # Search using the strategy
    results, errors, warnings, timing_info = await search_all_domains(
        query=strategy["query"],
        domains=strategy.get("domains"),
        max_results_per_domain=max_results
    )
    
    # Filter and rank results based on content type keywords
    filtered_results = []
    content_keywords = strategy.get("keywords", [])

    # Calculate max possible score for normalization
    max_possible_score = len(content_keywords) * 6  # 3 + 2 + 1, consistent with search scoring

    for result in results:
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        url = result.get("url", "").lower()

        # Calculate raw relevance score based on content-type keyword matches
        raw_score = 0
        for keyword in content_keywords:
            keyword_score = 0

            # Title matches are most important (3 points)
            if keyword in title:
                keyword_score += 3

            # Snippet matches are moderately important (2 points)
            if keyword in snippet:
                keyword_score += 2

            # URL matches are least important (1 point)
            if keyword in url:
                keyword_score += 1

            raw_score += keyword_score

        # Normalize to 0-100 scale
        if max_possible_score > 0:
            normalized_score = int((raw_score / max_possible_score) * 100)
        else:
            # If no content keywords, use the original query-based relevance
            normalized_score = result.get("relevance_score", 0)

        result["relevance_score"] = normalized_score
        filtered_results.append(result)

    # Sort by relevance score (highest first) and limit results
    filtered_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    return filtered_results[:max_results], errors, warnings, timing_info


def format_content_results(results: list[dict[str, Any]], content_type: str, topic: str) -> str:
    """Format content discovery results."""
    if not results:
        return f"No {content_type} content found for topic: {topic}"
    
    output = [f"Recommended {content_type.upper()} content for: {topic}\n"]
    output.append("=" * 60)
    
    for i, result in enumerate(results, 1):
        score = result.get("relevance_score", 0)
        output.append(f"\n{i}. {result.get('title', 'Untitled')} (Score: {score}/100)")
        if url := result.get('url'):
            output.append(f"   URL: {url}")
        if snippet := result.get('snippet'):
            output.append(f"   {snippet}")
        if domain := result.get('domain'):
            output.append(f"   Source: {domain}")
    
    # Add citations
    output.append("\n" + "=" * 60)
    output.append("\nRESOURCE LINKS:")
    output.append("-" * 60)
    for i, result in enumerate(results, 1):
        if url := result.get('url'):
            title = result.get('title', 'Untitled')
            output.append(f"[{i}] {title}")
            output.append(f"    {url}")
    
    return "\n".join(output)


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="search_nvidia",
            description=(
                "Search across multiple NVIDIA domains including developer resources, documentation, "
                "blogs, news, forums, research papers, NGC catalog, Omniverse docs, GitHub Pages, and more. "
                "This tool helps find relevant information about NVIDIA technologies, products, "
                "and services. Results include citations with URLs for reference and are categorized "
                "by domain type (documentation, blog, news, developer, build, research, catalog, forum, downloads, resources)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find information across NVIDIA domains"
                    },
                    "domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of specific NVIDIA domains to search. "
                            "If not provided, searches all default domains."
                        )
                    },
                    "max_results_per_domain": {
                        "type": "integer",
                        "description": "Maximum number of results to return per domain (default: 3)",
                        "default": 3
                    },
                    "min_relevance_score": {
                        "type": "integer",
                        "description": "Minimum relevance score threshold (0-100) to filter results (default: 33)",
                        "default": 33,
                        "minimum": 0,
                        "maximum": 100
                    }
                },
                "required": ["query"]
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Whether the operation was successful"
                    },
                    "summary": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "total_results": {"type": "integer"},
                            "domains_searched": {"type": "integer"},
                            "domains_with_results": {"type": "integer"},
                            "search_time_ms": {"type": "integer"},
                            "debug_info": {
                                "type": "object",
                                "properties": {
                                    "search_strategies": {"type": "array", "items": {"type": "string"}},
                                    "timing_breakdown": {"type": "object"}
                                }
                            }
                        },
                        "required": ["query", "total_results", "domains_searched", "domains_with_results", "search_time_ms"]
                    },
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "snippet": {"type": "string", "description": "Snippet with **bold** highlighting"},
                                "snippet_plain": {"type": "string", "description": "Plain text snippet without formatting"},
                                "domain": {"type": "string"},
                                "domain_category": {"type": "string", "enum": ["documentation", "blog", "news", "developer", "build", "research", "catalog", "forum", "downloads", "resources", "other"]},
                                "relevance_score": {"type": "integer", "minimum": 0, "maximum": 100},
                                "matched_keywords": {"type": "array", "items": {"type": "string"}},
                                "metadata": {"type": "object"},
                                "formatted_text": {"type": "string", "description": "Markdown-formatted text for direct use"}
                            },
                            "required": ["id", "title", "url", "snippet", "snippet_plain", "domain", "domain_category", "relevance_score", "matched_keywords", "metadata", "formatted_text"]
                        }
                    },
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "number": {"type": "integer"},
                                "url": {"type": "string"},
                                "title": {"type": "string"},
                                "domain": {"type": "string"}
                            },
                            "required": ["number", "url", "title", "domain"]
                        }
                    },
                    "warnings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "message": {"type": "string"},
                                "affected_domains": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["code", "message"]
                        }
                    },
                    "errors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "domain": {"type": "string"},
                                "code": {"type": "string"},
                                "message": {"type": "string"}
                            },
                            "required": ["code", "message"]
                        }
                    }
                },
                "required": ["success", "summary", "results", "citations", "warnings", "errors"]
            }
        ),
        Tool(
            name="discover_nvidia_content",
            description=(
                "Discover specific types of NVIDIA content such as videos, courses, tutorials, "
                "webinars, or blog posts. This tool helps find educational and learning resources "
                "from NVIDIA's various platforms. Returns ranked results with relevance scores "
                "and direct links to the content."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content_type": {
                        "type": "string",
                        "enum": ["video", "course", "tutorial", "webinar", "blog"],
                        "description": (
                            "Type of content to discover: "
                            "'video' for video tutorials and demonstrations, "
                            "'course' for training courses and certifications (DLI), "
                            "'tutorial' for step-by-step guides, "
                            "'webinar' for webinars and live sessions, "
                            "'blog' for blog posts and articles"
                        )
                    },
                    "topic": {
                        "type": "string",
                        "description": "The topic or technology to find content about (e.g., 'CUDA', 'Omniverse', 'AI')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of content items to return (default: 5)",
                        "default": 5
                    }
                },
                "required": ["content_type", "topic"]
            },
            outputSchema={
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Whether the operation was successful"
                    },
                    "summary": {
                        "type": "object",
                        "properties": {
                            "content_type": {"type": "string"},
                            "topic": {"type": "string"},
                            "total_found": {"type": "integer"},
                            "search_time_ms": {"type": "integer"},
                            "debug_info": {
                                "type": "object",
                                "properties": {
                                    "search_strategies": {"type": "array", "items": {"type": "string"}},
                                    "timing_breakdown": {"type": "object"}
                                }
                            }
                        },
                        "required": ["content_type", "topic", "total_found", "search_time_ms"]
                    },
                    "content": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "content_type": {"type": "string"},
                                "snippet": {"type": "string", "description": "Snippet with **bold** highlighting"},
                                "snippet_plain": {"type": "string", "description": "Plain text snippet without formatting"},
                                "relevance_score": {"type": "integer", "minimum": 0, "maximum": 100},
                                "domain": {"type": "string"},
                                "domain_category": {"type": "string", "enum": ["documentation", "blog", "news", "developer", "build", "research", "catalog", "forum", "downloads", "resources", "other"]},
                                "matched_keywords": {"type": "array", "items": {"type": "string"}},
                                "metadata": {"type": "object"},
                                "formatted_text": {"type": "string", "description": "Markdown-formatted text for direct use"}
                            },
                            "required": ["id", "title", "url", "content_type", "snippet", "snippet_plain", "relevance_score", "domain", "domain_category", "matched_keywords", "metadata", "formatted_text"]
                        }
                    },
                    "resource_links": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "number": {"type": "integer"},
                                "url": {"type": "string"},
                                "title": {"type": "string"},
                                "type": {"type": "string"}
                            },
                            "required": ["number", "url", "title", "type"]
                        }
                    },
                    "warnings": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "message": {"type": "string"},
                                "affected_domains": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["code", "message"]
                        }
                    },
                    "errors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "domain": {"type": "string"},
                                "code": {"type": "string"},
                                "message": {"type": "string"}
                            },
                            "required": ["code", "message"]
                        }
                    }
                },
                "required": ["success", "summary", "content", "resource_links", "warnings", "errors"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> CallToolResult:
    """Handle tool calls."""
    try:
        if name == "search_nvidia":
            # SECURITY: Use semaphore to limit concurrent searches
            async with _search_semaphore:
                query = arguments.get("query")
                if not query:
                    error_response = build_error_response_json(
                        "MISSING_PARAMETER",
                        "Query parameter is required"
                    )
                    return build_tool_result(error_response)

                # SECURITY: Validate query length
                if len(query) > MAX_QUERY_LENGTH:
                    error_response = build_error_response_json(
                        "INVALID_PARAMETER",
                        f"Query too long. Maximum length: {MAX_QUERY_LENGTH} characters"
                    )
                    return build_tool_result(error_response)

                domains = arguments.get("domains")
                max_results_per_domain = arguments.get("max_results_per_domain", 3)

                # SECURITY: Limit max_results_per_domain to prevent resource exhaustion
                if max_results_per_domain > MAX_RESULTS_PER_DOMAIN:
                    logger.warning(f"max_results_per_domain limited from {max_results_per_domain} to {MAX_RESULTS_PER_DOMAIN}")
                    max_results_per_domain = MAX_RESULTS_PER_DOMAIN

                # Validate caller-supplied domains
                validated_domains = None
                if domains is not None:
                    if not isinstance(domains, list):
                        error_response = build_error_response_json(
                            "INVALID_PARAMETER",
                            "domains must be a list of strings"
                        )
                        return build_tool_result(error_response)

                    invalid_domains = []
                    validated_domains = []

                    for domain in domains:
                        if not isinstance(domain, str):
                            error_response = build_error_response_json(
                                "INVALID_PARAMETER",
                                f"Invalid domain type: {type(domain).__name__}. Expected string."
                            )
                            return build_tool_result(error_response)

                        if validate_nvidia_domain(domain):
                            validated_domains.append(domain)
                        else:
                            invalid_domains.append(domain)

                    # Reject request if any invalid domain is present
                    if invalid_domains:
                        error_msg = (
                            f"Invalid domains detected. Only nvidia.com domains and subdomains are allowed. "
                            f"Invalid domains: {', '.join(invalid_domains)}"
                        )
                        logger.warning(error_msg)
                        error_response = build_error_response_json(
                            "INVALID_DOMAIN",
                            error_msg,
                            {"invalid_domains": invalid_domains}
                        )
                        return build_tool_result(error_response)

                    if not validated_domains:
                        error_response = build_error_response_json(
                            "NO_VALID_DOMAINS",
                            "No valid NVIDIA domains provided"
                        )
                        return build_tool_result(error_response)

                    logger.info(f"Validated {len(validated_domains)} caller-supplied domains")

                min_relevance_score = arguments.get("min_relevance_score", 33)

                logger.info(f"Searching NVIDIA domains for: {query}")

                # Get results with error tracking
                results, errors, warnings, timing_info = await search_all_domains(
                    query=query,
                    domains=validated_domains,
                    max_results_per_domain=max_results_per_domain,
                    min_relevance_score=min_relevance_score
                )

                # Build JSON response
                response = build_search_response_json(
                    results=results,
                    query=query,
                    domains_searched=len(validated_domains) if validated_domains else len(DEFAULT_DOMAINS),
                    search_time_ms=timing_info["total_time_ms"],
                    errors=errors,
                    warnings=warnings,
                    debug_info=timing_info.get("debug_info", {})
                )

                return build_tool_result(response)

        elif name == "discover_nvidia_content":
            # SECURITY: Use semaphore to limit concurrent searches
            async with _search_semaphore:
                content_type = arguments.get("content_type")
                topic = arguments.get("topic")

                if not content_type or not topic:
                    error_response = build_error_response_json(
                        "MISSING_PARAMETER",
                        "Both content_type and topic parameters are required"
                    )
                    return build_tool_result(error_response)

                # SECURITY: Validate topic length (same as query validation)
                if len(topic) > MAX_QUERY_LENGTH:
                    error_response = build_error_response_json(
                        "INVALID_PARAMETER",
                        f"Topic too long. Maximum length: {MAX_QUERY_LENGTH} characters"
                    )
                    return build_tool_result(error_response)

                max_results = arguments.get("max_results", 5)

                # SECURITY: Limit max_results to prevent resource exhaustion
                if max_results > MAX_RESULTS_PER_DOMAIN:
                    logger.warning(f"max_results limited from {max_results} to {MAX_RESULTS_PER_DOMAIN}")
                    max_results = MAX_RESULTS_PER_DOMAIN

                logger.info(f"Discovering {content_type} content for topic: {topic}")

                # Get results with error tracking
                results, errors, warnings, timing_info = await discover_content(
                    content_type=content_type,
                    topic=topic,
                    max_results=max_results
                )

                # Build JSON response
                response = build_content_response_json(
                    results=results,
                    content_type=content_type,
                    topic=topic,
                    search_time_ms=timing_info["total_time_ms"],
                    errors=errors,
                    warnings=warnings,
                    debug_info=timing_info.get("debug_info", {})
                )

                return build_tool_result(response)

        else:
            error_response = build_error_response_json(
                "UNKNOWN_TOOL",
                f"Unknown tool: {name}"
            )
            return build_tool_result(error_response)

    except Exception as e:
        # SECURITY: Sanitize error messages to avoid exposing internal details
        logger.exception(f"Unexpected error in tool call: {e}")
        error_response = build_error_response_json(
            "INTERNAL_ERROR",
            "An unexpected error occurred while processing your request. Please try again or contact support if the issue persists."
        )
        return build_tool_result(error_response)


async def run():
    """Run the MCP server with graceful shutdown handling."""
    # Set up signal handlers for graceful shutdown
    shutdown_event = asyncio.Event()

    def signal_handler(signum, _frame):
        """Handle shutdown signals."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        shutdown_event.set()

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    try:
        logger.info("MCP server ready and waiting for connections")
        async with stdio_server() as (read_stream, write_stream):
            # Run server in a task so we can cancel it on shutdown
            server_task = asyncio.create_task(
                app.run(
                    read_stream,
                    write_stream,
                    app.create_initialization_options()
                )
            )

            # Wait for either server completion or shutdown signal
            shutdown_task = asyncio.create_task(shutdown_event.wait())
            _done, pending = await asyncio.wait(
                [server_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # If shutdown was triggered, cancel the server
            if shutdown_event.is_set():
                logger.info("Cancelling server task...")
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    logger.info("Server task cancelled successfully")

            # Cancel any remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        logger.exception(f"Unexpected error in server: {e}")
        raise
    finally:
        logger.info("MCP server shutdown complete")


def main():
    """Main entry point."""
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        # This shouldn't normally be reached due to signal handlers,
        # but handle it gracefully just in case
        logger.info("Received keyboard interrupt")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
