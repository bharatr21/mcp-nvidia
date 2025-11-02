"""MCP server for searching across NVIDIA domains."""

import asyncio
import json
import logging
import os
from typing import Any, Sequence
from urllib.parse import quote_plus, urljoin

import httpx
from bs4 import BeautifulSoup
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Configure logging
log_level = os.getenv("MCP_NVIDIA_LOG_LEVEL", "INFO")
logging.basicConfig(level=getattr(logging, log_level.upper()))
logger = logging.getLogger(__name__)

# Default NVIDIA domains to search
DEFAULT_DOMAINS = [
    "https://developer.nvidia.com/",
    "https://blogs.nvidia.com/",
    "https://nvidianews.nvidia.com/",
    "https://docs.nvidia.com/",
    "https://build.nvidia.com/",
]

# Allow override via environment variable (comma-separated list)
if custom_domains := os.getenv("MCP_NVIDIA_DOMAINS"):
    DEFAULT_DOMAINS = [d.strip() for d in custom_domains.split(",") if d.strip()]
    logger.info(f"Using custom domains from environment: {DEFAULT_DOMAINS}")

# Create server instance
app = Server("mcp-nvidia")


def format_search_results(results: list[dict[str, Any]], query: str) -> str:
    """Format search results into a readable string with citations."""
    if not results:
        return f"No results found for query: {query}"
    
    output = [f"Search results for: {query}\n"]
    output.append("=" * 60)
    
    # Format main results
    for i, result in enumerate(results, 1):
        output.append(f"\n{i}. {result.get('title', 'Untitled')}")
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


async def search_nvidia_domain(
    client: httpx.AsyncClient,
    domain: str,
    query: str,
    max_results: int = 5
) -> list[dict[str, Any]]:
    """
    Search a specific NVIDIA domain using DuckDuckGo HTML search.
    
    Args:
        client: HTTP client for making requests
        domain: Domain to search (e.g., "developer.nvidia.com")
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, url, and snippet
    """
    results = []
    
    try:
        # Clean domain for site: operator
        clean_domain = domain.replace('https://', '').replace('http://', '').rstrip('/')
        
        # Use DuckDuckGo HTML search with site: operator for domain-specific search
        search_query = f"site:{clean_domain} {query}"
        ddg_url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_query)}"
        
        # Add headers to mimic a browser request
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }
        
        response = await client.get(ddg_url, headers=headers, follow_redirects=True, timeout=30.0)
        
        if response.status_code == 200:
            # Parse the HTML response using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all result divs
            result_divs = soup.find_all('div', class_='result')
            
            for i, result_div in enumerate(result_divs[:max_results]):
                try:
                    # Extract title and URL
                    title_link = result_div.find('a', class_='result__a')
                    if not title_link:
                        continue
                    
                    title = title_link.get_text(strip=True)
                    url = title_link.get('href', '')
                    
                    # Extract snippet
                    snippet_div = result_div.find('a', class_='result__snippet')
                    snippet = snippet_div.get_text(strip=True) if snippet_div else ""
                    
                    if title and url:
                        results.append({
                            "title": title,
                            "url": url,
                            "snippet": snippet,
                            "domain": clean_domain
                        })
                        
                except Exception as e:
                    logger.debug(f"Error parsing result item: {str(e)}")
                    continue
            
            # If no results found from parsing, add a fallback message
            if not results:
                results.append({
                    "title": f"Search performed for '{query}' on {clean_domain}",
                    "url": ddg_url,
                    "snippet": f"No specific results found. Try searching directly on {domain}",
                    "domain": clean_domain
                })
        else:
            logger.warning(f"Search request to {domain} returned status {response.status_code}")
            
    except Exception as e:
        logger.error(f"Error searching {domain}: {str(e)}")
    
    return results


async def search_all_domains(
    query: str,
    domains: list[str] | None = None,
    max_results_per_domain: int = 3
) -> list[dict[str, Any]]:
    """
    Search across all NVIDIA domains.
    
    Args:
        query: Search query
        domains: List of domains to search (uses DEFAULT_DOMAINS if None)
        max_results_per_domain: Maximum results per domain
        
    Returns:
        Aggregated list of search results from all domains
    """
    if domains is None:
        domains = DEFAULT_DOMAINS
    
    all_results = []
    
    async with httpx.AsyncClient() as client:
        # Search all domains concurrently
        tasks = [
            search_nvidia_domain(client, domain, query, max_results_per_domain)
            for domain in domains
        ]
        
        domain_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for results in domain_results:
            if isinstance(results, Exception):
                logger.error(f"Domain search failed: {results}")
                continue
            all_results.extend(results)
    
    return all_results


async def discover_content(
    content_type: str,
    topic: str,
    max_results: int = 5
) -> list[dict[str, Any]]:
    """
    Discover specific types of NVIDIA content (videos, courses, tutorials, etc.).
    
    Args:
        content_type: Type of content to find (video, course, tutorial, webinar, blog)
        topic: Topic or keyword to search for
        max_results: Maximum number of results to return
        
    Returns:
        List of content items with metadata
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
    results = await search_all_domains(
        query=strategy["query"],
        domains=strategy.get("domains"),
        max_results_per_domain=max_results
    )
    
    # Filter and rank results based on content type keywords
    filtered_results = []
    keywords = strategy.get("keywords", [])
    
    for result in results:
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        url = result.get("url", "").lower()
        
        # Calculate relevance score based on keyword matches
        score = 0
        for keyword in keywords:
            if keyword in title:
                score += 3
            if keyword in snippet:
                score += 2
            if keyword in url:
                score += 1
        
        result["relevance_score"] = score
        filtered_results.append(result)
    
    # Sort by relevance score (highest first) and limit results
    filtered_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
    
    return filtered_results[:max_results]


def format_content_results(results: list[dict[str, Any]], content_type: str, topic: str) -> str:
    """Format content discovery results."""
    if not results:
        return f"No {content_type} content found for topic: {topic}"
    
    output = [f"Recommended {content_type.upper()} content for: {topic}\n"]
    output.append("=" * 60)
    
    for i, result in enumerate(results, 1):
        score = result.get("relevance_score", 0)
        relevance = "⭐" * min(5, max(1, score // 2))  # 1-5 stars
        
        output.append(f"\n{i}. {result.get('title', 'Untitled')} {relevance}")
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
                "Search across NVIDIA domains including developer.nvidia.com, "
                "blogs.nvidia.com, nvidianews.nvidia.com, docs.nvidia.com, and "
                "build.nvidia.com for NVIDIA-specific information. This tool helps "
                "find relevant documentation, blog posts, news, and developer resources "
                "related to NVIDIA technologies, products, and services. Results include "
                "citations with URLs for reference."
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
                    }
                },
                "required": ["query"]
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
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> Sequence[TextContent]:
    """Handle tool calls."""
    if name == "search_nvidia":
        query = arguments.get("query")
        if not query:
            raise ValueError("Query parameter is required")
        
        domains = arguments.get("domains")
        max_results_per_domain = arguments.get("max_results_per_domain", 3)
        
        logger.info(f"Searching NVIDIA domains for: {query}")
        
        results = await search_all_domains(
            query=query,
            domains=domains,
            max_results_per_domain=max_results_per_domain
        )
        
        formatted_results = format_search_results(results, query)
        
        return [
            TextContent(
                type="text",
                text=formatted_results
            )
        ]
    
    elif name == "discover_nvidia_content":
        content_type = arguments.get("content_type")
        topic = arguments.get("topic")
        
        if not content_type or not topic:
            raise ValueError("Both content_type and topic parameters are required")
        
        max_results = arguments.get("max_results", 5)
        
        logger.info(f"Discovering {content_type} content for topic: {topic}")
        
        results = await discover_content(
            content_type=content_type,
            topic=topic,
            max_results=max_results
        )
        
        formatted_results = format_content_results(results, content_type, topic)
        
        return [
            TextContent(
                type="text",
                text=formatted_results
            )
        ]
    
    else:
        raise ValueError(f"Unknown tool: {name}")


async def run():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


def main():
    """Main entry point."""
    asyncio.run(run())


if __name__ == "__main__":
    main()
