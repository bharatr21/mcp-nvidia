"""Response building functions for MCP tool results."""

import json
from typing import Any

from mcp.types import CallToolResult, TextContent

from mcp_nvidia.lib.relevance import extract_matched_keywords
from mcp_nvidia.lib.utils import get_domain_category


def build_search_response_json(
    results: list[dict[str, Any]],
    query: str,
    domains_searched: int,
    search_time_ms: int,
    errors: list[dict[str, Any]] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    debug_info: dict[str, Any] | None = None,
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
        relevance_score = result.get("relevance_score", 0)
        published_date = result.get("published_date")
        content_type = result.get("content_type", "article")
        metadata = result.get("metadata", {})

        result_dict = {
            "id": i,
            "title": title,
            "url": url,
            "snippet": snippet,
            "domain": domain,
            "domain_category": get_domain_category(domain),
            "content_type": content_type,
            "relevance_score": relevance_score,
            "matched_keywords": extract_matched_keywords(query, result),
            "metadata": metadata,
        }

        # Add published_date only if it exists
        if published_date:
            result_dict["published_date"] = published_date

        structured_results.append(result_dict)

    # Build citations
    citations = []
    for i, result in enumerate(results, 1):
        citations.append(
            {
                "number": i,
                "url": result.get("url", ""),
                "title": result.get("title", "Untitled"),
                "domain": result.get("domain", ""),
            }
        )

    # Count domains with results
    domains_with_results = len({r.get("domain", "") for r in results if r.get("domain")})

    # Build summary
    summary = {
        "query": query,
        "total_results": len(results),
        "domains_searched": domains_searched,
        "domains_with_results": domains_with_results,
        "search_time_ms": search_time_ms,
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
        "errors": errors,
    }


def build_content_response_json(
    results: list[dict[str, Any]],
    content_type: str,
    topic: str,
    search_time_ms: int,
    errors: list[dict[str, Any]] | None = None,
    warnings: list[dict[str, Any]] | None = None,
    debug_info: dict[str, Any] | None = None,
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
        relevance_score = result.get("relevance_score", 0)
        published_date = result.get("published_date")
        detected_content_type = result.get("content_type", content_type.lower())
        metadata = result.get("metadata", {})

        content_dict = {
            "id": i,
            "title": title,
            "url": url,
            "content_type": detected_content_type,
            "snippet": snippet,
            "relevance_score": relevance_score,
            "domain": domain,
            "domain_category": get_domain_category(domain),
            "matched_keywords": extract_matched_keywords(topic, result),
            "metadata": metadata,
        }

        # Add published_date only if it exists
        if published_date:
            content_dict["published_date"] = published_date

        structured_content.append(content_dict)

    # Build resource links
    resource_links = []
    for i, result in enumerate(results, 1):
        resource_links.append(
            {
                "number": i,
                "url": result.get("url", ""),
                "title": result.get("title", "Untitled"),
                "type": content_type.lower(),
            }
        )

    # Build summary
    summary = {
        "content_type": content_type.lower(),
        "topic": topic,
        "total_found": len(results),
        "search_time_ms": search_time_ms,
    }

    # Add debug info if provided
    if debug_info is not None:
        summary["debug_info"] = debug_info
        # Add suggestions if present
        if debug_info.get("suggestions"):
            summary["suggestions"] = debug_info["suggestions"]
        # Add expanded topics if present
        if "expanded_topics" in debug_info:
            summary["expanded_topics"] = debug_info["expanded_topics"]

    return {
        "success": len(errors) == 0 or len(results) > 0,
        "summary": summary,
        "content": structured_content,
        "resource_links": resource_links,
        "warnings": warnings,
        "errors": errors,
    }


def build_error_response_json(
    error_code: str, error_message: str, details: dict[str, Any] | None = None
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
    error_response = {"success": False, "error": {"code": error_code, "message": error_message}}

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
        isError=not response.get("success", False),
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
        if url := result.get("url"):
            output.append(f"   URL: {url}")
        if snippet := result.get("snippet"):
            output.append(f"   {snippet}")
        if domain := result.get("domain"):
            output.append(f"   Source: {domain}")

    # Add citations section for easy reference
    output.append("\n" + "=" * 60)
    output.append("\nCITATIONS:")
    output.append("-" * 60)
    for i, result in enumerate(results, 1):
        if url := result.get("url"):
            title = result.get("title", "Untitled")
            output.append(f"[{i}] {title}")
            output.append(f"    {url}")

    return "\n".join(output)


def build_tool_result_with_ui(
    response: dict[str, Any],
    tool_name: str,
) -> CallToolResult:
    """
    Build CallToolResult with both text content and UI resource.

    Args:
        response: The JSON response dictionary
        tool_name: Name of the tool that generated this response

    Returns:
        CallToolResult with JSON content and optional UI resource
    """
    try:
        from mcp_ui_server import create_ui_resource

        from mcp_nvidia.ui import render_content_ui, render_search_ui

        if tool_name == "search_nvidia":
            html = render_search_ui(response)
        elif tool_name == "discover_nvidia_content":
            html = render_content_ui(response)
        else:
            html = None

        ui_content = []
        if html:
            ui_resource = create_ui_resource(
                {
                    "uri": f"ui://{tool_name}/results",
                    "content": {"type": "rawHtml", "htmlString": html},
                    "encoding": "text",
                }
            )
            ui_content.append(ui_resource)

        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(response, indent=2)), *ui_content],
            structuredContent=response,
            isError=not response.get("success", False),
        )
    except ImportError:
        return build_tool_result(response)
