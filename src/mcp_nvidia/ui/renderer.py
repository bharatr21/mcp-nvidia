"""Main renderer for MCP-UI components."""

from typing import Any

from mcp_nvidia.ui.templates import (
    render_content_fragment,
    render_error_ui,
    render_filter_fragment,
)
from mcp_nvidia.ui.templates import (
    render_content_ui as render_content_ui_template,
)
from mcp_nvidia.ui.templates import (
    render_search_ui as render_search_ui_template,
)


def render_search_ui(response: dict[str, Any]) -> str:
    """
    Render search results as interactive HTML UI.

    Args:
        response: The JSON response from build_search_response_json

    Returns:
        HTML string for MCP-UI rendering
    """
    if not response.get("success", False) and response.get("error"):
        return render_error_ui(response.get("error", {}))

    return render_search_ui_template(response)


def render_content_ui(response: dict[str, Any]) -> str:
    """
    Render content discovery results as interactive HTML UI.

    Args:
        response: The JSON response from build_content_response_json

    Returns:
        HTML string for MCP-UI rendering
    """
    if not response.get("success", False) and response.get("error"):
        return render_error_ui(response.get("error", {}))

    return render_content_ui_template(response)


def render_filter_ui(
    response: dict[str, Any],
    sort_by: str = "relevance",
    min_relevance_score: int = 17,
) -> str:
    """
    Render filter results as HTML fragment for HTMX updates.

    Args:
        response: The JSON response from search
        sort_by: Current sort option
        min_relevance_score: Current minimum relevance threshold

    Returns:
        HTML fragment for HTMX target
    """
    results = response.get("results", [])
    summary = response.get("summary", {})
    query = summary.get("query", "")
    total_results = summary.get("total_results", 0)
    search_time_ms = summary.get("search_time_ms", 0)

    return render_filter_fragment(
        results=results,
        query=query,
        sort_by=sort_by,
        min_relevance_score=min_relevance_score,
        total_results=total_results,
        search_time_ms=search_time_ms,
    )


def render_content_ui_fragment(response: dict[str, Any]) -> str:
    """
    Render content discovery results as HTML fragment for HTMX tab updates.

    Args:
        response: The JSON response from discover_content

    Returns:
        HTML fragment for HTMX target
    """
    content = response.get("content", [])
    return render_content_fragment(content)
