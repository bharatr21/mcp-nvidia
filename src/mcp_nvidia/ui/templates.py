"""HTML templates for MCP-UI components."""

import html
from typing import Any

from mcp_nvidia.ui.components import (
    render_citations,
    render_content_container,
    render_content_type_tabs,
    render_filter_panel,
    render_results_container,
    render_search_header,
    render_warnings,
)
from mcp_nvidia.ui.styles import STYLES


def render_search_ui(response: dict[str, Any]) -> str:
    """Render complete search results UI."""
    summary = response.get("summary", {})
    query = summary.get("query", "")
    total_results = summary.get("total_results", 0)
    search_time_ms = summary.get("search_time_ms", 0)
    results = response.get("results", [])
    citations = response.get("citations", [])
    warnings_list = response.get("warnings", [])

    header = render_search_header(query)
    filter_panel = render_filter_panel(
        query=query,
        sort_by="relevance",
        min_relevance_score=17,
        total_results=total_results,
        search_time_ms=search_time_ms,
    )
    results_container = render_results_container(results)
    citations_section = render_citations(citations)
    warnings_section = render_warnings(warnings_list)

    escaped_query = html.escape(query, quote=True)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NVIDIA MCP Search - {escaped_query}</title>
  <script src="https://unpkg.com/htmx.org@1.9.10"></script>
  {STYLES}
</head>
<body>
  <div class="mcp-nvidia-ui">
    {header}
    {warnings_section}
    {filter_panel}
    {results_container}
    {citations_section}
  </div>
</body>
</html>
"""


def render_content_ui(response: dict[str, Any]) -> str:
    """Render complete content discovery UI."""
    summary = response.get("summary", {})
    content_type = summary.get("content_type", "video")
    topic = summary.get("topic", "")
    total_found = summary.get("total_found", 0)
    search_time_ms = summary.get("search_time_ms", 0)
    content = response.get("content", [])
    warnings_list = response.get("warnings", [])

    # Escape user-provided values
    escaped_topic = html.escape(topic, quote=True)
    escaped_total = html.escape(str(total_found), quote=True)
    escaped_time = html.escape(str(search_time_ms), quote=True)

    header = f"""
    <div class="mcp-nvidia-header">
      <span class="mcp-nvidia-logo">NVIDIA</span>
      <span class="mcp-nvidia-title">Content Discovery</span>
    </div>
    <div class="mcp-nvidia-search-bar">
      <input type="text" class="mcp-nvidia-search-input" value="{escaped_topic}" placeholder="Discover content..." readonly>
    </div>
    <div class="mcp-nvidia-results-count" style="margin-bottom: 12px;">Found {escaped_total} results in {escaped_time}ms</div>
    """

    tabs = render_content_type_tabs(content_type, topic)
    content_container = render_content_container(content)
    warnings_section = render_warnings(warnings_list)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NVIDIA Content - {escaped_topic}</title>
  <script src="https://unpkg.com/htmx.org@1.9.10"></script>
  {STYLES}
</head>
<body>
  <div class="mcp-nvidia-ui">
    {header}
    {warnings_section}
    {tabs}
    {content_container}
  </div>
</body>
</html>
"""


def render_error_ui(error: dict[str, Any]) -> str:
    """Render error state UI."""
    code = error.get("code", "UNKNOWN")
    message = error.get("message", "An unknown error occurred")

    # Escape error details
    escaped_code = html.escape(code, quote=True)
    escaped_message = html.escape(message, quote=True)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Error - NVIDIA MCP</title>
  {STYLES}
</head>
<body>
  <div class="mcp-nvidia-ui">
    <div class="mcp-nvidia-error">
      <strong>Error [{escaped_code}]</strong>
      <p>{escaped_message}</p>
    </div>
  </div>
</body>
</html>
"""


def render_filter_fragment(
    results: list[dict[str, Any]],
    query: str,
    sort_by: str = "relevance",
    min_relevance_score: int = 17,
    total_results: int = 0,
    search_time_ms: int = 0,
) -> str:
    """Render filter results fragment for HTMX updates."""
    filter_panel = render_filter_panel(
        query=query,
        sort_by=sort_by,
        min_relevance_score=min_relevance_score,
        total_results=total_results,
        search_time_ms=search_time_ms,
    )
    results_container = render_results_container(results)

    return f"{filter_panel}{results_container}"


def render_content_fragment(content: list[dict[str, Any]]) -> str:
    """Render content discovery fragment for HTMX tab updates."""
    return render_content_container(content)
