"""UI component builders for MCP-UI."""

import html
import json
from typing import Any

CONTENT_TYPE_ICONS = {
    "tutorial": "📖",
    "video": "🎬",
    "course": "📚",
    "documentation": "📄",
    "blog_post": "📝",
    "blog": "📝",  # Normalize blog content type
    "forum_discussion": "💬",
    "announcement": "📢",
    "research_paper": "🔬",
    "news": "📰",
    "article": "📄",
    "guide": "📋",
    "webinar": "🎥",
}


def render_search_header(query: str) -> str:
    """Render the search header with query and meta info."""
    escaped_query = html.escape(query, quote=True)
    return f"""
    <div class="mcp-nvidia-header">
      <span class="mcp-nvidia-logo">NVIDIA</span>
      <span class="mcp-nvidia-title">MCP Search</span>
    </div>
    <div class="mcp-nvidia-search-bar">
      <input type="text" class="mcp-nvidia-search-input" value="{escaped_query}" placeholder="Search NVIDIA domains..." readonly>
    </div>
    """


def render_filter_panel(
    query: str,
    sort_by: str = "relevance",
    min_relevance_score: int = 17,
    total_results: int = 0,
    search_time_ms: int = 0,
) -> str:
    """Render the filter panel with HTMX-powered controls."""
    sort_options = ["relevance", "date", "domain"]
    sort_labels = {"relevance": "Relevance", "date": "Date", "domain": "Domain"}

    sort_html = "".join(
        f'<option value="{opt}" {"selected" if sort_by == opt else ""}>{sort_labels[opt]}</option>'
        for opt in sort_options
    )

    escaped_query = html.escape(query, quote=True)

    return f"""
    <div class="mcp-nvidia-filter-panel" hx-boost="true">
      <div class="mcp-nvidia-filter-group">
        <label class="mcp-nvidia-filter-label">Sort by:</label>
        <select class="mcp-nvidia-select" name="sort_by"
                hx-get="/ui/filter" hx-target="#mcp-nvidia-results"
                hx-swap="outerHTML"
                hx-trigger="change" hx-include="[name='min_relevance_score'], [name='query']">
          {sort_html}
        </select>
      </div>

      <div class="mcp-nvidia-filter-group">
        <label class="mcp-nvidia-filter-label">Min Relevance:</label>
        <div class="mcp-nvidia-range-container">
          <input type="range" class="mcp-nvidia-range" name="min_relevance_score"
                 min="0" max="100" value="{min_relevance_score}"
                 hx-get="/ui/filter" hx-target="#mcp-nvidia-results"
                 hx-swap="outerHTML"
                 hx-trigger="change" hx-include="[name='sort_by'], [name='query']">
          <span class="mcp-nvidia-range-value">{min_relevance_score}</span>
        </div>
      </div>

      <input type="hidden" name="query" value="{escaped_query}">

      <span class="mcp-nvidia-results-count">{total_results} results</span>
      <span class="mcp-nvidia-results-time">⏱️ {search_time_ms}ms</span>
    </div>
    """


def render_result_card(result: dict[str, Any], index: int) -> str:
    """Render a single search result card."""
    score = result.get("relevance_score", 0)
    title = result.get("title", "Untitled")
    url = result.get("url", "")
    snippet = result.get("snippet", "")
    domain = result.get("domain", "")
    content_type = result.get("content_type", "article")
    published_date = result.get("published_date")
    matched_keywords = result.get("matched_keywords", [])

    # Escape all user-provided fields
    escaped_title = html.escape(title, quote=True)
    escaped_snippet = html.escape(snippet, quote=True)
    escaped_domain = html.escape(domain, quote=True)
    escaped_content_type = html.escape(content_type.replace("_", " ").title(), quote=True)

    # Validate URL - only allow safe schemes
    safe_url = (
        html.escape(url, quote=True) if url and url.lower().startswith(("http://", "https://", "mailto:")) else ""
    )

    icon = CONTENT_TYPE_ICONS.get(content_type, "📄")

    # Escape date if present
    date_html = ""
    if published_date:
        escaped_date = html.escape(str(published_date), quote=True)
        date_html = f'<span class="mcp-nvidia-date">📅 {escaped_date}</span>'

    # Escape keywords
    escaped_keywords = [html.escape(kw, quote=True) for kw in matched_keywords[:5]]
    keywords_html = "".join(f'<span class="mcp-nvidia-keyword">{kw}</span>' for kw in escaped_keywords)

    keywords_section = f'<div class="mcp-nvidia-keywords">{keywords_html}</div>' if escaped_keywords else ""

    return f"""
    <div class="mcp-nvidia-result-card">
      <div class="mcp-nvidia-result-header">
        <span class="mcp-nvidia-relevance-badge" style="--score: {score}%">{score}</span>
        <a href="{safe_url}" target="_blank" rel="noopener noreferrer" class="mcp-nvidia-result-title">{escaped_title}</a>
        <span class="mcp-nvidia-content-type">{icon} {escaped_content_type}</span>
      </div>
      <div class="mcp-nvidia-result-meta">
        <span class="mcp-nvidia-domain-tag">{escaped_domain}</span>
        {date_html}
      </div>
      <p class="mcp-nvidia-result-snippet">{escaped_snippet}</p>
      {keywords_section}
      <div class="mcp-nvidia-result-actions">
        <a href="{safe_url}" target="_blank" rel="noopener noreferrer" class="mcp-nvidia-btn mcp-nvidia-btn-primary">→ Open</a>
        <button class="mcp-nvidia-btn mcp-nvidia-btn-secondary"
                hx-get="/ui/citation/{index}"
                hx-target="#mcp-nvidia-citations"
                hx-swap="innerHTML">📋 Copy Citation</button>
      </div>
    </div>
    """


def render_results_container(results: list[dict[str, Any]]) -> str:
    """Render all search results."""
    if not results:
        return """
        <div class="mcp-nvidia-results-container" id="mcp-nvidia-results">
          <div class="mcp-nvidia-empty-state">
            <div class="mcp-nvidia-empty-icon">🔍</div>
            <div class="mcp-nvidia-empty-title">No results found</div>
            <div class="mcp-nvidia-empty-message">Try adjusting your search query or filters</div>
          </div>
        </div>
        """

    cards = "".join(render_result_card(result, i + 1) for i, result in enumerate(results))
    return f'<div class="mcp-nvidia-results-container" id="mcp-nvidia-results">{cards}</div>'


def render_citations(citations: list[dict[str, Any]]) -> str:
    """Render the citations section."""
    if not citations:
        return ""

    citation_items = []
    for c in citations:
        # Escape all fields including number
        num = html.escape(str(c.get("number", "")), quote=True)
        escaped_title = html.escape(c.get("title", ""), quote=True)
        escaped_domain = html.escape(c.get("domain", ""), quote=True)

        # Validate URL
        url = c.get("url", "")
        safe_url = (
            html.escape(url, quote=True) if url and url.lower().startswith(("http://", "https://", "mailto:")) else ""
        )

        # Build link or span based on URL presence
        if safe_url:
            link_html = f'<a href="{safe_url}" target="_blank" rel="noopener noreferrer" class="mcp-nvidia-citation-link">{escaped_domain}</a>'
        else:
            link_html = f'<span class="mcp-nvidia-citation-link">{escaped_domain}</span>'

        citation_items.append(f"""
        <div class="mcp-nvidia-citation">
          <span class="mcp-nvidia-citation-number">[{num}]</span>
          <span>{escaped_title}</span>
          {link_html}
        </div>
        """)

    return f"""
    <div class="mcp-nvidia-citations">
      <div class="mcp-nvidia-citations-title">Citations</div>
      <div class="mcp-nvidia-citation-list">
        {"".join(citation_items)}
      </div>
    </div>
    """


def render_content_type_tabs(content_type: str, topic: str) -> str:
    """Render content type filter tabs."""
    content_types = ["video", "course", "tutorial", "webinar", "blog"]
    content_type_labels = {
        "video": "🎬 Videos",
        "course": "📚 Courses",
        "tutorial": "📖 Tutorials",
        "webinar": "🎥 Webinars",
        "blog": "📝 Blog Posts",
    }

    tabs_html = []
    for ct in content_types:
        # Build hx-vals JSON properly and escape it
        hx_vals_dict = {"content_type": ct, "topic": topic}
        hx_vals_json = json.dumps(hx_vals_dict)
        hx_vals_escaped = html.escape(hx_vals_json, quote=True)

        tab_html = (
            f'<button class="mcp-nvidia-tab {"active" if ct == content_type else ""}" '
            f'data-type="{ct}" '
            f'hx-get="/ui/content" hx-target="#mcp-nvidia-content-results" '
            f"hx-vals='{hx_vals_escaped}' "
            f'hx-trigger="click">{content_type_labels.get(ct, ct)}</button>'
        )
        tabs_html.append(tab_html)

    return f'<div class="mcp-nvidia-content-type-tabs">{"".join(tabs_html)}</div>'


def render_content_card(content: dict[str, Any]) -> str:
    """Render a content discovery card."""
    title = content.get("title", "Untitled")
    url = content.get("url", "")
    content_type = content.get("content_type", "article")
    snippet = content.get("snippet", "")
    domain = content.get("domain", "")
    relevance_score = content.get("relevance_score", 0)

    # Escape all user-provided fields
    escaped_title = html.escape(title, quote=True)
    escaped_snippet = html.escape(snippet, quote=True)
    escaped_domain = html.escape(domain, quote=True)

    # Validate URL
    safe_url = (
        html.escape(url, quote=True) if url and url.lower().startswith(("http://", "https://", "mailto:")) else ""
    )

    icon = CONTENT_TYPE_ICONS.get(content_type, "📄")

    return f"""
    <div class="mcp-nvidia-content-card">
      <div class="mcp-nvidia-content-thumbnail">{icon}</div>
      <div class="mcp-nvidia-content-info">
        <a href="{safe_url}" target="_blank" rel="noopener noreferrer" class="mcp-nvidia-content-title">{escaped_title}</a>
        <div class="mcp-nvidia-content-domain">{escaped_domain} · <span class="mcp-nvidia-content-score">Score: {relevance_score}</span></div>
        <p class="mcp-nvidia-content-snippet">{escaped_snippet}</p>
      </div>
    </div>
    """


def render_warnings(warnings: list[dict[str, Any]]) -> str:
    """Render warning messages."""
    if not warnings:
        return ""

    escaped_warnings = [html.escape(w.get("message", ""), quote=True) for w in warnings]
    return "".join(f'<div class="mcp-nvidia-warning">⚠️ {msg}</div>' for msg in escaped_warnings)


def render_content_container(content: list[dict[str, Any]]) -> str:
    """Render all content discovery results."""
    if not content:
        return """
        <div class="mcp-nvidia-results-container" id="mcp-nvidia-content-results">
          <div class="mcp-nvidia-empty-state">
            <div class="mcp-nvidia-empty-icon">📭</div>
            <div class="mcp-nvidia-empty-title">No content found</div>
            <div class="mcp-nvidia-empty-message">Try a different content type or topic</div>
          </div>
        </div>
        """

    cards = "".join(render_content_card(c) for c in content)
    return f'<div class="mcp-nvidia-results-container" id="mcp-nvidia-content-results">{cards}</div>'
