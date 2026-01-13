"""HTTP/SSE transport server for MCP NVIDIA.

This module provides an HTTP/SSE server for the MCP NVIDIA server,
making it accessible over HTTP instead of stdio.
"""

import logging
from pathlib import Path

from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Mount, Route

from mcp_nvidia import __version__
from mcp_nvidia.server import app as mcp_app

logger = logging.getLogger(__name__)

# Maximum citation index to prevent out-of-range requests
MAX_CITATION = 1000

# Create SSE transport
sse = SseServerTransport("/messages/")


async def handle_sse(scope, receive, send):
    """Handle SSE connections for MCP (raw ASGI callable)."""
    logger.info("New SSE connection established")

    async with sse.connect_sse(scope, receive, send) as (read_stream, write_stream):
        await mcp_app.run(read_stream, write_stream, mcp_app.create_initialization_options())


async def health_check(request: Request) -> Response:
    """Health check endpoint."""
    return JSONResponse(
        {
            "status": "healthy",
            "service": "mcp-nvidia",
            "transport": "http-sse",
            "version": __version__,
            "endpoints": {"sse": "/sse", "messages": "/messages/", "health": "/health"},
        }
    )


async def handle_ui_filter(request: Request) -> HTMLResponse:
    """Handle HTMX filter requests for search results."""
    query = request.query_params.get("query", "")
    sort_by = request.query_params.get("sort_by", "relevance")

    # Validate sort_by against allowlist
    valid_sort_options = {"relevance", "date", "domain"}
    if sort_by not in valid_sort_options:
        sort_by = "relevance"

    # Validate and clamp min_relevance_score to 0-100 range
    try:
        min_relevance_score = int(request.query_params.get("min_relevance_score", "17"))
        min_relevance_score = max(0, min(min_relevance_score, 100))
    except (ValueError, TypeError):
        min_relevance_score = 17

    # Handle missing UI dependencies gracefully
    try:
        from mcp_nvidia.lib import DEFAULT_DOMAINS, build_search_response_json, search_all_domains
        from mcp_nvidia.ui.renderer import render_filter_ui
    except (ImportError, ModuleNotFoundError):
        logger.warning("UI dependencies not available")
        return HTMLResponse(
            "<div class='mcp-nvidia-error'>UI features not available. Install with: pip install mcp-nvidia[ui]</div>",
            status_code=501,
        )

    try:
        results, errors, warnings, timing_info = await search_all_domains(
            query=query,
            domains=None,
            max_results_per_domain=10,
            min_relevance_score=min_relevance_score,
            sort_by=sort_by,
            date_from=None,
            date_to=None,
            max_total_results=None,
            allowed_domains=None,
            blocked_domains=None,
        )

        response = build_search_response_json(
            results=results,
            query=query,
            domains_searched=len(DEFAULT_DOMAINS),
            search_time_ms=timing_info["total_time_ms"],
            errors=errors,
            warnings=warnings,
            debug_info=timing_info.get("debug_info", {}),
        )

        html = render_filter_ui(
            response=response,
            sort_by=sort_by,
            min_relevance_score=min_relevance_score,
        )

        return HTMLResponse(html)
    except Exception as e:
        logger.exception(f"Error in filter endpoint: {e}")
        return HTMLResponse("<div class='mcp-nvidia-error'>An internal error occurred</div>", status_code=500)


async def handle_ui_content(request: Request) -> HTMLResponse:
    """Handle HTMX content type filter requests."""
    content_type = request.query_params.get("content_type", "video")
    topic = request.query_params.get("topic", "")

    # Validate content_type against allowlist
    valid_content_types = {"video", "course", "tutorial", "webinar", "blog"}
    if content_type not in valid_content_types:
        content_type = "video"

    try:
        from mcp_nvidia.lib import build_content_response_json, discover_content
        from mcp_nvidia.ui.renderer import render_content_ui_fragment
    except (ImportError, ModuleNotFoundError):
        logger.warning("UI dependencies not available")
        return HTMLResponse(
            "<div class='mcp-nvidia-error'>UI features not available. Install with: pip install mcp-nvidia[ui]</div>",
            status_code=501,
        )

    try:
        results, errors, warnings, timing_info = await discover_content(
            content_type=content_type,
            topic=topic,
            max_results=10,
            date_from=None,
        )

        response = build_content_response_json(
            results=results,
            content_type=content_type,
            topic=topic,
            search_time_ms=timing_info["total_time_ms"],
            errors=errors,
            warnings=warnings,
            debug_info=timing_info.get("debug_info", {}),
        )

        html = render_content_ui_fragment(response)

        return HTMLResponse(html)
    except Exception as e:
        logger.exception(f"Error in content endpoint: {e}")
        return HTMLResponse("<div class='mcp-nvidia-error'>An internal error occurred</div>", status_code=500)


async def handle_citation(request: Request) -> HTMLResponse:
    """Handle citation copy requests."""
    # Validate citation index
    index_str = request.path_params.get("index")

    if not index_str:
        return HTMLResponse(
            "<div class='mcp-nvidia-error'>Citation index is required</div>",
            status_code=400,
        )

    try:
        citation_num = int(index_str)
    except (ValueError, TypeError):
        return HTMLResponse(
            "<div class='mcp-nvidia-error'>Invalid citation index</div>",
            status_code=400,
        )

    # Clamp citation_num to valid range
    citation_num = max(1, min(citation_num, MAX_CITATION))

    citation_html = f"""
    <div class="mcp-nvidia-citation" hx-swap-oob="true" id="citation-{citation_num}">
      <span class="mcp-nvidia-citation-number">[{citation_num}]</span>
      <span>Copied to clipboard!</span>
    </div>
    """

    return HTMLResponse(citation_html)


# Create Starlette app
http_app = Starlette(
    debug=False,
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
        Route("/health", endpoint=health_check),
        Route("/", endpoint=health_check),
        Route("/ui/filter", endpoint=handle_ui_filter),
        Route("/ui/content", endpoint=handle_ui_content),
        Route("/ui/citation/{index}", endpoint=handle_citation),
    ],
)


def run_http_server(host: str = "0.0.0.0", port: int = 8000):  # nosec B104
    """
    Run the HTTP/SSE server.

    Args:
        host: Host to bind to (default: 0.0.0.0 for all interfaces)
        port: Port to listen on (default: 8000)
    """
    import uvicorn

    logger.info("Starting MCP NVIDIA HTTP/SSE server")
    logger.info(f"Log file: {Path.home() / '.mcp-nvidia' / 'server.log'}")

    logger.info(f"Starting HTTP/SSE server on {host}:{port}")

    uvicorn.run(http_app, host=host, port=port, log_level="info", access_log=True)
