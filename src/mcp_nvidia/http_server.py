"""HTTP/SSE transport server for MCP NVIDIA.

This module provides an HTTP/SSE server for the MCP NVIDIA server,
making it accessible over HTTP instead of stdio.
"""

import logging
from pathlib import Path

from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route

from mcp_nvidia import __version__
from mcp_nvidia.server import app as mcp_app

logger = logging.getLogger(__name__)


# Create SSE transport
sse = SseServerTransport("/messages/")


async def handle_sse(request: Request) -> Response:
    """Handle SSE connections for MCP."""
    logger.info("New SSE connection established")

    async with sse.connect_sse(request.scope, request.receive, request.scope["send"]) as (read_stream, write_stream):
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


# Create Starlette app
http_app = Starlette(
    debug=False,
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
        Route("/health", endpoint=health_check),
        Route("/", endpoint=health_check),
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
