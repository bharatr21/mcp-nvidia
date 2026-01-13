"""MCP server for searching across NVIDIA domains."""

import asyncio
import contextlib
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, Resource, Tool

# Import all necessary functions from lib modules
from mcp_nvidia.lib import (
    DEFAULT_DOMAINS,
    MAX_CONCURRENT_SEARCHES,
    MAX_QUERY_LENGTH,
    MAX_RESULTS_PER_DOMAIN,
    build_content_response_json,
    build_error_response_json,
    build_search_response_json,
    build_tool_result,
    build_tool_result_with_ui,
    discover_content,
    search_all_domains,
    validate_nvidia_domain,
)

# Import SDK generators
from mcp_nvidia.sdk_generator import generate_python_sdk, generate_typescript_sdk

# Configure logging
log_level = os.getenv("MCP_NVIDIA_LOG_LEVEL", "INFO")

# Set up logging to file (doesn't interfere with stdio)
log_dir = Path.home() / ".mcp-nvidia"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "server.log"

# Configure logging to file
logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),  # Also log to stderr for client capture
    ],
)
logger = logging.getLogger(__name__)

# Log startup
logger.info(f"MCP NVIDIA server starting (log level: {log_level})")
logger.info(f"Logs written to: {log_file}")

# Create server instance
app = Server("mcp-nvidia")

# SECURITY: Concurrency limits
_search_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SEARCHES)


def _get_tool_schemas() -> list[dict[str, Any]]:
    """Get shared tool schemas used for both runtime tools and SDK generation."""
    return [
        {
            "name": "search_nvidia",
            "description": (
                "Search across multiple NVIDIA domains including developer resources, documentation, "
                "blogs, news, forums, research papers, NGC catalog, Omniverse docs, GitHub Pages, and more. "
                "This tool helps find relevant information about NVIDIA technologies, products, "
                "and services. Results include citations with URLs for reference and are categorized "
                "by domain type (documentation, blog, news, developer, build, research, catalog, forum, downloads, resources)."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find information across NVIDIA domains",
                    },
                    "domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of specific NVIDIA domains to search. "
                            "If not provided, searches all default domains."
                        ),
                    },
                    "max_results_per_domain": {
                        "type": "integer",
                        "description": "Maximum number of results to return per domain (default: 3)",
                        "default": 3,
                    },
                    "min_relevance_score": {
                        "type": "integer",
                        "description": "Minimum relevance score threshold (0-100) to filter results (default: 17)",
                        "default": 17,
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["relevance", "date", "domain"],
                        "description": "Sort order for results: 'relevance' (default, highest score first), 'date' (newest first), or 'domain' (alphabetical by domain)",
                        "default": "relevance",
                    },
                    "date_from": {
                        "type": "string",
                        "format": "date",
                        "description": "Optional date filter in YYYY-MM-DD format. Only include content published on or after this date.",
                    },
                    "date_to": {
                        "type": "string",
                        "format": "date",
                        "description": "Optional date filter in YYYY-MM-DD format. Only include content published on or before this date.",
                    },
                    "max_total_results": {
                        "type": "integer",
                        "description": "Optional limit on total results across all domains (after filtering and sorting)",
                        "minimum": 1,
                    },
                    "allowed_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of domains to include. Only results from these domains will be returned.",
                    },
                    "blocked_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of domains to exclude. Results from these domains will be filtered out.",
                    },
                },
                "required": ["query"],
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean", "description": "Whether the operation was successful"},
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
                                    "timing_breakdown": {"type": "object"},
                                },
                            },
                        },
                        "required": [
                            "query",
                            "total_results",
                            "domains_searched",
                            "domains_with_results",
                            "search_time_ms",
                        ],
                    },
                    "results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "title": {"type": "string"},
                                "url": {"type": "string"},
                                "snippet": {
                                    "type": "string",
                                    "description": "Enhanced snippet with **bold** highlighting",
                                },
                                "domain": {"type": "string"},
                                "domain_category": {
                                    "type": "string",
                                    "enum": [
                                        "documentation",
                                        "blog",
                                        "news",
                                        "developer",
                                        "build",
                                        "research",
                                        "catalog",
                                        "forum",
                                        "downloads",
                                        "resources",
                                        "other",
                                    ],
                                },
                                "content_type": {
                                    "type": "string",
                                    "enum": [
                                        "announcement",
                                        "tutorial",
                                        "guide",
                                        "forum_discussion",
                                        "blog_post",
                                        "documentation",
                                        "research_paper",
                                        "news",
                                        "video",
                                        "course",
                                        "article",
                                    ],
                                    "description": "Detected content type based on title, snippet, and URL analysis",
                                },
                                "published_date": {
                                    "type": "string",
                                    "format": "date",
                                    "description": "Publication date in YYYY-MM-DD format (if available)",
                                },
                                "relevance_score": {
                                    "type": "integer",
                                    "description": "Relevance score from 0-100 based on keyword matching and TF-IDF",
                                },
                                "matched_keywords": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Keywords from the query that matched in this result",
                                },
                                "metadata": {
                                    "type": "object",
                                    "description": "Additional metadata extracted from the page (e.g., author, code presence)",
                                },
                            },
                            "required": ["id", "title", "url", "snippet", "domain", "relevance_score"],
                        },
                    },
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "number": {"type": "integer"},
                                "url": {"type": "string"},
                                "title": {"type": "string"},
                                "domain": {"type": "string"},
                            },
                        },
                    },
                    "warnings": {"type": "array", "items": {"type": "object"}},
                    "errors": {"type": "array", "items": {"type": "object"}},
                },
            },
        },
        {
            "name": "discover_nvidia_content",
            "description": (
                "Discover specific types of NVIDIA content such as videos, courses, tutorials, webinars, or blog posts. "
                "This tool helps find educational and learning resources from NVIDIA's various platforms. "
                "Returns ranked results with relevance scores and direct links to the content."
            ),
            "inputSchema": {
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
                        ),
                    },
                    "topic": {
                        "type": "string",
                        "description": "The topic or technology to find content about (e.g., 'CUDA', 'Omniverse', 'AI')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of content items to return (default: 5)",
                        "default": 5,
                    },
                    "date_from": {
                        "type": "string",
                        "format": "date",
                        "description": "Optional date filter in YYYY-MM-DD format. Only content published on or after this date will be included.",
                    },
                },
                "required": ["content_type", "topic"],
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "summary": {
                        "type": "object",
                        "properties": {
                            "content_type": {"type": "string"},
                            "topic": {"type": "string"},
                            "total_found": {"type": "integer"},
                            "search_time_ms": {"type": "integer"},
                            "suggestions": {
                                "type": "object",
                                "description": "Suggestions for alternative searches if no results found",
                            },
                            "expanded_topics": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Topic synonyms and related terms used for semantic matching",
                            },
                        },
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
                                "snippet": {"type": "string"},
                                "relevance_score": {"type": "integer"},
                                "domain": {"type": "string"},
                                "published_date": {"type": "string", "format": "date"},
                            },
                        },
                    },
                    "resource_links": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "number": {"type": "integer"},
                                "url": {"type": "string"},
                                "title": {"type": "string"},
                                "type": {"type": "string"},
                            },
                        },
                    },
                    "warnings": {"type": "array"},
                    "errors": {"type": "array"},
                },
            },
        },
    ]


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [Tool(**schema) for schema in _get_tool_schemas()]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> CallToolResult:
    """Handle tool calls."""
    try:
        if name == "search_nvidia":
            # SECURITY: Use semaphore to limit concurrent searches
            async with _search_semaphore:
                query = arguments.get("query")
                if not query:
                    error_response = build_error_response_json("MISSING_PARAMETER", "Query parameter is required")
                    return build_tool_result(error_response)

                # SECURITY: Validate query length
                if len(query) > MAX_QUERY_LENGTH:
                    error_response = build_error_response_json(
                        "INVALID_PARAMETER", f"Query too long. Maximum length: {MAX_QUERY_LENGTH} characters"
                    )
                    return build_tool_result(error_response)

                domains = arguments.get("domains")
                max_results_per_domain = arguments.get("max_results_per_domain", 3)

                # SECURITY: Limit max_results_per_domain to prevent resource exhaustion
                if max_results_per_domain > MAX_RESULTS_PER_DOMAIN:
                    logger.warning(
                        f"max_results_per_domain limited from {max_results_per_domain} to {MAX_RESULTS_PER_DOMAIN}"
                    )
                max_results_per_domain = min(max_results_per_domain, MAX_RESULTS_PER_DOMAIN)

                # Validate caller-supplied domains
                validated_domains = None
                if domains is not None:
                    if not isinstance(domains, list):
                        error_response = build_error_response_json(
                            "INVALID_PARAMETER", "domains must be a list of strings"
                        )
                        return build_tool_result(error_response)

                    invalid_domains = []
                    validated_domains = []

                    for domain in domains:
                        if not isinstance(domain, str):
                            error_response = build_error_response_json(
                                "INVALID_PARAMETER", f"Invalid domain type: {type(domain).__name__}. Expected string."
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
                            "INVALID_DOMAIN", error_msg, {"invalid_domains": invalid_domains}
                        )
                        return build_tool_result(error_response)

                    if not validated_domains:
                        error_response = build_error_response_json(
                            "NO_VALID_DOMAINS", "No valid NVIDIA domains provided"
                        )
                        return build_tool_result(error_response)

                    logger.info(f"Validated {len(validated_domains)} caller-supplied domains")

                min_relevance_score = arguments.get("min_relevance_score", 17)
                sort_by = arguments.get("sort_by", "relevance")
                date_from = arguments.get("date_from")
                date_to = arguments.get("date_to")
                max_total_results = arguments.get("max_total_results")
                allowed_domains = arguments.get("allowed_domains")
                blocked_domains = arguments.get("blocked_domains")

                # Validate sort_by parameter
                if sort_by not in ["relevance", "date", "domain"]:
                    error_response = build_error_response_json(
                        "INVALID_PARAMETER",
                        f"Invalid sort_by value: {sort_by}. Must be 'relevance', 'date', or 'domain'",
                    )
                    return build_tool_result(error_response)

                # Validate date parameters
                if date_from:
                    try:
                        datetime.strptime(date_from, "%Y-%m-%d")  # noqa: DTZ007
                    except ValueError:
                        error_response = build_error_response_json(
                            "INVALID_PARAMETER", f"Invalid date_from format. Expected YYYY-MM-DD, got: {date_from}"
                        )
                        return build_tool_result(error_response)

                if date_to:
                    try:
                        datetime.strptime(date_to, "%Y-%m-%d")  # noqa: DTZ007
                    except ValueError:
                        error_response = build_error_response_json(
                            "INVALID_PARAMETER", f"Invalid date_to format. Expected YYYY-MM-DD, got: {date_to}"
                        )
                        return build_tool_result(error_response)

                logger.info(
                    f"Searching NVIDIA domains for: {query} (sort_by={sort_by}, date_from={date_from}, date_to={date_to}, max_total_results={max_total_results})"
                )

                # Get results with error tracking
                results, errors, warnings, timing_info = await search_all_domains(
                    query=query,
                    domains=validated_domains,
                    max_results_per_domain=max_results_per_domain,
                    min_relevance_score=min_relevance_score,
                    sort_by=sort_by,
                    date_from=date_from,
                    date_to=date_to,
                    max_total_results=max_total_results,
                    allowed_domains=allowed_domains,
                    blocked_domains=blocked_domains,
                )

                # Build JSON response
                response = build_search_response_json(
                    results=results,
                    query=query,
                    domains_searched=len(validated_domains) if validated_domains else len(DEFAULT_DOMAINS),
                    search_time_ms=timing_info["total_time_ms"],
                    errors=errors,
                    warnings=warnings,
                    debug_info=timing_info.get("debug_info", {}),
                )

                return build_tool_result_with_ui(response, tool_name="search_nvidia")

        elif name == "discover_nvidia_content":
            async with _search_semaphore:
                content_type = arguments.get("content_type")
                topic = arguments.get("topic")
                date_from = arguments.get("date_from")

                if not content_type or not topic:
                    error_response = build_error_response_json(
                        "MISSING_PARAMETER", "Both content_type and topic parameters are required"
                    )
                    return build_tool_result(error_response)

                if len(topic) > MAX_QUERY_LENGTH:
                    error_response = build_error_response_json(
                        "INVALID_PARAMETER", f"Topic too long. Maximum length: {MAX_QUERY_LENGTH} characters"
                    )
                    return build_tool_result(error_response)

                if date_from:
                    try:
                        datetime.strptime(date_from, "%Y-%m-%d")  # noqa: DTZ007
                    except ValueError:
                        error_response = build_error_response_json(
                            "INVALID_PARAMETER", f"Invalid date_from format. Expected YYYY-MM-DD, got: {date_from}"
                        )
                        return build_tool_result(error_response)

                max_results = arguments.get("max_results", 5)

                if max_results > MAX_RESULTS_PER_DOMAIN:
                    logger.warning(f"max_results limited from {max_results} to {MAX_RESULTS_PER_DOMAIN}")
                    max_results = MAX_RESULTS_PER_DOMAIN

                logger.info(f"Discovering {content_type} content for topic: {topic} (date_from={date_from})")

                results, errors, warnings, timing_info = await discover_content(
                    content_type=content_type, topic=topic, max_results=max_results, date_from=date_from
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

                return build_tool_result_with_ui(response, tool_name="discover_nvidia_content")

        else:
            error_response = build_error_response_json("UNKNOWN_TOOL", f"Unknown tool: {name}")
            return build_tool_result(error_response)

    except Exception as e:
        # SECURITY: Sanitize error messages to avoid exposing internal details
        logger.exception(f"Error in tool {name}: {e}")
        error_response = build_error_response_json(
            "INTERNAL_ERROR", "An unexpected error occurred while processing the request"
        )
        return build_tool_result(error_response)


def _get_tool_definitions() -> list[dict[str, Any]]:
    """Get tool definitions for SDK generation."""
    return _get_tool_schemas()


def _generate_sdk_files() -> dict[str, dict[str, str]]:
    """
    Generate SDK files for all languages.

    This function is called eagerly at module load time to avoid race conditions
    and ensure SDK files are immediately available when the server starts.

    Returns:
        Dictionary mapping language names to file dictionaries
    """
    logger.info("Generating SDK files...")
    tools = _get_tool_definitions()

    sdk_files = {
        "typescript": generate_typescript_sdk(tools),
        "python": generate_python_sdk(tools),
    }

    logger.info(
        f"SDK files generated: {len(sdk_files['typescript'])} TypeScript files, {len(sdk_files['python'])} Python files"
    )

    return sdk_files


# SDK generation cache (eagerly initialized at module load to avoid race conditions)
# This is safe because SDK generation is deterministic and doesn't depend on runtime state
_sdk_files_cache: dict[str, dict[str, str]] = _generate_sdk_files()


@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available SDK resources."""
    sdk_files = _sdk_files_cache
    resources = []

    # Add TypeScript SDK files
    for filename in sdk_files["typescript"]:
        resources.append(
            Resource(
                uri=f"mcp-nvidia://sdk/typescript/{filename}",
                name=f"TypeScript SDK: {filename}",
                mimeType="text/plain" if filename.endswith(".md") else "text/typescript",
                description=f"TypeScript SDK file: {filename}",
            )
        )

    # Add Python SDK files
    for filename in sdk_files["python"]:
        resources.append(
            Resource(
                uri=f"mcp-nvidia://sdk/python/{filename}",
                name=f"Python SDK: {filename}",
                mimeType="text/plain" if filename.endswith(".md") else "text/x-python",
                description=f"Python SDK file: {filename}",
            )
        )

    logger.info(f"Listed {len(resources)} SDK resources")
    return resources


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read SDK resource by URI."""
    logger.info(f"Reading resource: {uri}")

    if not uri.startswith("mcp-nvidia://sdk/"):
        raise ValueError(f"Invalid resource URI: {uri}")

    # Parse URI: mcp-nvidia://sdk/[language]/[filename]
    parts = uri.replace("mcp-nvidia://sdk/", "").split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid resource URI format: {uri}")

    language, filename = parts

    if language not in ["typescript", "python"]:
        raise ValueError(f"Unknown SDK language: {language}")

    sdk_files = _sdk_files_cache

    if filename not in sdk_files[language]:
        raise ValueError(f"SDK file not found: {filename}")

    content = sdk_files[language][filename]
    logger.info(f"Successfully read resource {uri} ({len(content)} bytes)")

    return content


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
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

    try:
        logger.info("MCP server ready and waiting for connections")
        async with stdio_server() as (read_stream, write_stream):
            # Run server in a task so we can cancel it on shutdown
            server_task = asyncio.create_task(app.run(read_stream, write_stream, app.create_initialization_options()))

            # Wait for either server completion or shutdown signal
            shutdown_task = asyncio.create_task(shutdown_event.wait())
            _, pending = await asyncio.wait([server_task, shutdown_task], return_when=asyncio.FIRST_COMPLETED)

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
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    except Exception as e:
        logger.exception(f"Unexpected error in server: {e}")
        raise
    finally:
        logger.info("MCP server shutdown complete")


def main():
    """Main entry point with subcommands for different transports."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MCP NVIDIA Server - Search NVIDIA domains via Model Context Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run in stdio mode (default, for Claude Desktop)
  %(prog)s
  %(prog)s stdio

  # Run in HTTP/SSE mode (for remote access)
  %(prog)s http
  %(prog)s http --port 3000
  %(prog)s http --host 0.0.0.0 --port 8080

  # Enable debug logging
  MCP_NVIDIA_LOG_LEVEL=DEBUG %(prog)s
        """,
    )

    subparsers = parser.add_subparsers(dest="transport", help="Transport mode")

    # stdio subcommand (default)
    subparsers.add_parser("stdio", help="Run in stdio mode (default, for local MCP clients)")

    # http subcommand
    http_parser = subparsers.add_parser("http", help="Run in HTTP/SSE mode (for remote access)")

    http_parser.add_argument(
        "--host",
        default="0.0.0.0",  # nosec B104
        help="Host to bind to (default: 0.0.0.0 for all interfaces)",
    )

    http_parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")

    args = parser.parse_args()

    # Default to stdio if no subcommand specified
    if args.transport is None or args.transport == "stdio":
        # Run in stdio mode
        try:
            asyncio.run(run())
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            sys.exit(0)
        except Exception as e:
            logger.exception(f"Fatal error: {e}")
            sys.exit(1)

    elif args.transport == "http":
        # Run in HTTP mode
        try:
            from mcp_nvidia.http_server import run_http_server

            run_http_server(host=args.host, port=args.port)
        except ImportError as e:
            logger.exception(f"HTTP server dependencies not available: {e}")
            logger.exception("Install HTTP dependencies with: pip install mcp-nvidia")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            sys.exit(0)
        except Exception as e:
            logger.exception(f"Fatal error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
