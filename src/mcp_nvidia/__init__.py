"""MCP server for searching across NVIDIA domains."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("mcp-nvidia")
except importlib.metadata.PackageNotFoundError:
    # Package is not installed, use fallback version
    __version__ = "0.2.5"
