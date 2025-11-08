# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-XX

### Added

- Initial release of MCP NVIDIA search server
- **Two main tools**:
  - `search_nvidia`: Search across NVIDIA domains with relevance scoring
  - `discover_nvidia_content`: Find specific types of content (videos, courses, tutorials, webinars, blogs)
- **16 supported NVIDIA domains**:
  - blogs.nvidia.com, build.nvidia.com, catalog.ngc.nvidia.com
  - developer.download.nvidia.com, developer.nvidia.com, docs.api.nvidia.com
  - docs.nvidia.com, docs.omniverse.nvidia.com, forums.developer.nvidia.com
  - forums.nvidia.com, gameworksdocs.nvidia.com, ngc.nvidia.com
  - nvidia.github.io, nvidianews.nvidia.com, research.nvidia.com
  - resources.nvidia.com
- **Structured JSON output** with CallToolResult support (MCP 2025-06-18 spec)
- **Context-aware snippets** with automatic URL fetching and highlighted keywords
- **Relevance scoring** (0-100 scale) displayed in formatted text
- **Comprehensive security measures**:
  - SSRF protection with URL validation and redirect prevention
  - Rate limiting (200ms between DuckDuckGo API calls, ~5 searches/sec)
  - Concurrency control (max 5 simultaneous searches)
  - Input validation (500 char query limit, 10 results per domain max)
  - Domain whitelisting (nvidia.com and nvidia.github.io only)
  - Sanitized error messages
- **Concurrent domain searching** for fast results
- **Customizable domains** via `MCP_NVIDIA_DOMAINS` environment variable
- **Configurable logging** via `MCP_NVIDIA_LOG_LEVEL` environment variable
- Comprehensive test suite with pytest and pytest-asyncio
- CI/CD pipeline with GitHub Actions
- npx wrapper for easy installation (@bharatr21/mcp-nvidia)

### Dependencies

- mcp >= 1.1.0 (for CallToolResult.structuredContent support)
- httpx >= 0.28.0 (HTTP client)
- beautifulsoup4 >= 4.14.0 (HTML parsing)
- ddgs >= 9.0.0 (DuckDuckGo search)

### Security

- No API keys or credentials required
- All searches go through DuckDuckGo (no direct NVIDIA API access needed)
- Domain validation prevents SSRF attacks
- Rate limiting prevents abuse and API bans
- Resource exhaustion prevention with concurrency limits

[0.1.0]: https://github.com/bharatr21/mcp-nvidia/releases/tag/v0.1.0
