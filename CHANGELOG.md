# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.0] - 2026-09-12

This release pins the MCP SDK so existing installs keep working, adds hybrid keyword +
semantic ranking to `search_nvidia`, fixes `resources/read`, and gives notice of the
breaking 1.0.0 release that follows.

### Added

- **Hybrid search.** `search_nvidia` can rank results by semantic similarity alongside
  keyword matching and TF-IDF, using a small local embedding model
  (`BAAI/bge-small-en-v1.5` via fastembed; no torch and no API key). It is opt-in:
  `pip install "mcp-nvidia[embeddings]"`. The model downloads on first use, and the
  Docker image ships with it preinstalled.
- Configuration through `MCP_NVIDIA_EMBEDDING_MODEL` and `MCP_NVIDIA_EMBEDDING_THREADS`.
- A `SEMANTIC_RANKING_UNAVAILABLE` entry in `warnings` when semantic ranking is installed
  but fails, so degraded ranking is visible in the response.

### Changed

- **Result order changes for every `search_nvidia` query**, with or without the extra.
  The fixed 70% keyword / 30% TF-IDF score blend is replaced by reciprocal rank fusion
  over the available ranking signals.
- `relevance_score` is still an integer from 0 to 100, but it is now derived from a
  result's position in the fused ranking rather than from a weighted score.
- Candidates with no keyword match and low semantic similarity are dropped before
  ranking, and `min_relevance_score` then applies to the fused ranking. Queries with
  only weak matches can return fewer results.
- Duplicate results are removed before ranking instead of after the relevance cutoff, so a
  page returned by two overlapping domains (for example `ngc.nvidia.com` and
  `catalog.ngc.nvidia.com`) is no longer scored twice.

### Fixed

- Pinned the MCP SDK to `mcp>=1.28,<2.0.0`. The previous requirement, `mcp>=1.1.0`, has
  no upper bound, so a fresh `pip install mcp-nvidia` now resolves to `mcp` 2.2.0 — an
  SDK that removed the decorator handler API this release is built on. Installs that
  picked up `mcp` 2.x would fail at import. Verified against `mcp` 1.30.0, the latest 1.x.
- `resources/read` failed for every MCP client with
  `'AnyUrl' object has no attribute 'startswith'`. The SDK passes the resource URI as a
  pydantic `AnyUrl` rather than a string. This affected every supported `mcp` 1.x version,
  and 0.5.0 before it.

### ⚠️ Notice: 1.0.0 will be a breaking release

**mcp-nvidia 1.0.0 is not backward compatible with 0.x.** It requires the `mcp` 2.x SDK
(`mcp>=2.2.0,<3`) and the `2026-07-28` protocol revision. If you upgrade, expect all of
the following to change:

- **The remote transport moves.** The SSE endpoint `/sse` is removed and returns
  `410 Gone`. Remote clients must point at `/mcp` and use `"transport": "http"` instead
  of `"transport": "sse"`.
- **The SDK requirement flips.** 1.0.0 requires `mcp>=2.2.0` and will not run on `mcp` 1.x,
  just as 0.9.0 will not run on `mcp` 2.x. The two cannot be installed together.
- **An error code changes.** An invalid or unknown resource URI returns `-32602`
  (Invalid Params) instead of `-32002`.
- **Remote deployments need a host allow-list.** The server enables DNS-rebinding
  protection; a public deployment must advertise its hostname via `RAILWAY_PUBLIC_DOMAIN`
  (automatic on Railway) or `MCP_ALLOWED_HOSTS`, or requests are rejected with `421`.

Clients on the 2025-era protocol are still served by 1.0.0 over the same `/mcp` endpoint,
so upgrading the server does not force every client to upgrade at once.

**To stay on the 0.x line**, pin the package rather than the SDK:

```bash
pip install "mcp-nvidia<1.0.0"
```

## [0.4.0] - 2025-11-16

### Added

- **SDK Generation**: TypeScript and Python SDK generators with functional implementations
  - Python SDK with zero MCP overhead - direct function calls to `mcp_nvidia.lib`
  - TypeScript SDK with full type safety and MCP client integration
  - Both SDKs provide complete type definitions and JSDoc/docstring documentation
- **MCP Resources Protocol**: Code Execution Mode support via MCP Resources
  - `list_resources()`: Returns all available SDK files
  - `read_resource(uri)`: Returns SDK file contents
  - Virtual filesystem at `mcp-nvidia://sdk/` with 8 resource files
  - Cached generation on server startup for optimal performance
- **Modular Architecture**: Comprehensive refactoring to improve maintainability
  - 9 new modules in `src/mcp_nvidia/lib/`:
    - `constants.py`: Configuration and domain lists
    - `search.py`: Search orchestration and domain handling
    - `relevance.py`: Scoring and ranking algorithms
    - `snippet.py`: Context extraction and highlighting
    - `response_builders.py`: JSON response formatting
    - `content_discovery.py`: Content type detection
    - `deduplication.py`: Result deduplication
    - `utils.py`: Validation and utilities
    - `__init__.py`: Clean exports of all functions
  - SDK generation modules in `src/mcp_nvidia/sdk_generator/`:
    - `python_generator.py`: Python SDK generation
    - `typescript_generator.py`: TypeScript SDK generation
    - `__init__.py`: SDK generation orchestration

### Changed

- **server.py** reduced from ~2400 to ~800 lines by extracting business logic to lib/ modules
- Improved separation of concerns and code organization
- Better testability through modular design

### Technical

- Comprehensive test suite with 18 new tests:
  - 7 tests for SDK generation (TypeScript/Python, type conversion, multi-tool support)
  - 11 tests for MCP Resources (listing, reading, error handling, caching)
- Full backward compatibility - existing MCP tool calling works unchanged
- URI validation and error handling for resource protocol
- Snippet highlighting enhancements

### Documentation

- Added "SDK Resources (Code Execution Mode)" section to README
- Usage examples for both Python and TypeScript SDKs
- Documented differences between Python (direct) and TypeScript (MCP) approaches
- Updated architecture documentation with new components
- References to Anthropic's "Code Execution with MCP" and Cloudflare's "Code Mode"

### Breaking Changes

None - fully backward compatible

## [0.3.0] - 2025-11-10

### Added

- Initial implementation of features later refined in 0.4.0

## [0.2.5] - 2025-11-09

### Added

- **Publication date extraction**: Automatically extracts publication dates from HTML metadata (meta tags like
  `article:published_time`, `og:published_time`) and page content using regex patterns
- **Content type detection**: Identifies content as one of: announcement, tutorial, guide, forum_discussion,
  blog_post, documentation, research_paper, news, video, course, or article
- **Rich metadata extraction**:
  - Author name from various HTML sources (meta tags, author elements)
  - Approximate word count
  - Code snippet detection (`has_code`)
  - Video detection (`has_video`)
  - Image detection (`has_images`, `image_count`)
- **Flexible sorting options** via new `sort_by` parameter:
  - `relevance` (default): Sort by relevance score (highest first)
  - `date`: Sort by publication date (newest first)
  - `domain`: Sort alphabetically by domain name
- **New optional result fields**:
  - `published_date`: Publication date in YYYY-MM-DD format (when available)
  - `content_type`: Detected content type
  - `metadata`: Object containing author, word_count, has_code, has_video, has_images, image_count

### Changed

- `fetch_url_context()` now returns a tuple with enhanced snippet, publication date, and metadata
- Results are sorted based on `sort_by` parameter instead of only by relevance
- Updated JSON schemas to include new optional fields

### Removed

- `snippet_plain` field (redundant with `snippet`)
- `formatted_text` field (redundant - clients can format using individual fields)

### Dependencies

- Added `python-dateutil >= 2.8.0` for robust date parsing

### Technical

- Multi-strategy date extraction: HTML metadata first, then regex patterns in text
- Fallback date parsing with both dateutil and manual parsing
- Graceful degradation: metadata fields only appear when successfully extracted

## [0.2.0] - 2025-11-08

### Changes in 0.2.0

- Search algorithm improvements
- Enhanced relevance scoring

## [0.1.0] - 2025-11-05

### Features in 0.1.0

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

### Initial Dependencies

- mcp >= 1.1.0 (for CallToolResult.structuredContent support)
- httpx >= 0.28.0 (HTTP client)
- beautifulsoup4 >= 4.14.0 (HTML parsing)
- ddgs >= 9.0.0 (DuckDuckGo search)

### Security Features

- No API keys or credentials required
- All searches go through DuckDuckGo (no direct NVIDIA API access needed)
- Domain validation prevents SSRF attacks
- Rate limiting prevents abuse and API bans
- Resource exhaustion prevention with concurrency limits

[Unreleased]: https://github.com/bharatr21/mcp-nvidia/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/bharatr21/mcp-nvidia/releases/tag/v0.4.0
[0.3.0]: https://github.com/bharatr21/mcp-nvidia/releases/tag/v0.3.0
[0.2.5]: https://github.com/bharatr21/mcp-nvidia/releases/tag/v0.2.5
[0.2.0]: https://github.com/bharatr21/mcp-nvidia/releases/tag/v0.2.0
[0.1.0]: https://github.com/bharatr21/mcp-nvidia/releases/tag/v0.1.0
