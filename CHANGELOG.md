# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/bharatr21/mcp-nvidia/compare/v0.2.5...HEAD
[0.2.5]: https://github.com/bharatr21/mcp-nvidia/releases/tag/v0.2.5
[0.2.0]: https://github.com/bharatr21/mcp-nvidia/releases/tag/v0.2.0
[0.1.0]: https://github.com/bharatr21/mcp-nvidia/releases/tag/v0.1.0
