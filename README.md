# mcp-nvidia

MCP server to search across NVIDIA blogs and releases to empower LLMs to better answer NVIDIA-specific queries.

## Overview

This Model Context Protocol (MCP) server enables Large Language Models (LLMs) to search across
multiple NVIDIA domains to find relevant information about NVIDIA technologies, products, and services.

### Supported Domains (15 default, customizable)

The server searches across the following NVIDIA domains by default. You can customize this list using the
`MCP_NVIDIA_DOMAINS` environment variable or the `domains` parameter in search queries:

- **blogs.nvidia.com** - NVIDIA blog posts and articles
- **build.nvidia.com** - NVIDIA AI Foundation models and services
- **catalog.ngc.nvidia.com** - NGC catalog of GPU-accelerated software
- **developer.download.nvidia.com** - Developer downloads and resources
- **developer.nvidia.com** - Developer resources, SDKs, and technical documentation
- **docs.api.nvidia.com** - API documentation
- **docs.nvidia.com** - Comprehensive technical documentation
- **docs.omniverse.nvidia.com** - Omniverse documentation
- **forums.developer.nvidia.com** - Developer forums
- **forums.nvidia.com** - Community forums
- **ngc.nvidia.com** - NVIDIA GPU Cloud
- **nvidia.github.io** - NVIDIA GitHub Pages documentation
- **nvidianews.nvidia.com** - Official NVIDIA news and press releases
- **research.nvidia.com** - NVIDIA research publications
- **resources.nvidia.com** - NVIDIA resources and whitepapers

**Note**: For security, only nvidia.com domains and nvidia.github.io are allowed.

### Key Features

#### Search & Ranking

- **Advanced relevance scoring** combining multiple signals
- **Intelligent keyword extraction**
- **Context-aware search** with enhanced snippets and highlighting
- **Flexible sorting** by relevance, publication date, or domain
- **Content type detection** (tutorials, announcements, guides, forums, etc.)
- **Date extraction** from HTML metadata and page content

#### Rich Metadata

- **Publication dates** extracted from HTML metadata and text patterns
- **Content metadata** including author, word count, code/video/image detection
- **Structured content types** for easy filtering and categorization

#### Connectivity

- **stdio mode** (default) - for local MCP clients like Claude Desktop
- **HTTP/SSE mode** - for remote access and web-based clients
- **Structured JSON output** compatible with AI agents and LLMs

#### Customization

- **Domain-specific filtering** for targeted searches
- **Configurable domains** via environment variable
- **Adjustable relevance thresholds** and result limits
- **Multiple sort options** (relevance, date, domain)

## Installation

### Via npx (Easiest - recommended for MCP clients)

```bash
npx @bharatr21/mcp-nvidia
```

Or add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "nvidia": {
      "command": "npx",
      "args": ["-y", "@bharatr21/mcp-nvidia"]
    }
  }
}
```

**Note:** This requires Python 3.10+ to be installed on your system. The package will automatically use the Python backend.

### Via pip

```bash
pip install mcp-nvidia
```

### From source

```bash
git clone https://github.com/bharatr21/mcp-nvidia.git
cd mcp-nvidia
pip install -e .
```

## Usage

### Running the Server

The MCP server can be run directly from the command line:

```bash
mcp-nvidia
```

### Configuration

The server can be configured using environment variables:

- `MCP_NVIDIA_DOMAINS`: Comma-separated list of custom NVIDIA domains to search (overrides defaults)
  - **Security**: Only nvidia.com domains and subdomains are allowed. Invalid domains are automatically filtered out.
  - Example: `"https://developer.nvidia.com/,https://docs.nvidia.com/"`
- `MCP_NVIDIA_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

Example:

```bash
export MCP_NVIDIA_DOMAINS="https://developer.nvidia.com/,https://docs.nvidia.com/"
export MCP_NVIDIA_LOG_LEVEL="DEBUG"
mcp-nvidia
```

### Configuring with Claude Desktop

Add the following to your Claude Desktop configuration file:

**MacOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "nvidia": {
      "command": "mcp-nvidia"
    }
  }
}
```

With custom configuration (environment variables):

```json
{
  "mcpServers": {
    "nvidia": {
      "command": "mcp-nvidia",
      "env": {
        "MCP_NVIDIA_LOG_LEVEL": "DEBUG",
        "MCP_NVIDIA_DOMAINS": "https://developer.nvidia.com/,https://docs.nvidia.com/"
      }
    }
  }
}
```

If you installed from source, you may need to use the full path to the Python executable:

```json
{
  "mcpServers": {
    "nvidia": {
      "command": "/path/to/python",
      "args": ["-m", "mcp_nvidia"]
    }
  }
}
```

### HTTP Server Mode (Remote Access)

For remote access or public deployment, use HTTP/SSE mode:

```bash
# Run in HTTP mode (default port 8000)
mcp-nvidia http

# Custom host and port
mcp-nvidia http --host 0.0.0.0 --port 3000

# Or explicitly use stdio mode (default)
mcp-nvidia stdio
mcp-nvidia  # same as stdio
```

**Configure Claude AI for remote server:**

```json
{
  "mcpServers": {
    "nvidia": {
      "url": "https://your-server.com/sse",
      "transport": "sse"
    }
  }
}
```

## Available Tools

### search_nvidia

Search across NVIDIA domains for specific information. Results include citations with URLs for easy reference.

**Parameters:**

- `query` (required): The search query to find information across NVIDIA domains
- `domains` (optional): List of specific NVIDIA domains to search. If not provided, searches all default domains
- `max_results_per_domain` (optional): Maximum number of results to return per domain (default: 3)
- `min_relevance_score` (optional): Minimum relevance score threshold (0-100) to filter results (default: 17)
- `sort_by` (optional): Sort order for results - one of:
  - `relevance` (default): Sort by relevance score (highest first)
  - `date`: Sort by publication date (newest first)
  - `domain`: Sort alphabetically by domain name

**Example queries:**

- "CUDA programming best practices"
- "RTX 4090 specifications"
- "TensorRT optimization techniques"
- "Latest AI announcements"
- "Omniverse development tutorials"

**Example usage with sorting:**

```python
# Sort by publication date (newest first) - great for finding latest announcements
search_nvidia(query="NIM microservices", sort_by="date")

# Sort by relevance (default) - best for finding most relevant content
search_nvidia(query="CUDA optimization", sort_by="relevance")

# Sort by domain - useful for browsing content by source
search_nvidia(query="TensorRT", sort_by="domain")
```

**Features:**

- **Enhanced search using ddgs package** for reliable DuckDuckGo integration with domain filtering
- **Structured JSON output**: Returns data in a structured format with both machine-readable and
  human-readable fields
- **Context-aware snippets**: Automatically fetches surrounding text from source URLs and highlights the
  relevant snippet with `**bold**` formatting
- **Relevance scoring (0-100 scale)**: Each result includes a relevance score based on query term matches in
  title, snippet, and URL
  - Results are sorted by relevance score (highest first) when using default sorting
  - Results below the threshold are automatically filtered out
  - Score displayed as "Score: X/100" in formatted text
- **Flexible sorting options**:
  - `relevance`: Sort by relevance score (default)
  - `date`: Sort by publication date, newest first
  - `domain`: Sort alphabetically by domain name
- **Publication date extraction**: Automatically extracts dates from HTML metadata (meta tags, time elements)
  and page content using multiple strategies
- **Content type detection**: Identifies content as announcements, tutorials, guides, forum discussions,
  blog posts, documentation, research papers, news, videos, courses, or articles
- **Rich metadata extraction**:
  - Author name from various HTML sources
  - Approximate word count
  - Code snippet detection
  - Video and image detection with counts
- **Security controls**: Input validation and limits (customizable via code)
  - Query/topic length: 500 characters max
  - Results per domain: 10 max
  - Domain whitelist: nvidia.com and nvidia.github.io only
  - Rate limiting: 200ms between search API calls (~5 searches/sec)
  - Concurrent searches: 5 max simultaneous searches
- **Concurrent search** across multiple domains for fast results
- Formatted results with titles, URLs, enhanced snippets with context, and source domains
- Dedicated citations section with numbered references for easy copying

**Output Format:**
Results are returned as structured JSON with the following schema:

```json
{
  "success": true,
  "results": [
    {
      "title": "Page Title",
      "url": "https://example.nvidia.com/page",
      "snippet": "Enhanced snippet with **highlighted** keywords",
      "domain": "example.nvidia.com",
      "relevance_score": 85,
      "published_date": "2025-01-15",
      "content_type": "tutorial",
      "metadata": {
        "author": "John Doe",
        "word_count": 1234,
        "has_code": true,
        "has_video": false,
        "has_images": true,
        "image_count": 5
      }
    }
  ],
  "metadata": {
    "domains_searched": 16,
    "search_time_ms": 1234
  }
}
```

**Result Fields:**

- `title`: Page title
- `url`: Full URL to the resource
- `snippet`: Enhanced snippet with **bold** highlighting for matched keywords
- `domain`: Source domain (e.g., "docs.nvidia.com")
- `relevance_score`: Relevance score from 0-100 based on keyword matches
- `published_date` (optional): Publication date in YYYY-MM-DD format, extracted from HTML metadata or page content
- `content_type`: Detected content type, one of:
  - `announcement`: Product/feature announcements
  - `tutorial`: Step-by-step tutorials
  - `guide`: How-to guides and best practices
  - `forum_discussion`: Forum posts and discussions
  - `blog_post`: Blog articles
  - `documentation`: Technical documentation
  - `research_paper`: Research publications
  - `news`: News articles and press releases
  - `video`: Video content
  - `course`: Training courses
  - `article`: General articles (default)
- `metadata` (optional): Additional extracted metadata:
  - `author`: Content author name (if available)
  - `word_count`: Approximate word count
  - `has_code`: Whether the page contains code snippets
  - `has_video`: Whether the page contains embedded videos
  - `has_images`: Whether the page contains images
  - `image_count`: Number of images on the page

**Note**: The `published_date` and `metadata` fields are optional and only appear when successfully extracted from the page.

### discover_nvidia_content

Discover specific types of NVIDIA educational and learning content such as videos, courses, tutorials,
webinars, or blog posts.

**Parameters:**

- `content_type` (required): Type of content to find - one of:
  - `video`: Video tutorials and demonstrations
  - `course`: Training courses and certifications (NVIDIA DLI)
  - `tutorial`: Step-by-step guides and how-tos
  - `webinar`: Webinars and live sessions
  - `blog`: Blog posts and articles
- `topic` (required): The topic or technology to find content about (e.g., "CUDA", "Omniverse", "AI")
- `max_results` (optional): Maximum number of content items to return (default: 5)

**Example queries:**

- Find video tutorials: `discover_nvidia_content(content_type="video", topic="CUDA programming")`
- Find training courses: `discover_nvidia_content(content_type="course", topic="Deep Learning")`
- Find webinars: `discover_nvidia_content(content_type="webinar", topic="AI in Healthcare")`

**Features:**

- Content-specific search strategies optimized for each type
- **Relevance scoring on 0-100 scale** to highlight best matches
  - Score displayed as "Score: X/100" for transparency
- Direct links to videos, courses, tutorials, and other resources
- Resource links section for easy access to all discovered content

## Development

### Setting up a development environment

```bash
# Clone the repository
git clone https://github.com/bharatr21/mcp-nvidia.git
cd mcp-nvidia

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"
```

### Running tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest tests/ -v

# Run tests excluding slow tests (rate limiting)
pytest tests/ -v -m "not slow"
```

## Architecture

The server uses the Model Context Protocol (MCP) to expose search functionality to LLMs.

### Search Flow

```mermaid
sequenceDiagram
    participant LLM as LLM/AI Agent
    participant MCP as MCP Server
    participant Validator as Input Validator
    participant RateLimit as Rate Limiter
    participant DDGS as DuckDuckGo Search
    participant Fetcher as URL Fetcher

    LLM->>MCP: search_nvidia(query, domains)
    MCP->>Validator: Validate inputs
    Validator-->>MCP: ✓ Valid (or ✗ Error)

    loop For each domain (max 5 concurrent)
        MCP->>RateLimit: Check rate limit
        RateLimit-->>MCP: ✓ Proceed (wait 200ms)
        MCP->>DDGS: Search "site:domain query"
        DDGS-->>MCP: Search results

        loop For each result
            MCP->>Fetcher: Fetch URL context
            Fetcher-->>MCP: Enhanced snippet
        end
    end

    MCP->>MCP: Calculate relevance scores
    MCP->>MCP: Sort & format JSON
    MCP-->>LLM: Structured JSON response
```

### System Architecture

```mermaid
flowchart TD
    A[MCP Client<br/>Claude/LLMs] -->|JSON-RPC| B[MCP Server]
    B --> C{Tool Router}

    C -->|search_nvidia| D[Search Handler]
    C -->|discover_nvidia_content| E[Discovery Handler]

    D --> F[Security Layer]
    F --> G[Input Validator]
    F --> H[Rate Limiter]
    F --> I[Concurrency Control]

    G --> J[Domain Searcher]
    J --> K[DuckDuckGo API]

    J --> L[URL Context Fetcher]
    L --> M[BeautifulSoup Parser]

    J --> N[Relevance Scorer]
    N --> O[Response Builder]
    O --> P[Structured JSON]

    P -->|CallToolResult| A

    style F fill:#ffcccc
    style P fill:#ccffcc
    style K fill:#cce5ff
```

### Key Components

1. **Input Validation**: Validates query length, domain whitelist, and parameter limits
2. **Rate Limiting**: Enforces 200ms minimum interval between DuckDuckGo API calls
3. **Concurrent Search**: Searches up to 5 domains simultaneously with semaphore control
4. **Context Enhancement**: Fetches actual page content and highlights relevant snippets
5. **Relevance Scoring**: Calculates 0-100 scores based on keyword matches
6. **JSON Output**: Returns structured data compatible with both AI agents and humans

## Extending Domain Coverage

The list of searchable domains is configured in `src/mcp_nvidia/server.py` in the `DEFAULT_DOMAINS` constant.
To add more NVIDIA domains:

1. Edit `src/mcp_nvidia/server.py`
2. Add new domain URLs to the `DEFAULT_DOMAINS` list
3. Reinstall the package

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/bharatr21/mcp-nvidia).
