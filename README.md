# mcp-nvidia

MCP server to search across NVIDIA blogs and releases to empower LLMs to better answer NVIDIA specific queries.

## Overview

This Model Context Protocol (MCP) server enables Large Language Models (LLMs) to search across multiple NVIDIA domains to find relevant information about NVIDIA technologies, products, and services. The server searches across:

- **developer.nvidia.com** - Developer resources, SDKs, and technical documentation
- **blogs.nvidia.com** - NVIDIA blog posts and articles
- **nvidianews.nvidia.com** - Official NVIDIA news and press releases
- **docs.nvidia.com** - Comprehensive technical documentation
- **build.nvidia.com** - NVIDIA AI Foundation models and services

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
      "args": ["-m", "mcp_nvidia.server"]
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

**Example queries:**
- "CUDA programming best practices"
- "RTX 4090 specifications"
- "TensorRT optimization techniques"
- "Latest AI announcements"
- "Omniverse development tutorials"

**Features:**
- Concurrent search across multiple domains for fast results
- Formatted results with titles, URLs, snippets, and source domains
- Dedicated citations section with numbered references for easy copying

### discover_nvidia_content

Discover specific types of NVIDIA educational and learning content such as videos, courses, tutorials, webinars, or blog posts.

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
- **Relevance scoring on 0-100 scale** with star ratings (⭐) to highlight best matches
  - Score displayed as "Score: X/100" for transparency
  - Stars: 0-19 = ⭐, 20-39 = ⭐⭐, 40-59 = ⭐⭐⭐, 60-79 = ⭐⭐⭐⭐, 80-100 = ⭐⭐⭐⭐⭐
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
pytest tests/
```

## Architecture

The server uses the Model Context Protocol (MCP) to expose search functionality to LLMs. When a search is requested:

1. The query is distributed across all configured NVIDIA domains
2. Each domain is searched concurrently for efficiency
3. Results are aggregated and formatted
4. The formatted results are returned to the LLM for processing

## Extending Domain Coverage

The list of searchable domains is configured in `src/mcp_nvidia/server.py` in the `DEFAULT_DOMAINS` constant. To add more NVIDIA domains:

1. Edit `src/mcp_nvidia/server.py`
2. Add new domain URLs to the `DEFAULT_DOMAINS` list
3. Reinstall the package

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/bharatr21/mcp-nvidia).
