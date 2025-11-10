#!/usr/bin/env python3
"""
Pytest tests for the MCP NVIDIA server.

This module contains integration tests for the server's functionality
and security fixes.
"""

import json
import sys
from pathlib import Path

import pytest

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def get_server_command():
    """Get the MCP server command path."""
    # Try installed version first
    import shutil

    installed_path = shutil.which("mcp-nvidia")
    if installed_path:
        return Path(installed_path)

    # Fall back to dev installation
    repo_root = Path(__file__).parent.parent
    dev_venv = repo_root / "test_mcp_venv" / "bin" / "mcp-nvidia"
    if dev_venv.exists():
        return dev_venv

    return None


class MCPTestSession:
    """Helper class to manage MCP client session for tests."""

    def __init__(self):
        self.server_script = get_server_command()
        if not self.server_script or not self.server_script.exists():
            pytest.skip("MCP server not found. Install with: pip install -e .")

    async def __aenter__(self):
        """Create and initialize the MCP client session."""
        server_params = StdioServerParameters(command=str(self.server_script), args=[], env=None)

        self._stdio_context = stdio_client(server_params)
        read, write = await self._stdio_context.__aenter__()

        self._session_context = ClientSession(read, write)
        self._session = await self._session_context.__aenter__()
        await self._session.initialize()

        return self._session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up the MCP client session."""
        try:
            if hasattr(self, "_session_context"):
                await self._session_context.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if hasattr(self, "_stdio_context"):
                await self._stdio_context.__aexit__(exc_type, exc_val, exc_tb)


@pytest.mark.asyncio
async def test_structured_output():
    """Test that structured output is returned correctly."""
    async with MCPTestSession() as session:
        result = await session.call_tool("search_nvidia", arguments={"query": "TensorRT", "max_results_per_domain": 1})

        # Check that we have content
        assert result.content, "Response should have content"

        # Parse JSON
        response = json.loads(result.content[0].text)

        # Check expected fields
        expected_fields = ["success", "results"]
        for field in expected_fields:
            assert field in response, f"Missing field: {field}"

        # Check result structure if results exist
        if response.get("results"):
            result_item = response["results"][0]
            # Required fields
            required_fields = ["title", "url", "snippet", "domain", "relevance_score", "content_type"]
            for field in required_fields:
                assert field in result_item, f"Missing result field: {field}"

            # Validate content_type has an expected value
            assert result_item["content_type"] in [
                "article",
                "video",
                "blog",
                "tutorial",
                "documentation",
                "other",
            ], f"Unexpected content_type: {result_item['content_type']}"

            # Optional fields (may or may not be present)
            # published_date, metadata are optional and only appear when extracted
            if "published_date" in result_item:
                assert isinstance(result_item["published_date"], str), "published_date should be a string"
                assert result_item["published_date"], "published_date should not be empty"

            if "metadata" in result_item:
                assert isinstance(result_item["metadata"], dict), "metadata should be a dictionary"


@pytest.mark.asyncio
async def test_basic_search():
    """Test basic search functionality."""
    async with MCPTestSession() as session:
        result = await session.call_tool(
            "search_nvidia", arguments={"query": "CUDA programming", "max_results_per_domain": 2}
        )

        assert result.content, "Result should have content"
        response = json.loads(result.content[0].text)
        assert response.get("success"), "Search should succeed"


@pytest.mark.asyncio
async def test_query_length_validation():
    """Test that queries longer than 500 characters are rejected."""
    async with MCPTestSession() as session:
        long_query = "A" * 501
        result = await session.call_tool("search_nvidia", arguments={"query": long_query})
        response = json.loads(result.content[0].text)
        assert not response.get("success"), "Long query should be rejected"
        assert "too long" in response.get("error", {}).get("message", "").lower()


@pytest.mark.asyncio
async def test_max_results_validation():
    """Test that max_results_per_domain is capped at the limit."""
    async with MCPTestSession() as session:
        result = await session.call_tool(
            "search_nvidia",
            arguments={
                "query": "GPU",
                "max_results_per_domain": 100,  # Should be capped at 10
            },
        )
        response = json.loads(result.content[0].text)
        # Should still succeed but with capped results
        assert response.get("success"), "Search should succeed with capped results"


@pytest.mark.asyncio
async def test_invalid_domain_validation():
    """Test that non-NVIDIA domains are rejected."""
    async with MCPTestSession() as session:
        result = await session.call_tool("search_nvidia", arguments={"query": "test", "domains": ["https://evil.com/"]})
        response = json.loads(result.content[0].text)
        assert not response.get("success"), "Non-NVIDIA domain should be rejected"
        assert "invalid" in response.get("error", {}).get("message", "").lower()


@pytest.mark.asyncio
async def test_valid_custom_domain():
    """Test search with a valid custom NVIDIA domain."""
    async with MCPTestSession() as session:
        result = await session.call_tool(
            "search_nvidia",
            arguments={
                "query": "deep learning",
                "domains": ["https://developer.nvidia.com/"],
                "max_results_per_domain": 1,
            },
        )
        response = json.loads(result.content[0].text)
        assert response.get("success"), "Custom NVIDIA domain search should succeed"


@pytest.mark.asyncio
async def test_discover_content_basic():
    """Test basic content discovery."""
    async with MCPTestSession() as session:
        result = await session.call_tool(
            "discover_nvidia_content", arguments={"content_type": "video", "topic": "CUDA", "max_results": 3}
        )
        assert result.content, "Result should have content"
        response = json.loads(result.content[0].text)
        assert response.get("success"), "Content discovery should succeed"


@pytest.mark.asyncio
async def test_topic_length_validation():
    """Test that topics longer than 500 characters are rejected."""
    async with MCPTestSession() as session:
        long_topic = "B" * 501
        result = await session.call_tool(
            "discover_nvidia_content", arguments={"content_type": "blog", "topic": long_topic}
        )
        response = json.loads(result.content[0].text)
        assert not response.get("success"), "Long topic should be rejected"
        assert "too long" in response.get("error", {}).get("message", "").lower()


@pytest.mark.asyncio
async def test_discover_max_results_validation():
    """Test that max_results is capped for content discovery."""
    async with MCPTestSession() as session:
        result = await session.call_tool(
            "discover_nvidia_content",
            arguments={
                "content_type": "tutorial",
                "topic": "AI",
                "max_results": 100,  # Should be capped at 10
            },
        )
        response = json.loads(result.content[0].text)
        # Should still succeed but with capped results
        assert response.get("success"), "Content discovery should succeed with capped results"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_rate_limiting():
    """
    Test rate limiting by making multiple rapid requests.

    With 0.2s interval between DDGS calls, 6 requests should take at least 1.0s
    (5 intervals for 6 requests).
    """
    import time

    async with MCPTestSession() as session:
        start = time.time()

        for i in range(6):
            await session.call_tool(
                "search_nvidia", arguments={"query": f"test query {i}", "max_results_per_domain": 1}
            )

        elapsed = time.time() - start
        # With 0.2s (200ms) interval, 6 requests should take at least 1.0s
        # Allow some tolerance for timing variations
        assert elapsed >= 0.9, f"Rate limiting too fast: {elapsed:.2f}s (expected >=0.9s)"


def test_ad_url_blocking():
    """Test that ad URLs are correctly identified and blocked."""
    from mcp_nvidia.server import is_ad_url

    # Test DuckDuckGo ad URLs
    assert is_ad_url("https://duckduckgo.com/y.js?ad_domain=wyzant.com&ad_provider=bingv7aa&ad_type=txad")
    assert is_ad_url("https://duckduckgo.com/y.js?ad_domain=example.com")

    # Test URLs with ad parameters
    assert is_ad_url("https://example.com/?ad_domain=test.com")
    assert is_ad_url("https://example.com/?ad_provider=google")
    assert is_ad_url("https://example.com/?ad_type=banner")
    assert is_ad_url("https://example.com/?adurl=https://ad.com")
    assert is_ad_url("https://example.com/?adclick=123")

    # Test valid URLs should not be blocked
    assert not is_ad_url("https://developer.nvidia.com/cuda")
    assert not is_ad_url("https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/")
    assert not is_ad_url("https://nvidia.github.io/cuda-python/")


def test_keyword_extraction():
    """Test that keywords are correctly extracted from queries."""
    from mcp_nvidia.server import extract_keywords

    # Test basic keyword extraction
    keywords = extract_keywords("How to install CUDA for deep learning")
    assert "cuda" in keywords
    assert "deep" in keywords
    assert "learning" in keywords
    assert "install" in keywords
    # Stopwords should be filtered out
    assert "how" not in keywords
    assert "to" not in keywords
    assert "for" not in keywords

    # Test with technical terms
    keywords = extract_keywords("TensorRT optimization guide")
    assert "tensorrt" in keywords
    assert "optimization" in keywords
    assert "guide" in keywords

    # Test with mixed case
    keywords = extract_keywords("NVIDIA GPU Programming")
    assert "nvidia" in keywords
    assert "gpu" in keywords
    assert "programming" in keywords

    # Test empty or stopword-only queries
    keywords = extract_keywords("the and or")
    assert len(keywords) == 0

    # Test with punctuation
    keywords = extract_keywords("What is CUDA? How does it help?")
    assert "cuda" in keywords
    assert "help" in keywords
    # Stopwords should be filtered out
    assert "what" not in keywords
    assert "is" not in keywords
    assert "how" not in keywords
    assert "does" not in keywords
    assert "it" not in keywords


@pytest.mark.asyncio
async def test_search_results_no_ads():
    """Test that search results don't contain ad URLs."""
    async with MCPTestSession() as session:
        result = await session.call_tool("search_nvidia", arguments={"query": "CUDA", "max_results_per_domain": 5})

        assert result.content, "Result should have content"
        response = json.loads(result.content[0].text)

        # Check that no results contain ad URLs
        if response.get("results"):
            for result_item in response["results"]:
                url = result_item.get("url", "")
                # Import the function to check
                from mcp_nvidia.server import is_ad_url

                assert not is_ad_url(url), f"Ad URL found in results: {url}"
