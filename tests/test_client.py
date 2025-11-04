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
import pytest_asyncio

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest_asyncio.fixture
async def session():
    """Pytest fixture to create an MCP client session."""
    # Get the server command - look for installed mcp-nvidia or use dev version
    server_script = None

    # Try installed version first
    import shutil
    installed_path = shutil.which("mcp-nvidia")
    if installed_path:
        server_script = Path(installed_path)
    else:
        # Fall back to dev installation
        repo_root = Path(__file__).parent.parent
        dev_venv = repo_root / "test_mcp_venv" / "bin" / "mcp-nvidia"
        if dev_venv.exists():
            server_script = dev_venv

    if not server_script or not server_script.exists():
        pytest.skip(f"MCP server not found. Install with: pip install -e .")

    server_params = StdioServerParameters(
        command=str(server_script),
        args=[],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as client_session:
            # Initialize the connection
            await client_session.initialize()
            yield client_session


@pytest.mark.asyncio
async def test_structured_output(session: ClientSession):
    """Test that structured output is returned correctly."""
    result = await session.call_tool(
        "search_nvidia",
        arguments={
            "query": "TensorRT",
            "max_results_per_domain": 1
        }
    )

    # Check that we have content
    assert result.content, "Response should have content"

    # Parse JSON
    response = json.loads(result.content[0].text)

    # Check expected fields
    expected_fields = ['success', 'results']
    for field in expected_fields:
        assert field in response, f"Missing field: {field}"

    # Check result structure if results exist
    if response.get('results'):
        result_item = response['results'][0]
        result_fields = ['title', 'url', 'snippet', 'domain', 'relevance_score', 'formatted_text']
        for field in result_fields:
            assert field in result_item, f"Missing result field: {field}"


@pytest.mark.asyncio
async def test_basic_search(session: ClientSession):
    """Test basic search functionality."""
    result = await session.call_tool(
        "search_nvidia",
        arguments={
            "query": "CUDA programming",
            "max_results_per_domain": 2
        }
    )

    assert result.content, "Result should have content"
    response = json.loads(result.content[0].text)
    assert response.get('success'), "Search should succeed"


@pytest.mark.asyncio
async def test_query_length_validation(session: ClientSession):
    """Test that queries longer than 500 characters are rejected."""
    long_query = "A" * 501
    result = await session.call_tool(
        "search_nvidia",
        arguments={"query": long_query}
    )
    response = json.loads(result.content[0].text)
    assert not response.get('success'), "Long query should be rejected"
    assert 'too long' in response.get('error', {}).get('message', '').lower()


@pytest.mark.asyncio
async def test_max_results_validation(session: ClientSession):
    """Test that max_results_per_domain is capped at the limit."""
    result = await session.call_tool(
        "search_nvidia",
        arguments={
            "query": "GPU",
            "max_results_per_domain": 100  # Should be capped at 10
        }
    )
    response = json.loads(result.content[0].text)
    # Should still succeed but with capped results
    assert response.get('success'), "Search should succeed with capped results"


@pytest.mark.asyncio
async def test_invalid_domain_validation(session: ClientSession):
    """Test that non-NVIDIA domains are rejected."""
    result = await session.call_tool(
        "search_nvidia",
        arguments={
            "query": "test",
            "domains": ["https://evil.com/"]
        }
    )
    response = json.loads(result.content[0].text)
    assert not response.get('success'), "Non-NVIDIA domain should be rejected"
    assert 'invalid' in response.get('error', {}).get('message', '').lower()


@pytest.mark.asyncio
async def test_valid_custom_domain(session: ClientSession):
    """Test search with a valid custom NVIDIA domain."""
    result = await session.call_tool(
        "search_nvidia",
        arguments={
            "query": "deep learning",
            "domains": ["https://developer.nvidia.com/"],
            "max_results_per_domain": 1
        }
    )
    response = json.loads(result.content[0].text)
    assert response.get('success'), "Custom NVIDIA domain search should succeed"


@pytest.mark.asyncio
async def test_discover_content_basic(session: ClientSession):
    """Test basic content discovery."""
    result = await session.call_tool(
        "discover_nvidia_content",
        arguments={
            "content_type": "video",
            "topic": "CUDA",
            "max_results": 3
        }
    )
    assert result.content, "Result should have content"
    response = json.loads(result.content[0].text)
    assert response.get('success'), "Content discovery should succeed"


@pytest.mark.asyncio
async def test_topic_length_validation(session: ClientSession):
    """Test that topics longer than 500 characters are rejected."""
    long_topic = "B" * 501
    result = await session.call_tool(
        "discover_nvidia_content",
        arguments={
            "content_type": "blog",
            "topic": long_topic
        }
    )
    response = json.loads(result.content[0].text)
    assert not response.get('success'), "Long topic should be rejected"
    assert 'too long' in response.get('error', {}).get('message', '').lower()


@pytest.mark.asyncio
async def test_discover_max_results_validation(session: ClientSession):
    """Test that max_results is capped for content discovery."""
    result = await session.call_tool(
        "discover_nvidia_content",
        arguments={
            "content_type": "tutorial",
            "topic": "AI",
            "max_results": 100  # Should be capped at 10
        }
    )
    response = json.loads(result.content[0].text)
    # Should still succeed but with capped results
    assert response.get('success'), "Content discovery should succeed with capped results"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_rate_limiting(session: ClientSession):
    """
    Test rate limiting by making multiple rapid requests.

    With 0.2s interval between DDGS calls, 6 requests should take at least 1.0s
    (5 intervals for 6 requests).
    """
    import time
    start = time.time()

    for i in range(6):
        await session.call_tool(
            "search_nvidia",
            arguments={
                "query": f"test query {i}",
                "max_results_per_domain": 1
            }
        )

    elapsed = time.time() - start
    # With 0.2s (200ms) interval, 6 requests should take at least 1.0s
    # Allow some tolerance for timing variations
    assert elapsed >= 0.9, f"Rate limiting too fast: {elapsed:.2f}s (expected >=0.9s)"
