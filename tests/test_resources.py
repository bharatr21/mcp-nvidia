"""Test MCP Resources functionality."""

import pytest

from mcp_nvidia.server import list_resources, read_resource


@pytest.mark.asyncio
async def test_list_resources():
    """Test that list_resources returns all expected SDK files."""
    resources = await list_resources()

    # Should have TypeScript and Python SDK files
    assert len(resources) > 0

    # Convert to list of URIs for easier checking
    resource_uris = [str(r.uri) for r in resources]

    # Check for expected TypeScript files
    assert any("typescript/search_nvidia.ts" in uri for uri in resource_uris)
    assert any("typescript/discover_nvidia_content.ts" in uri for uri in resource_uris)
    assert any("typescript/index.ts" in uri for uri in resource_uris)
    assert any("typescript/README.md" in uri for uri in resource_uris)

    # Check for expected Python files
    assert any("python/search_nvidia.py" in uri for uri in resource_uris)
    assert any("python/discover_nvidia_content.py" in uri for uri in resource_uris)
    assert any("python/__init__.py" in uri for uri in resource_uris)
    assert any("python/README.md" in uri for uri in resource_uris)


@pytest.mark.asyncio
async def test_list_resources_structure():
    """Test that resources have correct structure."""
    resources = await list_resources()

    for resource in resources:
        # Check that all required fields are present
        assert resource.uri is not None
        assert resource.name is not None
        assert resource.mimeType is not None
        assert resource.description is not None

        # Check URI format
        uri_str = str(resource.uri)
        assert uri_str.startswith("mcp-nvidia://sdk/")

        # Check MIME type
        if uri_str.endswith(".ts"):
            assert resource.mimeType == "text/typescript"
        elif uri_str.endswith(".py"):
            assert resource.mimeType == "text/x-python"
        elif uri_str.endswith(".md"):
            assert resource.mimeType == "text/plain"


@pytest.mark.asyncio
async def test_read_typescript_resource():
    """Test reading TypeScript SDK resource."""
    uri = "mcp-nvidia://sdk/typescript/search_nvidia.ts"
    content = await read_resource(uri)

    # Check that content is returned
    assert content is not None
    assert len(content) > 0

    # Check for expected TypeScript content
    assert "SearchNvidiaInput" in content
    assert "SearchNvidiaOutput" in content
    assert "searchNvidia" in content
    assert "export" in content
    assert "interface" in content

    # Check for MCP Client interface
    assert "MCPClient" in content
    assert "callTool" in content
    assert "client: MCPClient" in content


@pytest.mark.asyncio
async def test_read_python_resource():
    """Test reading Python SDK resource."""
    uri = "mcp-nvidia://sdk/python/search_nvidia.py"
    content = await read_resource(uri)

    # Check that content is returned
    assert content is not None
    assert len(content) > 0

    # Check for expected Python content
    assert "SearchNvidiaInput" in content
    assert "SearchNvidiaOutput" in content
    assert "search_nvidia" in content
    assert "TypedDict" in content
    assert "async def" in content

    # Check for direct implementation
    assert "from mcp_nvidia.lib import" in content
    assert "search_all_domains" in content
    assert "build_search_response_json" in content


@pytest.mark.asyncio
async def test_read_readme_resource():
    """Test reading README resource."""
    # Test TypeScript README
    ts_readme = await read_resource("mcp-nvidia://sdk/typescript/README.md")
    assert "NVIDIA MCP TypeScript SDK" in ts_readme
    assert "search_nvidia" in ts_readme

    # Test Python README
    py_readme = await read_resource("mcp-nvidia://sdk/python/README.md")
    assert "NVIDIA MCP Python SDK" in py_readme
    assert "search_nvidia" in py_readme


@pytest.mark.asyncio
async def test_read_discover_content_resource():
    """Test reading discover_nvidia_content resources."""
    # TypeScript version
    ts_content = await read_resource("mcp-nvidia://sdk/typescript/discover_nvidia_content.ts")
    assert "DiscoverNvidiaContentInput" in ts_content
    assert "discoverNvidiaContent" in ts_content

    # Python version
    py_content = await read_resource("mcp-nvidia://sdk/python/discover_nvidia_content.py")
    assert "DiscoverNvidiaContentInput" in py_content
    assert "discover_nvidia_content" in py_content


@pytest.mark.asyncio
async def test_read_invalid_resource():
    """Test that reading an invalid resource raises an error."""
    with pytest.raises(ValueError, match="Invalid resource URI"):
        await read_resource("invalid://uri")

    with pytest.raises(ValueError, match="Invalid resource URI"):
        await read_resource("mcp-nvidia://invalid/path")


@pytest.mark.asyncio
async def test_read_nonexistent_resource():
    """Test that reading a non-existent resource raises an error."""
    with pytest.raises(ValueError, match="SDK file not found"):
        await read_resource("mcp-nvidia://sdk/typescript/nonexistent.ts")


@pytest.mark.asyncio
async def test_read_invalid_language():
    """Test that an invalid language raises an error."""
    with pytest.raises(ValueError, match="Unknown SDK language"):
        await read_resource("mcp-nvidia://sdk/invalid_lang/search_nvidia.ts")


@pytest.mark.asyncio
async def test_resource_caching():
    """Test that SDK resources are cached and reused."""
    # First call should generate cache
    resources1 = await list_resources()
    content1 = await read_resource("mcp-nvidia://sdk/typescript/search_nvidia.ts")

    # Second call should use cache
    resources2 = await list_resources()
    content2 = await read_resource("mcp-nvidia://sdk/typescript/search_nvidia.ts")

    # Results should be identical
    assert len(resources1) == len(resources2)
    assert content1 == content2


@pytest.mark.asyncio
async def test_all_resources_readable():
    """Test that all listed resources can be read."""
    resources = await list_resources()

    for resource in resources:
        uri = str(resource.uri)
        # Should not raise an exception
        content = await read_resource(uri)
        assert content is not None
        assert len(content) > 0
