"""Integration tests for the MCP NVIDIA server."""

import pytest
from unittest.mock import AsyncMock, patch
from mcp_nvidia.server import search_all_domains


@pytest.mark.asyncio
async def test_search_all_domains_with_mock():
    """Test search across all domains with mocked HTTP calls."""
    mock_results = [{
        "title": "Test CUDA Result",
        "url": "https://developer.nvidia.com/cuda",
        "snippet": "CUDA programming guide",
        "domain": "developer.nvidia.com"
    }]
    
    with patch('mcp_nvidia.server.search_nvidia_domain', new=AsyncMock(return_value=mock_results)):
        results = await search_all_domains(
            query="CUDA",
            max_results_per_domain=1
        )
        
        # Should have results from all domains
        assert len(results) > 0
        
        # Each result should have required fields
        for result in results:
            assert "title" in result
            assert "domain" in result


@pytest.mark.asyncio
async def test_search_specific_domains_with_mock():
    """Test search with specific domain list and mocked HTTP calls."""
    mock_results = [{
        "title": "Test GPU Result",
        "url": "https://developer.nvidia.com/gpu",
        "snippet": "GPU architecture",
        "domain": "developer.nvidia.com"
    }]
    
    with patch('mcp_nvidia.server.search_nvidia_domain', new=AsyncMock(return_value=mock_results)):
        results = await search_all_domains(
            query="GPU",
            domains=["https://developer.nvidia.com/"],
            max_results_per_domain=1
        )
        
        # Should have at least one result
        assert len(results) > 0
        
        # Check domain is correct
        for result in results:
            assert "developer.nvidia.com" in result.get("domain", "")
