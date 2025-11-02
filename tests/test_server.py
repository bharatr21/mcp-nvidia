"""Tests for the MCP NVIDIA server."""

import pytest
from mcp_nvidia.server import format_search_results, DEFAULT_DOMAINS


def test_format_search_results_empty():
    """Test formatting empty search results."""
    results = []
    query = "test query"
    formatted = format_search_results(results, query)
    assert "No results found" in formatted
    assert query in formatted


def test_format_search_results_with_data():
    """Test formatting search results with data."""
    results = [
        {
            "title": "Test Title",
            "url": "https://example.com",
            "snippet": "Test snippet",
            "domain": "example.com"
        }
    ]
    query = "test query"
    formatted = format_search_results(results, query)
    
    assert "Test Title" in formatted
    assert "https://example.com" in formatted
    assert "Test snippet" in formatted
    assert query in formatted


def test_default_domains_configured():
    """Test that default NVIDIA domains are configured."""
    assert len(DEFAULT_DOMAINS) > 0
    assert any("developer.nvidia.com" in domain for domain in DEFAULT_DOMAINS)
    assert any("blogs.nvidia.com" in domain for domain in DEFAULT_DOMAINS)
    assert any("docs.nvidia.com" in domain for domain in DEFAULT_DOMAINS)


def test_format_search_results_handles_missing_fields():
    """Test formatting results with missing optional fields."""
    results = [
        {
            "title": "Test Title",
        }
    ]
    query = "test"
    formatted = format_search_results(results, query)
    
    assert "Test Title" in formatted
    # Should not crash even with missing fields
