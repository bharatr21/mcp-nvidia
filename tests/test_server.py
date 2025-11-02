"""Tests for the MCP NVIDIA server."""

import pytest
from mcp_nvidia.server import format_search_results, format_content_results, DEFAULT_DOMAINS


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


def test_format_search_results_includes_citations():
    """Test that search results include a citations section."""
    results = [
        {
            "title": "Test Result",
            "url": "https://developer.nvidia.com/test",
            "snippet": "Test snippet"
        }
    ]
    query = "test"
    formatted = format_search_results(results, query)
    
    assert "CITATIONS:" in formatted
    assert "[1]" in formatted
    assert "https://developer.nvidia.com/test" in formatted


def test_format_content_results_with_data():
    """Test formatting content discovery results."""
    results = [
        {
            "title": "CUDA Tutorial",
            "url": "https://developer.nvidia.com/cuda-tutorial",
            "snippet": "Learn CUDA programming",
            "domain": "developer.nvidia.com",
            "relevance_score": 5
        }
    ]
    content_type = "tutorial"
    topic = "CUDA"
    formatted = format_content_results(results, content_type, topic)
    
    assert "CUDA Tutorial" in formatted
    assert "https://developer.nvidia.com/cuda-tutorial" in formatted
    assert "RESOURCE LINKS:" in formatted
    assert "⭐" in formatted  # Should include relevance stars


def test_format_content_results_empty():
    """Test formatting empty content discovery results."""
    results = []
    content_type = "video"
    topic = "AI"
    formatted = format_content_results(results, content_type, topic)
    
    assert "No video content found" in formatted
    assert topic in formatted
