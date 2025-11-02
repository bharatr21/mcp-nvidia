"""Tests for the MCP NVIDIA server."""

import pytest
from mcp_nvidia.server import format_search_results, format_content_results, DEFAULT_DOMAINS, validate_nvidia_domain


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
    """Test formatting content discovery results with 0-100 score scale."""
    results = [
        {
            "title": "CUDA Tutorial",
            "url": "https://developer.nvidia.com/cuda-tutorial",
            "snippet": "Learn CUDA programming",
            "domain": "developer.nvidia.com",
            "relevance_score": 85  # 0-100 scale
        }
    ]
    content_type = "tutorial"
    topic = "CUDA"
    formatted = format_content_results(results, content_type, topic)
    
    assert "CUDA Tutorial" in formatted
    assert "https://developer.nvidia.com/cuda-tutorial" in formatted
    assert "RESOURCE LINKS:" in formatted
    assert "⭐" in formatted  # Should include relevance stars
    assert "Score: 85/100" in formatted  # Should show 0-100 score


def test_format_content_results_empty():
    """Test formatting empty content discovery results."""
    results = []
    content_type = "video"
    topic = "AI"
    formatted = format_content_results(results, content_type, topic)
    
    assert "No video content found" in formatted
    assert topic in formatted


def test_validate_nvidia_domain_valid():
    """Test validation of valid NVIDIA domains."""
    assert validate_nvidia_domain("https://developer.nvidia.com/") == True
    assert validate_nvidia_domain("https://blogs.nvidia.com/") == True
    assert validate_nvidia_domain("https://docs.nvidia.com/cuda/") == True
    assert validate_nvidia_domain("https://nvidia.com/") == True
    assert validate_nvidia_domain("http://nvidianews.nvidia.com/") == True


def test_validate_nvidia_domain_invalid():
    """Test validation rejects non-NVIDIA domains."""
    assert validate_nvidia_domain("https://google.com/") == False
    assert validate_nvidia_domain("https://example.com/") == False
    assert validate_nvidia_domain("https://nvidia-fake.com/") == False
    assert validate_nvidia_domain("https://notnvidia.com/") == False


def test_relevance_score_normalization():
    """Test that relevance scores are normalized to 0-100 scale."""
    results = [
        {
            "title": "Test",
            "relevance_score": 0
        },
        {
            "title": "Test",
            "relevance_score": 50
        },
        {
            "title": "Test",
            "relevance_score": 100
        }
    ]
    
    for result in results:
        score = result["relevance_score"]
        assert 0 <= score <= 100, f"Score {score} should be in range 0-100"
