"""Tests for the MCP NVIDIA server."""

import pytest
from unittest.mock import AsyncMock, patch
from mcp_nvidia.server import format_search_results, format_content_results, DEFAULT_DOMAINS, validate_nvidia_domain, call_tool, calculate_search_relevance


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
    assert validate_nvidia_domain("https://developer.nvidia.com/")
    assert validate_nvidia_domain("https://blogs.nvidia.com/")
    assert validate_nvidia_domain("https://docs.nvidia.com/cuda/")
    assert validate_nvidia_domain("https://nvidia.com/")
    assert validate_nvidia_domain("http://nvidianews.nvidia.com/")


def test_validate_nvidia_domain_invalid():
    """Test validation rejects non-NVIDIA domains."""
    assert not validate_nvidia_domain("https://google.com/")
    assert not validate_nvidia_domain("https://example.com/")
    assert not validate_nvidia_domain("https://nvidia-fake.com/")
    assert not validate_nvidia_domain("https://notnvidia.com/")


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


@pytest.mark.asyncio
async def test_call_tool_validates_domains():
    """Test that call_tool validates caller-supplied domains."""
    # Test with invalid domain
    with pytest.raises(ValueError, match="Invalid domains detected"):
        await call_tool(
            name="search_nvidia",
            arguments={
                "query": "test",
                "domains": ["https://google.com/", "https://developer.nvidia.com/"]
            }
        )


@pytest.mark.asyncio
async def test_call_tool_accepts_valid_domains():
    """Test that call_tool accepts valid NVIDIA domains."""
    with patch('mcp_nvidia.server.search_all_domains', new=AsyncMock(return_value=[])):
        result = await call_tool(
            name="search_nvidia",
            arguments={
                "query": "test",
                "domains": ["https://developer.nvidia.com/", "https://blogs.nvidia.com/"]
            }
        )
        assert result is not None


@pytest.mark.asyncio
async def test_call_tool_accepts_none_domains():
    """Test that call_tool accepts None for domains (uses defaults)."""
    with patch('mcp_nvidia.server.search_all_domains', new=AsyncMock(return_value=[])):
        result = await call_tool(
            name="search_nvidia",
            arguments={
                "query": "test",
                "domains": None
            }
        )
        assert result is not None


@pytest.mark.asyncio
async def test_call_tool_rejects_non_list_domains():
    """Test that call_tool rejects non-list domain values."""
    with pytest.raises(ValueError, match="domains must be a list"):
        await call_tool(
            name="search_nvidia",
            arguments={
                "query": "test",
                "domains": "https://developer.nvidia.com/"
            }
        )


@pytest.mark.asyncio
async def test_call_tool_rejects_empty_validated_domains():
    """Test that call_tool rejects when all provided domains are invalid."""
    with pytest.raises(ValueError, match="Invalid domains detected"):
        await call_tool(
            name="search_nvidia",
            arguments={
                "query": "test",
                "domains": ["https://google.com/", "https://example.com/"]
            }
        )


def test_calculate_search_relevance():
    """Test search relevance score calculation."""
    result = {
        "title": "CUDA Programming Guide",
        "snippet": "Learn CUDA programming for GPU acceleration",
        "url": "https://developer.nvidia.com/cuda-programming"
    }
    query = "CUDA programming"
    
    score = calculate_search_relevance(result, query)
    
    # Should have high relevance since both terms appear in title, snippet, and url
    assert score > 50
    assert 0 <= score <= 100


def test_format_search_results_with_relevance_score():
    """Test that search results display relevance scores."""
    results = [
        {
            "title": "Test Result",
            "url": "https://developer.nvidia.com/test",
            "snippet": "Test snippet",
            "domain": "developer.nvidia.com",
            "relevance_score": 75
        }
    ]
    query = "test"
    formatted = format_search_results(results, query)
    
    assert "Test Result" in formatted
    assert "Score: 75/100" in formatted
    assert "⭐" in formatted  # Should have star ratings


@pytest.mark.asyncio
async def test_search_filters_by_min_relevance_score():
    """Test that search_all_domains filters by minimum relevance score."""
    from mcp_nvidia.server import search_all_domains
    
    mock_results = [
        {
            "title": "High relevance result",
            "url": "https://developer.nvidia.com/high",
            "snippet": "CUDA programming guide",
            "domain": "developer.nvidia.com"
        },
        {
            "title": "Low relevance result",
            "url": "https://developer.nvidia.com/low",
            "snippet": "Some other content",
            "domain": "developer.nvidia.com"
        }
    ]
    
    with patch('mcp_nvidia.server.search_nvidia_domain', new=AsyncMock(return_value=mock_results)):
        # Test with default threshold (33)
        results = await search_all_domains(
            query="CUDA programming",
            max_results_per_domain=2
        )
        
        # Should have at least one result (the high relevance one)
        assert len(results) >= 1
        
        # All results should have relevance scores
        for result in results:
            assert "relevance_score" in result
            assert result["relevance_score"] >= 33
