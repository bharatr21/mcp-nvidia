"""Tests for MCP-UI rendering functionality."""

from mcp_nvidia.ui.components import (
    CONTENT_TYPE_ICONS,
    render_citations,
    render_content_card,
    render_content_container,
    render_content_type_tabs,
    render_filter_panel,
    render_result_card,
    render_results_container,
    render_search_header,
    render_warnings,
)
from mcp_nvidia.ui.styles import STYLES
from mcp_nvidia.ui.templates import render_content_ui, render_error_ui, render_filter_fragment, render_search_ui


class TestStyles:
    """Tests for CSS styles."""

    def test_styles_defined(self):
        """Verify styles are defined."""
        assert isinstance(STYLES, str)
        assert len(STYLES) > 0

    def test_styles_contain_nvidia_colors(self):
        """Verify styles contain NVIDIA brand colors."""
        assert "#76b900" in STYLES


class TestSearchHeader:
    """Tests for search header component."""

    def test_render_search_header(self):
        """Test search header rendering."""
        html = render_search_header(query="CUDA programming")
        assert "NVIDIA" in html
        assert "MCP Search" in html
        assert "CUDA programming" in html


class TestFilterPanel:
    """Tests for filter panel component."""

    def test_render_filter_panel(self):
        """Test filter panel rendering."""
        html = render_filter_panel(
            query="CUDA",
            sort_by="relevance",
            min_relevance_score=50,
            total_results=10,
            search_time_ms=500,
        )
        assert 'name="sort_by"' in html
        assert 'name="min_relevance_score"' in html
        assert 'value="50"' in html
        assert 'hx-get="/ui/filter"' in html

    def test_filter_panel_sort_options(self):
        """Test sort options in filter panel."""
        html = render_filter_panel(query="test", sort_by="date", min_relevance_score=25)
        assert 'value="relevance"' in html
        assert 'value="date"' in html
        assert 'value="domain"' in html


class TestResultCard:
    """Tests for result card component."""

    def test_render_result_card_full(self):
        """Test result card with all fields."""
        result = {
            "title": "CUDA Programming Guide",
            "url": "https://docs.nvidia.com/cuda/cuda-programming-guide/",
            "snippet": "The CUDA programming model enables parallel computing.",
            "domain": "docs.nvidia.com",
            "content_type": "documentation",
            "relevance_score": 87,
            "published_date": "2024-11-15",
            "matched_keywords": ["cuda", "programming"],
        }
        html = render_result_card(result, index=1)
        assert "CUDA Programming Guide" in html
        assert "87" in html
        assert "docs.nvidia.com" in html
        assert "documentation" in html.lower()
        assert "2024-11-15" in html
        assert "cuda" in html.lower()
        assert "programming" in html.lower()

    def test_render_result_card_minimal(self):
        """Test result card with minimal fields."""
        result = {
            "title": "Test Result",
            "url": "https://example.com",
            "snippet": "Test snippet",
            "domain": "example.com",
            "content_type": "article",
            "relevance_score": 50,
        }
        html = render_result_card(result, index=1)
        assert "Test Result" in html
        assert "50" in html

    def test_content_type_icons(self):
        """Test content type icon mapping."""
        assert CONTENT_TYPE_ICONS["video"] == "🎬"
        assert CONTENT_TYPE_ICONS["tutorial"] == "📖"
        assert CONTENT_TYPE_ICONS["course"] == "📚"
        assert CONTENT_TYPE_ICONS["documentation"] == "📄"
        assert "unknown" not in CONTENT_TYPE_ICONS


class TestResultsContainer:
    """Tests for results container component."""

    def test_render_results_container_with_results(self):
        """Test rendering multiple results."""
        results = [
            {
                "title": "Result 1",
                "url": "http://1.com",
                "snippet": "Snippet 1",
                "domain": "d1.com",
                "content_type": "article",
                "relevance_score": 80,
            },
            {
                "title": "Result 2",
                "url": "http://2.com",
                "snippet": "Snippet 2",
                "domain": "d2.com",
                "content_type": "blog",
                "relevance_score": 70,
            },
        ]
        html = render_results_container(results)
        assert "Result 1" in html
        assert "Result 2" in html

    def test_render_results_container_empty(self):
        """Test rendering empty results."""
        html = render_results_container([])
        assert "No results found" in html


class TestCitations:
    """Tests for citations component."""

    def test_render_citations(self):
        """Test citations rendering."""
        citations = [
            {"number": 1, "title": "First Source", "url": "http://1.com", "domain": "source1.com"},
            {"number": 2, "title": "Second Source", "url": "http://2.com", "domain": "source2.com"},
        ]
        html = render_citations(citations)
        assert "[1]" in html
        assert "[2]" in html
        assert "First Source" in html
        assert "Second Source" in html

    def test_render_citations_empty(self):
        """Test empty citations."""
        html = render_citations([])
        assert html == ""


class TestContentTypeTabs:
    """Tests for content type tabs component."""

    def test_render_content_type_tabs(self):
        """Test content type tabs rendering."""
        html = render_content_type_tabs(content_type="video", topic="CUDA")
        assert "Videos" in html
        assert "Courses" in html
        assert "Tutorials" in html
        assert 'hx-get="/ui/content"' in html


class TestContentCard:
    """Tests for content card component."""

    def test_render_content_card(self):
        """Test content card rendering."""
        content = {
            "title": "CUDA Tutorial Video",
            "url": "https://youtube.com/watch?v=123",
            "content_type": "video",
            "snippet": "Learn CUDA basics",
            "domain": "youtube.com",
            "relevance_score": 85,
        }
        html = render_content_card(content)
        assert "CUDA Tutorial Video" in html
        assert "85" in html
        assert "youtube.com" in html


class TestContentContainer:
    """Tests for content container component."""

    def test_render_content_container_with_results(self):
        """Test rendering multiple content items."""
        content = [
            {
                "title": "Video 1",
                "url": "http://1.com",
                "content_type": "video",
                "snippet": "Snippet",
                "domain": "d1.com",
                "relevance_score": 80,
            },
        ]
        html = render_content_container(content)
        assert "Video 1" in html

    def test_render_content_container_empty(self):
        """Test rendering empty content."""
        html = render_content_container([])
        assert "No content found" in html


class TestWarnings:
    """Tests for warnings component."""

    def test_render_warnings(self):
        """Test warnings rendering."""
        warnings = [
            {"message": "Some warning message"},
            {"message": "Another warning"},
        ]
        html = render_warnings(warnings)
        assert "warning" in html.lower()
        assert "Some warning message" in html

    def test_render_warnings_empty(self):
        """Test empty warnings."""
        html = render_warnings([])
        assert html == ""


class TestTemplates:
    """Tests for HTML templates."""

    def test_render_search_ui(self):
        """Test complete search UI rendering."""
        response = {
            "success": True,
            "summary": {"query": "CUDA", "total_results": 3, "search_time_ms": 500},
            "results": [
                {
                    "title": "Result 1",
                    "url": "http://1.com",
                    "snippet": "Snippet",
                    "domain": "d1.com",
                    "content_type": "article",
                    "relevance_score": 80,
                },
            ],
            "citations": [{"number": 1, "title": "Source", "url": "http://1.com", "domain": "d1.com"}],
            "warnings": [],
        }
        html = render_search_ui(response)
        assert "<!DOCTYPE html>" in html
        assert "NVIDIA MCP Search" in html
        assert "Result 1" in html
        assert "htmx.org" in html

    def test_render_content_ui(self):
        """Test complete content UI rendering."""
        response = {
            "success": True,
            "summary": {"content_type": "video", "topic": "CUDA", "total_found": 2, "search_time_ms": 300},
            "content": [
                {
                    "title": "Video 1",
                    "url": "http://1.com",
                    "content_type": "video",
                    "snippet": "Snippet",
                    "domain": "d1.com",
                    "relevance_score": 85,
                },
            ],
            "warnings": [],
        }
        html = render_content_ui(response)
        assert "<!DOCTYPE html>" in html
        assert "NVIDIA Content" in html
        assert "Video 1" in html

    def test_render_error_ui(self):
        """Test error UI rendering."""
        error = {"code": "TEST_ERROR", "message": "Test error message"}
        html = render_error_ui(error)
        assert "Error [TEST_ERROR]" in html
        assert "Test error message" in html

    def test_render_filter_fragment(self):
        """Test filter fragment rendering."""
        response = {
            "summary": {"query": "test", "total_results": 5, "search_time_ms": 100},
            "results": [
                {
                    "title": "R1",
                    "url": "http://1.com",
                    "snippet": "S",
                    "domain": "d.com",
                    "content_type": "a",
                    "relevance_score": 75,
                },
            ],
        }
        html = render_filter_fragment(
            results=response["results"],
            query="test",
            sort_by="date",
            min_relevance_score=50,
            total_results=5,
            search_time_ms=100,
        )
        assert "test" in html
        assert "R1" in html


class TestHTMXAttributes:
    """Tests for HTMX integration attributes."""

    def test_filter_panel_has_htmx(self):
        """Verify filter panel has HTMX attributes."""
        html = render_filter_panel(query="test", sort_by="relevance", min_relevance_score=17)
        assert 'hx-get="/ui/filter"' in html
        assert 'hx-target="#mcp-nvidia-results"' in html

    def test_result_card_has_htmx(self):
        """Verify result card has HTMX attributes for citations."""
        result = {
            "title": "Test",
            "url": "http://test.com",
            "snippet": "Test",
            "domain": "test.com",
            "content_type": "article",
            "relevance_score": 50,
        }
        html = render_result_card(result, index=1)
        assert "/ui/citation/" in html
