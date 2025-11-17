"""Utility functions for validation, date extraction, and content detection."""

import logging
import re
from datetime import datetime
from typing import Any

from bs4 import BeautifulSoup

from mcp_nvidia.lib.constants import DOMAIN_CATEGORY_MAP

logger = logging.getLogger(__name__)

# Try to import dateutil for better date parsing
try:
    from dateutil import parser as date_parser

    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False


def is_ad_url(url: str) -> bool:
    """
    Check if a URL is an advertisement or tracking URL.

    Args:
        url: URL string to check

    Returns:
        True if the URL is an ad/tracking URL, False otherwise
    """
    try:
        url_lower = url.lower()

        # Block DuckDuckGo ad URLs
        if "duckduckgo.com/y.js" in url_lower:
            return True

        # Block URLs with ad-related query parameters
        ad_patterns = [
            "ad_domain=",
            "ad_provider=",
            "ad_type=",
            "adurl=",
            "adclick=",
        ]

        return any(pattern in url_lower for pattern in ad_patterns)
    except Exception as e:
        logger.debug(f"Error checking ad URL {url}: {e}")
        return False


def extract_date_from_text(text: str) -> str | None:
    """
    Extract publication date from text using regex patterns and dateutil.

    Args:
        text: Text to extract date from (snippet, title, etc.)

    Returns:
        ISO format date string (YYYY-MM-DD) or None if no date found
    """
    if not text:
        return None

    # Common date patterns in snippets
    date_patterns = [
        # "January 16, 2025" or "Jan 16, 2025"
        r"(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}",
        # "2025-01-16" or "2025/01/16"
        r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        # "01-16-2025" or "01/16/2025"
        r"\d{1,2}[-/]\d{1,2}[-/]\d{4}",
        # "16 January 2025"
        r"\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}",
    ]

    for pattern in date_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            date_str = match.group(0)
            try:
                # Try dateutil parser if available
                if HAS_DATEUTIL:
                    parsed_date = date_parser.parse(date_str, fuzzy=True)
                    return parsed_date.strftime("%Y-%m-%d")
                # Fallback to manual parsing
                # Try common formats
                for fmt in [
                    "%B %d, %Y",
                    "%b %d, %Y",
                    "%Y-%m-%d",
                    "%Y/%m/%d",
                    "%m-%d-%Y",
                    "%m/%d/%Y",
                    "%d %B %Y",
                    "%d %b %Y",
                ]:
                    try:
                        parsed_date = datetime.strptime(date_str.replace(",", ""), fmt)  # noqa: DTZ007
                        return parsed_date.strftime("%Y-%m-%d")
                    except ValueError:
                        continue
            except Exception as e:
                logger.debug(f"Error parsing date '{date_str}': {e}")
                continue

    return None


def extract_date_from_html(soup: BeautifulSoup) -> str | None:
    """
    Extract publication date from HTML metadata.

    Args:
        soup: BeautifulSoup object of the page

    Returns:
        ISO format date string (YYYY-MM-DD) or None if no date found
    """
    # Check common meta tags
    meta_tags = [
        ("property", "article:published_time"),
        ("property", "og:published_time"),
        ("name", "date"),
        ("name", "publish-date"),
        ("name", "article:published_time"),
        ("itemprop", "datePublished"),
        ("itemprop", "dateCreated"),
    ]

    for attr, value in meta_tags:
        tag = soup.find("meta", {attr: value})
        if tag and tag.get("content"):
            date_str = tag.get("content")
            try:
                if HAS_DATEUTIL:
                    parsed_date = date_parser.parse(date_str)
                    return parsed_date.strftime("%Y-%m-%d")
                # Try ISO format first
                if "T" in date_str:
                    parsed_date = datetime.fromisoformat(date_str.replace("Z", "+00:00").split("+")[0].split("T")[0])
                    return parsed_date.strftime("%Y-%m-%d")
            except Exception as e:
                logger.debug(f"Error parsing date from meta tag '{date_str}': {e}")
                continue

    # Check time tags with datetime attribute
    time_tag = soup.find("time", {"datetime": True})
    if time_tag:
        date_str = time_tag.get("datetime")
        try:
            if HAS_DATEUTIL:
                parsed_date = date_parser.parse(date_str)
                return parsed_date.strftime("%Y-%m-%d")
            if "T" in date_str:
                parsed_date = datetime.fromisoformat(date_str.replace("Z", "+00:00").split("+")[0].split("T")[0])
                return parsed_date.strftime("%Y-%m-%d")
        except Exception as e:
            logger.debug(f"Error parsing date from time tag '{date_str}': {e}")

    return None


def detect_content_type(title: str, snippet: str, url: str, domain_category: str) -> str:
    """
    Detect the content type of a search result.

    Args:
        title: Page title
        snippet: Page snippet
        url: Page URL
        domain_category: Domain category (blog, forum, documentation, etc.)

    Returns:
        Content type: announcement, tutorial, guide, forum_discussion, blog_post, documentation, research_paper, news, video, course, or article
    """
    title_lower = title.lower()
    snippet_lower = snippet.lower()
    url_lower = url.lower()
    combined = f"{title_lower} {snippet_lower} {url_lower}"

    # Announcement detection
    announcement_keywords = ["announc", "releas", "introduc", "launch", "unveil", "availab"]
    if any(kw in title_lower for kw in announcement_keywords):
        return "announcement"

    # Tutorial/Guide detection
    tutorial_keywords = [
        "tutorial",
        "how to",
        "how-to",
        "step by step",
        "getting started",
        "quick start",
        "walkthrough",
    ]
    if any(kw in combined for kw in tutorial_keywords):
        if "guide" in combined:
            return "guide"
        return "tutorial"

    # Video detection
    if any(kw in combined for kw in ["video", "watch", "youtube", "webinar", "livestream"]):
        return "video"

    # Course detection
    if any(kw in combined for kw in ["course", "training", "certification", "dli", "deep learning institute"]):
        return "course"

    # Forum discussion
    if domain_category == "forum" or "forum" in url_lower or "discuss" in title_lower:
        return "forum_discussion"

    # Research paper
    if domain_category == "research" or any(kw in combined for kw in ["paper", "research", "arxiv", "publication"]):
        return "research_paper"

    # News
    if domain_category == "news" or "news" in url_lower:
        return "news"

    # Blog post
    if domain_category == "blog" or "blog" in url_lower:
        return "blog_post"

    # Documentation
    if (
        domain_category == "documentation"
        or "docs" in url_lower
        or any(kw in combined for kw in ["api reference", "documentation", "reference guide"])
    ):
        return "documentation"

    # Default to article
    return "article"


def extract_metadata_from_html(soup: BeautifulSoup) -> dict[str, Any]:
    """
    Extract metadata from HTML content.

    Args:
        soup: BeautifulSoup object of the page

    Returns:
        Dictionary with metadata fields
    """
    metadata = {}

    # Extract author
    author = None
    author_tags = [
        soup.find("meta", {"name": "author"}),
        soup.find("meta", {"property": "article:author"}),
        soup.find("meta", {"name": "article:author"}),
        soup.find("span", {"class": re.compile(r"author", re.I)}),
        soup.find("a", {"rel": "author"}),
    ]

    for tag in author_tags:
        if tag:
            author = tag.get("content") if tag.name == "meta" else tag.get_text(strip=True)
            if author:
                # Clean author name
                author = re.sub(r"^(by|author:)\s*", "", author, flags=re.I).strip()
                if author and len(author) < 100:
                    metadata["author"] = author
                    break

    # Get text content for analysis
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.decompose()

    text = soup.get_text()
    text = re.sub(r"\s+", " ", text).strip()

    # Word count (approximate)
    if text:
        word_count = len(text.split())
        metadata["word_count"] = word_count

    # Detect if page has code examples
    code_tags = soup.find_all(["code", "pre", "div"], class_=re.compile(r"code|highlight|syntax", re.I))
    metadata["has_code"] = len(code_tags) > 0

    # Detect if page has video
    video_tags = soup.find_all(["video", "iframe"], src=re.compile(r"youtube|vimeo|video", re.I))
    metadata["has_video"] = len(video_tags) > 0

    # Detect if page has images
    img_tags = soup.find_all("img")
    metadata["has_images"] = len(img_tags) > 0
    if len(img_tags) > 0:
        metadata["image_count"] = len(img_tags)

    return metadata


def get_domain_category(domain: str) -> str:
    """
    Categorize an NVIDIA domain.

    Args:
        domain: Domain URL or hostname

    Returns:
        Category string: documentation, blog, news, developer, build, research, catalog, forum, downloads, resources, or other
    """
    domain_lower = domain.lower()

    # Check patterns in order (most specific first)
    for pattern, category in DOMAIN_CATEGORY_MAP:
        if pattern in domain_lower:
            return category

    return "other"
