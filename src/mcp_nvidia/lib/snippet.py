"""Snippet extraction and context fetching functions."""

import logging
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from mcp_nvidia.lib.constants import validate_nvidia_domain
from mcp_nvidia.lib.utils import extract_date_from_html, extract_date_from_text, extract_metadata_from_html

logger = logging.getLogger(__name__)


def extract_sentence_snippet(text: str, match_pos: int, max_length: int = 400) -> str:
    """
    Extract a snippet that ends at sentence boundaries for better readability.

    Args:
        text: Full text content
        match_pos: Position of the match in the text
        max_length: Maximum snippet length

    Returns:
        Snippet ending at a sentence boundary
    """
    # Sentence delimiters
    sentence_ends = {". ", "! ", "? ", ".\n", "!\n", "?\n"}

    # Find start position (go back up to max_length/2 characters)
    start = max(0, match_pos - max_length // 2)

    # Try to start at a sentence boundary if we're not at the beginning
    if start > 0:
        for i in range(start, min(start + 100, match_pos)):
            if i > 0 and text[i - 1 : i + 1] in {". ", "! ", "? "}:
                start = i
                break

    # Find end position (go forward up to max_length/2 characters)
    end = min(len(text), match_pos + max_length // 2)

    # Try to end at a sentence boundary
    found_sentence_end = False
    for i in range(match_pos, end):
        if i + 2 <= len(text):
            two_char = text[i : i + 2]
            if two_char in sentence_ends or (i + 1 == len(text) and text[i] in ".!?"):
                end = i + 1
                found_sentence_end = True
                break

    # If no sentence boundary found, try to break at a space
    if not found_sentence_end and end < len(text):
        for i in range(end, max(match_pos, end - 50), -1):
            if text[i] == " ":
                end = i
                break

    snippet = text[start:end].strip()

    # Clean up multiple spaces and formatting artifacts
    snippet = re.sub(r"\s+", " ", snippet)
    snippet = snippet.replace(" . ", ". ").replace(" , ", ", ")

    # Add ellipsis if truncated
    if start > 0 and not snippet.startswith("..."):
        snippet = "..." + snippet
    if end < len(text) and not snippet.endswith("..."):
        snippet = snippet + "..."

    return snippet


async def fetch_url_context(
    client: httpx.AsyncClient, url: str, snippet: str, context_chars: int = 200
) -> tuple[str, str | None, dict[str, Any]]:
    """
    Fetch the webpage and extract surrounding context, date, and metadata.

    Args:
        client: HTTP client for making requests
        url: URL to fetch
        snippet: Snippet text to find in the page
        context_chars: Number of characters to include on each side of snippet

    Returns:
        Tuple of (enhanced_snippet, published_date, metadata)
    """
    metadata = {}
    published_date = None

    try:
        # SECURITY: Re-validate URL before fetching to prevent SSRF
        if not validate_nvidia_domain(url):
            logger.warning(f"Skipping fetch for non-NVIDIA URL: {url}")
            return snippet, published_date, metadata

        # SECURITY: Validate URL scheme (only allow http/https)
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            logger.warning(f"Skipping fetch for non-HTTP(S) URL: {url}")
            return snippet, published_date, metadata

        # SECURITY: Disable redirects to prevent redirect-based SSRF
        response = await client.get(url, timeout=10.0, follow_redirects=False)
        if response.status_code != 200:
            return snippet, published_date, metadata

        # Parse HTML to get text content
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract date from HTML metadata
        published_date = extract_date_from_html(soup)

        # Extract metadata from HTML
        metadata = extract_metadata_from_html(soup)

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text content
        text = soup.get_text()

        # Clean up whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Try to find the snippet or similar text in the page
        snippet_clean = re.sub(r"\s+", " ", snippet).strip().lower()
        text_lower = text.lower()

        # Find position of snippet in text
        pos = text_lower.find(snippet_clean[:50])  # Use first 50 chars for matching

        enhanced_snippet = snippet
        if pos != -1:
            # Extract context around the snippet with sentence boundaries
            context = extract_sentence_snippet(text, pos, max_length=context_chars * 2)

            # Highlight the snippet portion using case-insensitive matching
            # Try full snippet first, then fall back to progressively shorter prefixes
            context_lower = context.lower()
            snippet_start = -1
            search_slice = snippet_clean

            # Try to find the full snippet (up to 200 chars max to avoid excessive length)
            max_search_length = min(len(snippet_clean), 200)
            search_slice = snippet_clean[:max_search_length]
            snippet_start = context_lower.find(search_slice)

            # Fall back to shorter prefixes if full snippet not found
            if snippet_start == -1:
                for length in [100, 50, 30]:
                    if len(snippet_clean) > length:
                        search_slice = snippet_clean[:length]
                        snippet_start = context_lower.find(search_slice)
                        if snippet_start != -1:
                            break

            if snippet_start != -1:
                snippet_end = snippet_start + len(search_slice)
                enhanced_snippet = (
                    context[:snippet_start] + "**" + context[snippet_start:snippet_end] + "**" + context[snippet_end:]
                )
            else:
                enhanced_snippet = context

        # If date not found in HTML, try extracting from text
        if not published_date:
            published_date = extract_date_from_text(text[:2000])  # Check first 2000 chars

        return enhanced_snippet, published_date, metadata

    except Exception as e:
        logger.debug(f"Error fetching context from {url}: {e!s}")
        return snippet, published_date, metadata
