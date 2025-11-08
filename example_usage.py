#!/usr/bin/env python3
"""
Example usage script for the MCP NVIDIA search server.

This script demonstrates how the search and content discovery features work.
Note: This is for demonstration purposes and requires network access.
"""

import asyncio

from mcp_nvidia.server import discover_content, format_content_results, format_search_results, search_all_domains


async def main():
    """Run example searches."""

    # Example 1: Search for CUDA across all domains
    try:
        results = await search_all_domains(query="CUDA programming guide", max_results_per_domain=2)
        format_search_results(results, "CUDA programming guide")
    except Exception as e:  # nosec B110
        print(f"Example 1 failed: {e}")

    # Example 2: Search specific domain
    try:
        results = await search_all_domains(
            query="RTX GPU architecture", domains=["https://developer.nvidia.com/"], max_results_per_domain=2
        )
        format_search_results(results, "RTX GPU architecture")
    except Exception as e:  # nosec B110
        print(f"Example 2 failed: {e}")

    # Example 3: Discover video tutorials
    try:
        results = await discover_content(content_type="video", topic="AI training", max_results=3)
        format_content_results(results, "video", "AI training")
    except Exception as e:  # nosec B110
        print(f"Example 3 failed: {e}")

    # Example 4: Find training courses
    try:
        results = await discover_content(content_type="course", topic="Deep Learning", max_results=3)
        format_content_results(results, "course", "Deep Learning")
    except Exception as e:  # nosec B110
        print(f"Example 4 failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
