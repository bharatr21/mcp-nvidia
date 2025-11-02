#!/usr/bin/env python3
"""
Example usage script for the MCP NVIDIA search server.

This script demonstrates how the search functionality works.
Note: This is for demonstration purposes and requires network access.
"""

import asyncio
from mcp_nvidia.server import search_all_domains, format_search_results


async def main():
    """Run example searches."""
    print("=" * 70)
    print("MCP NVIDIA Search Server - Example Usage")
    print("=" * 70)
    print()
    
    # Example 1: Search for CUDA across all domains
    print("Example 1: Searching for 'CUDA' across all NVIDIA domains...")
    print("-" * 70)
    try:
        results = await search_all_domains(
            query="CUDA programming guide",
            max_results_per_domain=2
        )
        formatted = format_search_results(results, "CUDA programming guide")
        print(formatted)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n")
    
    # Example 2: Search specific domain
    print("Example 2: Searching for 'RTX' on developer.nvidia.com only...")
    print("-" * 70)
    try:
        results = await search_all_domains(
            query="RTX GPU architecture",
            domains=["https://developer.nvidia.com/"],
            max_results_per_domain=2
        )
        formatted = format_search_results(results, "RTX GPU architecture")
        print(formatted)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n")
    print("=" * 70)
    print("Note: Actual results depend on network connectivity and")
    print("search engine availability. In restricted environments,")
    print("you may see fallback messages instead of real search results.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
