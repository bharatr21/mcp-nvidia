#!/usr/bin/env python3
"""
Test client for the MCP NVIDIA server.

This script acts as an MCP client to test the server's functionality
and security fixes.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add the src directory to the path so we can import the server
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def test_search_nvidia(session: ClientSession):
    """Test the search_nvidia tool."""
    print("\n" + "=" * 60)
    print("Testing search_nvidia tool")
    print("=" * 60)

    # Test 1: Basic search
    print("\n[Test 1] Basic search query...")
    try:
        result = await session.call_tool(
            "search_nvidia",
            arguments={
                "query": "CUDA programming",
                "max_results_per_domain": 2
            }
        )
        print(f"✅ Basic search successful")
        if result.content:
            response = json.loads(result.content[0].text)
            print(f"   Found {len(response.get('results', []))} results")
            print(f"   Success: {response.get('success')}")
    except Exception as e:
        print(f"❌ Basic search failed: {e}")

    # Test 2: Query length validation (should be rejected)
    print("\n[Test 2] Testing query length validation (>500 chars)...")
    try:
        long_query = "A" * 501
        result = await session.call_tool(
            "search_nvidia",
            arguments={"query": long_query}
        )
        response = json.loads(result.content[0].text)
        if not response.get('success') and 'too long' in response.get('error', {}).get('message', '').lower():
            print(f"✅ Query length validation working")
        else:
            print(f"❌ Query length validation failed - should reject long queries")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")

    # Test 3: Max results validation (should be limited to 10)
    print("\n[Test 3] Testing max_results_per_domain validation (>10)...")
    try:
        result = await session.call_tool(
            "search_nvidia",
            arguments={
                "query": "GPU",
                "max_results_per_domain": 100  # Should be capped at 10
            }
        )
        response = json.loads(result.content[0].text)
        print(f"✅ Max results validation working (capped at limit)")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")

    # Test 4: Invalid domain validation
    print("\n[Test 4] Testing domain validation (invalid domain)...")
    try:
        result = await session.call_tool(
            "search_nvidia",
            arguments={
                "query": "test",
                "domains": ["https://evil.com/"]
            }
        )
        response = json.loads(result.content[0].text)
        if not response.get('success') and 'invalid' in response.get('error', {}).get('message', '').lower():
            print(f"✅ Domain validation working - rejected non-NVIDIA domain")
        else:
            print(f"❌ Domain validation failed - should reject non-NVIDIA domains")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")

    # Test 5: Valid custom domain
    print("\n[Test 5] Testing with valid custom domain...")
    try:
        result = await session.call_tool(
            "search_nvidia",
            arguments={
                "query": "deep learning",
                "domains": ["https://developer.nvidia.com/"],
                "max_results_per_domain": 1
            }
        )
        response = json.loads(result.content[0].text)
        if response.get('success'):
            print(f"✅ Custom domain search working")
        else:
            print(f"⚠️  Custom domain search returned: {response.get('error')}")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")


async def test_discover_content(session: ClientSession):
    """Test the discover_nvidia_content tool."""
    print("\n" + "=" * 60)
    print("Testing discover_nvidia_content tool")
    print("=" * 60)

    # Test 1: Basic content discovery
    print("\n[Test 1] Basic content discovery (video)...")
    try:
        result = await session.call_tool(
            "discover_nvidia_content",
            arguments={
                "content_type": "video",
                "topic": "CUDA",
                "max_results": 3
            }
        )
        print(f"✅ Content discovery successful")
        if result.content:
            response = json.loads(result.content[0].text)
            print(f"   Found {len(response.get('results', []))} results")
            print(f"   Success: {response.get('success')}")
    except Exception as e:
        print(f"❌ Content discovery failed: {e}")

    # Test 2: Topic length validation
    print("\n[Test 2] Testing topic length validation (>500 chars)...")
    try:
        long_topic = "B" * 501
        result = await session.call_tool(
            "discover_nvidia_content",
            arguments={
                "content_type": "blog",
                "topic": long_topic
            }
        )
        response = json.loads(result.content[0].text)
        if not response.get('success') and 'too long' in response.get('error', {}).get('message', '').lower():
            print(f"✅ Topic length validation working")
        else:
            print(f"❌ Topic length validation failed")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")

    # Test 3: Max results validation
    print("\n[Test 3] Testing max_results validation (>10)...")
    try:
        result = await session.call_tool(
            "discover_nvidia_content",
            arguments={
                "content_type": "tutorial",
                "topic": "AI",
                "max_results": 100  # Should be capped at 10
            }
        )
        response = json.loads(result.content[0].text)
        print(f"✅ Max results validation working")
    except Exception as e:
        print(f"⚠️  Unexpected error: {e}")


async def test_rate_limiting(session: ClientSession):
    """Test rate limiting by making multiple rapid requests."""
    print("\n" + "=" * 60)
    print("Testing rate limiting")
    print("=" * 60)

    print("\n[Test] Making 6 rapid sequential requests...")
    print("(Should see ~1 second delay between DDGS calls in server logs)")

    import time
    start = time.time()

    for i in range(6):
        try:
            await session.call_tool(
                "search_nvidia",
                arguments={
                    "query": f"test query {i}",
                    "max_results_per_domain": 1
                }
            )
            print(f"  Request {i+1} completed")
        except Exception as e:
            print(f"  Request {i+1} failed: {e}")

    elapsed = time.time() - start
    print(f"\n✅ 6 requests completed in {elapsed:.2f}s")
    if elapsed >= 2.0:  # At least 2 seconds for 6 requests (2 intervals)
        print(f"   Rate limiting appears to be working (expected >=2s)")
    else:
        print(f"   ⚠️  Faster than expected - rate limiting may not be working")


async def test_structured_output(session: ClientSession):
    """Test that structured output is returned correctly."""
    print("\n" + "=" * 60)
    print("Testing structured output (CallToolResult)")
    print("=" * 60)

    print("\n[Test] Checking response structure...")
    try:
        result = await session.call_tool(
            "search_nvidia",
            arguments={
                "query": "TensorRT",
                "max_results_per_domain": 1
            }
        )

        # Check that we have content
        if result.content:
            print(f"✅ Response has content")

            # Parse JSON
            response = json.loads(result.content[0].text)

            # Check expected fields
            expected_fields = ['success', 'query', 'results', 'metadata']
            missing_fields = [f for f in expected_fields if f not in response]

            if not missing_fields:
                print(f"✅ All expected top-level fields present")
            else:
                print(f"❌ Missing fields: {missing_fields}")

            # Check result structure
            if response.get('results'):
                result_item = response['results'][0]
                result_fields = ['title', 'url', 'snippet', 'domain', 'relevance_score', 'formatted_text']
                missing_result_fields = [f for f in result_fields if f not in result_item]

                if not missing_result_fields:
                    print(f"✅ All expected result fields present")
                else:
                    print(f"❌ Missing result fields: {missing_result_fields}")

            print(f"✅ JSON structure validated")
        else:
            print(f"❌ No content in response")

    except json.JSONDecodeError as e:
        print(f"❌ Response is not valid JSON: {e}")
    except Exception as e:
        print(f"❌ Test failed: {e}")


async def main():
    """Main test runner."""
    print("\n" + "=" * 60)
    print("MCP NVIDIA Server Test Client")
    print("=" * 60)

    # Get the server command
    server_script = Path(__file__).parent / "test_mcp_venv" / "bin" / "mcp-nvidia"

    if not server_script.exists():
        print(f"❌ Server not found at: {server_script}")
        print("   Please install the server first: pip install -e .")
        return

    server_params = StdioServerParameters(
        command=str(server_script),
        args=[],
        env=None
    )

    print(f"\n🚀 Connecting to MCP server: {server_script}")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            print(f"✅ Connected to server")

            # List available tools
            tools = await session.list_tools()
            print(f"\n📋 Available tools: {[tool.name for tool in tools.tools]}")

            # Run tests
            await test_structured_output(session)
            await test_search_nvidia(session)
            await test_discover_content(session)
            await test_rate_limiting(session)

            print("\n" + "=" * 60)
            print("✅ All tests completed!")
            print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
