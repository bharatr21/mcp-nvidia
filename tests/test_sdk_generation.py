"""Test SDK generation functionality."""

import pytest

from mcp_nvidia.sdk_generator import generate_python_sdk, generate_typescript_sdk


@pytest.fixture
def sample_tools():
    """Sample tool definitions for testing."""
    return [
        {
            "name": "search_nvidia",
            "description": "Search NVIDIA domains",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Max results"},
                },
                "required": ["query"],
            },
            "outputSchema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "results": {"type": "array"},
                },
            },
        }
    ]


def test_typescript_generation(sample_tools):
    """Test TypeScript SDK generation."""
    sdk_files = generate_typescript_sdk(sample_tools)

    # Check that expected files are generated
    assert "search_nvidia.ts" in sdk_files
    assert "index.ts" in sdk_files
    assert "README.md" in sdk_files

    # Check content of main file
    content = sdk_files["search_nvidia.ts"]
    assert "SearchNvidiaInput" in content
    assert "SearchNvidiaOutput" in content
    assert "searchNvidia" in content
    assert "export" in content
    assert "async function" in content

    # Check that it has proper JSDoc comments
    assert "/**" in content
    assert " * Search NVIDIA domains" in content

    # Check for MCP Client interface
    assert "MCPClient" in content
    assert "callTool(name: string, args: any)" in content

    # Check that function takes client parameter
    assert "client: MCPClient" in content
    assert 'await client.callTool("search_nvidia", input)' in content


def test_python_generation(sample_tools):
    """Test Python SDK generation."""
    sdk_files = generate_python_sdk(sample_tools)

    # Check that expected files are generated
    assert "search_nvidia.py" in sdk_files
    assert "__init__.py" in sdk_files
    assert "README.md" in sdk_files

    # Check content of main file
    content = sdk_files["search_nvidia.py"]
    assert "SearchNvidiaInput" in content
    assert "SearchNvidiaOutput" in content
    assert "search_nvidia" in content
    assert "TypedDict" in content
    assert "async def" in content

    # Check that it has proper docstrings
    assert '"""' in content

    # Check for direct implementation imports
    assert "from mcp_nvidia.lib import search_all_domains" in content
    assert "await search_all_domains(" in content
    assert "build_search_response_json" in content

    # Check that it mentions direct call
    assert "directly calls the implementation" in content.lower()


def test_typescript_type_handling():
    """Test TypeScript type conversion."""
    tools = [
        {
            "name": "test_tool",
            "description": "Test tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "str_field": {"type": "string"},
                    "int_field": {"type": "integer"},
                    "bool_field": {"type": "boolean"},
                    "array_field": {"type": "array", "items": {"type": "string"}},
                    "enum_field": {"type": "string", "enum": ["a", "b", "c"]},
                },
                "required": ["str_field"],
            },
            "outputSchema": {"type": "object", "properties": {}},
        }
    ]

    sdk_files = generate_typescript_sdk(tools)
    content = sdk_files["test_tool.ts"]

    # Verify type conversions
    assert "str_field: string" in content
    assert "int_field?: number" in content
    assert "bool_field?: boolean" in content
    assert "array_field?: Array<string>" in content
    assert '"a" | "b" | "c"' in content


def test_python_type_handling():
    """Test Python type conversion."""
    tools = [
        {
            "name": "test_tool",
            "description": "Test tool",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "str_field": {"type": "string"},
                    "int_field": {"type": "integer"},
                    "bool_field": {"type": "boolean"},
                    "array_field": {"type": "array", "items": {"type": "string"}},
                    "enum_field": {"type": "string", "enum": ["a", "b", "c"]},
                },
                "required": ["str_field"],
            },
            "outputSchema": {"type": "object", "properties": {}},
        }
    ]

    sdk_files = generate_python_sdk(tools)
    content = sdk_files["test_tool.py"]

    # Verify type conversions
    assert "str_field: str" in content
    assert "int_field: NotRequired[int]" in content
    assert "bool_field: NotRequired[bool]" in content
    assert "array_field: NotRequired[list[str]]" in content
    assert "Literal" in content


def test_index_file_generation(sample_tools):
    """Test that index files are generated correctly."""
    # TypeScript
    ts_files = generate_typescript_sdk(sample_tools)
    assert "index.ts" in ts_files
    assert 'export * from "./search_nvidia"' in ts_files["index.ts"]

    # Python
    py_files = generate_python_sdk(sample_tools)
    assert "__init__.py" in py_files
    assert "from .search_nvidia import" in py_files["__init__.py"]


def test_readme_generation(sample_tools):
    """Test that README files are generated."""
    # TypeScript
    ts_files = generate_typescript_sdk(sample_tools)
    assert "README.md" in ts_files
    assert "NVIDIA MCP TypeScript SDK" in ts_files["README.md"]

    # Python
    py_files = generate_python_sdk(sample_tools)
    assert "README.md" in py_files
    assert "NVIDIA MCP Python SDK" in py_files["README.md"]


def test_multiple_tools():
    """Test generation with multiple tools."""
    tools = [
        {
            "name": "tool_one",
            "description": "First tool",
            "inputSchema": {"type": "object", "properties": {"param": {"type": "string"}}, "required": ["param"]},
            "outputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "tool_two",
            "description": "Second tool",
            "inputSchema": {"type": "object", "properties": {"param": {"type": "string"}}, "required": ["param"]},
            "outputSchema": {"type": "object", "properties": {}},
        },
    ]

    ts_files = generate_typescript_sdk(tools)
    assert "tool_one.ts" in ts_files
    assert "tool_two.ts" in ts_files

    py_files = generate_python_sdk(tools)
    assert "tool_one.py" in py_files
    assert "tool_two.py" in py_files


def test_typescript_rejects_non_object_schema():
    """Test that TypeScript generator rejects non-object top-level schemas."""
    # Scalar top-level schema (string)
    scalar_tool = {
        "name": "scalar_tool",
        "description": "Tool with scalar schema",
        "inputSchema": {"type": "string", "description": "A string input"},
        "outputSchema": {"type": "object", "properties": {}},
    }

    with pytest.raises(ValueError, match="top-level schema must be type 'object'"):
        generate_typescript_sdk([scalar_tool])

    # Union top-level schema
    union_tool = {
        "name": "union_tool",
        "description": "Tool with union schema",
        "inputSchema": {"type": ["string", "number"]},
        "outputSchema": {"type": "object", "properties": {}},
    }

    with pytest.raises(ValueError, match="top-level schema must be type 'object'"):
        generate_typescript_sdk([union_tool])


def test_typescript_rejects_object_without_properties():
    """Test that TypeScript generator rejects object schemas without properties."""
    # Object without properties (would be Record<string, any>)
    empty_object_tool = {
        "name": "empty_object_tool",
        "description": "Tool with empty object schema",
        "inputSchema": {
            "type": "object",
            "additionalProperties": {"type": "string"},
            # Note: no "properties" field
        },
        "outputSchema": {"type": "object", "properties": {}},
    }

    with pytest.raises(ValueError, match="object schema must have 'properties'"):
        generate_typescript_sdk([empty_object_tool])
