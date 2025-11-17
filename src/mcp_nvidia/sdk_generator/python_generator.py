"""Python SDK generator for mcp-nvidia tools."""

from typing import Any


def _json_type_to_python_type(json_type: str | list, schema: dict[str, Any] | None = None) -> str:
    """Convert JSON schema type to Python type hint."""
    if isinstance(json_type, list):
        # Handle union types
        types = [_json_type_to_python_type(t, schema) for t in json_type]
        return " | ".join(types)

    type_map = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list[Any]",
        "object": "dict[str, Any]",
        "null": "None",
    }

    python_type = type_map.get(json_type, "Any")

    # Handle enum
    if schema and "enum" in schema:
        enum_values = [f'"{val}"' if isinstance(val, str) else str(val) for val in schema["enum"]]
        return f"Literal[{', '.join(enum_values)}]"

    # Handle array items
    if json_type == "array" and schema and "items" in schema:
        items_type = _generate_python_type(schema["items"])
        return f"list[{items_type}]"

    return python_type


def _generate_python_type(schema: dict[str, Any]) -> str:
    """Generate Python type hint from JSON schema."""
    if "type" not in schema:
        return "Any"

    json_type = schema["type"]

    # Handle enum
    if "enum" in schema:
        return _json_type_to_python_type(json_type, schema)

    # Handle array
    if json_type == "array":
        return _json_type_to_python_type(json_type, schema)

    # Handle object - will use TypedDict
    if json_type == "object" and "properties" in schema:
        return "dict[str, Any]"  # TypedDict will be defined separately

    # Handle primitive types
    return _json_type_to_python_type(json_type, schema)


def _generate_typed_dict(name: str, schema: dict[str, Any], indent: int = 0) -> list[str]:
    """Generate a Python TypedDict from JSON schema."""
    lines = []
    indent_str = "    " * indent

    # Add description as docstring
    description = schema.get("description", "")
    if description:
        lines.append(f'{indent_str}"""')
        lines.append(f"{indent_str}{description}")
        lines.append(f'{indent_str}"""')
        lines.append("")

    # Generate TypedDict
    lines.append(f"{indent_str}class {name}(TypedDict, total=False):")

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    if not properties:
        lines.append(f"{indent_str}    pass")
        return lines

    # Add docstring for the TypedDict
    if description:
        lines.append(f'{indent_str}    """')
        lines.append(f"{indent_str}    {description}")
        lines.append(f'{indent_str}    """')
        lines.append("")

    # First, add required fields
    for prop_name in required:
        if prop_name in properties:
            prop_schema = properties[prop_name]
            prop_description = prop_schema.get("description", "")
            prop_type = _generate_python_type(prop_schema)

            # Add docstring comment
            if prop_description:
                lines.append(f"{indent_str}    # {prop_description}")

            lines.append(f"{indent_str}    {prop_name}: {prop_type}")

    # Then add optional fields with NotRequired
    for prop_name, prop_schema in properties.items():
        if prop_name not in required:
            prop_description = prop_schema.get("description", "")
            prop_type = _generate_python_type(prop_schema)

            # Add docstring comment
            if prop_description:
                lines.append(f"{indent_str}    # {prop_description}")

            lines.append(f"{indent_str}    {prop_name}: NotRequired[{prop_type}]")

    return lines


def _to_snake_case(name: str) -> str:
    """Convert PascalCase or camelCase to snake_case."""
    import re

    # Insert underscore before uppercase letters
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _to_pascal_case(snake_str: str) -> str:
    """Convert snake_case to PascalCase."""
    components = snake_str.split("_")
    return "".join(x.title() for x in components)


def _generate_function(
    tool_name: str, input_schema: dict[str, Any], output_schema: dict[str, Any], description: str
) -> list[str]:
    """Generate Python async function with type hints and docstring."""
    lines = []

    # Generate type class names
    input_type_name = f"{_to_pascal_case(tool_name)}Input"
    output_type_name = f"{_to_pascal_case(tool_name)}Output"

    # Generate TypedDicts
    lines.extend(_generate_typed_dict(input_type_name, input_schema))
    lines.append("")
    lines.append("")
    lines.extend(_generate_typed_dict(output_type_name, output_schema))
    lines.append("")
    lines.append("")

    # Generate function
    function_name = tool_name
    lines.append(f"async def {function_name}(input: {input_type_name}) -> {output_type_name}:")

    # Generate docstring
    lines.append('    """')
    lines.append(f"    {description}")
    lines.append("")
    lines.append("    This function directly calls the implementation without MCP protocol overhead.")
    lines.append("")
    lines.append("    Args:")
    lines.append(f"        input: {input_type_name} - Input parameters for {tool_name}")
    lines.append("")

    # Document parameters
    properties = input_schema.get("properties", {})
    for prop_name, prop_schema in properties.items():
        prop_description = prop_schema.get("description", "")
        if prop_description:
            lines.append(f"        input.{prop_name}: {prop_description}")

    lines.append("")
    lines.append("    Returns:")
    lines.append(f"        {output_type_name}: The results")
    lines.append('    """')

    # Generate implementation based on tool name
    if tool_name == "search_nvidia":
        lines.append("    from mcp_nvidia.lib import search_all_domains, build_search_response_json, DEFAULT_DOMAINS")
        lines.append("")
        lines.append("    # Call the implementation directly")
        lines.append("    results, errors, warnings, timing_info = await search_all_domains(")
        lines.append('        query=input["query"],')
        lines.append('        domains=input.get("domains"),')
        lines.append('        max_results_per_domain=input.get("max_results_per_domain", 3),')
        lines.append('        min_relevance_score=input.get("min_relevance_score", 17),')
        lines.append('        sort_by=input.get("sort_by", "relevance"),')
        lines.append('        date_from=input.get("date_from"),')
        lines.append('        date_to=input.get("date_to"),')
        lines.append('        max_total_results=input.get("max_total_results"),')
        lines.append('        allowed_domains=input.get("allowed_domains"),')
        lines.append('        blocked_domains=input.get("blocked_domains"),')
        lines.append("    )")
        lines.append("")
        lines.append("    # Build response")
        lines.append("    return build_search_response_json(")
        lines.append("        results=results,")
        lines.append('        query=input["query"],')
        lines.append('        domains_searched=len(input.get("domains", DEFAULT_DOMAINS)),')
        lines.append('        search_time_ms=timing_info["total_time_ms"],')
        lines.append("        errors=errors,")
        lines.append("        warnings=warnings,")
        lines.append('        debug_info=timing_info.get("debug_info", {}),')
        lines.append("    )")
    elif tool_name == "discover_nvidia_content":
        lines.append("    from mcp_nvidia.lib import discover_content, build_content_response_json")
        lines.append("")
        lines.append("    # Call the implementation directly")
        lines.append("    results, errors, warnings, timing_info = await discover_content(")
        lines.append('        content_type=input["content_type"],')
        lines.append('        topic=input["topic"],')
        lines.append('        max_results=input.get("max_results", 5),')
        lines.append('        date_from=input.get("date_from"),')
        lines.append("    )")
        lines.append("")
        lines.append("    # Build response")
        lines.append("    return build_content_response_json(")
        lines.append("        results=results,")
        lines.append('        content_type=input["content_type"],')
        lines.append('        topic=input["topic"],')
        lines.append('        search_time_ms=timing_info["total_time_ms"],')
        lines.append("        errors=errors,")
        lines.append("        warnings=warnings,")
        lines.append('        debug_info=timing_info.get("debug_info", {}),')
        lines.append("    )")
    else:
        # Generic fallback
        lines.append('    raise NotImplementedError(f"Direct implementation not available for {tool_name}")')

    return lines


def generate_python_sdk(tools: list[dict[str, Any]]) -> dict[str, str]:
    """
    Generate Python SDK files from tool definitions.

    Args:
        tools: List of tool definitions with name, description, inputSchema, outputSchema

    Returns:
        Dictionary mapping file paths to file contents
    """
    sdk_files = {}

    # Generate a file for each tool
    for tool in tools:
        tool_name = tool["name"]
        description = tool.get("description", "")
        input_schema = tool.get("inputSchema", {})
        output_schema = tool.get("outputSchema", {})

        # Generate the Python file
        lines = []
        lines.append('"""')
        lines.append(f"Python SDK for {tool_name}")
        lines.append("")
        lines.append("Generated from MCP tool schema.")
        lines.append("This file provides Python type definitions for the MCP tool.")
        lines.append("To use this tool, you need an MCP client that can invoke tools.")
        lines.append('"""')
        lines.append("")

        # Imports
        lines.append("from typing import Any, Literal, TypedDict")
        lines.append("")
        lines.append("try:")
        lines.append("    from typing import NotRequired")
        lines.append("except ImportError:")
        lines.append("    from typing_extensions import NotRequired")
        lines.append("")
        lines.append("")

        # Generate function with types
        function_lines = _generate_function(tool_name, input_schema, output_schema, description)
        lines.extend(function_lines)

        # Add example usage
        lines.append("")
        lines.append("")
        lines.append("# Example usage (direct function call - no MCP overhead):")
        lines.append('"""')

        # Add example code
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        example_params = []

        for prop_name, prop_schema in properties.items():
            if prop_name in required:
                prop_type = prop_schema.get("type", "string")
                if prop_type == "string":
                    example_value = f'"{prop_name}_value"'
                elif prop_type in {"integer", "number"}:
                    example_value = str(prop_schema.get("default", 1))
                elif prop_type == "boolean":
                    example_value = "True"
                elif prop_type == "array":
                    example_value = "[]"
                else:
                    example_value = "{}"
                example_params.append(f'    "{prop_name}": {example_value}')

        lines.append(f"result = await {tool_name}({{")
        lines.append(",\n".join(example_params))
        lines.append("})")
        lines.append("")
        lines.append("# Process results")
        lines.append("# The function returns structured data with full type safety")
        lines.append('"""')

        sdk_files[f"{tool_name}.py"] = "\n".join(lines)

    # Generate __init__.py that exports all tools
    init_lines = []
    init_lines.append('"""')
    init_lines.append("NVIDIA MCP Python SDK")
    init_lines.append("")
    init_lines.append("This SDK provides Python type definitions for all NVIDIA MCP tools.")
    init_lines.append("These types can be used with MCP clients to get full type safety.")
    init_lines.append('"""')
    init_lines.append("")

    # Import all tools
    for tool in tools:
        tool_name = tool["name"]
        # Import the function and types
        pascal_name = _to_pascal_case(tool_name)
        init_lines.append(f"from .{tool_name} import (")
        init_lines.append(f"    {tool_name},")
        init_lines.append(f"    {pascal_name}Input,")
        init_lines.append(f"    {pascal_name}Output,")
        init_lines.append(")")

    init_lines.append("")
    init_lines.append("__all__ = [")
    for tool in tools:
        tool_name = tool["name"]
        pascal_name = _to_pascal_case(tool_name)
        init_lines.append(f'    "{tool_name}",')
        init_lines.append(f'    "{pascal_name}Input",')
        init_lines.append(f'    "{pascal_name}Output",')
    init_lines.append("]")

    sdk_files["__init__.py"] = "\n".join(init_lines)

    # Generate README
    readme_lines = []
    readme_lines.append("# NVIDIA MCP Python SDK")
    readme_lines.append("")
    readme_lines.append("Python type definitions for NVIDIA MCP tools.")
    readme_lines.append("")
    readme_lines.append("## Available Tools")
    readme_lines.append("")

    for tool in tools:
        tool_name = tool["name"]
        description = tool.get("description", "")
        readme_lines.append(f"### {tool_name}")
        readme_lines.append("")
        readme_lines.append(description)
        readme_lines.append("")

    readme_lines.append("## Usage")
    readme_lines.append("")
    readme_lines.append("This SDK is designed to be used with MCP clients. The type definitions")
    readme_lines.append("provide full type safety when calling NVIDIA MCP tools.")
    readme_lines.append("")
    readme_lines.append("```python")
    readme_lines.append("from mcp_nvidia_sdk import search_nvidia, SearchNvidiaInput")
    readme_lines.append("")
    readme_lines.append("# Full type safety for inputs and outputs")
    readme_lines.append("result = await search_nvidia({")
    readme_lines.append('    "query": "CUDA programming",')
    readme_lines.append('    "max_results_per_domain": 5')
    readme_lines.append("})")
    readme_lines.append("```")
    readme_lines.append("")
    readme_lines.append("## Installation")
    readme_lines.append("")
    readme_lines.append("These type definitions are provided by the mcp-nvidia server and")
    readme_lines.append("are available as MCP resources. You can access them through any MCP")
    readme_lines.append("client that supports resource reading.")

    sdk_files["README.md"] = "\n".join(readme_lines)

    return sdk_files
