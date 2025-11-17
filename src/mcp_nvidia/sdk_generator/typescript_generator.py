"""TypeScript SDK generator for mcp-nvidia tools."""

from typing import Any


def _json_type_to_ts_type(json_type: str | list, schema: dict[str, Any] | None = None) -> str:
    """Convert JSON schema type to TypeScript type.

    Args:
        json_type: JSON schema type string or list of types for unions
        schema: Full schema dict with additional constraints (enum, items, etc.)

    Returns:
        TypeScript type string

    Note:
        For union types (json_type as list), schema-level constraints apply to the whole union.
        Per-branch constraints in unions (e.g., different array item types) are not supported.
    """
    if isinstance(json_type, list):
        # Handle union types like ["string", "null"]
        # Note: This creates a simple union. Per-branch schema details (like different
        # array item types for each branch) are not preserved.
        # If the schema has shared constraints (enum), they're applied to the whole union.
        if schema and "enum" in schema:
            # Enum applies to the whole union
            enum_values = [f'"{val}"' if isinstance(val, str) else str(val) for val in schema["enum"]]
            return " | ".join(enum_values)
        return " | ".join(_json_type_to_ts_type(t) for t in json_type)

    type_map = {
        "string": "string",
        "integer": "number",
        "number": "number",
        "boolean": "boolean",
        "array": "Array<any>",  # Will be refined if items specified
        "object": "Record<string, any>",  # Will be refined if properties specified
        "null": "null",
    }

    ts_type = type_map.get(json_type, "any")

    # Handle enum
    if schema and "enum" in schema:
        enum_values = [f'"{val}"' if isinstance(val, str) else str(val) for val in schema["enum"]]
        return " | ".join(enum_values)

    # Handle array items
    if json_type == "array" and schema and "items" in schema:
        items_type = _generate_ts_type(schema["items"])
        return f"Array<{items_type}>"

    return ts_type


def _generate_ts_type(schema: dict[str, Any], type_name: str | None = None) -> str:
    """Generate TypeScript type definition from JSON schema."""
    if "type" not in schema:
        return "any"

    json_type = schema["type"]

    # Handle enum
    if "enum" in schema:
        return _json_type_to_ts_type(json_type, schema)

    # Handle array
    if json_type == "array":
        return _json_type_to_ts_type(json_type, schema)

    # Handle object - generate interface
    if json_type == "object" and "properties" in schema:
        lines = []
        if type_name:
            lines.append(f"interface {type_name} {{")
        else:
            lines.append("{")

        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for prop_name, prop_schema in properties.items():
            description = prop_schema.get("description", "")
            is_required = prop_name in required
            optional_marker = "" if is_required else "?"

            # Add JSDoc comment for the property
            if description:
                lines.append(f"  /** {description} */")

            prop_type = _generate_ts_type(prop_schema)
            lines.append(f"  {prop_name}{optional_marker}: {prop_type};")

        lines.append("}")
        return "\n".join(lines)

    # Handle primitive types
    return _json_type_to_ts_type(json_type, schema)


def _generate_ts_interface(name: str, schema: dict[str, Any], is_export: bool = True) -> str:
    """Generate a TypeScript interface from a JSON schema.

    Args:
        name: The interface name
        schema: JSON schema (must be type: "object" with properties)
        is_export: Whether to add export keyword

    Raises:
        ValueError: If schema is not an object with properties
    """
    # Validate that schema is an object with properties
    schema_type = schema.get("type")
    if schema_type != "object":
        raise ValueError(
            f"Cannot generate interface {name}: top-level schema must be type 'object', "
            f"got '{schema_type}'. Non-object schemas are not currently supported for interfaces."
        )

    if "properties" not in schema:
        raise ValueError(
            f"Cannot generate interface {name}: object schema must have 'properties'. "
            f"Got schema without properties (might be an empty object or Record type)."
        )

    lines = []

    # Add description as JSDoc
    description = schema.get("description", "")
    if description:
        lines.append("/**")
        lines.append(f" * {description}")
        lines.append(" */")

    # Generate interface
    export_keyword = "export " if is_export else ""
    lines.append(f"{export_keyword}interface {name} {{")

    properties = schema.get("properties", {})
    required = schema.get("required", [])

    for prop_name, prop_schema in properties.items():
        prop_description = prop_schema.get("description", "")
        is_required = prop_name in required
        optional_marker = "" if is_required else "?"

        # Add JSDoc comment for the property
        if prop_description:
            lines.append(f"  /** {prop_description} */")

        prop_type = _generate_ts_type(prop_schema)
        lines.append(f"  {prop_name}{optional_marker}: {prop_type};")

    lines.append("}")
    return "\n".join(lines)


def _generate_function_signature(
    tool_name: str, input_schema: dict[str, Any], output_schema: dict[str, Any], description: str
) -> str:
    """Generate TypeScript function signature with JSDoc."""
    lines = []

    # Generate input/output type names
    input_type_name = f"{_to_pascal_case(tool_name)}Input"
    output_type_name = f"{_to_pascal_case(tool_name)}Output"

    # Generate type definitions
    input_interface = _generate_ts_interface(input_type_name, input_schema)
    output_interface = _generate_ts_interface(output_type_name, output_schema)

    lines.append(input_interface)
    lines.append("")
    lines.append(output_interface)
    lines.append("")

    # Generate MCP Client interface
    lines.append("/**")
    lines.append(" * MCP Client interface for calling tools")
    lines.append(" */")
    lines.append("export interface MCPClient {")
    lines.append("  callTool(name: string, args: any): Promise<any>;")
    lines.append("}")
    lines.append("")

    # Generate JSDoc for the function
    lines.append("/**")
    lines.append(f" * {description}")
    lines.append(" *")
    lines.append(" * This function calls the MCP tool via the provided client.")
    lines.append(" *")

    # Add parameter documentation
    lines.append(" * @param input - The input parameters")
    properties = input_schema.get("properties", {})
    for prop_name, prop_schema in properties.items():
        prop_description = prop_schema.get("description", "")
        if prop_description:
            lines.append(f" * @param input.{prop_name} {prop_description}")

    lines.append(" * @param client - MCP client instance to use for calling the tool")
    lines.append(f" * @returns {output_type_name}")
    lines.append(" */")

    # Generate function signature
    function_name = _to_camel_case(tool_name)
    lines.append(f"export async function {function_name}(")
    lines.append(f"  input: {input_type_name},")
    lines.append("  client: MCPClient")
    lines.append(f"): Promise<{output_type_name}> {{")
    lines.append(f'  return await client.callTool("{tool_name}", input) as {output_type_name};')
    lines.append("}")

    return "\n".join(lines)


def _to_pascal_case(snake_str: str) -> str:
    """Convert snake_case to PascalCase."""
    components = snake_str.split("_")
    return "".join(x.title() for x in components)


def _to_camel_case(snake_str: str) -> str:
    """Convert snake_case to camelCase."""
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def generate_typescript_sdk(tools: list[dict[str, Any]]) -> dict[str, str]:
    """
    Generate TypeScript SDK files from tool definitions.

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

        # Generate the TypeScript file
        lines = []
        lines.append("/**")
        lines.append(f" * TypeScript SDK for {tool_name}")
        lines.append(" * Generated from MCP tool schema")
        lines.append(" * ")
        lines.append(" * This file provides TypeScript type definitions for the MCP tool.")
        lines.append(" * To use this tool, you need an MCP client that can invoke tools.")
        lines.append(" */")
        lines.append("")

        # Generate function with types
        function_code = _generate_function_signature(tool_name, input_schema, output_schema, description)
        lines.append(function_code)

        # Add example usage
        lines.append("")
        lines.append("/**")
        lines.append(" * Example usage (requires MCP client):")
        lines.append(" * ```typescript")
        lines.append(" * const client: MCPClient = getMCPClient();")
        lines.append(f" * const result = await {_to_camel_case(tool_name)}({{")

        # Add example parameters
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        example_params = []
        for _i, (prop_name, prop_schema) in enumerate(properties.items()):
            if prop_name in required:
                prop_type = prop_schema.get("type", "string")
                if prop_type == "string":
                    example_value = f'"{prop_name}_value"'
                elif prop_type in {"integer", "number"}:
                    example_value = prop_schema.get("default", 1)
                elif prop_type == "boolean":
                    example_value = "true"
                elif prop_type == "array":
                    example_value = "[]"
                else:
                    example_value = "{}"
                example_params.append(f" *   {prop_name}: {example_value}")

        lines.append(",\n".join(example_params))
        lines.append(" * }, client);")
        lines.append(" * ```")
        lines.append(" */")

        sdk_files[f"{tool_name}.ts"] = "\n".join(lines)

    # Generate index.ts that exports all tools
    index_lines = []
    index_lines.append("/**")
    index_lines.append(" * NVIDIA MCP TypeScript SDK")
    index_lines.append(" * ")
    index_lines.append(" * This SDK provides TypeScript type definitions for all NVIDIA MCP tools.")
    index_lines.append(" * These types can be used with MCP clients to get full type safety.")
    index_lines.append(" */")
    index_lines.append("")

    for tool in tools:
        tool_name = tool["name"]
        index_lines.append(f'export * from "./{tool_name}";')

    sdk_files["index.ts"] = "\n".join(index_lines)

    # Generate README
    readme_lines = []
    readme_lines.append("# NVIDIA MCP TypeScript SDK")
    readme_lines.append("")
    readme_lines.append("TypeScript type definitions for NVIDIA MCP tools.")
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
    readme_lines.append("```typescript")
    readme_lines.append('import { searchNvidia } from "./search_nvidia";')
    readme_lines.append("")
    readme_lines.append("// Full type safety for inputs and outputs")
    readme_lines.append("const result = await searchNvidia({")
    readme_lines.append('  query: "CUDA programming",')
    readme_lines.append("  max_results_per_domain: 5")
    readme_lines.append("});")
    readme_lines.append("```")

    sdk_files["README.md"] = "\n".join(readme_lines)

    return sdk_files
