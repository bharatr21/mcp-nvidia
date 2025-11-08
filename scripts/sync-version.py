#!/usr/bin/env python3
"""
Sync version from pyproject.toml to package.json.

This script ensures that package.json always has the same version as pyproject.toml,
making pyproject.toml the single source of truth for version management.

Usage:
    python scripts/sync-version.py
"""

import json
import sys
from pathlib import Path

# Conditional import: tomllib (Python 3.11+) or tomli (fallback)
try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

# Project root is one level up from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_version_from_pyproject() -> str:
    """Extract version from pyproject.toml [project] section."""
    pyproject_path = PROJECT_ROOT / "pyproject.toml"

    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found", file=sys.stderr)
        sys.exit(1)

    # Read TOML file in binary mode
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    # Extract version from [project] section
    version = data.get("project", {}).get("version")
    if not version:
        print("Error: Could not find version in [project] section of pyproject.toml", file=sys.stderr)
        sys.exit(1)

    return version


def update_package_json(version: str) -> None:
    """Update version in package.json."""
    package_json_path = PROJECT_ROOT / "package.json"

    if not package_json_path.exists():
        print(f"Error: {package_json_path} not found", file=sys.stderr)
        sys.exit(1)

    # Read package.json
    with package_json_path.open() as f:
        package_data = json.load(f)

    old_version = package_data.get("version", "unknown")

    # Update version
    package_data["version"] = version

    # Write back with proper formatting
    with package_json_path.open("w") as f:
        json.dump(package_data, f, indent=2)
        f.write("\n")  # Add trailing newline

    if old_version != version:
        print(f"✅ Updated package.json: {old_version} → {version}")
    else:
        print(f"✅ package.json already at version {version}")


def main():
    """Main entry point."""
    version = get_version_from_pyproject()
    print(f"📦 pyproject.toml version: {version}")
    update_package_json(version)
    print("\n✨ Version sync complete!")


if __name__ == "__main__":
    main()
