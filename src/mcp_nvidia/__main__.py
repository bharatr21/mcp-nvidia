"""Main entry point for running mcp_nvidia as a module."""

import sys


def main():
    """Handle command-line invocation with -m flag."""
    # Check for --version flag
    if len(sys.argv) > 1 and sys.argv[1] == "--version":
        from mcp_nvidia import __version__

        print(__version__)
        sys.exit(0)

    # Otherwise, run the server
    from mcp_nvidia.server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
