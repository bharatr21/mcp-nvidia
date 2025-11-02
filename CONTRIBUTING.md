# Contributing to mcp-nvidia

Thank you for your interest in contributing to mcp-nvidia! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/bharatr21/mcp-nvidia.git
   cd mcp-nvidia
   ```

2. **Install uv (if not already installed)**
   ```bash
   pip install uv
   ```

3. **Create a virtual environment and install dependencies**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip install -e ".[dev]"
   ```

## Running Tests

```bash
# Activate virtual environment
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_server.py -v

# Run with coverage
pytest tests/ --cov=mcp_nvidia --cov-report=html
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose

## Adding New Features

1. **Add a new domain**
   - Edit `src/mcp_nvidia/server.py`
   - Add the new domain URL to the `DEFAULT_DOMAINS` list
   - Update the README.md to reflect the new domain

2. **Add new search functionality**
   - Implement your feature in `src/mcp_nvidia/server.py`
   - Add tests in `tests/`
   - Update documentation in README.md

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Ensure all tests pass
- Update documentation as needed
- Keep commits atomic and well-described

## Testing Guidelines

- Write tests for all new features
- Maintain or improve code coverage
- Use mocks for external API calls
- Test edge cases and error conditions

## Questions?

If you have questions or need help, please:
- Open an issue on GitHub
- Provide as much context as possible
- Include code examples if relevant

Thank you for contributing!
