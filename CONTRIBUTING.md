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
   pre-commit install  # Install git hooks
   ```

   **Or use the Makefile (Linux/macOS):**
   ```bash
   make install-dev
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

## Code Quality & Linting

This project uses modern Python tooling to ensure code quality:

### Ruff - Linting & Formatting

We use **Ruff**, an extremely fast Python linter and formatter that replaces multiple tools (flake8, isort, black, pylint, etc.).

**Check for issues:**
```bash
# Check for linting issues
ruff check .

# Check formatting
ruff format --check .
```

**Auto-fix issues:**
```bash
# Fix linting issues automatically
ruff check . --fix

# Format code
ruff format .
```

**Quick fix everything:**
```bash
ruff check . --fix && ruff format .
```

### Pre-commit Hooks (Recommended)

Install pre-commit hooks to automatically check code before commits:

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Run manually on all files
pre-commit run --all-files
```

Once installed, hooks will run automatically on `git commit`.

### Type Checking

We use **mypy** for static type checking:

```bash
# Check types
mypy src/
```

### Code Style Guidelines

- Follow PEP 8 (enforced by Ruff)
- Line length: 120 characters
- Use type hints where appropriate
- Write docstrings for all public functions and classes
- Keep functions focused and single-purpose
- Avoid relative imports (use absolute imports)

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
