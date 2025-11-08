# Development Guide

Quick reference for common development tasks.

## Quick Start

```bash
# Setup
uv venv && source .venv/bin/activate
make install-dev

# Or manually:
uv pip install -e ".[dev]"
pre-commit install
```

## Daily Workflow

### Before Committing

```bash
# Quick check and fix everything
make fix

# Or step by step:
make lint-fix    # Fix linting issues
make format      # Format code
make test        # Run tests
```

### Pre-commit Hooks

Hooks run automatically on `git commit`, but you can run manually:

```bash
make pre-commit
# Or: pre-commit run --all-files
```

## Common Commands

### Testing

```bash
make test          # Run tests
make test-cov      # Run tests with coverage report
pytest tests/ -v   # Verbose test output
pytest tests/test_server.py -v  # Run specific test file
pytest tests/ -k "test_search"  # Run tests matching pattern
```

### Linting & Formatting

```bash
make lint          # Check for linting issues
make lint-fix      # Auto-fix linting issues
make format        # Format code
make format-check  # Check formatting without changing
make typecheck     # Run mypy type checker
make check         # Run all checks (lint + format + types)
```

### Cleaning

```bash
make clean         # Remove build artifacts and cache
```

### Building & Publishing

```bash
make build         # Build distribution packages
make publish-test  # Publish to TestPyPI
make publish       # Publish to PyPI
```

## Code Quality Tools

### Ruff (Linter & Formatter)

Ruff replaces: flake8, isort, black, pylint, pyupgrade, and more.

**Configuration:** `pyproject.toml` under `[tool.ruff]`

**Common commands:**

```bash
ruff check .              # Check for issues
ruff check . --fix        # Auto-fix issues
ruff format .             # Format code
ruff format --check .     # Check formatting
```

**Selected rules:**

- E/W: pycodestyle (PEP 8)
- F: pyflakes
- I: isort (import sorting)
- N: pep8-naming
- UP: pyupgrade
- S: flake8-bandit (security)
- B: flake8-bugbear
- C4: flake8-comprehensions
- PL: pylint
- And many more! See `pyproject.toml` for full list

### Mypy (Type Checker)

**Configuration:** `pyproject.toml` under `[tool.mypy]`

```bash
mypy src/                 # Check types in src/
mypy --strict src/        # Strict mode (more checks)
```

### Pre-commit

**Configuration:** `.pre-commit-config.yaml`

**Hooks enabled:**

- Ruff (linting + formatting)
- General file checks (trailing whitespace, file endings, etc.)
- Bandit (security checks)
- Markdownlint (markdown formatting)

```bash
pre-commit install                 # Install hooks (one-time)
pre-commit run --all-files        # Run on all files
pre-commit autoupdate             # Update hook versions
```

## CI/CD

### GitHub Actions Workflows

**`.github/workflows/lint.yml`** - Code quality checks:

- Ruff linting
- Ruff formatting
- Mypy type checking
- Test coverage

**`.github/workflows/test.yml`** - Tests across Python versions:

- Python 3.10, 3.11, 3.12
- Full test suite
- Package build verification

**`.github/workflows/publish.yml`** - Publishing:

- Automated releases to PyPI/npm

All workflows run on push to `main` and on pull requests.

## Ignored Violations

Some rules are intentionally ignored (see `pyproject.toml`):

- `E501`: Line too long (handled by formatter)
- `S101`: Use of assert (OK in tests)
- `S603`, `S607`: Subprocess warnings (we validate inputs)
- `PLR0913`: Too many arguments (sometimes necessary)
- `TRY003`: Long exception messages (common pattern)
- `COM812`, `ISC001`: Conflicts with formatter

## Tips

### Fix Everything at Once

```bash
# The nuclear option - fix all auto-fixable issues
make fix && make test

# Or manually:
ruff check . --fix && ruff format . && pytest tests/
```

### Watch Mode (for development)

```bash
# Install pytest-watch
uv pip install pytest-watch

# Run tests on file changes
ptw tests/ -- -v
```

### Debug Ruff Rules

```bash
# See what rule failed
ruff check . --output-format=full

# Explain a specific rule
ruff rule S101

# Show all enabled rules
ruff rule --all
```

### Selective Pre-commit

```bash
# Run specific hook
pre-commit run ruff --all-files

# Skip hooks temporarily
git commit --no-verify
```

## Troubleshooting

### "Import could not be resolved" (mypy)

Add the module to `[[tool.mypy.overrides]]` in `pyproject.toml`:

```toml
[[tool.mypy.overrides]]
module = ["your_module.*"]
ignore_missing_imports = true
```

### Ruff and Formatter Conflicts

If you see conflicting rules, they're likely already ignored in our config.
Check `pyproject.toml` under `[tool.ruff.lint] ignore`.

### Pre-commit Hook Fails

```bash
# See what failed
pre-commit run --all-files --verbose

# Update hooks
pre-commit autoupdate

# Reinstall hooks
pre-commit uninstall
pre-commit install
```

## Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Pre-commit Documentation](https://pre-commit.com/)
- [pytest Documentation](https://docs.pytest.org/)
