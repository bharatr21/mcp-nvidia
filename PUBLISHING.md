# Publishing Guide for mcp-nvidia

This guide explains how to publish the mcp-nvidia package to both PyPI (for `pip install`) and npm (for `npx` usage).

## Publishing to Both Registries

The package is designed to be published to both PyPI and npm:
- **PyPI**: For Python users and direct installation
- **npm**: For easy use with MCP clients via `npx @bharatr21/mcp-nvidia`

The npm package acts as a wrapper that executes the Python backend.

---

## Part 1: Publishing to PyPI

### Prerequisites

1. **PyPI Account**: Create accounts on both:
   - [Test PyPI](https://test.pypi.org/account/register/) - for testing
   - [PyPI](https://pypi.org/account/register/) - for production

2. **API Tokens**: Generate API tokens from:
   - Test PyPI: https://test.pypi.org/manage/account/token/
   - PyPI: https://pypi.org/manage/account/token/

3. **Install Build Tools**:
   ```bash
   uv pip install build twine
   ```

## Build the Package

```bash
# Ensure you're in the project root
cd /path/to/mcp-nvidia

# Create a clean build
rm -rf dist/ build/ src/*.egg-info

# Build the package (creates both wheel and source distribution)
python -m build
```

This will create files in the `dist/` directory:
- `mcp_nvidia-0.1.0.tar.gz` (source distribution)
- `mcp_nvidia-0.1.0-py3-none-any.whl` (wheel distribution)

## Test on Test PyPI First

Before publishing to the real PyPI, test on Test PyPI:

```bash
# Upload to Test PyPI
python -m twine upload --repository testpypi dist/*

# You'll be prompted for:
# - Username: __token__
# - Password: <your Test PyPI API token>
```

Test the installation from Test PyPI:

```bash
# Create a test environment
python -m venv test_env
source test_env/bin/activate

# Install from Test PyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ mcp-nvidia

# Test the installation
mcp-nvidia --help  # Should start the server

# Clean up
deactivate
rm -rf test_env
```

## Publish to PyPI

Once you've verified everything works on Test PyPI:

```bash
# Upload to PyPI
python -m twine upload dist/*

# You'll be prompted for:
# - Username: __token__
# - Password: <your PyPI API token>
```

## Verify Publication

After publishing, verify the package is available:

```bash
# Install from PyPI
pip install mcp-nvidia

# Check the package info
pip show mcp-nvidia

# Test it works
mcp-nvidia
```

---

## Part 2: Publishing to npm

### Prerequisites

1. **npm Account**: Create an account at https://www.npmjs.com/signup
2. **Login to npm**:
   ```bash
   npm login
   ```

### Publish to npm

```bash
# Ensure you're in the project root
cd /path/to/mcp-nvidia

# Publish to npm (first time, you may need to add --access public for scoped packages)
npm publish --access public
```

### Test npm Package

```bash
# Test with npx
npx @bharatr21/mcp-nvidia

# Or install globally
npm install -g @bharatr21/mcp-nvidia
mcp-nvidia
```

### Update Claude Desktop Config for npm

Users can now use the npm package directly:

```json
{
  "mcpServers": {
    "nvidia": {
      "command": "npx",
      "args": ["-y", "@bharatr21/mcp-nvidia"]
    }
  }
}
```

---

## Version Management

When releasing new versions:

1. Update the version in **both** `pyproject.toml` and `package.json`
2. Update the version in `src/mcp_nvidia/__init__.py`
3. Create a git tag:
   ```bash
   git tag -a v0.1.0 -m "Release version 0.1.0"
   git push origin v0.1.0
   ```
4. Build and publish to both PyPI and npm following the steps above

## Automated Publishing with GitHub Actions

You can automate publishing using GitHub Actions. Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: '3.10'
    
    - name: Install dependencies
      run: |
        pip install build twine
    
    - name: Build package
      run: python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

Store your PyPI API token as a GitHub secret named `PYPI_API_TOKEN`.

## Troubleshooting

### Package Name Already Taken

If `mcp-nvidia` is already taken on PyPI, you'll need to choose a different name:
1. Update the `name` field in `pyproject.toml`
2. Consider names like: `nvidia-mcp-server`, `mcp-nvidia-search`, etc.

### Import Errors

Ensure your package structure matches:
```
src/
  mcp_nvidia/
    __init__.py
    server.py
```

### Missing Dependencies

If users report missing dependencies, verify:
- All dependencies are listed in `pyproject.toml`
- Version constraints are not too restrictive

## Best Practices

1. **Test thoroughly** before publishing
2. **Use semantic versioning**: MAJOR.MINOR.PATCH
3. **Keep a changelog**: Document changes between versions
4. **Tag releases** in git for traceability
5. **Never delete published versions** (PyPI policy)

## Support

For issues with:
- **Package building**: Check the `build` tool documentation
- **Publishing**: Check the `twine` tool documentation
- **PyPI policies**: See https://pypi.org/help/

## Next Steps

After successful publication:
1. Announce the package on relevant forums/communities
2. Update the README with installation instructions
3. Monitor issues and feedback from users
4. Plan for regular maintenance and updates
