FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for scikit-learn/nltk
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

RUN pip install --no-cache-dir .

CMD ["sh", "-c", "exec python -m uvicorn mcp_nvidia.http_server:http_app --host 0.0.0.0 --port ${PORT:-8000} --log-level info"]
