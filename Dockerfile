FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for scikit-learn/nltk
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Hybrid search: install the embeddings extra and bake the model weights into the image,
# so containers never download them on a cold start.
ENV FASTEMBED_CACHE_PATH=/opt/models
RUN pip install --no-cache-dir ".[embeddings]"
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

EXPOSE 8080

CMD ["sh", "-c", "exec python -m uvicorn mcp_nvidia.http_server:http_app --host 0.0.0.0 --port ${PORT:-8000} --log-level info"]
