"""Constants and configuration for MCP NVIDIA server."""

import asyncio
import logging
import os
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Default NVIDIA domains to search
DEFAULT_DOMAINS = [
    "https://blogs.nvidia.com/",
    "https://build.nvidia.com/",
    "https://catalog.ngc.nvidia.com/",
    "https://developer.download.nvidia.com/",
    "https://developer.nvidia.com/",
    "https://docs.api.nvidia.com/",
    "https://docs.nvidia.com/",
    "https://docs.omniverse.nvidia.com/",
    "https://forums.developer.nvidia.com/",
    "https://forums.nvidia.com/",
    "https://ngc.nvidia.com/",
    "https://nvidia.github.io/",
    "https://nvidianews.nvidia.com/",
    "https://research.nvidia.com/",
    "https://resources.nvidia.com/",
]

# SECURITY: Rate limiting for DDGS calls
DDGS_RATE_LIMIT_LOCK = asyncio.Lock()
# PERFORMANCE NOTE: DuckDuckGo imposes rate limits. The minimum interval of 0.2s between calls
# is intentionally conservative to avoid HTTP 429 (rate limit) errors. This is the primary
# bottleneck for search performance. We search multiple domains concurrently to mitigate this,
# but the overall search time will still be ~15-50 seconds depending on the number of domains.
DDGS_MIN_INTERVAL = 0.2  # Minimum 0.2 second (200ms) between DDGS calls

# SECURITY: Concurrency limits
MAX_CONCURRENT_SEARCHES = 5

# SECURITY: Input validation limits
MAX_QUERY_LENGTH = 500
MAX_RESULTS_PER_DOMAIN = 10

# NVIDIA Product Name Variants and Acronyms
# Maps product names to their common variations, acronyms, and alternate spellings
NVIDIA_PRODUCT_VARIANTS = {
    "cuda-q": ["cuda quantum", "cuda-quantum", "cudaquantum", "qoda"],
    "cuda quantum": ["cuda-q", "cuda-quantum", "cudaquantum", "qoda"],
    "cuquantum": ["cuquantum", "cu-quantum"],
    "nim": ["nvidia nim", "nim microservice", "nim microservices"],
    "nemo": ["nvidia nemo", "nemo framework"],
    "triton": ["triton inference server", "nvidia triton"],
    "tensorrt": ["tensor rt", "trt"],
    "cudnn": ["cuda dnn", "cudnn library"],
    "nccl": ["nvidia collective communications library", "nccl library"],
    "cudf": ["cu dataframe", "rapids cudf"],
    "rapids": ["rapids ai", "rapids suite"],
    "isaac sim": ["isaac simulator", "nvidia isaac sim"],
    "isaac ros": ["isaac robot operating system"],
    "omniverse": ["nvidia omniverse", "omniverse platform"],
    "jetson": ["nvidia jetson", "jetson platform"],
    "drive": ["nvidia drive", "drive av"],
    "dgx": ["nvidia dgx", "dgx system", "dgx systems"],
    "a100": ["nvidia a100", "a100 gpu"],
    "h100": ["nvidia h100", "h100 gpu", "hopper"],
    "l40s": ["nvidia l40s", "l40s gpu"],
    "geforce rtx": ["rtx", "geforce"],
    "quadro": ["nvidia quadro", "quadro rtx"],
    "ai enterprise": ["nvidia ai enterprise", "nvaie"],
    "clara": ["nvidia clara", "clara healthcare"],
    "parabricks": ["nvidia parabricks", "parabricks genomics"],
    "bionemo": ["nvidia bionemo", "bionemo framework"],
    "modulus": ["nvidia modulus", "modulus physics ml"],
    "earth-2": ["nvidia earth-2", "earth 2", "e2"],
    "metropolis": ["nvidia metropolis", "metropolis platform"],
    "merlin": ["nvidia merlin", "merlin recommender"],
    "riva": ["nvidia riva", "riva speech ai"],
    "maxine": ["nvidia maxine", "maxine video ai"],
    "broadcast": ["nvidia broadcast", "rtx broadcast"],
    "canvas": ["nvidia canvas", "gaugan"],
}

# Domain category mapping - order matters! More specific patterns first
DOMAIN_CATEGORY_MAP = [
    # Forums (most specific first)
    ("forums.developer.nvidia.com", "forum"),
    ("forums.nvidia.com", "forum"),
    # Downloads
    ("developer.download.nvidia.com", "downloads"),
    # Documentation (specific subdomains first)
    ("nvidia.github.io", "documentation"),
    ("docs.api.nvidia.com", "documentation"),
    ("docs.omniverse.nvidia.com", "documentation"),
    ("gameworksdocs.nvidia.com", "documentation"),
    ("docs.nvidia.com", "documentation"),
    # Catalog
    ("catalog.ngc.nvidia.com", "catalog"),
    ("ngc.nvidia.com", "catalog"),
    # Resources
    ("resources.nvidia.com", "resources"),
    # Blog, News, Research
    ("blogs.nvidia.com", "blog"),
    ("nvidianews.nvidia.com", "news"),
    ("research.nvidia.com", "research"),
    # Build and Developer (broader matches last)
    ("build.nvidia.com", "build"),
    ("developer.nvidia.com", "developer"),
]


STOPWORDS_FALLBACK = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "will",
    "with",
    "this",
    "but",
    "or",
    "not",
    "can",
}


def validate_nvidia_domain(domain: str) -> bool:
    """
    Validate that a domain is a valid NVIDIA domain or subdomain.

    Args:
        domain: URL string to validate

    Returns:
        True if domain is nvidia.com, a subdomain, or nvidia.github.io, False otherwise
    """
    try:
        parsed = urlparse(domain)
        hostname = parsed.netloc or parsed.path.split("/")[0]
        hostname = hostname.lower()

        # Check if it's nvidia.com or a subdomain of nvidia.com
        if hostname == "nvidia.com" or hostname.endswith(".nvidia.com"):
            return True

        # Allow NVIDIA's official GitHub Pages (specifically nvidia.github.io only)
        if hostname == "nvidia.github.io":
            return True

        logger.warning(f"Domain validation failed for: {domain} (hostname: {hostname})")
        return False
    except Exception as e:
        logger.exception(f"Error validating domain {domain}: {e}")
        return False


# Allow override via environment variable (comma-separated list)
def _init_domains():
    """Initialize domains from environment variable if provided."""
    global DEFAULT_DOMAINS

    if custom_domains := os.getenv("MCP_NVIDIA_DOMAINS"):
        raw_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]
        validated_domains = []

        for domain in raw_domains:
            if validate_nvidia_domain(domain):
                validated_domains.append(domain)
            else:
                logger.warning(f"Skipping invalid domain (not nvidia.com): {domain}")

        if validated_domains:
            DEFAULT_DOMAINS = validated_domains
            logger.info(f"Using custom domains from environment: {DEFAULT_DOMAINS}")
        else:
            logger.warning("No valid NVIDIA domains found in MCP_NVIDIA_DOMAINS. Using defaults.")


# Initialize domains on module import
_init_domains()
