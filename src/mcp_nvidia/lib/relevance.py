"""Relevance scoring, keyword extraction, and query expansion functions."""

import logging
from typing import Any

from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mcp_nvidia.lib.constants import NVIDIA_PRODUCT_VARIANTS, STOPWORDS_FALLBACK

logger = logging.getLogger(__name__)

# Import NLTK stopwords
try:
    import nltk
    from nltk.corpus import stopwords

    try:
        STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(stopwords.words("english"))
    except Exception:
        STOPWORDS = STOPWORDS_FALLBACK
except ImportError:
    STOPWORDS = STOPWORDS_FALLBACK


def extract_keywords(query: str) -> list[str]:
    """
    Extract meaningful keywords from a query string.

    Filters out stopwords and very short words to get actual keywords.

    Args:
        query: Search query string

    Returns:
        List of keywords (non-stopwords, meaningful words)
    """
    # Split and normalize
    words = query.lower().split()

    # Filter out stopwords and very short words
    keywords = []
    for word in words:
        # Remove common punctuation
        cleaned = word.strip(".,!?;:()\"'")

        # Keep if:
        # - Not a stopword
        # - Length >= 2 characters
        # - Contains at least one letter (to avoid pure numbers/symbols unless they're tech terms)
        if cleaned and cleaned not in STOPWORDS and len(cleaned) >= 2 and any(c.isalpha() for c in cleaned):
            keywords.append(cleaned)

    return keywords


def calculate_fuzzy_match_score(keyword: str, text: str, threshold: int = 80) -> float:
    """
    Calculate fuzzy match score for a keyword in text.

    Uses fuzzy matching to handle typos and variations.

    Args:
        keyword: Keyword to search for
        text: Text to search in
        threshold: Minimum similarity threshold (0-100)

    Returns:
        Float score 0.0-1.0 based on best fuzzy match
    """
    if keyword in text:
        return 1.0  # Exact match

    # Split text into words and find best fuzzy match
    words = text.split()
    best_score = 0

    for word in words:
        score = fuzz.ratio(keyword, word)
        if score > best_score and score >= threshold:
            best_score = score

    # Normalize to 0-1 range
    return best_score / 100.0 if best_score >= threshold else 0.0


def extract_phrases(query: str) -> list[str]:
    """
    Extract multi-word phrases from query (2-3 word phrases).

    Args:
        query: Search query string

    Returns:
        List of phrases (2-3 words)
    """
    # Clean and split query
    words = []
    for word in query.lower().split():
        # Remove punctuation
        cleaned = word.strip(".,!?;:()\"'")
        if cleaned:
            words.append(cleaned)

    phrases = []

    # Extract 2-word phrases
    for i in range(len(words) - 1):
        # Skip if both words are stopwords
        if words[i] not in STOPWORDS or words[i + 1] not in STOPWORDS:
            phrase = f"{words[i]} {words[i + 1]}"
            phrases.append(phrase)

    # Extract 3-word phrases
    for i in range(len(words) - 2):
        # Include if at least one word is not a stopword
        if any(w not in STOPWORDS for w in [words[i], words[i + 1], words[i + 2]]):
            phrase = f"{words[i]} {words[i + 1]} {words[i + 2]}"
            phrases.append(phrase)

    return phrases


def get_domain_boost(domain: str, query: str) -> float:
    """
    Calculate domain-specific boost based on query intent.

    Technical queries get higher boost for docs domains.

    Args:
        domain: Domain name
        query: Search query

    Returns:
        Boost multiplier (1.0 = no boost, >1.0 = boost)
    """
    domain_lower = domain.lower()
    query_lower = query.lower()

    # Technical indicator keywords
    technical_keywords = [
        "api",
        "sdk",
        "documentation",
        "guide",
        "tutorial",
        "install",
        "setup",
        "configuration",
        "code",
        "programming",
        "develop",
        "cuda",
        "tensorrt",
        "triton",
        "nccl",
        "cutlass",
    ]

    is_technical_query = any(kw in query_lower for kw in technical_keywords)

    # Boost documentation domains for technical queries
    if is_technical_query:
        if "docs." in domain_lower or "documentation" in domain_lower:
            return 1.3
        if "developer." in domain_lower:
            return 1.2
        if "github.io" in domain_lower:
            return 1.15

    # Boost research domain for research queries
    if (
        any(kw in query_lower for kw in ["research", "paper", "publication", "whitepaper"])
        and "research." in domain_lower
    ):
        return 1.25

    # Boost blog/news for announcement queries
    if any(kw in query_lower for kw in ["announce", "release", "news", "launch"]) and (
        "blog" in domain_lower or "news" in domain_lower
    ):
        return 1.2

    return 1.0  # No boost


def calculate_tfidf_scores(results: list[dict[str, Any]], query: str) -> list[float]:
    """
    Calculate TF-IDF based relevance scores for search results.

    Args:
        results: List of search result dictionaries
        query: Search query string

    Returns:
        List of TF-IDF scores (0-1) for each result
    """
    if not results:
        return []

    # Build corpus from results (combine title + snippet)
    corpus = []
    for result in results:
        title = result.get("title", "")
        snippet = result.get("snippet_plain", result.get("snippet", ""))
        # Remove markdown formatting
        snippet = snippet.replace("**", "")
        doc = f"{title} {snippet}"
        corpus.append(doc)

    # Add query as first document for comparison
    corpus = [query, *corpus]

    try:
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words=list(STOPWORDS),
            ngram_range=(1, 2),  # Unigrams and bigrams
            max_features=1000,
        )

        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(corpus)
        query_vector = tfidf_matrix[0:1]
        result_vectors = tfidf_matrix[1:]

        similarities = cosine_similarity(query_vector, result_vectors)[0]

        return similarities.tolist()

    except Exception as e:
        logger.debug(f"TF-IDF calculation failed: {e}")
        # Return neutral scores on failure
        return [0.5] * len(results)


def expand_query_with_product_variants(query: str) -> str:
    """
    Expand query with NVIDIA product name variants and acronyms.

    This helps match different ways users refer to the same product.
    Example: "CUDA-Q" -> "CUDA-Q OR CUDA Quantum OR cudaquantum"

    Args:
        query: Original search query

    Returns:
        Expanded query string with product variants
    """
    query_lower = query.lower()
    expanded_terms = [query]  # Keep original query

    # Check if query contains any known NVIDIA product
    for product, variants in NVIDIA_PRODUCT_VARIANTS.items():
        if product in query_lower:
            # Add all variants except the ones already in the query
            for variant in variants:
                if variant.lower() not in query_lower:
                    expanded_terms.append(variant)
            break  # Only expand first matching product to avoid over-expansion

    # Combine terms with OR operator if expanded
    if len(expanded_terms) > 1:
        # Limit to 3-4 variants to avoid overly complex queries
        return " ".join(expanded_terms[:4])
    return query


def expand_topic_with_synonyms(topic: str) -> list[str]:
    """
    Expand a topic with related terms and synonyms for better semantic matching.

    Uses a hybrid approach:
    1. NVIDIA product variants (for product name acronyms/variations)
    2. Domain-specific NVIDIA terminology (hardcoded, curated)
    3. WordNet for general English synonyms (automatic, linguistic)

    Args:
        topic: The original topic string

    Returns:
        List of related terms including the original topic (max 15 terms)
    """
    topic_lower = topic.lower()

    # Check for NVIDIA product variants first
    expanded_terms = [topic]
    for product, variants in NVIDIA_PRODUCT_VARIANTS.items():
        if product in topic_lower or any(variant in topic_lower for variant in variants):
            # Add all variants
            expanded_terms.extend(variants)
            expanded_terms.append(product)
            break  # Only expand first matching product

    # Domain-specific synonym/related term mappings
    # Curated NVIDIA-specific terminology and common domain mappings
    synonym_map = {
        # Life Sciences / Biology
        "biochemistry": ["biochemistry", "life sciences", "biology", "molecular biology", "protein", "genomics"],
        "protein": ["protein", "protein folding", "alphafold", "molecular structure"],
        "biology": ["biology", "life sciences", "biochemistry", "molecular biology", "genomics"],
        "genomics": ["genomics", "dna", "rna", "sequencing", "parabricks", "genome analysis"],
        "drug discovery": ["drug discovery", "pharmaceutical", "molecular dynamics", "protein docking", "bionemo"],
        "molecular": ["molecular", "molecular dynamics", "protein", "biochemistry"],
        # AI / ML
        "llm": ["llm", "large language model", "language model", "nemo", "megatron", "transformer"],
        "generative ai": ["generative ai", "gen ai", "genai", "llm", "diffusion", "stable diffusion"],
        "deep learning": ["deep learning", "neural network", "machine learning", "ai", "training"],
        "training": ["training", "fine-tuning", "pre-training", "model training"],
        # GPU / Computing
        "gpu": ["gpu", "cuda", "graphics card", "accelerator"],
        "cuda": ["cuda", "gpu programming", "parallel computing"],
        "tensorrt": ["tensorrt", "inference", "optimization"],
        # Infrastructure
        "kubernetes": ["kubernetes", "k8s", "container", "orchestration"],
        "docker": ["docker", "container", "containerization"],
        # Gaming / Graphics
        "rtx": ["rtx", "ray tracing", "graphics", "geforce", "gaming"],
        "ray tracing": ["ray tracing", "rtx", "graphics", "rendering"],
        # Autonomous Vehicles
        "autonomous": ["autonomous", "self-driving", "drive", "av", "autonomous vehicle"],
        "self-driving": ["self-driving", "autonomous", "drive", "av"],
        # Robotics
        "robotics": ["robotics", "isaac", "manipulation", "navigation"],
        "robot": ["robot", "robotics", "isaac", "automation"],
        # Virtual Worlds
        "omniverse": ["omniverse", "usd", "3d", "simulation", "digital twin"],
        "metaverse": ["metaverse", "virtual world", "omniverse", "3d"],
    }

    # Check for exact matches or partial matches in synonym map
    for key, synonyms in synonym_map.items():
        # Check if the key is in the topic or vice versa
        if key in topic_lower or topic_lower in key:
            expanded_terms.extend(synonyms)
            break

    terms_before_wordnet = len(expanded_terms)

    # Enhance with WordNet for general English synonyms
    # This catches terms not in our curated map
    try:
        from nltk.corpus import wordnet

        synsets = wordnet.synsets(topic_lower.replace(" ", "_"))[:3]  # Top 3 word senses

        for syn in synsets:
            # Get top 3 synonyms per sense
            for lemma in syn.lemmas()[:3]:
                synonym = lemma.name().replace("_", " ")
                # Add if not already present (case-insensitive check)
                if synonym.lower() not in [t.lower() for t in expanded_terms]:
                    expanded_terms.append(synonym)

        logger.debug(f"WordNet expanded '{topic}' with {len(expanded_terms) - terms_before_wordnet} additional terms")

    except (LookupError, AttributeError) as e:
        # WordNet data not available or error in lookup
        logger.debug(f"WordNet lookup skipped for '{topic}': {e}")
    except Exception as e:
        # Catch any other errors to prevent breaking the search
        logger.debug(f"WordNet error for '{topic}': {e}")

    # Remove duplicates while preserving order
    seen = set()
    unique_terms = []
    for term in expanded_terms:
        term_lower = term.lower()
        if term_lower not in seen:
            seen.add(term_lower)
            unique_terms.append(term)

    # Limit to 15 terms to avoid overly broad searches
    return unique_terms[:15]


def extract_matched_keywords(query: str, result: dict[str, Any]) -> list[str]:
    """
    Extract which meaningful keywords from the query matched in the result.

    Only returns actual keywords (non-stopwords) that appear in the result.

    Args:
        query: Search query string
        result: Search result dictionary

    Returns:
        List of matched keywords from the query
    """
    # Extract only meaningful keywords from query
    keywords = extract_keywords(query)

    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()
    url = result.get("url", "").lower()

    matched = []
    for keyword in keywords:
        if keyword in title or keyword in snippet or keyword in url:
            matched.append(keyword)

    return matched


def calculate_search_relevance(result: dict[str, Any], query: str, domain_boost: float = 1.0) -> int:
    """
    Calculate relevance score for a search result using multiple signals.

    Uses:
    - Exact keyword matching
    - Fuzzy keyword matching (handles typos)
    - Phrase matching (multi-word phrases)
    - Domain-specific boosting

    Args:
        result: Search result dictionary
        query: Search query string
        domain_boost: Domain boost multiplier (default 1.0)

    Returns:
        Relevance score from 0-100
    """
    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()
    url = result.get("url", "").lower()

    # Extract meaningful keywords only (no stopwords)
    keywords = extract_keywords(query)
    phrases = extract_phrases(query)

    if not keywords:
        return 0

    # === Part 1: Exact keyword matching (base score) ===
    base_score = 0
    max_score_per_keyword = 6  # 3 + 2 + 1

    for keyword in keywords:
        keyword_score = 0

        # Title matches are most important (3 points)
        if keyword in title:
            keyword_score += 3

        # Snippet matches are moderately important (2 points)
        if keyword in snippet:
            keyword_score += 2

        # URL matches are least important (1 point)
        if keyword in url:
            keyword_score += 1

        base_score += keyword_score

    # === Part 2: Fuzzy keyword matching (bonus points) ===
    fuzzy_bonus = 0
    for keyword in keywords:
        # Only apply fuzzy if no exact match
        if keyword not in title and keyword not in snippet:
            title_fuzzy = calculate_fuzzy_match_score(keyword, title, threshold=80)
            snippet_fuzzy = calculate_fuzzy_match_score(keyword, snippet, threshold=80)

            # Award partial points for fuzzy matches
            fuzzy_bonus += title_fuzzy * 1.5  # Up to 1.5 points for title fuzzy
            fuzzy_bonus += snippet_fuzzy * 1.0  # Up to 1.0 points for snippet fuzzy

    # === Part 3: Phrase matching (bonus points) ===
    phrase_bonus = 0
    for phrase in phrases:
        if phrase in title:
            phrase_bonus += 2.0  # Bonus for phrase in title
        elif phrase in snippet:
            phrase_bonus += 1.0  # Bonus for phrase in snippet

    # === Part 4: Combine scores ===
    raw_score = base_score + fuzzy_bonus + phrase_bonus

    # Max possible: keywords * 6 + fuzzy bonus (2.5 per keyword) + phrase bonus (2 per phrase)
    max_possible_score = (len(keywords) * max_score_per_keyword) + (len(keywords) * 2.5) + (len(phrases) * 2.0)

    # Normalize to 0-100 scale
    normalized_score = int(raw_score / max_possible_score * 100) if max_possible_score > 0 else 0

    # Apply domain boost
    boosted_score = int(normalized_score * domain_boost)

    # Cap at 100
    return min(boosted_score, 100)
