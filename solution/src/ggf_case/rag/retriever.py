"""
Code retrieval system for RAG.

Turkce: Coklu stratejili RAG retrieval sistemi (keyword, BM25, embedding, hybrid).
Turkce: Config ile strateji ve reranker secimi desteklenir.

BASELINE: Keyword retrieval is provided and working.
TODO: Implement multi-strategy retrieval for Phase 1. ----
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Optional

from rich.console import Console

from .indexer import CodeChunk, CodebaseIndex

console = Console()

_EMBEDDING_MODEL_CACHE: dict[str, object] = {}
_EMBEDDING_INDEX_CACHE: dict[tuple[str, str], object] = {}


@dataclass
class RetrievalResult:
    """A retrieved chunk with relevance score."""
    chunk: CodeChunk
    score: float
    method: str  # "keyword", "bm25", "embedding", "hybrid"


# ============================================================================
# BASELINE: Keyword retrieval (PROVIDED — do not modify)
# ============================================================================

def keyword_score(query: str, chunk: CodeChunk) -> float:
    """
    Simple keyword-based relevance scoring.

    Scores based on:
    - Term frequency in content
    - Symbol name matches (weighted higher)
    - File path relevance
    """
    query_terms = set(re.findall(r'\w+', query.lower()))
    if not query_terms:
        return 0.0

    score = 0.0
    content_lower = chunk.content.lower()

    # Term frequency scoring
    for term in query_terms:
        count = content_lower.count(term)
        if count > 0:
            score += min(count, 5) * 1.0  # Cap at 5 occurrences

    # Symbol name matching (higher weight)
    for symbol in chunk.symbols:
        symbol_lower = symbol.lower()
        for term in query_terms:
            if term in symbol_lower:
                score += 5.0  # Strong signal
            if symbol_lower == term:
                score += 10.0  # Exact match

    # File path relevance
    path_lower = chunk.file_path.lower()
    for term in query_terms:
        if term in path_lower:
            score += 3.0

    return score


def retrieve_keyword(
    index: CodebaseIndex,
    query: str,
    top_k: int = 8,
    file_filter: Optional[list[str]] = None,
) -> list[RetrievalResult]:
    """
    Retrieve top-k code chunks using keyword matching.

    This is the baseline retrieval method that works without
    any external dependencies.

    Args:
        index: The codebase index to search.
        query: Natural language query.
        top_k: Number of results to return.
        file_filter: Optional list of file paths to restrict search to.

    Returns:
        List of RetrievalResult sorted by relevance score.
    """
    results: list[RetrievalResult] = []

    for chunk in index.chunks:
        # Apply file filter if provided
        if file_filter:
            if not any(f in chunk.file_path for f in file_filter):
                continue

        score = keyword_score(query, chunk)
        if score > 0:
            results.append(RetrievalResult(
                chunk=chunk,
                score=score,
                method="keyword",
            ))

    # Sort by score descending
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


# ============================================================================
# BASELINE: Embedding retrieval stub (PROVIDED — candidates can improve)
# ============================================================================

def try_embedding_retrieval(
    index: CodebaseIndex,
    query: str,
    top_k: int = 8,
    model_name: str = "all-MiniLM-L6-v2",
) -> Optional[list[RetrievalResult]]:
    """
    Attempt embedding-based retrieval using sentence-transformers.

    Returns None if sentence-transformers is not installed.
    Candidates can improve this function for better retrieval.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        console.print("[yellow]sentence-transformers not installed, falling back to keyword retrieval[/yellow]")
        return None

    def _index_cache_key() -> str:
        parts = [f"{k}:{v}" for k, v in sorted(index.file_hashes.items())]
        raw = "|".join(parts)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    try:
        model = _EMBEDDING_MODEL_CACHE.get(model_name)
        if model is None:
            model = SentenceTransformer(model_name)
            _EMBEDDING_MODEL_CACHE[model_name] = model

        # Encode query
        query_embedding = model.encode(query, normalize_embeddings=True)

        # Encode all chunks (cached by index hash)
        cache_key = (model_name, _index_cache_key())
        chunk_embeddings = _EMBEDDING_INDEX_CACHE.get(cache_key)
        if chunk_embeddings is None:
            chunk_texts = [
                f"File: {c.file_path}\nSymbols: {', '.join(c.symbols)}\n{c.content}"
                for c in index.chunks
            ]
            if not chunk_texts:
                return []
            chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True)
            _EMBEDDING_INDEX_CACHE[cache_key] = chunk_embeddings

        # Compute cosine similarities
        similarities = np.dot(chunk_embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(RetrievalResult(
                chunk=index.chunks[idx],
                score=float(similarities[idx]),
                method="embedding",
            ))

        return results

    except Exception as e:
        console.print(f"[yellow]Embedding retrieval failed: {e}[/yellow]")
        return None


# ============================================================================
# TODO: Implement multi-strategy retrieval ----
# ============================================================================

def retrieve(
    index: CodebaseIndex,
    query: str,
    top_k: Optional[int] = 8,
    file_filter: Optional[list[str]] = None,
    use_embeddings: bool = True,
    embedding_model: Optional[str] = "all-MiniLM-L6-v2",
    strategy: Optional[str] = "keyword",
) -> list[RetrievalResult]:
    """
    Main retrieval function. Supports multiple strategies.

    Currently only "keyword" strategy works.
    TODO: Implement "bm25", "hybrid", and "embedding" strategies for Phase 1. ----
    """
    # Resolve defaults from config when not provided
    if strategy is None or embedding_model is None or top_k is None:
        from ..config import get_settings
        settings = get_settings()
        if strategy is None:
            strategy = settings.retrieval_strategy
        if embedding_model is None:
            embedding_model = settings.embedding_model
        if top_k is None:
            top_k = settings.top_k

    if strategy == "bm25":
        try:
            from .bm25 import retrieve_bm25
            results = retrieve_bm25(index, query, top_k, file_filter)
        except Exception as e:
            console.print(f"[yellow]BM25 retrieval failed: {e}. Falling back to keyword[/yellow]")
            results = retrieve_keyword(index, query, top_k, file_filter)
        return _apply_reranker(results, query)

    elif strategy == "hybrid":
        try:
            from .hybrid import retrieve_hybrid
            results = retrieve_hybrid(index, query, top_k, file_filter, embedding_model=embedding_model)
        except Exception as e:
            console.print(f"[yellow]Hybrid retrieval failed: {e}. Falling back to keyword[/yellow]")
            results = retrieve_keyword(index, query, top_k, file_filter)
        return _apply_reranker(results, query)

    elif strategy == "embedding":
        if use_embeddings:
            results = try_embedding_retrieval(index, query, top_k, embedding_model)
            if results is not None:
                return _apply_reranker(results, query)
        results = retrieve_keyword(index, query, top_k, file_filter)
        return _apply_reranker(results, query)

    else:  # keyword (default)
        results = retrieve_keyword(index, query, top_k, file_filter)
        return _apply_reranker(results, query)


def _apply_reranker(results: list[RetrievalResult], query: str) -> list[RetrievalResult]:
    """
    Apply reranker if enabled in config.
    """
    from ..config import get_settings
    settings = get_settings()
    results = _boost_and_dedupe(results, query)
    if not settings.reranker_enabled:
        return results
    try:
        from .reranker import create_reranker
        reranker = create_reranker("simple")
        return reranker.rerank(results)
    except Exception:
        return results


def _boost_and_dedupe(results: list[RetrievalResult], query: str) -> list[RetrievalResult]:
    """Boost file-path matches and keep only best chunk per file."""
    if not results:
        return results
    query_terms = set(re.findall(r"\w+", query.lower()))
    best_by_file: dict[str, RetrievalResult] = {}
    for r in results:
        path_lower = r.chunk.file_path.lower()
        score = r.score
        for term in query_terms:
            if term and term in path_lower:
                score += 2.0
        boosted = RetrievalResult(chunk=r.chunk, score=score, method=r.method)
        existing = best_by_file.get(r.chunk.file_path)
        if existing is None or boosted.score > existing.score:
            best_by_file[r.chunk.file_path] = boosted
    deduped = list(best_by_file.values())
    deduped.sort(key=lambda x: x.score, reverse=True)
    return deduped


# ============================================================================
# Context formatting (PROVIDED — do not modify)
# ============================================================================

def format_context(results: list[RetrievalResult], max_tokens: int = 4000) -> str:
    """
    Format retrieval results into a context string for the LLM.

    Args:
        results: Retrieved code chunks.
        max_tokens: Approximate max character budget (rough estimate).

    Returns:
        Formatted context string.
    """
    parts: list[str] = []
    total_len = 0

    for r in results:
        header = f"=== {r.chunk.file_path} (lines {r.chunk.start_line}-{r.chunk.end_line}, score: {r.score:.2f}) ==="
        section = f"{header}\n{r.chunk.content}\n"

        if total_len + len(section) > max_tokens * 4:  # rough char-to-token ratio
            break

        parts.append(section)
        total_len += len(section)

    return "\n".join(parts)
