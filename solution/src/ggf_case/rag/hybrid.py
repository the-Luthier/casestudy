"""
Hybrid retrieval combining keyword, BM25, and embeddings.
Turkce: Anahtar kelime, BM25 ve embedding sonuclarini birlestiren hibrit arama.
"""

# TODO: Implement hybrid retrieval ----

from typing import Optional

from .indexer import CodebaseIndex
from .retriever import RetrievalResult, retrieve_keyword, try_embedding_retrieval
from .bm25 import retrieve_bm25
def weighted_combination(result_lists, weights=None):
	"""
	Combine multiple result lists by weighted sum of normalized scores.
	result_lists: list of lists of RetrievalResult
	weights: list of floats, same length as result_lists
	"""
	scores = {}
	if weights is None:
		weights = [1.0] * len(result_lists)
	# Normalize scores in each list
	for method_idx, results in enumerate(result_lists):
		if not results:
			continue
		max_score = max(r.score for r in results) or 1.0
		for r in results:
			key = id(r.chunk)
			norm_score = (r.score / max_score) * weights[method_idx]
			if key not in scores:
				scores[key] = {'score': 0.0, 'chunk': r.chunk, 'methods': set()}
			scores[key]['score'] += norm_score
			scores[key]['methods'].add(r.method)
	fused = [RetrievalResult(chunk=v['chunk'], score=v['score'], method='hybrid') for v in scores.values()]
	fused.sort(key=lambda r: r.score, reverse=True)
	return fused

def hybrid_retrieve(
	index: CodebaseIndex,
	query: str,
	top_k: int = 8,
	file_filter: Optional[list[str]] = None,
	embedding_model: str = "all-MiniLM-L6-v2",
	weights: Optional[list[float]] = None,
	method: str = "rrf",  # or "weighted"
) -> list[RetrievalResult]:
	"""
	Main entry point for hybrid retrieval. Supports RRF or weighted combination.
	"""
	bm25_results = retrieve_bm25(index, query, top_k=top_k*2, file_filter=file_filter)
	keyword_results = retrieve_keyword(index, query, top_k=top_k*2, file_filter=file_filter)
	embedding_results = try_embedding_retrieval(index, query, top_k=top_k*2, model_name=embedding_model) or []
	lists = [bm25_results, embedding_results, keyword_results]
	if weights is None:
		weights = [1.4, 1.4, 0.2]
	if method == "weighted":
		fused = weighted_combination(lists, weights=weights)
	else:
		fused = reciprocal_rank_fusion(lists, weights=weights)
	return fused[:top_k]

from typing import Optional
from .indexer import CodebaseIndex
from .retriever import RetrievalResult, retrieve_keyword, try_embedding_retrieval
from .bm25 import retrieve_bm25

def reciprocal_rank_fusion(result_lists, weights=None, k=60):
	"""
	RRF: Combines multiple ranked lists using reciprocal rank fusion.
	result_lists: list of lists of RetrievalResult
	weights: list of floats, same length as result_lists
	k: RRF constant (default 60)
	"""
	scores = {}
	if weights is None:
		weights = [1.0] * len(result_lists)
	for method_idx, results in enumerate(result_lists):
		for rank, r in enumerate(results):
			key = id(r.chunk)
			score = weights[method_idx] / (k + rank)
			if key not in scores:
				scores[key] = {'score': 0.0, 'chunk': r.chunk, 'methods': set()}
			scores[key]['score'] += score
			scores[key]['methods'].add(r.method)
	# Convert to RetrievalResult list
	fused = [RetrievalResult(chunk=v['chunk'], score=v['score'], method='hybrid') for v in scores.values()]
	fused.sort(key=lambda r: r.score, reverse=True)
	return fused

def retrieve_hybrid(
	index: CodebaseIndex,
	query: str,
	top_k: int = 8,
	file_filter: Optional[list[str]] = None,
	embedding_model: str = "all-MiniLM-L6-v2",
	weights: Optional[list[float]] = None,
) -> list[RetrievalResult]:
	"""
	Hybrid retrieval: combines BM25, embedding, and keyword using RRF.
	"""
	bm25_results = retrieve_bm25(index, query, top_k=top_k*2, file_filter=file_filter)
	keyword_results = retrieve_keyword(index, query, top_k=top_k*2, file_filter=file_filter)
	embedding_results = try_embedding_retrieval(index, query, top_k=top_k*2, model_name=embedding_model) or []
	lists = [bm25_results, embedding_results, keyword_results]
	if weights is None:
		weights = [1.4, 1.4, 0.2]  # BM25, embedding, keyword
	fused = reciprocal_rank_fusion(lists, weights=weights)
	return fused[:top_k]
