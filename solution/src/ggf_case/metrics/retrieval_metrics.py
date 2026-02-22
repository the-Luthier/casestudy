"""
Retrieval quality metrics (precision, recall, MRR, NDCG, hit rate).
Turkce: Retrieval kalitesi icin metrikler (precision, recall, MRR, NDCG, hit rate).
"""

# TODO: Implement retrieval quality metrics ----

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass
class RetrievalScores:
	precision_at_k: float
	recall_at_k: float
	mrr: float
	ndcg_at_k: float
	hit_rate: float
	num_queries: int


def _precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
	if k <= 0:
		return 0.0
	top_k = _dedupe_preserve_order(retrieved)[:k]
	if not top_k:
		return 0.0
	hits = sum(1 for r in top_k if r in relevant)
	return hits / k


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
	if not relevant:
		return 0.0
	top_k = _dedupe_preserve_order(retrieved)[:k]
	hits = {r for r in top_k if r in relevant}
	return len(hits) / len(relevant)


def _mrr(retrieved: list[str], relevant: set[str]) -> float:
	for idx, r in enumerate(_dedupe_preserve_order(retrieved), start=1):
		if r in relevant:
			return 1.0 / idx
	return 0.0


def _dcg(retrieved: list[str], relevant: set[str], k: int) -> float:
	dcg = 0.0
	seen: set[str] = set()
	for idx, r in enumerate(_dedupe_preserve_order(retrieved)[:k], start=1):
		if r in relevant and r not in seen:
			dcg += 1.0 / math.log2(idx + 1)
			seen.add(r)
	return dcg


def _ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
	if not relevant:
		return 0.0
	dcg = _dcg(retrieved, relevant, k)
	ideal_hits = min(k, len(relevant))
	ideal_dcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
	if ideal_dcg == 0:
		return 0.0
	return dcg / ideal_dcg


def _hit_rate(retrieved: list[str], relevant: set[str], k: int) -> float:
	return 1.0 if any(r in relevant for r in _dedupe_preserve_order(retrieved)[:k]) else 0.0


def compute_retrieval_scores(queries: Iterable[dict], k: int = 5) -> RetrievalScores:
	"""
	Compute average retrieval metrics over multiple queries.
	Each query: {"retrieved": [...], "relevant": [...]}
	"""
	total_precision = 0.0
	total_recall = 0.0
	total_mrr = 0.0
	total_ndcg = 0.0
	total_hit = 0.0
	num_queries = 0

	for q in queries:
		retrieved = q.get("retrieved", []) or []
		relevant_list = q.get("relevant", []) or []
		relevant = set(relevant_list)

		total_precision += _precision_at_k(retrieved, relevant, k)
		total_recall += _recall_at_k(retrieved, relevant, k)
		total_mrr += _mrr(retrieved, relevant)
		total_ndcg += _ndcg_at_k(retrieved, relevant, k)
		total_hit += _hit_rate(retrieved, relevant, k)
		num_queries += 1

	if num_queries == 0:
		return RetrievalScores(0.0, 0.0, 0.0, 0.0, 0.0, 0)

	return RetrievalScores(
		precision_at_k=total_precision / num_queries,
		recall_at_k=total_recall / num_queries,
		mrr=total_mrr / num_queries,
		ndcg_at_k=total_ndcg / num_queries,
		hit_rate=total_hit / num_queries,
		num_queries=num_queries,
	)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
	seen: set[str] = set()
	deduped: list[str] = []
	for item in items:
		if item in seen:
			continue
		seen.add(item)
		deduped.append(item)
	return deduped


def precision_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
	"""Public precision@k wrapper."""
	return _precision_at_k(retrieved, set(relevant), k)


def recall_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
	"""Public recall@k wrapper."""
	return _recall_at_k(retrieved, set(relevant), k)


def mrr(retrieved: list[str], relevant: Iterable[str]) -> float:
	"""Public MRR wrapper."""
	return _mrr(retrieved, set(relevant))


def ndcg_at_k(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
	"""Public NDCG@k wrapper."""
	return _ndcg_at_k(retrieved, set(relevant), k)


def hit_rate(retrieved: list[str], relevant: Iterable[str], k: int) -> float:
	"""Public hit rate wrapper."""
	return _hit_rate(retrieved, set(relevant), k)
