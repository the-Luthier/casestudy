"""
Reranking utilities for retrieval results.
Turkce: Retrieval sonuclarini yeniden siralama araclari.
"""

# TODO: Implement reranking ----
# Reranker base class
class Reranker:
	def rerank(self, results):
		"""
		Rerank a list of RetrievalResult objects. Should return a new sorted list.
		"""
		raise NotImplementedError("Reranker must implement rerank method.")

# No-operation reranker (returns input as is)
class NoOpReranker(Reranker):
	def rerank(self, results):
		return results

# Simple heuristic reranker (stable, score-first)
class SimpleReranker(Reranker):
	def __init__(self, length_penalty: float = 0.0):
		self.length_penalty = length_penalty

	def rerank(self, results):
		# Prefer higher scores, then shorter chunks if penalty is set
		def key_fn(r):
			content_len = len(r.chunk.content) if r and r.chunk and r.chunk.content else 0
			return (r.score, -self.length_penalty * content_len)
		return sorted(results, key=key_fn, reverse=True)

# Factory function
def create_reranker(name: str = None, **kwargs) -> Reranker:
	"""
	Create a reranker instance by name. Default is NoOpReranker.
	"""
	if name is None or name.lower() == "noop":
		return NoOpReranker()
	if name.lower() in ("simple", "heuristic"):
		return SimpleReranker(length_penalty=kwargs.get("length_penalty", 0.0))
	# Placeholder for future reranker types
	raise ValueError(f"Unknown reranker: {name}")
