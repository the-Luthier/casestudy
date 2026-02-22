"""
BM25 retrieval implementation.
Turkce: BM25 arama ve siralama mantigi.
"""

# TODO: Implement BM25 retrieval ----
import math
import re
from collections import defaultdict, Counter
from typing import Optional
from .indexer import CodebaseIndex, CodeChunk
from .retriever import RetrievalResult

class BM25Retriever:
	def __init__(self, index: CodebaseIndex, k1: float = 1.5, b: float = 0.75):
		self.index = index
		self.k1 = k1
		self.b = b
		self.inverted_index = defaultdict(list)  # term -> list of (chunk_idx, freq)
		self.doc_freq = defaultdict(int)  # term -> doc freq
		self.doc_lens = []
		self.avgdl = 0.0
		self.N = len(index.chunks)
		self._build_index()

	def _tokenize(self, text: str) -> list[str]:
		return re.findall(r'\w+', text.lower())

	def _build_index(self):
		total_len = 0
		for i, chunk in enumerate(self.index.chunks):
			tokens = self._tokenize(chunk.content)
			total_len += len(tokens)
			self.doc_lens.append(len(tokens))
			counts = Counter(tokens)
			for term, freq in counts.items():
				self.inverted_index[term].append((i, freq))
				self.doc_freq[term] += 1
		self.avgdl = total_len / self.N if self.N > 0 else 0.0

	def _idf(self, term: str) -> float:
		df = self.doc_freq.get(term, 0)
		if df == 0:
			return 0.0
		return math.log(1 + (self.N - df + 0.5) / (df + 0.5))

	def score(self, query: str, chunk_idx: int) -> float:
		chunk = self.index.chunks[chunk_idx]
		tokens = self._tokenize(chunk.content)
		dl = len(tokens)
		score = 0.0
		query_terms = self._tokenize(query)
		counts = Counter(tokens)
		for term in set(query_terms):
			if term not in self.doc_freq:
				continue
			idf = self._idf(term)
			freq = counts.get(term, 0)
			denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
			score += idf * (freq * (self.k1 + 1)) / (denom + 1e-8)
		return score


def tokenize(text: str) -> list[str]:
	"""Tokenize text for BM25."""
	return re.findall(r"\w+", text.lower())


def build_bm25_index(index: CodebaseIndex, k1: float = 1.5, b: float = 0.75) -> BM25Retriever:
	"""Build a BM25 retriever over a codebase index."""
	return BM25Retriever(index, k1=k1, b=b)


def bm25_score(query: str, chunk: CodeChunk, retriever: Optional[BM25Retriever] = None,
			 index: Optional[CodebaseIndex] = None) -> float:
	"""Score a single chunk with BM25."""
	if retriever is None:
		if index is None:
			raise ValueError("index is required if retriever is not provided")
		retriever = BM25Retriever(index)
	try:
		chunk_idx = retriever.index.chunks.index(chunk)
		return retriever.score(query, chunk_idx)
	except ValueError:
		return 0.0

def retrieve_bm25(
	index: CodebaseIndex,
	query: str,
	top_k: int = 8,
	file_filter: Optional[list[str]] = None,
	k1: float = 1.5,
	b: float = 0.75,
) -> list[RetrievalResult]:
	"""
	Retrieve top-k code chunks using BM25 ranking.
	"""
	retriever = BM25Retriever(index, k1=k1, b=b)
	results = []
	for i, chunk in enumerate(index.chunks):
		if file_filter:
			if not any(f in chunk.file_path for f in file_filter):
				continue
		score = retriever.score(query, i)
		if score > 0:
			results.append(RetrievalResult(
				chunk=chunk,
				score=score,
				method="bm25",
			))
	results.sort(key=lambda r: r.score, reverse=True)
	return results[:top_k]
