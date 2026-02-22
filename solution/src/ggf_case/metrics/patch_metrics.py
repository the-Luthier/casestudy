"""
Patch quality metrics for diff evaluation.
Turkce: Diff kalitesi icin metrikler.
"""

# TODO: Implement patch quality metrics ----

import re
from typing import List


def _normalize_diff(diff: str) -> str:
	return "\n".join(line.rstrip() for line in diff.strip().splitlines())


def exact_match(predicted: str, gold: str) -> float:
	"""
	Return 1.0 if diffs match exactly after normalization, else 0.0.
	"""
	return 1.0 if _normalize_diff(predicted) == _normalize_diff(gold) else 0.0


def _extract_hunks(diff: str) -> List[str]:
	"""
	Extract unified diff hunks as strings starting from @@ lines.
	"""
	lines = diff.splitlines()
	hunks: List[str] = []
	current: List[str] = []
	for line in lines:
		if line.startswith("@@"):
			if current:
				hunks.append("\n".join(current))
			current = [line]
		elif current:
			current.append(line)
	if current:
		hunks.append("\n".join(current))
	return hunks


def hunk_match_rate(predicted: str, gold: str) -> float:
	"""
	Compute fraction of gold hunks that appear verbatim in predicted diff.
	"""
	pred_hunks = {_normalize_diff(h) for h in _extract_hunks(predicted)}
	gold_hunks = [_normalize_diff(h) for h in _extract_hunks(gold)]
	if not gold_hunks:
		return 1.0 if not pred_hunks else 0.0
	match_count = sum(1 for h in gold_hunks if h in pred_hunks)
	return match_count / len(gold_hunks)
