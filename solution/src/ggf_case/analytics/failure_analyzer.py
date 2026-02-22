"""
Failure analysis utilities for evaluation results.
Turkce: Degerlendirme sonuclarinda hata analizi ve siniflandirma.
"""

# TODO: Implement failure analysis ----

import json
from dataclasses import dataclass, field
from typing import Iterable


FAILURE_CATEGORIES = [
	"RETRIEVAL_MISS",
	"GENERATION_ERROR",
	"APPLY_FAILURE",
	"BUILD_FAILURE",
	"CHECK_FAILURE",
]


@dataclass
class FailureReport:
	total_failures: int
	by_category: dict[str, int]
	patterns: list[str] = field(default_factory=list)
	recommendations: list[str] = field(default_factory=list)
	correlation_notes: list[str] = field(default_factory=list)


class FailureAnalyzer:
	"""Analyze task failures and attribute root causes."""

	def classify_failure(self, result: dict) -> str:
		"""Classify a single failure based on result flags and error text."""
		error = (result.get("error") or "").lower()
		if result.get("retrieval_count", 0) == 0:
			return "RETRIEVAL_MISS"
		if "diff" in error or "patch" in error or not result.get("patch_generated", True):
			return "GENERATION_ERROR"
		if "apply" in error or not result.get("patch_applied", True):
			return "APPLY_FAILURE"
		if "build" in error:
			return "BUILD_FAILURE"
		return "CHECK_FAILURE"

	def analyze_results(self, results: Iterable[dict]) -> FailureReport:
		"""Analyze a list of task results and aggregate failure causes."""
		by_category = {c: 0 for c in FAILURE_CATEGORIES}
		failures = []

		for r in results:
			if r.get("success"):
				continue
			category = self.classify_failure(r)
			by_category[category] = by_category.get(category, 0) + 1
			failures.append({"result": r, "category": category})

		patterns = self._identify_patterns(failures)
		recommendations = self._generate_recommendations(by_category)
		correlation_notes = self._analyze_correlations(results)

		return FailureReport(
			total_failures=len(failures),
			by_category=by_category,
			patterns=patterns,
			recommendations=recommendations,
			correlation_notes=correlation_notes,
		)

	def _identify_patterns(self, failures: list[dict]) -> list[str]:
		"""Identify recurring patterns in failures."""
		pattern_notes: list[str] = []
		if not failures:
			return pattern_notes

		retrieval_miss = sum(1 for f in failures if f["category"] == "RETRIEVAL_MISS")
		if retrieval_miss >= max(2, len(failures) // 2):
			pattern_notes.append("Many failures are due to RETRIEVAL_MISS; improve retrieval strategy.")

		generation_err = sum(1 for f in failures if f["category"] == "GENERATION_ERROR")
		if generation_err >= 2:
			pattern_notes.append("Multiple GENERATION_ERROR cases; strengthen prompt or diff validation.")

		return pattern_notes

	def _generate_recommendations(self, by_category: dict[str, int]) -> list[str]:
		"""Generate recommendations based on failure distribution."""
		recs: list[str] = []
		if by_category.get("RETRIEVAL_MISS", 0) > 0:
			recs.append("Increase top_k or switch to hybrid retrieval.")
		if by_category.get("GENERATION_ERROR", 0) > 0:
			recs.append("Tighten prompt constraints and enforce structured output.")
		if by_category.get("APPLY_FAILURE", 0) > 0:
			recs.append("Reduce patch size and improve diff correctness.")
		if by_category.get("BUILD_FAILURE", 0) > 0:
			recs.append("Add TypeScript type checks to prompt and tests.")
		if by_category.get("CHECK_FAILURE", 0) > 0:
			recs.append("Expand context window or refine acceptance criteria parsing.")
		return recs

	def _analyze_correlations(self, results: Iterable[dict]) -> list[str]:
		"""Basic correlation notes between retrieval quality and success."""
		notes: list[str] = []
		total = 0
		low_retrieval_fail = 0
		for r in results:
			total += 1
			if not r.get("success") and r.get("retrieval_count", 0) <= 1:
				low_retrieval_fail += 1
		if total > 0 and low_retrieval_fail > 0:
			notes.append("Correlation: low retrieval_count is associated with failures.")
		return notes

	def print_report(self, report: FailureReport) -> None:
		"""Print a human-readable report."""
		print("Failure Analysis Report")
		print(f"Total failures: {report.total_failures}")
		print("By category:")
		for k, v in report.by_category.items():
			print(f"  {k}: {v}")
		if report.patterns:
			print("Patterns:")
			for p in report.patterns:
				print(f"  - {p}")
		if report.recommendations:
			print("Recommendations:")
			for r in report.recommendations:
				print(f"  - {r}")
		if report.correlation_notes:
			print("Correlations:")
			for n in report.correlation_notes:
				print(f"  - {n}")

	def export_report(self, report: FailureReport, path) -> None:
		"""Export report to JSON."""
		data = {
			"total_failures": report.total_failures,
			"by_category": report.by_category,
			"patterns": report.patterns,
			"recommendations": report.recommendations,
			"correlation_notes": report.correlation_notes,
		}
		with open(path, "w", encoding="utf-8") as f:
			f.write(json.dumps(data, indent=2))
