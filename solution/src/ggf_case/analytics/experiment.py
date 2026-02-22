"""
A/B experiment framework for comparing configurations.
Turkce: Konfigurasyonlari karsilastirmak icin A/B deney catisi.
"""

# TODO: Implement A/B experiment framework ----

import json
import math
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class ExperimentConfig:
	name: str
	runs: int = 5


@dataclass
class StatisticalResult:
	t_stat: float
	p_value: float
	cohens_d: float


@dataclass
class ExperimentReport:
	config_a: str
	config_b: str
	mean_a: float
	mean_b: float
	stat: StatisticalResult
	notes: list[str] = field(default_factory=list)


class ExperimentRunner:
	"""Run paired A/B experiments on metrics."""

	def _paired_t_test(self, a: list[float], b: list[float]) -> StatisticalResult:
		if len(a) != len(b) or len(a) < 2:
			return StatisticalResult(0.0, 1.0, 0.0)

		diffs = [ai - bi for ai, bi in zip(a, b)]
		mean_diff = sum(diffs) / len(diffs)
		var = sum((d - mean_diff) ** 2 for d in diffs) / (len(diffs) - 1)
		std = math.sqrt(var) if var > 0 else 0.0
		t_stat = (mean_diff / (std / math.sqrt(len(diffs)))) if std > 0 else 0.0

		# Approximate p-value using normal distribution for simplicity
		p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

		cohens_d = (mean_diff / std) if std > 0 else 0.0

		return StatisticalResult(t_stat=t_stat, p_value=p_value, cohens_d=cohens_d)

	def generate_report(self, config_a: ExperimentConfig, config_b: ExperimentConfig,
						scores_a: Iterable[float], scores_b: Iterable[float]) -> ExperimentReport:
		a = list(scores_a)
		b = list(scores_b)
		mean_a = sum(a) / len(a) if a else 0.0
		mean_b = sum(b) / len(b) if b else 0.0

		stat = self._paired_t_test(a, b)
		notes = []
		if stat.p_value < 0.05:
			notes.append("Statistically significant difference (p < 0.05).")
		else:
			notes.append("No statistically significant difference (p >= 0.05).")

		return ExperimentReport(
			config_a=config_a.name,
			config_b=config_b.name,
			mean_a=mean_a,
			mean_b=mean_b,
			stat=stat,
			notes=notes,
		)

	def export_report(self, report: ExperimentReport, path) -> None:
		"""Export report to JSON."""
		data = {
			"config_a": report.config_a,
			"config_b": report.config_b,
			"mean_a": report.mean_a,
			"mean_b": report.mean_b,
			"t_stat": report.stat.t_stat,
			"p_value": report.stat.p_value,
			"cohens_d": report.stat.cohens_d,
			"notes": report.notes,
		}
		with open(path, "w", encoding="utf-8") as f:
			f.write(json.dumps(data, indent=2))
