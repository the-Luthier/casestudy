"""
Model evaluation and comparison utilities.
Turkce: Model karsilastirma ve degerlendirme araclari.
"""

# TODO: Implement model evaluation and comparison ----

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class TaskComparison:
	task_id: str
	base_success: bool
	tuned_success: bool


@dataclass
class ComparisonReport:
	base_pass_rate: float
	tuned_pass_rate: float
	total_tasks: int
	per_task: list[TaskComparison] = field(default_factory=list)


class ModelEvaluator:
	"""Compare base vs fine-tuned model results."""

	def generate_comparison(self, base_results: Iterable[dict], tuned_results: Iterable[dict]) -> ComparisonReport:
		"""
		Generate a comparison report from two result lists.
		Turkce: Iki sonuc listesinden karsilastirma raporu uretir.
		"""
		base_map = {r.get("task_id"): r for r in base_results}
		tuned_map = {r.get("task_id"): r for r in tuned_results}

		per_task: list[TaskComparison] = []
		task_ids = sorted(set(base_map.keys()) | set(tuned_map.keys()))
		base_pass = 0
		tuned_pass = 0

		for task_id in task_ids:
			base_ok = bool(base_map.get(task_id, {}).get("success"))
			tuned_ok = bool(tuned_map.get(task_id, {}).get("success"))
			if base_ok:
				base_pass += 1
			if tuned_ok:
				tuned_pass += 1
			per_task.append(TaskComparison(task_id=task_id, base_success=base_ok, tuned_success=tuned_ok))

		total = len(task_ids)
		base_rate = (base_pass / total) * 100 if total else 0.0
		tuned_rate = (tuned_pass / total) * 100 if total else 0.0

		return ComparisonReport(
			base_pass_rate=base_rate,
			tuned_pass_rate=tuned_rate,
			total_tasks=total,
			per_task=per_task,
		)
