"""
Training data curation utilities for fine-tuning.
Turkce: Fine-tuning icin egitim verisi hazirlama ve dogrulama araclari.
"""

# TODO: Implement training data curation ----

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from ..llm.prompts import SYSTEM_PROMPT


@dataclass
class ValidationReport:
	total_examples: int
	valid_examples: int
	invalid_examples: int
	avg_input_tokens: float
	avg_output_tokens: float
	task_distribution: dict[str, int] = field(default_factory=dict)
	quality_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class TrainValSplit:
	train: list[dict]
	val: list[dict]
	train_size: int
	val_size: int


class DataCurator:
	"""Load, validate, and format fine-tuning data."""

	def load_examples(self, path: Path) -> list[dict]:
		"""Load JSONL examples from disk."""
		examples: list[dict] = []
		for line in path.read_text(encoding="utf-8").splitlines():
			if not line.strip():
				continue
			examples.append(json.loads(line))
		return examples

	def validate_examples(self, examples: Iterable[dict]) -> ValidationReport:
		"""Validate schema and compute basic statistics."""
		total = 0
		valid = 0
		invalid = 0
		input_tokens = 0
		output_tokens = 0
		task_dist: dict[str, int] = {}
		quality_dist: dict[str, int] = {}

		for ex in examples:
			total += 1
			task_id = ex.get("task_id")
			input_prompt = ex.get("input_prompt")
			expected_output = ex.get("expected_output")
			metadata = ex.get("metadata", {})

			if not task_id or not input_prompt or not expected_output or not isinstance(metadata, dict):
				invalid += 1
				continue

			valid += 1
			task_dist[task_id] = task_dist.get(task_id, 0) + 1
			quality = metadata.get("quality", "unknown")
			quality_dist[quality] = quality_dist.get(quality, 0) + 1

			input_tokens += len(str(input_prompt).split())
			output_tokens += len(str(expected_output).split())

		avg_in = (input_tokens / valid) if valid else 0.0
		avg_out = (output_tokens / valid) if valid else 0.0

		return ValidationReport(
			total_examples=total,
			valid_examples=valid,
			invalid_examples=invalid,
			avg_input_tokens=avg_in,
			avg_output_tokens=avg_out,
			task_distribution=task_dist,
			quality_distribution=quality_dist,
		)

	def split_train_val(
		self,
		examples: list[dict],
		val_ratio: float = 0.2,
		stratify_by_task: bool = True,
		seed: int = 42,
	) -> TrainValSplit:
		"""Split examples into train/val, optionally stratified by task."""
		rng = random.Random(seed)
		train: list[dict] = []
		val: list[dict] = []

		if stratify_by_task:
			by_task: dict[str, list[dict]] = {}
			for ex in examples:
				by_task.setdefault(ex.get("task_id", "unknown"), []).append(ex)
			for task_id, group in by_task.items():
				rng.shuffle(group)
				cut = max(1, int(len(group) * (1 - val_ratio)))
				train.extend(group[:cut])
				val.extend(group[cut:])
		else:
			shuffled = list(examples)
			rng.shuffle(shuffled)
			cut = max(1, int(len(shuffled) * (1 - val_ratio)))
			train = shuffled[:cut]
			val = shuffled[cut:]

		return TrainValSplit(train=train, val=val, train_size=len(train), val_size=len(val))

	def format_for_openai(
		self,
		examples: Iterable[dict],
		include_quality: Optional[str] = None,
	) -> list[dict]:
		"""Format examples into OpenAI fine-tuning JSONL chat format."""
		formatted: list[dict] = []
		for ex in examples:
			metadata = ex.get("metadata", {})
			quality = metadata.get("quality")
			if include_quality and quality != include_quality:
				continue
			formatted.append({
				"messages": [
					{"role": "system", "content": SYSTEM_PROMPT},
					{"role": "user", "content": ex.get("input_prompt", "")},
					{"role": "assistant", "content": ex.get("expected_output", "")},
				]
			})
		return formatted

	def export_jsonl(self, rows: Iterable[dict], path: Path) -> None:
		"""Write formatted rows to JSONL."""
		with path.open("w", encoding="utf-8") as f:
			for row in rows:
				f.write(json.dumps(row) + "\n")
