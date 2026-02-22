"""
Structured output handling and CoT prompting helpers.
Turkce: Yapilandirilmis cikti ve dusunce zinciri (CoT) yardimcilari.
"""

# TODO: Implement structured output handling and chain-of-thought prompting ----

import json
import re
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError, field_validator


class PatchAnalysis(BaseModel):
	"""Structured analysis of the proposed patch."""
	target_files: list[str] = Field(default_factory=list)
	summary: str = ""
	risks: list[str] = Field(default_factory=list)
	confidence: float = 0.0


class PatchResponse(BaseModel):
	"""Structured response for patch generation."""
	diff: str
	analysis: Optional[PatchAnalysis] = None

	@field_validator("diff")
	@classmethod
	def validate_diff(cls, value: str) -> str:
		stripped = value.lstrip()
		if not (stripped.startswith("--- a/") or stripped.startswith("diff --git")):
			raise ValueError("diff must start with unified diff header '--- a/' or 'diff --git'")
		if "+++ b/" not in value:
			raise ValueError("diff must include '+++ b/' header")
		if "@@" not in value:
			raise ValueError("diff must include hunk markers '@@'")
		return value


class AnalysisResponse(BaseModel):
	"""Structured response for analysis-only outputs."""
	analysis: PatchAnalysis


T = TypeVar("T", bound=BaseModel)


def _extract_code_block(text: str) -> Optional[str]:
	match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
	if match:
		return match.group(1).strip()
	return None


def _extract_brace_block(text: str) -> Optional[str]:
	start = text.find("{")
	if start == -1:
		return None
	depth = 0
	for i in range(start, len(text)):
		if text[i] == "{":
			depth += 1
		elif text[i] == "}":
			depth -= 1
			if depth == 0:
				return text[start:i + 1]
	return None


def extract_json_from_response(response_text: str) -> dict[str, Any]:
	"""
	Extract JSON object from an LLM response using multiple strategies.
	"""
	# 1) Direct parse
	try:
		parsed = json.loads(response_text)
		if isinstance(parsed, dict):
			return parsed
	except json.JSONDecodeError:
		pass

	# 2) Code block parse
	code_block = _extract_code_block(response_text)
	if code_block:
		try:
			parsed = json.loads(code_block)
			if isinstance(parsed, dict):
				return parsed
		except json.JSONDecodeError:
			pass

	# 3) Brace matching parse
	brace_block = _extract_brace_block(response_text)
	if brace_block:
		try:
			parsed = json.loads(brace_block)
			if isinstance(parsed, dict):
				return parsed
		except json.JSONDecodeError:
			pass

	raise ValueError("Could not extract JSON from response")


def parse_structured_response(response_text: str, model: Type[T]) -> T:
	"""
	Parse and validate an LLM response into a Pydantic model.
	"""
	data = extract_json_from_response(response_text)
	try:
		return model.model_validate(data)
	except ValidationError as e:
		raise ValueError(f"Structured response validation failed: {e}") from e


def build_json_mode_prompt(model: Type[BaseModel]) -> str:
	"""
	Build a JSON mode instruction prompt from a Pydantic model schema.
	"""
	schema = model.model_json_schema()
	schema_text = json.dumps(schema, indent=2)
	return (
		"Return ONLY valid JSON that matches this schema. "
		"Do not include markdown or extra text.\n\n"
		f"JSON Schema:\n{schema_text}"
	)


COT_PATCH_TEMPLATE = """
You will think step by step and produce a minimal unified diff patch.
Steps:
1) Identify target files and symbols from context.
2) Analyze current behavior and required changes.
3) Plan minimal edits and consider edge cases.
4) Generate the unified diff only.
""".strip()


COT_ANALYSIS_TEMPLATE = """
You will think step by step and produce a structured analysis.
Steps:
1) Identify target files and relevant symbols.
2) Summarize current behavior and gaps.
3) List risks and edge cases.
4) Provide confidence level.
""".strip()


def build_cot_patch_prompt(base_prompt: str) -> str:
	"""Attach chain-of-thought guidance to a base patch prompt."""
	return f"{COT_PATCH_TEMPLATE}\n\n{base_prompt}"


def build_cot_analysis_prompt(base_prompt: str) -> str:
	"""Attach chain-of-thought guidance to a base analysis prompt."""
	return f"{COT_ANALYSIS_TEMPLATE}\n\n{base_prompt}"
