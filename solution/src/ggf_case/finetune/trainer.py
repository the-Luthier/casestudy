"""
Fine-tuning orchestration using OpenAI-compatible APIs.
Turkce: OpenAI uyumlu API ile fine-tuning islemleri.
"""

# TODO: Implement fine-tuning orchestration ----

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

from ..config import Settings


@dataclass
class FineTuneConfig:
	"""Hyperparameters for fine-tuning."""
	model: str
	suffix: str = "ggf-case"
	n_epochs: int = 5  # number of epochs
	batch_size: int = 2  # batch size
	learning_rate_multiplier: float = 1.0  # learning rate multiplier


@dataclass
class FineTuneJob:
	job_id: str
	status: str
	model: str
	fine_tuned_model: Optional[str] = None
	error: Optional[str] = None


class FineTuneTrainer:
	"""OpenAI fine-tuning API wrapper."""

	def __init__(self, settings: Settings) -> None:
		self.settings = settings
		self.base_url = settings.openai_base_url.rstrip("/")
		self.api_key = settings.openai_api_key

	def _headers(self) -> dict[str, str]:
		headers = {"Authorization": f"Bearer {self.api_key}"}
		return headers

	def upload_training_file(self, path: Path) -> str:
		"""
		Upload JSONL training file and return file ID.
		Turkce: JSONL egitim dosyasini yukler ve dosya ID dondurur.
		"""
		url = f"{self.base_url}/files"
		files = {"file": (path.name, path.read_bytes())}
		data = {"purpose": "fine-tune"}
		with httpx.Client(timeout=60.0) as client:
			resp = client.post(url, headers=self._headers(), files=files, data=data)
			resp.raise_for_status()
			payload = resp.json()
		return payload["id"]

	def create_job(self, training_file_id: str, config: FineTuneConfig) -> FineTuneJob:
		"""
		Create a fine-tuning job and return its metadata.
		Turkce: Fine-tuning job'i olusturur ve metadata dondurur.
		"""
		url = f"{self.base_url}/fine_tuning/jobs"
		body = {
			"model": config.model,
			"training_file": training_file_id,
			"suffix": config.suffix,
			"hyperparameters": {
				"n_epochs": config.n_epochs,
				"batch_size": config.batch_size,
				"learning_rate_multiplier": config.learning_rate_multiplier,
			},
		}
		with httpx.Client(timeout=60.0) as client:
			resp = client.post(url, headers=self._headers(), json=body)
			resp.raise_for_status()
			payload = resp.json()
		return FineTuneJob(
			job_id=payload.get("id"),
			status=payload.get("status", "unknown"),
			model=payload.get("model", config.model),
			fine_tuned_model=payload.get("fine_tuned_model"),
			error=payload.get("error"),
		)

	def get_job_status(self, job_id: str) -> FineTuneJob:
		"""
		Fetch fine-tuning job status.
		Turkce: Fine-tuning job durumunu getirir.
		"""
		url = f"{self.base_url}/fine_tuning/jobs/{job_id}"
		with httpx.Client(timeout=30.0) as client:
			resp = client.get(url, headers=self._headers())
			resp.raise_for_status()
			payload = resp.json()
		return FineTuneJob(
			job_id=payload.get("id"),
			status=payload.get("status", "unknown"),
			model=payload.get("model", ""),
			fine_tuned_model=payload.get("fine_tuned_model"),
			error=payload.get("error"),
		)

	def list_jobs(self, limit: int = 5) -> list[FineTuneJob]:
		"""
		List recent fine-tuning jobs.
		Turkce: Son fine-tuning job'larini listeler.
		"""
		url = f"{self.base_url}/fine_tuning/jobs?limit={limit}"
		with httpx.Client(timeout=30.0) as client:
			resp = client.get(url, headers=self._headers())
			resp.raise_for_status()
			payload = resp.json()
		jobs = []
		for item in payload.get("data", []):
			jobs.append(FineTuneJob(
				job_id=item.get("id"),
				status=item.get("status", "unknown"),
				model=item.get("model", ""),
				fine_tuned_model=item.get("fine_tuned_model"),
				error=item.get("error"),
			))
		return jobs
