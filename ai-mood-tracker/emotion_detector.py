"""Core emotion detection utilities for the AI Mood Tracker.

This module wraps emotion inference backends (DeepFace or FER), provides
utilities for logging predictions, and exposes helpers for building daily
summaries and charts consumed by the Streamlit frontend.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import cv2
import numpy as np
import pandas as pd


try:  # pragma: no cover - optional dependency
	from deepface import DeepFace  # type: ignore

	_HAS_DEEPFACE = True
except (ImportError, ValueError, Exception):  # pragma: no cover - optional dependency
	DeepFace = None
	_HAS_DEEPFACE = False

try:  # pragma: no cover - optional dependency
	from fer import FER  # type: ignore

	_HAS_FER = True
except ImportError:  # pragma: no cover - optional dependency
	FER = None
	_HAS_FER = False


LOGGER = logging.getLogger(__name__)


EMOTION_COLUMNS = ["timestamp", "emotion", "confidence", "source"]


@dataclass
class EmotionResult:
	"""Container for a single emotion prediction."""

	dominant_emotion: str
	confidence: float
	emotions: Dict[str, float]
	timestamp: datetime


@dataclass
class DailySummary:
	"""Aggregated emotion metrics for a single day."""

	target_date: date
	total_scans: int
	dominant_emotion: Optional[str]
	distribution: Dict[str, float]
	average_confidence: Optional[float]


class EmotionAnalyzer:
	"""Emotion inference wrapper with DeepFace/FER fallbacks."""

	def __init__(
		self,
		detector_backend: str = "opencv",
		enforce_detection: bool = False,
		backend_priority: Sequence[str] = ("deepface", "fer"),
	) -> None:
		self.detector_backend = detector_backend
		self.enforce_detection = enforce_detection
		self.backend = self._select_backend(backend_priority)
		self._fer_detector: Optional[object] = None

		if self.backend is None:
			raise ImportError(
				"No emotion detection backend available. Install 'deepface' or 'fer'."
			)

		LOGGER.info("Using emotion backend: %s", self.backend)

	def _select_backend(self, backend_priority: Sequence[str]) -> Optional[str]:
		for backend in backend_priority:
			if backend == "deepface" and _HAS_DEEPFACE:
				return backend
			if backend == "fer" and _HAS_FER:
				return backend
		return None

	def analyze(self, frame: np.ndarray) -> Optional[EmotionResult]:
		"""Infer emotion for a single BGR frame.

		Returns ``None`` when no face/emotion is detected.
		"""

		if frame is None:
			LOGGER.debug("Received empty frame for analysis")
			return None

		if self.backend == "deepface":
			return self._analyze_with_deepface(frame)
		if self.backend == "fer":
			return self._analyze_with_fer(frame)
		raise RuntimeError("Unsupported backend configured")

	def _analyze_with_deepface(self, frame: np.ndarray) -> Optional[EmotionResult]:
		if DeepFace is None:  # pragma: no cover - defensive
			return None

		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		try:
			analysis = DeepFace.analyze(  # type: ignore[attr-defined]
				rgb_frame,
				actions=["emotion"],
				enforce_detection=self.enforce_detection,
				detector_backend=self.detector_backend,
				silent=True,
			)
		except Exception as exc:  # pragma: no cover - backend specific
			LOGGER.debug("DeepFace failed to analyze frame: %s", exc)
			return None

		if isinstance(analysis, Iterable):
			# DeepFace may return a list when multiple faces detected.
			analysis = next(iter(analysis), None)
			if analysis is None:
				return None

		emotions = analysis.get("emotion") or analysis.get("emotions")
		if not emotions:
			return None

		normalized = _normalize_emotions(emotions)
		if not normalized:
			return None

		dominant = analysis.get("dominant_emotion") or _get_dominant(normalized)
		confidence = normalized.get(dominant, 0.0)

		return EmotionResult(
			dominant_emotion=dominant,
			confidence=float(confidence),
			emotions=normalized,
			timestamp=datetime.utcnow(),
		)

	def _analyze_with_fer(self, frame: np.ndarray) -> Optional[EmotionResult]:
		if FER is None:  # pragma: no cover - defensive
			return None

		if self._fer_detector is None:
			self._fer_detector = FER(mtcnn=True)

		rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		try:
			detections = self._fer_detector.detect_emotions(rgb_frame)
		except Exception as exc:  # pragma: no cover - backend specific
			LOGGER.debug("FER failed to analyze frame: %s", exc)
			return None

		if not detections:
			return None

		emotions = detections[0].get("emotions")
		if not emotions:
			return None

		normalized = _normalize_emotions(emotions)
		if not normalized:
			return None

		dominant = _get_dominant(normalized)

		return EmotionResult(
			dominant_emotion=dominant,
			confidence=float(normalized.get(dominant, 0.0)),
			emotions=normalized,
			timestamp=datetime.utcnow(),
		)


def ensure_log_file(log_path: Path) -> None:
	"""Ensure the CSV log exists with the expected columns."""

	log_path.parent.mkdir(parents=True, exist_ok=True)
	if not log_path.exists():
		log_path.write_text(",".join(EMOTION_COLUMNS) + "\n", encoding="utf-8")


def append_log_entry(log_path: Path, result: EmotionResult, source: str = "webcam") -> None:
	"""Append a single emotion prediction to the CSV log."""

	ensure_log_file(log_path)
	line = f"{result.timestamp.isoformat()},{result.dominant_emotion},{result.confidence:.4f},{source}\n"
	with log_path.open("a", encoding="utf-8") as handle:
		handle.write(line)


def load_log_dataframe(log_path: Path) -> pd.DataFrame:
	"""Load the emotion log into a dataframe (empty when missing)."""

	if not log_path.exists() or log_path.stat().st_size == 0:
		return pd.DataFrame(columns=EMOTION_COLUMNS)

	df = pd.read_csv(log_path, parse_dates=["timestamp"])
	if "timestamp" in df.columns:
		df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
	return df.dropna(subset=["timestamp", "emotion", "confidence"], how="any")


def get_daily_summary(log_path: Path, target_date: Optional[date] = None) -> DailySummary:
	"""Compute aggregated stats for the given date (defaults to today)."""

	target_date = target_date or datetime.utcnow().date()
	df = load_log_dataframe(log_path)
	if df.empty:
		return DailySummary(target_date, 0, None, {}, None)

	df["date"] = df["timestamp"].dt.date
	day_df = df[df["date"] == target_date]

	if day_df.empty:
		return DailySummary(target_date, 0, None, {}, None)

	counts = day_df["emotion"].value_counts()
	total = int(counts.sum())
	dominant = counts.idxmax()
	distribution = (counts / total).to_dict()
	avg_conf = float(day_df["confidence"].mean()) if "confidence" in day_df else None

	return DailySummary(target_date, total, dominant, distribution, avg_conf)


def get_recent_entries(log_path: Path, limit: int = 10) -> pd.DataFrame:
	"""Return the most recent log records for quick inspection."""

	df = load_log_dataframe(log_path)
	if df.empty:
		return df
	return df.sort_values("timestamp", ascending=False).head(limit)


def save_distribution_chart(
	distribution: Dict[str, float],
	output_path: Path,
	title: str,
) -> None:
	"""Persist a bar chart for the provided emotion distribution."""

	output_path.parent.mkdir(parents=True, exist_ok=True)

	import matplotlib

	matplotlib.use("Agg")  # Use non-interactive backend for servers
	import matplotlib.pyplot as plt

	plt.figure(figsize=(6, 4))
	if distribution:
		emotions = list(distribution.keys())
		values = [distribution[e] for e in emotions]
		plt.bar(emotions, values, color="#6C63FF")
		plt.ylim(0, 1)
		plt.ylabel("Share of detections")
	else:
		plt.text(0.5, 0.5, "No data", ha="center", va="center")
		plt.xticks([])
		plt.yticks([])

	plt.title(title)
	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()


def _normalize_emotions(emotions: Dict[str, float]) -> Dict[str, float]:
	floats = {k: float(v) for k, v in emotions.items() if _is_finite_number(v)}
	total = sum(floats.values())
	if total <= 0:
		return {}
	return {emotion: value / total for emotion, value in floats.items()}


def _get_dominant(emotions: Dict[str, float]) -> str:
	return max(emotions, key=emotions.get)


def _is_finite_number(value: object) -> bool:
	try:
		return np.isfinite(float(value))
	except (TypeError, ValueError):
		return False
