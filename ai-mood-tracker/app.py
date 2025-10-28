"""Streamlit entry point for the AI Mood Tracker."""

from __future__ import annotations

from datetime import datetime
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

from emotion_detector import (
	EmotionAnalyzer,
	EmotionResult,
	append_log_entry,
	ensure_log_file,
	get_daily_summary,
	get_recent_entries,
	save_distribution_chart,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VISUALS_DIR = BASE_DIR / "visuals"
LOG_PATH = DATA_DIR / "mood_log.csv"
CHART_PATH = VISUALS_DIR / "daily_mood_chart.png"


st.set_page_config(
	page_title="AI Mood Tracker",
	page_icon="😃",
	layout="wide",
)


def _init_session_state() -> None:
	defaults = {
		"running": False,
		"start_time": None,
		"auto_stop_minutes": 5,
		"frame_rate": 2,
		"camera_index": 0,
		"last_result": None,
		"status": "Idle",
	}
	for key, value in defaults.items():
		st.session_state.setdefault(key, value)


def _start_session() -> None:
	st.session_state.running = True
	st.session_state.start_time = time.time()
	st.session_state.status = "Capturing"
	st.session_state.last_result = None


def _stop_session() -> None:
	st.session_state.running = False
	st.session_state.start_time = None
	st.session_state.status = "Stopped"


def _load_analyzer() -> EmotionAnalyzer:
	try:
		return EmotionAnalyzer()
	except ImportError as exc:
		st.error(
			"Emotion backend not available. Install `deepface` or `fer` "
			"via `pip install deepface` or `pip install fer`."
		)
		st.stop()


def _capture_frame(camera_index: int) -> Optional[np.ndarray]:
	capture = cv2.VideoCapture(camera_index)
	if not capture.isOpened():
		return None
	success, frame = capture.read()
	capture.release()
	if not success:
		return None
	return frame


def _draw_prediction(frame: np.ndarray, result: EmotionResult) -> np.ndarray:
	output = frame.copy()
	label = f"{result.dominant_emotion.title()} ({result.confidence * 100:.1f}%)"
	cv2.putText(
		output,
		label,
		(16, 32),
		cv2.FONT_HERSHEY_SIMPLEX,
		0.9,
		(255, 255, 255),
		2,
		cv2.LINE_AA,
	)
	return output


@st.cache_resource(show_spinner=False)
def get_analyzer() -> EmotionAnalyzer:
	return _load_analyzer()


def main() -> None:
	ensure_log_file(LOG_PATH)
	_init_session_state()

	analyzer = get_analyzer()

	with st.sidebar:
		st.header("Session Controls")
		st.number_input(
			"Camera index",
			value=st.session_state.camera_index,
			key="camera_index",
		)
		st.slider(
			"Auto-stop after (minutes)",
			min_value=0,
			max_value=30,
			value=st.session_state.auto_stop_minutes,
			key="auto_stop_minutes",
		)
		st.slider(
			"Frame capture rate (fps)",
			min_value=1,
			max_value=10,
			value=st.session_state.frame_rate,
			key="frame_rate",
		)

		st.button("▶️ Start session", on_click=_start_session, disabled=st.session_state.running)
		st.button("⏹ Stop session", on_click=_stop_session, disabled=not st.session_state.running)

		st.markdown(
			"Camera access requires local execution. If you see a blank frame, "
			"verify that your webcam is not being used by another application."
		)

	col_live, col_summary = st.columns((2, 1))

	with col_live:
		st.subheader("Live Emotion Feed")
		frame_placeholder = st.empty()
		info_placeholder = st.empty()
		status_placeholder = st.empty()

		current_result: Optional[EmotionResult] = None
		frame = None
		if st.session_state.running:
			frame = _capture_frame(st.session_state.camera_index)
			if frame is None:
				st.warning("Unable to read from camera. Session stopped.")
				_stop_session()
			else:
				current_result = analyzer.analyze(frame)
				if current_result:
					append_log_entry(LOG_PATH, current_result)
					st.session_state.last_result = current_result
					st.session_state.status = "Mood detected"
				else:
					st.session_state.status = "No face detected"

		last_result: Optional[EmotionResult] = st.session_state.last_result

		if frame is not None:
			display_frame = frame
			if current_result:
				display_frame = _draw_prediction(frame, current_result)
			frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
			frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
		else:
			frame_placeholder.info("Session is idle. Press **Start session** to begin.")

		if last_result:
			info_placeholder.metric(
				"Current mood",
				last_result.dominant_emotion.title(),
				f"{last_result.confidence * 100:.1f}% confidence",
			)
			st.caption(
				f"Last update: {datetime.fromtimestamp(last_result.timestamp.timestamp()).strftime('%H:%M:%S')}"
			)
		else:
			info_placeholder.metric("Current mood", "--", "Waiting for detection")

		status_text = st.session_state.status
		if st.session_state.running and st.session_state.start_time:
			elapsed = time.time() - st.session_state.start_time
			status_text = f"{status_text} • {elapsed / 60:.1f} min"
		status_placeholder.caption(f"Status: {status_text}")

	with col_summary:
		st.subheader("Today's Summary")
		summary = get_daily_summary(LOG_PATH)
		dominant_display = summary.dominant_emotion.title() if summary.dominant_emotion else "--"
		avg_conf_display = (
			f"{summary.average_confidence * 100:.1f}%" if summary.average_confidence is not None else "--"
		)
		st.metric("Dominant mood", dominant_display)
		st.metric("Total scans", summary.total_scans)
		st.metric("Average confidence", avg_conf_display)

		save_distribution_chart(
			summary.distribution,
			CHART_PATH,
			f"Emotion distribution • {summary.target_date.isoformat()}",
		)
		st.image(
			str(CHART_PATH),
			caption="Emotion distribution for today",
			use_container_width=True,
		)

		recent = get_recent_entries(LOG_PATH, limit=10)
		if not recent.empty:
			table = recent.copy()
			table["timestamp"] = table["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
			st.dataframe(table, use_container_width=True)
		else:
			st.info("No detections logged yet today.")

		if LOG_PATH.exists():
			with LOG_PATH.open("rb") as log_file:
				st.download_button(
					"Download mood log",
					data=log_file,
					file_name="mood_log.csv",
					mime="text/csv",
				)

	if st.session_state.running:
		if st.session_state.auto_stop_minutes:
			elapsed = time.time() - (st.session_state.start_time or time.time())
			if elapsed >= st.session_state.auto_stop_minutes * 60:
				st.info("Session auto-stopped based on the configured duration.")
				_stop_session()
				st.rerun()
				return

		time.sleep(1.0 / st.session_state.frame_rate)
		st.rerun()


if __name__ == "__main__":
	main()
