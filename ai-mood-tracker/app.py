"""Streamlit entry point for AI Mood Tracker - Cloud Version with Browser Camera Access."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional
from PIL import Image
import io

try:
	import cv2
	_HAS_CV2 = True
	_CV2_ERROR = None
except Exception as _cv_err:  # pragma: no cover - optional runtime dependency
	cv2 = None
	_HAS_CV2 = False
	_CV2_ERROR = str(_cv_err)

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
	_HAS_DEEPFACE,
	_HAS_FER,
	_DEEPFACE_ERROR,
	_FER_ERROR,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VISUALS_DIR = BASE_DIR / "visuals"
LOG_PATH = DATA_DIR / "mood_log.csv"
CHART_PATH = VISUALS_DIR / "daily_mood_chart.png"


st.set_page_config(
	page_title="AI Mood Tracker",
	page_icon="�",
	layout="wide",
)


@st.cache_resource
def get_analyzer() -> EmotionAnalyzer:
	"""Load and cache the emotion analyzer."""
	try:
		return EmotionAnalyzer()
	except ImportError as exc:
		st.error("No emotion detection backend available.")
		st.error(str(exc))
		st.info("Try installing: `pip install deepface fer`")
		st.stop()


def analyze_image(image_data, analyzer: EmotionAnalyzer) -> Optional[EmotionResult]:
	"""Analyze emotion from uploaded/captured image."""
	try:
		# Convert PIL Image to numpy array
		if isinstance(image_data, Image.Image):
			img = np.array(image_data)
		else:
			img = np.array(Image.open(io.BytesIO(image_data.read())))
		
		# Convert RGB->BGR for backends that expect BGR. Prefer cv2 if available.
		if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
			if _HAS_CV2:
				img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			else:
				img_bgr = img[:, :, ::-1]
		else:
			img_bgr = img
		
		# Analyze emotion
		result = analyzer.analyze(img_bgr)
		return result
	except Exception as e:
		st.error(f"Error analyzing image: {e}")
		return None


def draw_emotion_overlay(image: Image.Image, result: EmotionResult) -> Image.Image:
	"""Draw emotion label on image."""
	img_array = np.array(image)
	
	# If OpenCV is available use it for drawing; otherwise use Pillow
	text = f"{result.dominant_emotion}: {result.confidence:.1f}%"
	if _HAS_CV2:
		img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
		cv2.putText(
			img_bgr,
			text,
			(10, 30),
			cv2.FONT_HERSHEY_SIMPLEX,
			1.0,
			(0, 255, 0),
			2,
		)
		img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
		return Image.fromarray(img_rgb)
	else:
		from PIL import ImageDraw, ImageFont

		pil_img = Image.fromarray(img_array)
		draw = ImageDraw.Draw(pil_img)
		try:
			font = ImageFont.load_default()
		except Exception:
			font = None
		draw.text((10, 10), text, fill=(0, 255, 0), font=font)
		return pil_img


def main() -> None:
	ensure_log_file(LOG_PATH)

	st.title("AI Mood Tracker")
	st.markdown("*Detect and track your emotions in real-time using AI*")

	# Display backend availability info
	with st.sidebar:
		st.markdown("### Backend Status")
		if _HAS_DEEPFACE:
			st.success("DeepFace available")
		else:
			st.warning("DeepFace unavailable")
			if _DEEPFACE_ERROR:
				with st.expander("Error details"):
					st.code(_DEEPFACE_ERROR[:200])
		
		if _HAS_FER:
			st.success("FER available")
		else:
			st.warning("FER unavailable")
			if _FER_ERROR:
				with st.expander("Error details"):
					st.code(_FER_ERROR[:200])
		
		if not _HAS_DEEPFACE and not _HAS_FER:
			st.error("No emotion backend available!")
			st.stop()

	analyzer = get_analyzer()

	# Main layout
	col1, col2 = st.columns([2, 1])

	with col1:
		st.subheader("Camera Input")
		
		# Camera input - this will ask for camera permission in browser
		camera_photo = st.camera_input("Take a photo to analyze your emotion")
		
		if camera_photo is not None:
			# Analyze the captured photo
			with st.spinner("Analyzing emotion..."):
				result = analyze_image(camera_photo, analyzer)
			
			if result:
				# Show image with emotion overlay
				img = Image.open(camera_photo)
				img_with_overlay = draw_emotion_overlay(img, result)
				st.image(img_with_overlay, caption=f"Detected: {result.dominant_emotion} ({result.confidence:.1f}%)", use_container_width=True)
				
				# Log the result
				append_log_entry(LOG_PATH, result, source="browser_camera")
				
				st.success(f"Logged: **{result.dominant_emotion}** ({result.confidence:.1f}%)")
			else:
				st.warning("No face detected. Please try again with better lighting.")

	with col2:
		st.subheader("Today's Summary")
		
		summary = get_daily_summary(LOG_PATH)
		
		if summary and summary.total_scans > 0:
			st.metric("Total Detections", summary.total_scans)
			st.metric("Dominant Emotion", summary.dominant_emotion)
			
			# Show emotion distribution
			st.markdown("**Emotion Distribution:**")
			for emotion, pct in summary.distribution.items():
				st.progress(pct, text=f"{emotion}: {pct*100:.1f}%")
			
			# Generate and show chart
			save_distribution_chart(summary.distribution, CHART_PATH, "Daily Mood Distribution")
			if CHART_PATH.exists():
				st.image(str(CHART_PATH), caption="Daily Mood Distribution", use_container_width=True)
		else:
			st.info("No data logged today. Take a photo to get started!")

	# Recent entries
	st.subheader("Recent Entries")
	recent = get_recent_entries(LOG_PATH, limit=10)
	
	if not recent.empty:
		# Format the dataframe
		recent_display = recent.copy()
		recent_display['timestamp'] = recent_display['timestamp'].dt.strftime('%H:%M:%S')
		recent_display['confidence'] = recent_display['confidence'].apply(lambda x: f"{x:.1f}%")
		st.dataframe(recent_display, use_container_width=True, hide_index=True)
	else:
		st.info("No entries yet. Start by taking a photo!")

	# Instructions
	with st.expander("How to use"):
		st.markdown("""
		1. **Allow Camera Access**: Click the camera button above and allow browser access to your camera
		2. **Take a Photo**: Click the capture button to take a photo
		3. **View Results**: Your emotion will be detected and logged automatically
		4. **Track Progress**: Check the summary and recent entries to see your mood patterns
		
		**Note**: This works entirely in your browser - no video is stored, only emotion data is logged.
		""")


if __name__ == "__main__":
	main()
