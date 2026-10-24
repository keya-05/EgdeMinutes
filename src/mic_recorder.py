"""Browser microphone recorder as a minimal Streamlit component.

Streamlit's built-in st.audio_input did not start recording on iOS Safari in
testing, so this component uses the browser's MediaRecorder API directly
(AAC/MP4 on iOS, Opus/WebM on Chrome, Edge and Firefox) and returns the raw
recording to Python. The audio is decoded later by PyAV in transcription.py.
"""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit.components.v1 as components

_component = components.declare_component(
    "mic_recorder", path=str(Path(__file__).parent / "mic_recorder_frontend")
)

_SUFFIXES = {"mp4": ".m4a", "aac": ".m4a", "webm": ".webm", "ogg": ".ogg", "wav": ".wav"}


def mic_recorder(key: str | None = None) -> tuple[bytes, str, dict] | None:
    """Render the recorder. Returns (audio_bytes, file_suffix, info) once a
    recording has been made, otherwise None."""
    value = _component(key=key, default=None)
    if not value or not value.get("b64"):
        return None
    mime = value.get("mime", "")
    suffix = next((s for k, s in _SUFFIXES.items() if k in mime), ".m4a")
    info = {"mime": mime, "size": value.get("size"), "duration_ms": value.get("duration_ms")}
    return base64.b64decode(value["b64"]), suffix, info
