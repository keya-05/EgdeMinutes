"""Speech-to-text using faster-whisper (CTranslate2 port of OpenAI Whisper).

Whisper is a Transformer encoder-decoder: the encoder reads a log-Mel
spectrogram of the audio, the decoder autoregressively generates text tokens.
All inference here runs locally on the CPU of the machine running this code.
"""

from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from functools import lru_cache

SUPPORTED_EXTENSIONS = ("wav", "mp3", "m4a", "flac", "ogg", "webm", "mp4", "mov")
MODEL_SIZES = ("tiny", "base", "small")
DEFAULT_MODEL_SIZE = "tiny"


class TranscriptionError(RuntimeError):
    """Raised when audio cannot be transcribed."""


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    audio_duration_s: float
    processing_time_s: float
    model_size: str


@lru_cache(maxsize=2)
def load_whisper_model(model_size: str = DEFAULT_MODEL_SIZE):
    """Load (and cache) a Whisper model. Downloads weights on first use."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc
    try:
        # int8 quantisation keeps memory low (~75 MB for tiny) on CPU.
        return WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as exc:  # network errors, corrupt cache, etc.
        raise TranscriptionError(
            f"Could not load Whisper '{model_size}' model. The first run needs "
            f"internet access to download it. Details: {exc}"
        ) from exc


def decode_audio(audio_path: str, sampling_rate: int = 16000):
    """Decode any PyAV-supported file to mono float32 at 16 kHz.

    faster-whisper's own decoder passes an argument removed in PyAV 19, so we
    decode here and give Whisper a NumPy array instead of a file path.
    """
    import av
    import numpy as np

    resampler = av.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    chunks = []
    with av.open(audio_path, mode="r") as container:
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):  # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def transcribe_audio(
    audio_path: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    language: str | None = None,
) -> TranscriptionResult:
    """Transcribe an audio file. `language=None` enables auto-detection."""
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise TranscriptionError("The audio file is empty or missing.")

    model = load_whisper_model(model_size)
    start = time.perf_counter()
    try:
        audio = decode_audio(audio_path)
    except Exception as exc:
        raise TranscriptionError(
            f"Could not decode the audio file (unsupported or corrupt format?). Details: {exc}"
        ) from exc
    if audio.size < 16000 * 0.5:
        raise TranscriptionError("The recording is empty or shorter than half a second.")
    try:
        segments, info = model.transcribe(
            audio, language=language, beam_size=1, vad_filter=True
        )
        # `segments` is a lazy generator; decoding happens while iterating.
        text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as exc:
        raise TranscriptionError(
            f"Speech recognition failed. Details: {exc}"
        ) from exc

    if not text:
        raise TranscriptionError(
            "No speech was detected in the audio. Try a louder or longer recording."
        )
    return TranscriptionResult(
        text=text,
        language=info.language,
        language_probability=round(info.language_probability, 3),
        audio_duration_s=round(info.duration, 2),
        processing_time_s=round(time.perf_counter() - start, 2),
        model_size=model_size,
    )


def transcribe_bytes(
    data: bytes,
    suffix: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    language: str | None = None,
) -> TranscriptionResult:
    """Write uploaded/recorded bytes to a temp file, transcribe, then delete it."""
    if not data:
        raise TranscriptionError("The recording or uploaded file is empty.")
    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="edgeminutes_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return transcribe_audio(path, model_size=model_size, language=language)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
