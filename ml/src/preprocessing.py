"""Audio preprocessing utilities for Speech Emotion Recognition.

Provides loading, validation, and normalization of audio files prior to
feature extraction. All resampling uses librosa; file I/O uses soundfile.
"""

from __future__ import annotations

import pathlib
from typing import Dict, Tuple, Union

import librosa
import numpy as np
import soundfile as sf


def load_audio(
    path: Union[str, pathlib.Path],
    target_sr: int = 16000,
) -> Tuple[np.ndarray, int]:
    """Load an audio file, convert to mono, and resample to *target_sr*.

    Args:
        path: Path to the audio file (any format supported by soundfile).
        target_sr: Target sample rate in Hz.

    Returns:
        Tuple of (waveform as 1-D float32 ndarray, sample rate).

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If *target_sr* is not positive.
    """
    p = pathlib.Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {path}")
    if target_sr <= 0:
        raise ValueError(f"target_sr must be positive, got {target_sr}")

    y, sr = sf.read(str(p), dtype="float32")

    # Convert stereo (or multi-channel) to mono by averaging channels
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Resample if the native sample rate differs from the target
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return y, sr


def validate_audio(
    y: np.ndarray,
    sr: int,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
) -> Dict[str, object]:
    """Validate basic properties of an audio waveform.

    Args:
        y: 1-D audio waveform.
        sr: Sample rate in Hz.
        min_duration: Minimum acceptable duration in seconds.
        max_duration: Maximum acceptable duration in seconds.

    Returns:
        Dict with keys:
            duration      – float, clip length in seconds.
            is_silent     – bool, True when RMS energy < 1e-4.
            is_too_short  – bool, True when duration < *min_duration*.
            is_too_long   – bool, True when duration > *max_duration*.
    """
    duration = float(len(y)) / sr
    rms = float(np.sqrt(np.mean(y ** 2)))

    return {
        "duration": round(duration, 6),
        "is_silent": rms < 1e-4,
        "is_too_short": duration < min_duration,
        "is_too_long": duration > max_duration,
    }


TRIM_TOP_DB: float = 30.0


def prepare_waveform(y: np.ndarray, top_db: float = TRIM_TOP_DB) -> np.ndarray:
    """Trim leading/trailing silence, then peak-normalize.

    Used identically at training time and at inference time (Rule 3), so
    a 10 s browser recording with pauses at either end is processed the same
    way as a short acted corpus clip. If trimming would remove everything,
    the untrimmed signal is kept.
    """
    trimmed, _ = librosa.effects.trim(y, top_db=top_db)
    if trimmed.size == 0:
        trimmed = y
    return normalize_audio(trimmed)


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """Peak-normalize an audio waveform to [-1, 1].

    If the waveform is all zeros the input is returned unchanged.

    Args:
        y: 1-D audio waveform.

    Returns:
        Peak-normalized waveform as float32 ndarray.
    """
    peak = np.max(np.abs(y))
    if peak == 0:
        return y.astype(np.float32)
    return (y / peak).astype(np.float32)
