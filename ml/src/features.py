"""Feature extraction for Speech Emotion Recognition.

Extracts a fixed-length 396-dimensional feature vector from an audio waveform
using librosa. The vector is the concatenation of per-feature mean and standard
deviation statistics computed over time frames.

Feature breakdown (mean + std each):
    40 MFCCs            →  80
    12 Chroma STFT      →  24
   128 Mel spectrogram  → 256
     7 Spectral contrast→  14
     6 Tonnetz          →  12
     1 RMS energy       →   2
     1 ZCR              →   2
     1 Spectral centroid→   2
     1 Spectral bandwidth→  2
     1 Spectral rolloff →   2
                   Total: 396
"""

from __future__ import annotations

from typing import List

import librosa
import numpy as np

# Number of features expected from `extract_features`.
FEATURE_DIM: int = 396


def extract_features(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract a fixed-length feature vector from an audio waveform.

    Args:
        y: 1-D audio waveform (float).
        sr: Sample rate in Hz.

    Returns:
        1-D numpy array of shape ``(396,)`` containing concatenated
        mean and standard-deviation statistics for each feature group.

    Raises:
        ValueError: If *y* is empty or not 1-D.
    """
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("y must be a non-empty 1-D array")

    parts: List[np.ndarray] = []

    # --- MFCCs (40 coefficients) ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    parts.append(np.mean(mfccs, axis=1))
    parts.append(np.std(mfccs, axis=1))

    # --- Chroma STFT (12 bins) ---
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    parts.append(np.mean(chroma, axis=1))
    parts.append(np.std(chroma, axis=1))

    # --- Mel spectrogram (128 bands) ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    parts.append(np.mean(mel_db, axis=1))
    parts.append(np.std(mel_db, axis=1))

    # --- Spectral contrast (7 bands) ---
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    parts.append(np.mean(contrast, axis=1))
    parts.append(np.std(contrast, axis=1))

    # --- Tonnetz (6 dimensions) ---
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    parts.append(np.mean(tonnetz, axis=1))
    parts.append(np.std(tonnetz, axis=1))

    # --- RMS energy ---
    rms = librosa.feature.rms(y=y)
    parts.append(np.mean(rms, axis=1))
    parts.append(np.std(rms, axis=1))

    # --- Zero-crossing rate ---
    zcr = librosa.feature.zero_crossing_rate(y=y)
    parts.append(np.mean(zcr, axis=1))
    parts.append(np.std(zcr, axis=1))

    # --- Spectral centroid ---
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    parts.append(np.mean(centroid, axis=1))
    parts.append(np.std(centroid, axis=1))

    # --- Spectral bandwidth ---
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    parts.append(np.mean(bandwidth, axis=1))
    parts.append(np.std(bandwidth, axis=1))

    # --- Spectral rolloff ---
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    parts.append(np.mean(rolloff, axis=1))
    parts.append(np.std(rolloff, axis=1))

    feature_vector = np.concatenate(parts)
    assert feature_vector.shape == (FEATURE_DIM,), (
        f"Expected {FEATURE_DIM} features, got {feature_vector.shape[0]}"
    )
    return feature_vector


def get_feature_names() -> List[str]:
    """Return the ordered list of feature names matching `extract_features` output.

    Returns:
        List of 396 human-readable feature name strings.
    """
    names: List[str] = []

    # Helper to generate mean/std name pairs
    def _add(prefix: str, n: int) -> None:
        for i in range(n):
            names.append(f"{prefix}_{i}_mean")
        for i in range(n):
            names.append(f"{prefix}_{i}_std")

    def _add_scalar(prefix: str) -> None:
        names.append(f"{prefix}_mean")
        names.append(f"{prefix}_std")

    _add("mfcc", 40)
    _add("chroma", 12)
    _add("mel", 128)
    _add("spectral_contrast", 7)
    _add("tonnetz", 6)
    _add_scalar("rms")
    _add_scalar("zcr")
    _add_scalar("spectral_centroid")
    _add_scalar("spectral_bandwidth")
    _add_scalar("spectral_rolloff")

    assert len(names) == FEATURE_DIM, (
        f"Expected {FEATURE_DIM} names, got {len(names)}"
    )
    return names
