"""Audio file handling: format detection, conversion via ffmpeg, temp cleanup.

Accepts WAV, MP3, WebM, OGG, M4A. Converts to 16 kHz mono WAV in a temp
directory. Always deletes temp files, including on errors.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from backend.app.errors import DecodeFailed, UnsupportedFormat

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".webm", ".ogg", ".m4a", ".mp4", ".wma"}
ALLOWED_MIMETYPES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3",
    "audio/webm",
    "audio/ogg", "audio/vorbis",
    "audio/x-m4a", "audio/mp4", "audio/aac",
    "video/webm",  # browser MediaRecorder often uses video/webm for audio
    "video/mp4",
    "audio/x-ms-wma",
    "application/octet-stream",  # fallback: rely on extension
}


def _detect_extension(filename: str, content_type: Optional[str]) -> str:
    """Determine the audio file extension from name or MIME type."""
    ext = Path(filename).suffix.lower()
    if ext in ALLOWED_EXTENSIONS:
        return ext
    # Try to infer from content type
    mime_ext_map = {
        "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
        "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
        "audio/webm": ".webm", "video/webm": ".webm",
        "audio/ogg": ".ogg",
        "audio/x-m4a": ".m4a", "audio/mp4": ".m4a", "audio/aac": ".m4a",
    }
    if content_type and content_type in mime_ext_map:
        return mime_ext_map[content_type]
    raise UnsupportedFormat(ext or "(unknown)")


def ffmpeg_path() -> Optional[str]:
    """Locate ffmpeg: FFMPEG_PATH env var, then PATH, then a winget install.

    The winget fallback means a terminal opened before ffmpeg was installed
    (stale PATH) still works.
    """
    explicit = os.getenv("FFMPEG_PATH")
    if explicit and Path(explicit).is_file():
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    winget = Path(os.getenv("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    if winget.is_dir():
        for exe in sorted(winget.glob("Gyan.FFmpeg*/*/bin/ffmpeg.exe"), reverse=True):
            return str(exe)
    return None


def _has_ffmpeg() -> bool:
    return ffmpeg_path() is not None


def convert_to_wav(
    input_path: Path,
    output_path: Path,
    target_sr: int = 16000,
) -> Path:
    """Convert any supported audio to 16 kHz mono WAV using ffmpeg.

    Raises DecodeFailed if ffmpeg fails or is unavailable.
    """
    if not _has_ffmpeg():
        # If the file is already a WAV, just copy it
        if input_path.suffix.lower() == ".wav":
            shutil.copy2(input_path, output_path)
            return output_path
        raise DecodeFailed("ffmpeg is not installed — cannot convert non-WAV audio.")

    cmd = [
        ffmpeg_path(), "-y",
        "-i", str(input_path),
        "-ac", "1",            # mono
        "-ar", str(target_sr), # target sample rate
        "-sample_fmt", "s16",  # 16-bit PCM
        "-f", "wav",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace")[-300:]
            raise DecodeFailed(f"ffmpeg exited with code {result.returncode}: {stderr}")
    except FileNotFoundError:
        raise DecodeFailed("ffmpeg not found on PATH.")
    except subprocess.TimeoutExpired:
        raise DecodeFailed("ffmpeg timed out after 30s.")

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise DecodeFailed("ffmpeg produced an empty output file.")

    return output_path


class TempAudioDir:
    """Context manager that creates a temp directory and guarantees cleanup.

    Rule 5: Audio is deleted after inference by default.
    """

    def __init__(self) -> None:
        self._dir: Optional[str] = None

    def __enter__(self) -> Path:
        self._dir = tempfile.mkdtemp(prefix="ser_audio_")
        return Path(self._dir)

    def __exit__(self, *args: object) -> None:
        if self._dir and os.path.isdir(self._dir):
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None


def save_upload_and_convert(
    file_bytes: bytes,
    filename: str,
    content_type: Optional[str],
    temp_dir: Path,
    target_sr: int = 16000,
) -> Path:
    """Save uploaded bytes, detect format, convert to WAV, return path.

    Args:
        file_bytes: Raw upload bytes.
        filename: Original filename from the upload.
        content_type: MIME type from the upload.
        temp_dir: Temporary directory (managed by TempAudioDir).
        target_sr: Target sample rate for the output WAV.

    Returns:
        Path to the converted 16 kHz mono WAV file.
    """
    ext = _detect_extension(filename, content_type)

    # Save raw upload
    raw_path = temp_dir / f"upload{ext}"
    raw_path.write_bytes(file_bytes)

    # Convert to standard WAV
    wav_path = temp_dir / "converted.wav"

    if ext == ".wav":
        # Still run through ffmpeg for normalisation if available,
        # otherwise just use the raw file
        if _has_ffmpeg():
            convert_to_wav(raw_path, wav_path, target_sr)
        else:
            wav_path = raw_path
    else:
        convert_to_wav(raw_path, wav_path, target_sr)

    return wav_path

