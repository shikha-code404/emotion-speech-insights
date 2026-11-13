"""Build audio metadata manifest for Emo-DB and SAVEE datasets.

Scans ml/data/raw, extracts acoustic metadata, validates audio integrity,
and generates ml/data_manifest/audio_manifest.csv adhering to speaker-disjoint
rules and ethical guidelines.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import sys
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
import soundfile as sf

# -----------------------------------------------------------------------------
# Constants & Mappings
# -----------------------------------------------------------------------------

# Emo-DB (Berlin Database of Emotional Speech)
EMODB_SPEAKERS: Set[str] = {
    "03", "08", "09", "10", "11", "12", "13", "14", "15", "16"
}

EMODB_EMOTIONS: Dict[str, str] = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happiness",
    "T": "sadness",
    "N": "neutral",
}

# SAVEE (Surrey Audio-Visual Expressed Emotion)
SAVEE_SPEAKERS: Set[str] = {"DC", "JE", "JK", "KL"}

# Prefixes ordered with 2-char prefixes first to prevent partial matching
SAVEE_PREFIXES: List[tuple[str, str]] = [
    ("sa", "sadness"),
    ("su", "surprise"),
    ("a", "anger"),
    ("d", "disgust"),
    ("f", "fear"),
    ("h", "happiness"),
    ("n", "neutral"),
]

MANIFEST_COLUMNS: List[str] = [
    "filepath",
    "corpus",
    "speaker_id",
    "emotion",
    "language",
    "sample_rate",
    "channels",
    "duration_seconds",
    "file_size",
    "file_hash",
]


# -----------------------------------------------------------------------------
# Filename Parsers
# -----------------------------------------------------------------------------

def parse_emodb_filename(filename: Union[str, pathlib.Path]) -> Dict[str, str]:
    """Parse an Emo-DB audio filename.

    Naming convention:
      Characters 0-1: Speaker ID (e.g., '03')
      Character 5: Emotion code (W, L, E, A, F, T, N)

    Args:
        filename: Name or path of the audio file (e.g., '03a01Fa.wav').

    Returns:
        Dict with corpus, speaker_id, emotion, language.

    Raises:
        ValueError: If format, speaker, or emotion code is invalid.
    """
    stem = pathlib.Path(filename).stem
    if len(stem) < 6:
        raise ValueError(
            f"Invalid Emo-DB filename stem '{stem}' in '{filename}'. "
            "Must have at least 6 characters."
        )

    speaker_id = stem[0:2]
    if speaker_id not in EMODB_SPEAKERS:
        raise ValueError(
            f"Unknown Emo-DB speaker ID '{speaker_id}' in '{filename}'. "
            f"Expected one of: {sorted(EMODB_SPEAKERS)}"
        )

    emotion_code = stem[5]
    if emotion_code not in EMODB_EMOTIONS:
        raise ValueError(
            f"Unknown Emo-DB emotion code '{emotion_code}' in '{filename}'. "
            f"Expected one of: {sorted(EMODB_EMOTIONS.keys())}"
        )

    return {
        "corpus": "Emo-DB",
        "speaker_id": speaker_id,
        "emotion": EMODB_EMOTIONS[emotion_code],
        "language": "German",
    }


def parse_savee_filename(
    filename: Union[str, pathlib.Path],
    speaker_folder: Optional[str] = None,
) -> Dict[str, str]:
    """Parse a SAVEE audio filename.

    Folder convention:
      Folder name indicates speaker ID: DC, JE, JK, KL.
    Filename prefix indicates emotion:
      a=anger, d=disgust, f=fear, h=happiness, n=neutral, sa=sadness, su=surprise.

    Args:
        filename: Name or path of the audio file (e.g., 'sa01.wav' or 'DC/sa01.wav').
        speaker_folder: Explicit speaker folder name if known.

    Returns:
        Dict with corpus, speaker_id, emotion, language.

    Raises:
        ValueError: If speaker or emotion prefix is invalid.
    """
    path_obj = pathlib.Path(filename)
    stem = path_obj.stem

    # Determine speaker
    speaker = speaker_folder
    if speaker is None:
        # Check if parent folder matches speaker name
        parent_name = path_obj.parent.name.upper()
        if parent_name in SAVEE_SPEAKERS:
            speaker = parent_name
        elif "_" in stem:
            # Fallback if filename is formatted as <SPEAKER>_<clip>.wav
            prefix_speaker, rest = stem.split("_", 1)
            if prefix_speaker.upper() in SAVEE_SPEAKERS:
                speaker = prefix_speaker.upper()
                stem = rest

    if speaker is None or speaker.upper() not in SAVEE_SPEAKERS:
        raise ValueError(
            f"Could not determine valid SAVEE speaker for '{filename}'. "
            f"Expected one of: {sorted(SAVEE_SPEAKERS)}"
        )
    speaker = speaker.upper()

    # Determine emotion from prefix
    lower_stem = stem.lower()
    matched_emotion: Optional[str] = None
    for prefix, emotion_name in SAVEE_PREFIXES:
        if lower_stem.startswith(prefix):
            matched_emotion = emotion_name
            break

    if matched_emotion is None:
        raise ValueError(
            f"Unknown SAVEE emotion prefix in '{filename}' (stem='{stem}'). "
            f"Expected prefixes: {[p for p, _ in SAVEE_PREFIXES]}"
        )

    return {
        "corpus": "SAVEE",
        "speaker_id": speaker,
        "emotion": matched_emotion,
        "language": "English",
    }


# -----------------------------------------------------------------------------
# Audio & File Metadata Utilities
# -----------------------------------------------------------------------------

def compute_file_hash(filepath: Union[str, pathlib.Path], block_size: int = 65536) -> str:
    """Compute SHA-256 hash of a file for integrity check and deduplication."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(block_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_audio_info(filepath: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """Inspect and validate audio file headers and duration.

    Fails loudly if file is empty, missing, or corrupt.
    """
    p = pathlib.Path(filepath)
    if not p.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {filepath}")

    file_size = p.stat().st_size
    if file_size == 0:
        raise ValueError(f"Empty audio file (0 bytes): {filepath}")

    try:
        info = sf.info(str(p))
    except Exception as exc:
        raise ValueError(f"Corrupt or unreadable audio file '{filepath}': {exc}") from exc

    if info.samplerate <= 0 or info.channels <= 0 or info.duration <= 0:
        raise ValueError(
            f"Invalid audio parameters in '{filepath}': "
            f"sample_rate={info.samplerate}, channels={info.channels}, duration={info.duration}"
        )

    return {
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "duration_seconds": round(float(info.duration), 6),
        "file_size": int(file_size),
        "file_hash": compute_file_hash(p),
    }


# -----------------------------------------------------------------------------
# Corpus Scanners
# -----------------------------------------------------------------------------

def scan_emodb(emodb_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Scan and parse Emo-DB corpus files."""
    if not emodb_dir.exists():
        return []

    records: List[Dict[str, Any]] = []
    seen_hashes: Dict[str, str] = {}
    wav_files = sorted(emodb_dir.rglob("*.wav"))

    for wav_path in wav_files:
        meta = parse_emodb_filename(wav_path)
        audio_info = extract_audio_info(wav_path)

        file_hash = audio_info["file_hash"]
        if file_hash in seen_hashes:
            raise ValueError(
                f"Duplicate file hash within Emo-DB corpus: {file_hash}\n"
                f"File 1: {seen_hashes[file_hash]}\n"
                f"File 2: {wav_path}"
            )
        seen_hashes[file_hash] = str(wav_path)

        # Standardize path with forward slashes relative to workspace root if possible
        try:
            rel_path = wav_path.relative_to(pathlib.Path.cwd()).as_posix()
        except ValueError:
            rel_path = wav_path.as_posix()

        records.append({
            "filepath": rel_path,
            "corpus": meta["corpus"],
            "speaker_id": meta["speaker_id"],
            "emotion": meta["emotion"],
            "language": meta["language"],
            "sample_rate": audio_info["sample_rate"],
            "channels": audio_info["channels"],
            "duration_seconds": audio_info["duration_seconds"],
            "file_size": audio_info["file_size"],
            "file_hash": file_hash,
        })

    return records


def scan_savee(savee_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Scan and parse SAVEE corpus files, deduplicating nested mirror trees."""
    if not savee_dir.exists():
        return []

    records: List[Dict[str, Any]] = []
    seen_hashes: Dict[str, str] = {}
    wav_files = sorted(savee_dir.rglob("*.wav"))

    for wav_path in wav_files:
        # Identify speaker folder from path hierarchy
        speaker = None
        for part in reversed(wav_path.parts[:-1]):
            if part.upper() in SAVEE_SPEAKERS:
                speaker = part.upper()
                break

        meta = parse_savee_filename(wav_path.name, speaker_folder=speaker)
        audio_info = extract_audio_info(wav_path)
        file_hash = audio_info["file_hash"]

        # Deduplicate archive duplicates (e.g. AudioData/AudioData/ nested copies)
        if file_hash in seen_hashes:
            existing_path = seen_hashes[file_hash]
            # If same base filename, it's the duplicate tree: keep first copy
            if pathlib.Path(existing_path).name == wav_path.name:
                continue
            # Genuine distinct clip collision within unique set
            raise ValueError(
                f"Unexpected duplicate file hash across distinct clips in SAVEE: {file_hash}\n"
                f"File 1: {existing_path}\n"
                f"File 2: {wav_path}"
            )

        seen_hashes[file_hash] = str(wav_path)

        try:
            rel_path = wav_path.relative_to(pathlib.Path.cwd()).as_posix()
        except ValueError:
            rel_path = wav_path.as_posix()

        records.append({
            "filepath": rel_path,
            "corpus": meta["corpus"],
            "speaker_id": meta["speaker_id"],
            "emotion": meta["emotion"],
            "language": meta["language"],
            "sample_rate": audio_info["sample_rate"],
            "channels": audio_info["channels"],
            "duration_seconds": audio_info["duration_seconds"],
            "file_size": audio_info["file_size"],
            "file_hash": file_hash,
        })

    return records


# -----------------------------------------------------------------------------
# Manifest Generation
# -----------------------------------------------------------------------------

def build_manifest(
    raw_dir: Union[str, pathlib.Path] = "ml/data/raw",
    output_path: Union[str, pathlib.Path] = "ml/data_manifest/audio_manifest.csv",
) -> pd.DataFrame:
    """Scan raw directory, construct unified manifest, and save CSV.

    Args:
        raw_dir: Base directory containing raw corpus subdirectories.
        output_path: Target CSV path for the generated manifest.

    Returns:
        pd.DataFrame containing the manifest records.
    """
    raw_path = pathlib.Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_dir}")

    # Discover and scan Emo-DB
    emodb_candidates = [
        raw_path / "emodb",
        raw_path / "Emo-DB",
        raw_path / "EMODB",
        raw_path / "emodb" / "wav",
    ]
    emodb_dir = next((p for p in emodb_candidates if p.exists()), raw_path)
    emodb_records = scan_emodb(emodb_dir)

    # Discover and scan SAVEE
    savee_candidates = [
        raw_path / "savee",
        raw_path / "SAVEE",
        raw_path / "savee" / "AudioData",
        raw_path / "AudioData",
    ]
    savee_dir = next((p for p in savee_candidates if p.exists()), raw_path)
    savee_records = scan_savee(savee_dir)

    all_records = emodb_records + savee_records
    if not all_records:
        raise ValueError(f"No valid audio clips found in {raw_dir}")

    df = pd.DataFrame(all_records, columns=MANIFEST_COLUMNS)

    # Sort deterministically
    df = df.sort_values(by=["corpus", "speaker_id", "filepath"]).reset_index(drop=True)

    out_file = pathlib.Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"Manifest successfully generated: {out_file} ({len(df)} clips)")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build audio manifest for SER corpora")
    parser.add_argument("--raw-dir", default="ml/data/raw", help="Path to raw audio folder")
    parser.add_argument("--output", default="ml/data_manifest/audio_manifest.csv", help="Output CSV path")
    args = parser.parse_args()

    try:
        build_manifest(raw_dir=args.raw_dir, output_path=args.output)
    except Exception as e:
        print(f"Error building manifest: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

