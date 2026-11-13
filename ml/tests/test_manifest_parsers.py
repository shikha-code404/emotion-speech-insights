"""Unit tests for Emo-DB and SAVEE filename parsers and audio validators."""

import pathlib
import pytest
import soundfile as sf
import numpy as np

from ml.src.build_manifest import (
    EMODB_EMOTIONS,
    SAVEE_PREFIXES,
    parse_emodb_filename,
    parse_savee_filename,
    extract_audio_info,
    compute_file_hash,
)


# -----------------------------------------------------------------------------
# Emo-DB Parser Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("code,expected_emotion", [
    ("W", "anger"),
    ("L", "boredom"),
    ("E", "disgust"),
    ("A", "fear"),
    ("F", "happiness"),
    ("T", "sadness"),
    ("N", "neutral"),
])
def test_emodb_all_emotion_codes(code: str, expected_emotion: str):
    """Test that every valid Emo-DB emotion code is correctly parsed."""
    filename = f"03a01{code}a.wav"
    result = parse_emodb_filename(filename)

    assert result["corpus"] == "Emo-DB"
    assert result["speaker_id"] == "03"
    assert result["emotion"] == expected_emotion
    assert result["language"] == "German"


@pytest.mark.parametrize("speaker_id", [
    "03", "08", "09", "10", "11", "12", "13", "14", "15", "16"
])
def test_emodb_all_valid_speakers(speaker_id: str):
    """Test that all 10 known Emo-DB speakers are supported."""
    filename = f"{speaker_id}a01Fa.wav"
    result = parse_emodb_filename(filename)
    assert result["speaker_id"] == speaker_id
    assert result["emotion"] == "happiness"


def test_emodb_invalid_emotion_code():
    """Fail loudly on unknown Emo-DB emotion codes."""
    with pytest.raises(ValueError, match="Unknown Emo-DB emotion code"):
        parse_emodb_filename("03a01Za.wav")


def test_emodb_invalid_speaker():
    """Fail loudly on unknown Emo-DB speaker ID."""
    with pytest.raises(ValueError, match="Unknown Emo-DB speaker ID"):
        parse_emodb_filename("99a01Fa.wav")


def test_emodb_stem_too_short():
    """Fail loudly on malformed short filenames."""
    with pytest.raises(ValueError, match="Must have at least 6 characters"):
        parse_emodb_filename("03a.wav")


# -----------------------------------------------------------------------------
# SAVEE Parser Tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("prefix,expected_emotion", [
    ("a", "anger"),
    ("d", "disgust"),
    ("f", "fear"),
    ("h", "happiness"),
    ("n", "neutral"),
    ("sa", "sadness"),
    ("su", "surprise"),
])
def test_savee_all_emotion_codes(prefix: str, expected_emotion: str):
    """Test that every SAVEE emotion code/prefix is correctly parsed."""
    filename = f"{prefix}05.wav"
    result = parse_savee_filename(filename, speaker_folder="DC")

    assert result["corpus"] == "SAVEE"
    assert result["speaker_id"] == "DC"
    assert result["emotion"] == expected_emotion
    assert result["language"] == "English"


@pytest.mark.parametrize("speaker", ["DC", "JE", "JK", "KL"])
def test_savee_all_valid_speakers(speaker: str):
    """Test that all 4 SAVEE male speakers are supported."""
    filename = "n15.wav"
    result = parse_savee_filename(filename, speaker_folder=speaker)
    assert result["speaker_id"] == speaker
    assert result["emotion"] == "neutral"


def test_savee_speaker_from_parent_folder():
    """Test determining speaker from directory hierarchy."""
    path = pathlib.Path("AudioData/JE/sa12.wav")
    result = parse_savee_filename(path)
    assert result["speaker_id"] == "JE"
    assert result["emotion"] == "sadness"


def test_savee_speaker_from_compound_stem():
    """Test fallback when filename has format <SPEAKER>_<clip>.wav."""
    result = parse_savee_filename("KL_su08.wav")
    assert result["speaker_id"] == "KL"
    assert result["emotion"] == "surprise"


def test_savee_invalid_emotion_prefix():
    """Fail loudly on unknown SAVEE emotion prefix."""
    with pytest.raises(ValueError, match="Unknown SAVEE emotion prefix"):
        parse_savee_filename("zz01.wav", speaker_folder="DC")


def test_savee_invalid_speaker():
    """Fail loudly on invalid SAVEE speaker."""
    with pytest.raises(ValueError, match="Could not determine valid SAVEE speaker"):
        parse_savee_filename("sa01.wav", speaker_folder="INVALID")


# -----------------------------------------------------------------------------
# Audio & Integrity Validator Tests
# -----------------------------------------------------------------------------

def test_missing_audio_path_fails_loudly(tmp_path: pathlib.Path):
    """Ensure missing paths raise FileNotFoundError."""
    missing = tmp_path / "non_existent.wav"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        extract_audio_info(missing)


def test_empty_audio_fails_loudly(tmp_path: pathlib.Path):
    """Ensure empty 0-byte audio raises ValueError."""
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="Empty audio file"):
        extract_audio_info(empty_file)


def test_corrupt_audio_fails_loudly(tmp_path: pathlib.Path):
    """Ensure corrupt header audio raises ValueError."""
    corrupt_file = tmp_path / "corrupt.wav"
    corrupt_file.write_bytes(b"RIFF\x00\x00\x00\x00NOT_A_WAV_FORMAT")
    with pytest.raises(ValueError, match="Corrupt or unreadable"):
        extract_audio_info(corrupt_file)


def test_valid_synthetic_wav_extraction(tmp_path: pathlib.Path):
    """Verify valid synthetic audio extracts correct metadata and hash."""
    test_wav = tmp_path / "test_synth.wav"
    sr = 16000
    duration_sec = 0.5
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    data = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    sf.write(str(test_wav), data, sr)

    info = extract_audio_info(test_wav)
    assert info["sample_rate"] == sr
    assert info["channels"] == 1
    assert abs(info["duration_seconds"] - duration_sec) < 0.01
    assert info["file_size"] > 0
    assert len(info["file_hash"]) == 64  # SHA-256 length


# -----------------------------------------------------------------------------
# Scanner & Duplicate Deduplication Tests
# -----------------------------------------------------------------------------

def test_emodb_duplicate_hash_fails_loudly(tmp_path: pathlib.Path):
    """Emo-DB must fail loudly if duplicate audio hashes exist within the corpus."""
    from ml.src.build_manifest import scan_emodb

    sr = 16000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    data = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    f1 = tmp_path / "03a01Fa.wav"
    f2 = tmp_path / "03a02Fa.wav"
    sf.write(str(f1), data, sr)
    sf.write(str(f2), data, sr)  # Identical content -> identical hash

    with pytest.raises(ValueError, match="Duplicate file hash within Emo-DB corpus"):
        scan_emodb(tmp_path)


def test_savee_deduplicates_nested_mirror_tree(tmp_path: pathlib.Path):
    """SAVEE must deduplicate nested mirror trees by file hash and keep one copy."""
    from ml.src.build_manifest import scan_savee

    sr = 16000
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    data = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    primary = tmp_path / "AudioData" / "DC" / "a01.wav"
    nested = tmp_path / "AudioData" / "AudioData" / "DC" / "a01.wav"
    primary.parent.mkdir(parents=True, exist_ok=True)
    nested.parent.mkdir(parents=True, exist_ok=True)

    sf.write(str(primary), data, sr)
    sf.write(str(nested), data, sr)

    records = scan_savee(tmp_path)
    assert len(records) == 1
    assert records[0]["speaker_id"] == "DC"
    assert records[0]["emotion"] == "anger"

