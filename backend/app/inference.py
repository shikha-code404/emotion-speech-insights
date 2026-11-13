"""Speech model loading and prediction.

The model folder (settings.MODEL_DIR) describes itself in model_card.json:
  "format": "sklearn-joblib"          -> model.joblib on ml/src/features.py features
  "format": "pytorch-wav2vec2-head"   -> ml/src/models/wav2vec_head.Wav2Vec2Predictor
Folders from before model cards (metadata.json + model.joblib) still load as sklearn.

All audio preprocessing and features are imported from ml/src (Rule 3).
Temp audio lives in a TempAudioDir and is deleted after inference (Rule 5).
The /api/analyze response is the same whichever format is loaded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np

from backend.app.audio import TempAudioDir, save_upload_and_convert
from backend.app.config import settings
from backend.app.errors import AudioSilent, AudioTooLong, AudioTooShort, ModelNotLoaded
from backend.app.safety import compute_uncertainty
from backend.app.schemas import EmotionScore, PredictionResponse
from ml.src.features import extract_features
from ml.src.preprocessing import load_audio, prepare_waveform, validate_audio

META_KEYS = ("version", "model_family", "labels", "corpora", "metrics", "limitations", "sample_rate")


@dataclass
class SpeechModel:
    predict_proba: Callable[[np.ndarray], dict]   # prepared 16 kHz waveform -> {label: probability}
    meta: dict
    format: str

    @property
    def labels(self) -> list[str]:
        return list(self.meta["labels"])


_model: Optional[SpeechModel] = None


def _sklearn_predictor(pipeline, labels: list[str], sr: int) -> Callable[[np.ndarray], dict]:
    if sorted(pipeline.classes_) != sorted(labels):
        raise RuntimeError("model classes do not match the model card labels")

    def predict(y: np.ndarray) -> dict:
        proba = pipeline.predict_proba(extract_features(y, sr=sr).reshape(1, -1))[0]
        return dict(zip(pipeline.classes_, (float(p) for p in proba)))

    return predict


def load_model(model_dir: Path | str = settings.MODEL_DIR) -> Optional[SpeechModel]:
    global _model
    d = Path(model_dir)
    card_path = d / "model_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        fmt = card["format"]
        meta = {k: card[k] for k in META_KEYS}
        sr = int(meta["sample_rate"])
        if fmt == "sklearn-joblib":
            predict = _sklearn_predictor(joblib.load(d / card["files"]["model"]), meta["labels"], sr)
        elif fmt == "pytorch-wav2vec2-head":
            from ml.src.models.wav2vec_head import Wav2Vec2Predictor  # torch is only needed for this format

            predictor = Wav2Vec2Predictor(d)
            if predictor.labels != meta["labels"]:
                raise RuntimeError("label_map.json does not match the model card labels")
            predict = predictor.predict_proba
        else:
            raise RuntimeError(f"unknown model format {fmt!r} in {card_path}")
    elif (d / "model.joblib").exists() and (d / "metadata.json").exists():  # legacy folder
        fmt = "sklearn-joblib"
        meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        meta.setdefault("sample_rate", 16000)
        predict = _sklearn_predictor(joblib.load(d / "model.joblib"), meta["labels"], int(meta["sample_rate"]))
    else:
        _model = None
        return None
    _model = SpeechModel(predict, meta, fmt)
    return _model


def get_model() -> Optional[SpeechModel]:
    return _model


def analyze_bytes(file_bytes: bytes, filename: str, content_type: Optional[str]) -> PredictionResponse:
    model = _model
    if model is None:
        raise ModelNotLoaded()
    sr = int(model.meta.get("sample_rate", 16000))

    with TempAudioDir() as tmp:
        wav = save_upload_and_convert(file_bytes, filename, content_type, tmp, target_sr=sr)
        y, sr = load_audio(wav, target_sr=sr)
    # Temp files are gone here; only the in-memory waveform remains.

    check = validate_audio(y, sr, settings.MIN_DURATION_S, settings.MAX_DURATION_S)
    duration = float(check["duration"])
    if check["is_too_short"]:
        raise AudioTooShort(duration, settings.MIN_DURATION_S)
    if check["is_too_long"]:
        raise AudioTooLong(duration, settings.MAX_DURATION_S)
    if check["is_silent"]:
        raise AudioSilent()

    by_class = model.predict_proba(prepare_waveform(y))
    del y
    scores = {label: float(by_class[label]) for label in model.labels}

    predicted = max(scores, key=scores.get)
    return PredictionResponse(
        predicted_emotion=predicted,
        confidence=round(scores[predicted], 4),
        scores=[EmotionScore(emotion=k, score=round(v, 4)) for k, v in scores.items()],
        uncertainty=compute_uncertainty(scores),
        model_version=model.meta["version"],
        audio_duration_seconds=round(duration, 2),
    )
