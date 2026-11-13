"""API tests for both modules (FastAPI TestClient, in-process)."""

import io
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.app.audio import ffmpeg_path
from backend.app.config import settings
from backend.app.main import app

REPO = Path(__file__).resolve().parents[2]
MODEL_READY = any((Path(settings.MODEL_DIR) / f).exists() for f in ("model_card.json", "model.joblib"))
SURVEY_READY = (REPO / "ml" / "artifacts" / "survey-v1" / "questions.json").exists()
CORPUS_CLIP = next((REPO / "ml" / "data" / "raw" / "savee").rglob("a01.wav"), None) if (REPO / "ml/data/raw/savee").exists() else None

needs_model = pytest.mark.skipif(not MODEL_READY, reason="speech model not trained")
needs_survey = pytest.mark.skipif(not SURVEY_READY, reason="survey artifacts not exported")

DEPRESSION = (3, 5, 10, 13, 16, 17, 21)
ANXIETY = (2, 4, 7, 9, 15, 19, 20)
SPEECH_BANNED = re.compile(r"depress|anxi|stress|disorder|risk score", re.IGNORECASE)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _wav_bytes(seconds: float, freq: float = 220.0, amp: float = 0.3, sr: int = 16000) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    buf = io.BytesIO()
    sf.write(buf, (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr, format="WAV")
    return buf.getvalue()


def _ser_temp_dirs() -> set[str]:
    return {p.name for p in Path(tempfile.gettempdir()).glob("ser_audio_*")}


# ---------------------------------------------------------------------------
# Speech module
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@needs_model
def test_model_info_lists_labels_and_limitations(client):
    body = client.get("/api/model-info").json()
    assert body["labels"] == ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
    assert body["limitations"] and body["survey"]["development_data_synthetic"] is True


@needs_model
@pytest.mark.skipif(CORPUS_CLIP is None, reason="corpus audio not present")
def test_analyze_corpus_clip_and_cleanup(client):
    before = _ser_temp_dirs()
    r = client.post("/api/analyze", files={"audio": ("clip.wav", CORPUS_CLIP.read_bytes(), "audio/wav")},
                    data={"consent": "true"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction_type"] == "vocal_emotion"
    assert abs(sum(s["score"] for s in body["scores"]) - 1) < 0.01
    assert body["uncertainty"] in {"low", "moderate", "high"}
    assert not SPEECH_BANNED.search(r.text), "speech response must not use clinical/survey wording"
    assert _ser_temp_dirs() <= before, "temp audio must be deleted after inference (Rule 5)"


@needs_model
@pytest.mark.parametrize("files,data,code", [
    ({"audio": ("a.wav", _wav_bytes(3), "audio/wav")}, {}, "consent_required"),
    ({"audio": ("a.wav", _wav_bytes(0.4), "audio/wav")}, {"consent": "true"}, "audio_too_short"),
    ({"audio": ("a.wav", _wav_bytes(3, amp=0.0), "audio/wav")}, {"consent": "true"}, "audio_silent"),
    ({"audio": ("a.txt", b"hello", "text/plain")}, {"consent": "true"}, "unsupported_format"),
])
def test_analyze_errors(client, files, data, code):
    r = client.post("/api/analyze", files=files, data=data)
    assert r.json()["error_code"] == code


@needs_model
@pytest.mark.skipif(ffmpeg_path() is None, reason="ffmpeg not installed")
def test_decode_failure_does_not_leak_server_details(client):
    r = client.post("/api/analyze", files={"audio": ("x.wav", b"not audio", "audio/wav")}, data={"consent": "true"})
    body = r.json()
    assert body["error_code"] == "decode_failed"
    assert "ffmpeg" not in body["message"].lower() and "temp" not in body["message"].lower()


# ---------------------------------------------------------------------------
# Survey module
# ---------------------------------------------------------------------------

def _answers(dep=0, anx=0, stress=0):
    return {f"q{i}": dep if i in DEPRESSION else anx if i in ANXIETY else stress for i in range(1, 22)}


@needs_survey
def test_questions_have_21_items_and_4_options(client):
    body = client.get("/api/survey/questions").json()
    assert [i["id"] for i in body["items"]] == [f"q{i}" for i in range(1, 22)]
    assert [o["value"] for o in body["answer_scale"]] == [0, 1, 2, 3]
    assert "past week" in body["instructions"]
    assert "subscale" not in body["items"][0]


@needs_survey
def test_score_all_zero_is_normal_without_message(client):
    body = client.post("/api/survey/score", json={"consent": True, "answers": _answers()}).json()
    assert body["dominant_state"] == "Normal"
    assert body["tied_states"] == []
    assert body["supportive_message"] is None
    assert "not a diagnosis" in body["notice"]


@needs_survey
def test_score_known_answer_case(client):
    # Hand-computed: Depression (q3,5,10,13,16,17,21) = 1+0+2+1+0+1+0 = 5 -> 10 Mild;
    # Anxiety (q2,4,7,9,15,19,20) = 0+1+0+0+1+0+0 = 2 -> 4 Normal;
    # Stress (q1,6,8,11,12,14,18) = 3+2+2+3+2+2+2 = 16 -> 32 Severe. Dominant: Stress, no tie.
    values = {1: 3, 2: 0, 3: 1, 4: 1, 5: 0, 6: 2, 7: 0, 8: 2, 9: 0, 10: 2, 11: 3,
              12: 2, 13: 1, 14: 2, 15: 1, 16: 0, 17: 1, 18: 2, 19: 0, 20: 0, 21: 0}
    body = client.post("/api/survey/score",
                       json={"consent": True, "answers": {f"q{k}": v for k, v in values.items()}}).json()
    rows = {s["subscale"]: (s["score"], s["band"]) for s in body["subscales"]}
    assert rows == {"Depression": (10, "Mild"), "Anxiety": (4, "Normal"), "Stress": (32, "Severe")}
    # tied_states lists the subscales in the top non-Normal band; one entry means no tie.
    assert body["dominant_state"] == "Stress" and body["tied_states"] == ["Stress"]
    assert body["supportive_message"]


@needs_survey
def test_score_tie_is_reported(client):
    # Depression 7*1*2=14 (Moderate), Anxiety 5*... -> use 1s on 5 anxiety items: 10 (Moderate)
    answers = _answers(dep=1)
    for i in (2, 4, 7, 9, 15):
        answers[f"q{i}"] = 1
    body = client.post("/api/survey/score", json={"consent": True, "answers": answers}).json()
    assert body["tied_states"] == ["Depression", "Anxiety"]
    assert body["supportive_message"]
    rows = {s["subscale"]: s for s in body["subscales"]}
    assert rows["Depression"]["score"] == 14 and rows["Anxiety"]["band"] == "Moderate"
    assert rows["Stress"]["bands"][-1] == {"band": "Extremely Severe", "min": 34, "max": 42}


@needs_survey
@pytest.mark.parametrize("payload,code", [
    ({"consent": False, "answers": _answers()}, "consent_required"),
    ({"consent": True, "answers": {"q1": 1}}, "invalid_answers"),
    ({"consent": True, "answers": {**_answers(), "q5": 4}}, "invalid_answers"),
    ({"answers": _answers()}, "consent_required"),
    ({"consent": True}, "invalid_request"),
])
def test_score_errors(client, payload, code):
    r = client.post("/api/survey/score", json=payload)
    assert (r.status_code, r.json()["error_code"]) == ({"consent_required": 400}.get(code, 422), code)


# ---------------------------------------------------------------------------
# Both artifact formats load from model_card.json and give the same response shape
# ---------------------------------------------------------------------------

ARTIFACTS = REPO / "ml" / "artifacts"


@pytest.mark.parametrize("folder,fmt", [("ser-baseline-v1", "sklearn-joblib"), ("ser-w2v2-v2", "pytorch-wav2vec2-head")])
@pytest.mark.skipif(CORPUS_CLIP is None, reason="corpus audio not present")
def test_each_artifact_format_serves_the_same_schema(folder, fmt):
    from backend.app import inference
    from backend.app.schemas import PredictionResponse

    if not (ARTIFACTS / folder / "model_card.json").exists():
        pytest.skip(f"{folder} not exported")
    previous = inference.get_model()
    try:
        model = inference.load_model(ARTIFACTS / folder)
        assert model.format == fmt
        assert model.labels == ["anger", "disgust", "fear", "happiness", "neutral", "sadness"]
        resp = inference.analyze_bytes(CORPUS_CLIP.read_bytes(), "clip.wav", "audio/wav")
        assert set(resp.model_dump()) == set(PredictionResponse.model_fields)
        assert [s.emotion for s in resp.scores] == model.labels
        assert abs(sum(s.score for s in resp.scores) - 1) < 0.01
        assert not SPEECH_BANNED.search(resp.model_dump_json())
    finally:
        inference._model = previous


def test_default_model_dir_is_the_selected_model():
    assert Path(settings.MODEL_DIR).name == "ser-w2v2-v2"


# ---------------------------------------------------------------------------
# Audit checks (blueprint section 11 and scope addendum)
# ---------------------------------------------------------------------------

@pytest.fixture
def isolated_tmp(tmp_path, monkeypatch):
    """Point Python's temp directory (used by TempAudioDir and by Starlette's spooled uploads) at an empty folder."""
    d = tmp_path / "tmp"
    d.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(d))
    return d


@needs_model
@pytest.mark.parametrize("name,payload,consent,code", [
    ("ok.wav", None, "true", None),                              # success (corpus clip)
    ("empty.wav", b"", "true", "decode_failed"),
    ("corrupt.wav", b"\x00" * 2_500_000, "true", "decode_failed"),  # >1 MB: Starlette spools it to disk
    ("short.wav", _wav_bytes(0.4), "true", "audio_too_short"),
    ("silent.wav", _wav_bytes(3, amp=0.0), "true", "audio_silent"),
    ("notes.txt", b"hello", "true", "unsupported_format"),
    ("noconsent.wav", _wav_bytes(3), "false", "consent_required"),
])
def test_no_audio_left_on_disk_after_any_request(client, isolated_tmp, name, payload, consent, code):
    if payload is None:
        if CORPUS_CLIP is None:
            pytest.skip("corpus audio not present")
        payload = CORPUS_CLIP.read_bytes()
    mime = "text/plain" if name.endswith(".txt") else "audio/wav"
    r = client.post("/api/analyze", files={"audio": (name, payload, mime)}, data={"consent": consent})
    assert (r.json().get("error_code") if code else r.status_code) == (code or 200), r.text
    assert list(isolated_tmp.iterdir()) == [], "temporary audio must be deleted after every request (Rule 5)"


def test_upload_over_size_limit_is_rejected(client, isolated_tmp):
    big = b"\x00" * (settings.max_upload_bytes + 1)
    r = client.post("/api/analyze", files={"audio": ("big.wav", big, "audio/wav")}, data={"consent": "true"})
    assert r.status_code == 413 and r.json()["error_code"] == "file_too_large"
    assert list(isolated_tmp.iterdir()) == []


@needs_survey
@pytest.mark.parametrize("pattern", ["zeros", "tie", "max", "mixed"])
def test_every_survey_result_says_self_report_and_not_a_diagnosis(client, pattern):
    answers = {"zeros": _answers(), "tie": {**_answers(dep=1), **{f"q{i}": 1 for i in (2, 4, 7, 9, 15)}},
               "max": _answers(3, 3, 3), "mixed": _answers(2, 1, 3)}[pattern]
    body = client.post("/api/survey/score", json={"consent": True, "answers": answers}).json()
    assert body["result_type"] == "dass21_self_report_screening"
    assert "self-report screening result" in body["notice"] and "not a diagnosis" in body["notice"]
    assert not re.search(r"diagnosed|disorder|patient|risk score", json.dumps(body), re.IGNORECASE)


@needs_survey
def test_survey_answers_are_never_written_to_disk_or_logs(client, isolated_tmp, monkeypatch, caplog, capsys):
    import builtins
    import logging

    writes = []
    real_open = builtins.open

    def spy_open(file, mode="r", *a, **k):
        if any(c in mode for c in "wax+"):
            writes.append(str(file))
        return real_open(file, mode, *a, **k)

    monkeypatch.setattr(builtins, "open", spy_open)
    caplog.set_level(logging.DEBUG)
    marker = _answers(2, 1, 3)
    client.post("/api/survey/score", json={"consent": True, "answers": marker})
    client.post("/api/survey/score", json={"consent": True, "answers": {**marker, "q7": 9}})  # invalid
    out = capsys.readouterr()
    assert writes == [], f"files opened for writing during scoring: {writes}"
    assert list(isolated_tmp.iterdir()) == []
    logged = caplog.text + out.out + out.err
    assert '"q1"' not in logged and "'q1'" not in logged and "answers" not in logged.lower()


def test_lifestyle_model_is_never_loaded_or_routed():
    import subprocess
    import sys

    paths = [getattr(r, "path", "") for r in app.routes]
    assert not any("lifestyle" in p for p in paths), paths
    # Fresh interpreter: pytest's own process also imports the ML tests, which load train_lifestyle.
    probe = ("import sys, backend.app.main; "
             "print([m for m in sys.modules if 'lifestyle' in m])")
    out = subprocess.run([sys.executable, "-c", probe], cwd=REPO, capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stdout
    backend_src = "".join(p.read_text(encoding="utf-8") for p in (REPO / "backend" / "app").glob("*.py"))
    assert "lifestyle" not in backend_src.lower()


def test_speech_code_and_voice_card_have_no_survey_or_clinical_words():
    files = [REPO / "backend/app/inference.py", REPO / "backend/app/audio.py",
             REPO / "frontend/src/components/VoiceResultCard.tsx", REPO / "frontend/src/pages/VoiceCheck.tsx"]
    for f in files:
        text = f.read_text(encoding="utf-8")
        hits = SPEECH_BANNED.findall(text)
        assert not hits, f"{f.name}: {hits}"


@needs_model
def test_model_info_speech_fields_have_no_survey_or_clinical_words(client):
    body = client.get("/api/model-info").json()
    speech = {k: v for k, v in body.items() if k != "survey"}
    assert not SPEECH_BANNED.search(json.dumps(speech))
    assert any("acted speech" in l for l in body["limitations"])
    assert body["survey"]["development_data_synthetic"] is True
