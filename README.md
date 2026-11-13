# emotion-speech-insights: Voice & Wellbeing Check

A learning prototype with two independent modules:

1. **Voice check:** record or upload a 5–15 second clip, and a speech-emotion model estimates which of six
   emotions the voice expresses *in that clip* (anger, disgust, fear, happiness, neutral, sadness). It is trained on
   acted speech and says nothing about a person's mood or mental health.
2. **Wellbeing questionnaire:** the 21-item DASS-21. Your own answers are scored with the standard fixed formula
   into Depression, Anxiety and Stress scores and bands, presented as a **self-report screening result, not a
   diagnosis**.

The two results are always shown separately and **never combined**. Neither module is a diagnostic tool. Audio is
deleted on the server right after analysis; answers and results are never stored and live only in the browser tab
until you leave the results page.

| Part | Stack | Folder |
|---|---|---|
| Website | React, Vite, TypeScript, Tailwind, Recharts | `frontend/` |
| API | FastAPI, Python 3.14 | `backend/` |
| Models, training, shared preprocessing | scikit-learn, PyTorch, transformers, librosa | `ml/` |

Project rules (speaker-disjoint evaluation, no leakage, one copy of preprocessing, wording, audio deletion) are in
[CLAUDE.md](CLAUDE.md). The original specification and its scope addendum are in
[docs/blueprint.md](docs/blueprint.md).

---

## Repository layout

```
frontend/          website: src/pages, src/components, src/services/api.ts, e2e/ (Playwright)
backend/           API: app/ (routes, inference, survey scoring), tests/, requirements*.txt
ml/                src/ (preprocessing, features, models, training, evaluation), tests/,
                   data/raw/ (datasets, git-ignored), artifacts/ (trained models; the Wav2Vec backbone,
                   caches and release tarball are git-ignored), reports/ (metrics, plots,
                   model_selection.md), configs/ (run configs)
deploy/            install.sh (shared by Docker builds), package_artifacts.py, huggingface/ (Space files)
Dockerfile         API image for any Docker host
start.ps1          starts API + website locally in two windows (Windows)
```

---

## Setup (local)

Requirements: Python 3.14, Node.js 18+, ffmpeg (Windows: `winget install Gyan.FFmpeg`; the backend also finds a
winget install or `FFMPEG_PATH`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt      # everything: API + training + tests
cd frontend; npm install; cd ..
```

The trained models are not in the repo. Either train them (below) or download the release tarball
(`ser-artifacts-*.tar.gz`) and extract it in the repo root.

Run:

```powershell
.\start.ps1                                   # opens the API (:8000) and the website (:5173)
# or by hand:
python -m uvicorn backend.app.main:app --port 8000
cd frontend; npm run dev                      # http://localhost:5173
```

Environment variables are documented in [backend/.env.example](backend/.env.example) and
[frontend/.env.example](frontend/.env.example).

---

## Reproducing training and evaluation

Put the datasets in `ml/data/raw/` first (see [Datasets](#datasets)). All commands run from the repo root; every
run uses fixed seeds and writes its config to `ml/configs/`.

```powershell
python -m ml.src.build_manifest                    # -> ml/data_manifest/audio_manifest.csv (labels, speakers, hashes)
python -m ml.src.eda                               # class / speaker / duration plots -> ml/reports/
python -m ml.src.train_speech                      # SVM + Random Forest baseline -> ml/artifacts/ser-baseline-v1
python -m ml.src.compare_speech                    # experiments A, B, C: majority, SVM, RF, log-Mel CNN (~3 h on CPU)
python -m ml.src.compare_wav2vec                   # Wav2Vec 2.0 frozen-backbone heads (downloads facebook/wav2vec2-base)
python -m ml.src.compare_wav2vec --experiments D --stages train   # same, in the deployed setting
python -m ml.src.selection_report                  # calibration, ECE, per-class recall, confusion pairs
python -m ml.src.export_wav2vec                    # selected model -> ml/artifacts/ser-w2v2-v2
python -m ml.src.survey.train_lifestyle            # survey analysis + ml/artifacts/survey-v1/questions.json
```

Evaluation protocol:
- **Folds:** every speech model uses the same speaker-grouped folds (`ml/src/splits.py`): leave-one-speaker-out, or
  leave-one-corpus-out for experiment C. No clip is ever scored by a model that heard its speaker.
- **Training-fold-only fitting:** scalers, normalisation and calibration are fit on training folds only.
- **Metrics:** macro-F1, unweighted accuracy, per-class recall, confusion matrices, fold-to-fold variation and ECE
  (`ml/src/evaluate.py`).
- **Results and choice:** [ml/reports/model_selection.md](ml/reports/model_selection.md) has the full results and
  the reasoning behind the model choice.

Tests:

```powershell
python -m pytest                                    # backend/tests + ml/tests (scoring, leakage, folds, API, privacy)
cd frontend; npx playwright install chromium; npm run e2e   # end-to-end, starts both servers itself
```

---

## Model cards (brief)

### Speech: `ser-w2v2-v2` (served)

| | |
|---|---|
| Predicts | the vocal emotion expressed in one clip: anger, disgust, fear, happiness, neutral, sadness |
| Model | `facebook/wav2vec2-base` (Apache-2.0), frozen, cut after transformer layer 6; mean pooling; linear head |
| Input | 16 kHz mono waveform, silence trimmed, peak-normalised (`ml/src/preprocessing.py`) |
| Training data | Emo-DB + SAVEE, acted speech, 874 clips, 14 speakers, 6 shared emotions |
| Evaluation | leave-one-speaker-out, 14 speakers: macro-F1 0.774, unweighted accuracy 0.771, ECE 0.063 |
| Across corpora | trained on one corpus, tested on the other: macro-F1 0.245, with over-confident probabilities |
| Size / speed | 51.8M parameters, 198 MB; ~0.5–0.65 s per 4 s clip on a 4-thread laptop CPU; ~700 MB RAM |
| Not for | any statement about a person's mental health, decisions about people, monitoring over time |

The previous model, `ser-baseline-v1` (SVM on 396 MFCC/spectral statistics), is kept for rollback. It scores
macro-F1 0.596 in the same evaluation and is at chance across corpora. To roll back, set
`MODEL_DIR=ml/artifacts/ser-baseline-v1`.

### Survey: DASS-21 scoring (served) and lifestyle model (analysis only)

| | |
|---|---|
| Served | deterministic DASS-21 scoring (`ml/src/survey/scoring.py`): item sums × 2, standard bands, documented tie rule; no machine learning |
| Output | Depression, Anxiety and Stress scores (0–42) and bands, labelled a self-report screening result, not a diagnosis |
| Verified | reproduces all 1,200 stored scores, bands and labels of the development dataset exactly |
| Lifestyle model | predicts the dataset's label from lifestyle answers only; macro-F1 about 0.58. **Analysis-only**: never served, no endpoint, not in the Docker image |
| Data | **fully synthetic**: these metrics show the data generator's rules, not real-world accuracy |

---

## API

Base URL: `http://localhost:8000` locally. Errors return `{"error_code": "...", "message": "..."}`.

| Method & path | Purpose |
|---|---|
| `GET /api/health` | `{"status", "model_loaded", "survey_loaded"}` |
| `GET /api/model-info` | speech model version, family, labels, corpora, metrics, limitations; survey module info |
| `POST /api/analyze` | multipart: `audio` (WAV/MP3/WebM/OGG/M4A, ≤ 10 MB, 1–30 s), `consent=true`. Returns `predicted_emotion`, `confidence`, `scores[]`, `uncertainty` (low / moderate / high), `model_version`, `audio_duration_seconds`, `notice` |
| `GET /api/survey/questions` | the 21 items, the 0–3 answer scale and instructions ("over the past week") |
| `POST /api/survey/score` | JSON `{"consent": true, "answers": {"q1": 0..3, …, "q21": 0..3}}`. Returns the three subscales with score, band and band ranges, `dominant_state`, `tied_states`, `supportive_message`, `notice` |

Error codes:
- **Speech:** `consent_required`, `file_too_large`, `unsupported_format`, `audio_too_short`, `audio_too_long`,
  `audio_silent`, `decode_failed`, `model_not_loaded`.
- **Survey:** `invalid_answers`, `survey_not_loaded`.
- **Either:** `invalid_request`.

---

## Deployment (free tiers)

- **Website:** Vercel, project root `frontend/`, configured by `frontend/vercel.json`.
- **API:** a Docker container on a Hugging Face Space (free CPU tier, 16 GB RAM). The selected model needs about
  700 MB of RAM, more than the 512 MB of Render's and Railway's free tiers.
- **Model files:** the Space builds from `deploy/huggingface/Dockerfile`, which clones this repo and downloads the
  model files from a GitHub Release (`python deploy/package_artifacts.py` makes the tarball).
- **Other Docker hosts:** the root `Dockerfile` works anywhere. With the SVM (`SER_MODEL=ser-baseline-v1`) it skips
  PyTorch and fits in 512 MB.

Production environment variables:

| Variable | Where | Production value |
|---|---|---|
| `MODEL_DIR` | Space (set by the Dockerfile) | `/app/ml/artifacts/ser-w2v2-v2` |
| `SURVEY_DIR` | Space (set by the Dockerfile) | `/app/ml/artifacts/survey-v1` |
| `MAX_UPLOAD_MB` | Space variable | `10` |
| `MIN_DURATION_S` | Space variable | `1` |
| `MAX_DURATION_S` | Space variable | `30` |
| `ALLOWED_ORIGINS` | Space variable | `https://<your-project>.vercel.app` (exact origin; `*` is refused) |
| `VITE_API_URL` | Vercel env var (build time) | `https://<hf-username>-<space-name>.hf.space` |

---

## Limitations

- **Acted speech.** Both corpora are actors performing emotions in studios. Everyday speech is subtler, and on an
  unfamiliar recording setup every model tested dropped to about 0.25 macro-F1. The served model can be confidently
  wrong there.
- **German vs English.** Emo-DB is German (10 actors); SAVEE is English (4 male actors). Language, accent and
  recording setup are mixed up with corpus. The Wav2Vec backbone was pretrained on English read speech.
- **Small data.** 874 clips from 14 speakers; results vary a lot between speakers.
- **Synthetic survey data.** The questionnaire was developed and tested on a fully synthetic dataset; no metric
  from it describes real people.
- **Not diagnostic.** The voice check estimates the emotion expressed in one clip. The questionnaire is a self-report
  screening result. Neither is a diagnosis or a substitute for a qualified professional.

---

## Datasets

The data is **not** included in this repository. Obtain each dataset from its source under its own terms and place
it as shown.

| Dataset | Source | Place in |
|---|---|---|
| Emo-DB (Berlin Database of Emotional Speech): 535 German utterances, 10 actors, 7 emotions | Burkhardt et al., "A Database of German Emotional Speech", Interspeech 2005; http://emodb.bilderbar.info/ | `ml/data/raw/emodb/wav/*.wav` |
| SAVEE (Surrey Audio-Visual Expressed Emotion): 480 English utterances, 4 male actors, 7 emotions | Haq & Jackson, University of Surrey, 2009; http://kahlan.eps.surrey.ac.uk/savee/ (registration required) | `ml/data/raw/savee/AudioData/AudioData/<speaker>/*.wav` (one copy of the tree only) |
| Survey dataset: 1,200 **synthetic** DASS-21 + lifestyle responses | project-provided `survey_dataset_1200.xlsx`, artificially generated (see its "Notes" sheet) | `ml/data/raw/survey/survey_dataset_1200.xlsx` |
| DASS-21 questionnaire | Lovibond & Lovibond (1995); the authors state the DASS is in the public domain | items ship in `questions.json` |
| Wav2Vec 2.0 base | Baevski et al., 2020; `facebook/wav2vec2-base` on Hugging Face (Apache-2.0) | downloaded by `compare_wav2vec` |
