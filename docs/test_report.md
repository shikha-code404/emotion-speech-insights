# Final test report

Date: 13 November 2026. Machine: Windows 11 laptop, CPU only, Python 3.14, Node 18+, ffmpeg 9.0.2.
Served speech model: `ser-w2v2-v2`. Survey module: `survey-v1` (deterministic DASS-21 scoring).

**Result: every automated test and live check passes after two fixes (one code fix, one test fix, both below).**
Lint was not run because no linter is configured in the project.

## Summary

| Check | Result | Notes |
|---|---|---|
| **Part 1: automated tests** | | |
| ML: Emo-DB and SAVEE filename parsers | Pass, 41/41 | `ml/tests/test_manifest_parsers.py`: every emotion code, speaker, invalid names, corrupt or empty audio, SAVEE mirror-tree dedup |
| ML: speaker-disjoint splits | Pass, 18/18 | `test_speech_folds_disjoint.py` (experiments A, B, C, deployed setting, inner splits) and `test_splits_and_cnn.py` |
| ML: DASS-21 scoring rules | Pass, 15/15 | `test_survey_scoring_rules.py`: item partition, doubling, every band boundary, invalid answers, tie rule |
| ML: 1,200-row scoring match | Pass, 10/10 | `test_survey_scoring_dataset.py`: scores, bands, emotional state and severity level match all 1,200 rows exactly; no ties occur in the dataset |
| ML: lifestyle leakage test | Pass, 16/16 | `test_survey_no_leakage.py`: no DASS item, score, band or label reaches the lifestyle model |
| ML: survey workbook loading | Pass, 10/10 | `test_survey_load.py` |
| Backend (pytest + TestClient) | Pass, 36/36 | Valid WAV, missing consent on both modules, oversized file, fake .wav, empty file, silent clip, too-short clip, survey known-answer case, tie case, out-of-range and missing answers, all-zero answers, temp-file cleanup after every outcome, no answers in logs, lifestyle model not loaded |
| **Python total** | **Pass, 146/146** | `python -m pytest` |
| Frontend type check | Pass | `npm run typecheck` (tsc -b), no errors |
| Frontend lint | **Not run** | No ESLint or other linter is configured; adding one would be new tooling |
| Frontend production build | Pass | Builds with `VITE_API_URL=https://…`; correctly refuses to build when the URL is missing. Note: Vite warns the main JS chunk is over 500 kB (Recharts); harmless |
| Playwright end to end | Pass, 9/9 | Voice only (fake microphone), questionnaire only (keyboard, back keeps answers), both (upload + tie, two separate cards), too-short error, About limitations + phone width, no browser storage, uncertain state (high, moderate), keyboard-only at 375 px |
| **Part 2: live API (uvicorn on :8000, real model)** | | |
| `GET /api/health` | Pass | `status ok`, `model_loaded true`, `survey_loaded true` |
| `GET /api/model-info` | Pass | version `ser-w2v2-v2`, 6 labels |
| `GET /api/survey/questions` | Pass | 21 items, answer scale 0–3 |
| `POST /api/analyze`, SAVEE clip per emotion | Pass, 12/12 classes correct | Two speakers (DC, KL), table below. See caveat: these clips were in the deployed model's training data |
| `POST /api/analyze`, browser-style WebM | Pass | Opus 48 kHz stereo WebM converted by ffmpeg; predicted happiness (actual happiness), 4.31 s, ~0.6–0.8 s |
| `POST /api/survey/score`, three answer sets | Pass | Table below |
| Error cases | Pass, 19/19 | 17 on the normal server + `model_not_loaded` and `survey_not_loaded` on a second server with empty model folders. 1 failed before the fix |
| **Part 3: safety and privacy** | | |
| No temp audio left after all live calls | Pass | The server's temp folder was redirected to an empty directory; after every success and error call it held no `ser_audio_*` folders or audio files |
| No survey answers in any log | Pass | Server log (54 lines) contains no item IDs or answers; also covered by `test_survey_answers_are_never_written_to_disk_or_logs` |
| Voice module wording | Pass | Voice card, Voice Check page, `inference.py`, `audio.py` and the speech fields of `/api/model-info` contain no depression, anxiety, stress or risk-score wording. "diagnosis" appears only in "this is not a diagnosis" |
| Questionnaire results notice | Pass | The card is titled "DASS-21 self-report screening result" and always renders the API `notice` ("…self-report screening result… It is not a diagnosis…"); every score response carries it (tested on 4 answer patterns) |
| About page | Pass | States "acted speech only" and that the survey dataset "is fully synthetic" (read from the API, with the same text as fallback) |
| Nothing sensitive tracked by git | Pass | No `.env` (only the two `.env.example`), no audio, no workbook, no secrets (pattern search for API keys and tokens found none). Tracked CSV/NPZ files are metrics and predictions only (labels, speaker IDs, file hashes, probabilities) |

### Live predictions, one SAVEE clip per emotion

| Clip | Actual | Predicted | Confidence | Uncertainty |
|---|---|---|---|---|
| DC/a03 | anger | anger | 0.736 | low |
| DC/d03 | disgust | disgust | 0.694 | moderate |
| DC/f03 | fear | fear | 0.826 | low |
| DC/h03 | happiness | happiness | 0.967 | low |
| DC/n03 | neutral | neutral | 0.722 | low |
| DC/sa03 | sadness | sadness | 0.870 | low |
| DC/su03 | surprise | fear | 0.901 | low (surprise is not a model class) |
| KL/a03 | anger | anger | 0.308 | high |
| KL/d03 | disgust | disgust | 0.348 | high |
| KL/f03 | fear | fear | 0.478 | moderate |
| KL/h03 | happiness | happiness | 0.479 | moderate |
| KL/n03 | neutral | neutral | 0.898 | low |
| KL/sa03 | sadness | sadness | 0.482 | moderate |
| KL/su03 | surprise | fear | 0.366 | high (surprise is not a model class) |

**Caveat:** the deployed model was trained on all 14 speakers, so these clips were seen in training. This check proves the
live pipeline works end to end; it does **not** measure accuracy. The honest estimate is the leave-one-speaker-out
result (macro-F1 0.774) in `ml/reports/model_selection.md`. The first request took ~2.8 s (warm-up); later ones took
0.34–0.51 s.

### Live survey scoring

| Answer set | Depression | Anxiety | Stress | Dominant | Tied | Supportive message |
|---|---|---|---|---|---|---|
| All zeros | 0 Normal | 0 Normal | 0 Normal | Normal | none | none |
| Severe anxiety (anxiety items 3, stress items 1) | 0 Normal | 42 Extremely Severe | 14 Normal | Anxiety | none | shown |
| Tie | 14 Moderate | 10 Moderate | 0 Normal | Depression | Depression, Anxiety | shown |

Every response had `result_type: dass21_self_report_screening` and the "not a diagnosis" notice.

### Live error cases

| Request | Expected | Got |
|---|---|---|
| analyze, no consent field | 400 `consent_required` | 400 `consent_required` |
| analyze, `consent=false` | 400 `consent_required` | 400 `consent_required` |
| analyze, 11 MB file | 413 `file_too_large` | 413 `file_too_large` |
| analyze, `.txt` upload | 415 `unsupported_format` | 415 `unsupported_format` |
| analyze, text renamed `.wav` | 422 `decode_failed` | 422 `decode_failed` |
| analyze, empty `.wav` | 422 `decode_failed` | 422 `decode_failed` |
| analyze, 3 s of silence | 422 `audio_silent` | 422 `audio_silent` |
| analyze, 0.5 s clip | 422 `audio_too_short` | 422 `audio_too_short` |
| analyze, 35 s clip | 422 `audio_too_long` | 422 `audio_too_long` |
| analyze, no file | 422 `invalid_request` | 422 `invalid_request` |
| analyze, server without a model | 503 `model_not_loaded` | 503 `model_not_loaded` |
| score, no consent field | 400 `consent_required` | 400 `consent_required` (**after fix**; was 422 `invalid_request`) |
| score, `consent: false` | 400 `consent_required` | 400 `consent_required` |
| score, answer 4 | 422 `invalid_answers` | 422 `invalid_answers` |
| score, answer −1 | 422 `invalid_answers` | 422 `invalid_answers` |
| score, q21 missing | 422 `invalid_answers` | 422 `invalid_answers` |
| score, unknown item q22 | 422 `invalid_answers` | 422 `invalid_answers` |
| score, body not JSON | 422 `invalid_request` | 422 `invalid_request` |
| questions, server without `questions.json` | 503 `survey_not_loaded` | 503 `survey_not_loaded` |

## Failures found and fixed

1. **Survey request with no `consent` field gave the wrong error (code fix).**
   - Cause: `SurveyScoreRequest.consent` in `backend/app/schemas.py` was a required field, so pydantic rejected
     the body as `422 invalid_request` before the consent check ran. `/api/analyze` treats a missing consent as
     "not given" (`400 consent_required`), so the two modules disagreed.
   - Fix: `consent: bool = False`, matching the voice endpoint. The existing test that asserted the old behaviour
     now expects `consent_required`, keeps an `invalid_request` case (body with no answers) and also checks
     status codes. The website always sends `consent`, so it is unaffected.
2. **New known-answer test had a wrong expectation (test fix, no code change).**
   - The new `test_score_known_answer_case` expected `tied_states == []` for a result with one top subscale. The
     API's documented contract lists the subscales in the highest non-Normal band, so one entry means no tie,
     and the website only shows a tie for two or more. The test now expects `["Stress"]`.

Tests added in this pass: `test_score_known_answer_case` (hand-computed answers → Depression 10 Mild, Anxiety 4
Normal, Stress 32 Severe), and the missing-body case in `test_score_errors`.

## Not tested, and why

- **Lint:** no linter is configured in `frontend/` or the Python code. Type checking (`tsc`) is the strictest static
  check available.
- **Real microphone, real browsers other than Chromium, real phones:** Playwright uses Chromium with a fake
  microphone and a 375 px viewport. Safari/iOS MediaRecorder behaviour and real permission prompts need a human
  (see the manual checklist).
- **Docker image build:** Docker is not installed on this machine. The install script was verified instead in a
  clean virtual environment from the release tarball (previous pass).
- **Live hosted URLs:** not deployed yet, so the Hugging Face Space and Vercel site were not tested.
- **Speech accuracy on new speakers in the live check:** every SAVEE clip was in the deployed model's training
  data (see caveat above).

## Notes (not failures)

- `/api/survey/score` still works when `questions.json` is missing (only `/api/survey/questions` returns
  `survey_not_loaded`). Scoring is a fixed formula that does not need the question text, and the website cannot
  reach scoring without loading the questions, so this was left as is.
- `frontend/tsconfig.tsbuildinfo` is a build cache file that is not in `.gitignore`; do not commit it.

## Known limitations

- **Acted speech:** both corpora are actors performing emotions in studios. On an unfamiliar recording setup every
  model tested dropped to about 0.25 macro-F1, and the served model can be confidently wrong there.
- **Language and corpus mix:** Emo-DB is German (10 actors), SAVEE is English (4 male actors). There are no female
  English voices; surprise and boredom are not predicted.
- **Small data:** 874 clips from 14 speakers; results vary a lot between speakers (compare speakers DC and KL above).
- **Synthetic survey data:** the survey module was developed on a fully synthetic dataset. The 1,200-row match proves
  the formula is implemented correctly, not anything about real people.
- **Not diagnostic:** the voice check estimates the emotion expressed in one clip; the questionnaire is a self-report
  screening result. Neither is a diagnosis or a substitute for a qualified professional.
