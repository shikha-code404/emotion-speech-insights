# Model selection

Date: 12 November 2026. Sources: everything in `ml/reports/` (listed at the end).

## Summary

- **Speech module: switch from the SVM to a frozen Wav2Vec 2.0 backbone cut after layer 6, with a linear head**
  (`ml/artifacts/ser-w2v2-v2`). In the deployed setting (both corpora, 6 emotions, each of 14 speakers held out in
  turn) it scores macro-F1 **0.774** against the SVM's **0.596**, is better calibrated (ECE 0.063 vs 0.098), and
  answers in about 0.65 s for a 4-second clip on this CPU. The SVM failed one of the four rules set before the
  comparison: trained on one corpus and tested on the other, it is no better than chance.
- **Main risk of the switch:** on recordings unlike its training data, Wav2Vec is often confidently wrong. Every model
  does badly across corpora (best macro-F1 about 0.25), and Wav2Vec's confidence there stays high. See
  [Risks](#risks-of-the-recommendation).
- **Survey module: no model is selected.** DASS-21 scoring is a fixed formula. The lifestyle model is analysis-only,
  trained on a fully synthetic dataset, and is never served.

## Speech module

### How the models were compared

Every model was scored on the same speaker-grouped folds (`ml/src/splits.py`) with the same metrics
(`ml/src/evaluate.py`). No clip was ever scored by a model that had heard its speaker.

| Setting | Data | Folds | Why it matters |
|---|---|---|---|
| A | Emo-DB (German), 7 emotions, 535 clips | 10, one per held-out speaker | within-corpus, more speakers |
| B | SAVEE (English), 7 emotions, 480 clips | 4, one per held-out speaker | within-corpus, very few speakers |
| C | both corpora, 6 shared emotions, 874 clips | 2: train on one corpus, test on the other | closest test of a new recording setup |
| Deployed | both corpora, 6 shared emotions, 874 clips | 14, one per held-out speaker | the setting the served model is trained for |

**Models compared:**
- **Majority baseline:** always predicts the most common class.
- **SVM (RBF) and Random Forest:** trained on 396 pooled MFCC/spectral statistics.
- **Log-Mel CNN:** 24k parameters; not tuned.
- **Wav2Vec 2.0 base, frozen, with a linear head:** two variants, using the last layer, or a middle layer chosen
  inside each fold on a training speaker.

Fine-tuning Wav2Vec (step 2) was **not run**: it would take about 3.9 hours for setting A on this CPU
(`comparison.md`).

**Metrics:**
- **Macro-F1 and UA (unweighted accuracy, the mean per-class recall)** are pooled over all held-out predictions.
- **Per fold (mean ± sd)** is the fold-to-fold variation.
- **ECE (expected calibration error)** is the clip-weighted gap between the top probability and the actual accuracy,
  over 10 confidence bins.

### Deployed setting (both corpora, 6 emotions, 14 held-out speakers)

| Model | Macro-F1 | UA | Macro-F1 per speaker (mean ± sd) | UA per speaker (mean ± sd) | ECE |
|---|---|---|---|---|---|
| SVM (RBF), served until now | 0.596 | 0.588 | 0.632 ± 0.188 | 0.641 ± 0.172 | 0.098 |
| W2V2 frozen, last layer | 0.588 | 0.591 | 0.634 ± 0.219 | 0.656 ± 0.216 | 0.188 |
| **W2V2 frozen, middle layer** | **0.774** | **0.771** | **0.804 ± 0.181** | **0.811 ± 0.165** | **0.063** |

The CNN and Random Forest were not run in this setting. Both lost to the SVM within each corpus (below), so they
could not pass rule 1.

Per-class recall:

| Emotion | SVM | W2V2 last layer | W2V2 middle layer |
|---|---|---|---|
| anger | 0.75 | 0.79 | 0.87 |
| disgust | 0.48 | 0.35 | 0.70 |
| fear | 0.49 | 0.47 | 0.67 |
| happiness | 0.53 | 0.53 | 0.69 |
| neutral | 0.72 | 0.71 | 0.93 |
| sadness | 0.56 | 0.70 | 0.77 |

### Within each corpus (A and B)

| Model | A macro-F1 | A UA | A per fold (mean ± sd) | A ECE | B macro-F1 | B UA | B per fold (mean ± sd) | B ECE |
|---|---|---|---|---|---|---|---|---|
| Majority baseline | 0.055 | 0.143 | 0.057 ± 0.009 | 0.763 | 0.057 | 0.143 | 0.057 ± 0.000 | 0.750 |
| SVM (RBF) | 0.675 | 0.673 | 0.662 ± 0.094 | 0.055 | 0.378 | 0.375 | 0.333 ± 0.196 | 0.170 |
| Random Forest | 0.583 | 0.583 | 0.549 ± 0.157 | 0.138 | 0.359 | 0.354 | 0.314 ± 0.184 | 0.060 |
| Log-Mel CNN | 0.599 | 0.606 | 0.523 ± 0.082 | 0.087 | 0.309 | 0.327 | 0.275 ± 0.117 | 0.065 |
| W2V2 frozen, last layer | 0.802 | 0.799 | 0.754 ± 0.120 | 0.035 | 0.377 | 0.382 | 0.343 ± 0.133 | 0.307 |
| W2V2 frozen, middle layer | **0.878** | **0.876** | 0.862 ± 0.053 | 0.042 | **0.657** | **0.664** | 0.654 ± 0.124 | 0.099 |

- The CNN does not beat the SVM. In A it scored higher on only 1 of 10 held-out speakers (Wilcoxon p = 0.006); in B
  on 1 of 4. On this little data that is an acceptable, expected finding.
- SAVEE's 4 speakers make every B score noisy (sd up to 0.2).
- Per-class recall for A and B for every model is in `selection/tables.md`.

### Across corpora (C): train on one corpus, test on the other

| Model | Macro-F1 | UA | Test on Emo-DB (macro-F1 / UA) | Test on SAVEE (macro-F1 / UA) | ECE |
|---|---|---|---|---|---|
| Majority baseline | 0.073 | 0.120 | 0.049 / 0.167 | 0.042 / 0.167 | 0.841 |
| SVM (RBF) | 0.090 | 0.184 | 0.094 / 0.183 | 0.087 / 0.183 | 0.353 |
| Random Forest | 0.171 | 0.183 | 0.112 / 0.132 | 0.144 / 0.213 | 0.184 |
| Log-Mel CNN | 0.242 | 0.257 | 0.158 / 0.177 | 0.236 / 0.286 | 0.131 |
| W2V2 frozen, last layer | 0.218 | 0.247 | 0.195 / 0.227 | 0.196 / 0.260 | 0.634 |
| W2V2 frozen, middle layer | 0.245 | 0.271 | 0.283 / 0.300 | 0.160 / 0.215 | 0.555 |

Chance UA is 0.167 (6 emotions).

- **SVM:** at chance. It labels 89% of anger clips and 88% of happiness clips as "fear", so it learned cues tied to
  one corpus rather than to emotion.
- **CNN and Wav2Vec:** do somewhat better, but every model is poor here.
- **What this means for the demo:** a user's microphone and voice form a third "corpus", so real-world accuracy
  should be expected near this table, not near the deployed-setting one.

### Main confusion pairs (true → predicted, share of that true emotion)

| Setting | SVM | W2V2 middle layer |
|---|---|---|
| Deployed | happiness → anger 21%; fear → disgust 20%; neutral → disgust 13% | disgust → sadness 18%; happiness → anger 11%; fear → sadness 10% |
| A (Emo-DB) | happiness → anger 28%; neutral → boredom 18%; anger → happiness 12% | fear → happiness 12%; happiness → anger 10%; boredom → neutral 7% |
| B (SAVEE) | surprise → fear 32%; sadness → neutral 28%; neutral → disgust 27% | disgust → sadness 25%; neutral → anger 12%; neutral → sadness 10% |
| C (cross) | anger → fear 89%; happiness → fear 88%; neutral → fear 67% | anger → fear 61%; happiness → fear 51%; sadness → neutral 49% |

- **Happiness ↔ anger** is the most persistent confusion. Both are high-arousal, so they sound alike in pitch and
  energy.
- **Low-arousal emotions** (neutral, sadness, boredom, disgust) get mixed up with each other.

### Calibration

Reliability plots: `selection/calibration_deployed.png`, `selection/calibration_expA.png`, `_expB.png`, `_expC.png`.
Points below the diagonal mean over-confident.

- **Deployed setting:**
  - The middle-layer Wav2Vec head is the best calibrated (ECE 0.063): its 447 clips at about 95% confidence are
    right about 93% of the time.
  - The SVM is mildly over-confident (ECE 0.098).
  - The last-layer head is clearly over-confident (ECE 0.188).
- **Across corpora:** every model is over-confident, and Wav2Vec most of all (ECE 0.56–0.63). About 500 clips sit at
  around 95% confidence with only 20–35% accuracy.

### Size and CPU latency (4-second clip, full pipeline from raw audio)

| Model | Size on disk | Parameters / structure | CPU time, session 1 | CPU time, session 2 |
|---|---|---|---|---|
| SVM (RBF), deployed | 2.4 MB | 758 support vectors | 162–208 ms | 314 ms |
| Random Forest | 4.5 MB | about 37,600 tree nodes | 288–476 ms | not measured |
| Log-Mel CNN | 103 KB | 24,311 parameters | 12–18 ms | not measured |
| W2V2, full 12 layers | 360 MB | 94.4M parameters | 510–580 ms | 919 ms (backbone only) |
| **W2V2, cut after layer 6 (served)** | **198 MB** | **51.8M parameters** | not measured | **648 ms** |

- **Machine:** this laptop's CPU, 4 threads, one clip at a time.
- **Sessions:** session 1 is the comparison runs (`comparison.md`). Session 2 is one side-by-side benchmark run later.
  The machine ran about 2× slower in session 2, so compare models within a column, not across columns.
- **Classical models' time** is mostly feature extraction (librosa), not the classifier.

### Decision rules (fixed before the latency and calibration results were seen)

1. **Accuracy:** at least as good as the current SVM in the deployed setting, or within each corpus where no
   deployed-setting score exists.
2. **Generalisation:** clearly above chance across corpora. A user's microphone is effectively a new corpus.
3. **CPU budget:** under 1 second for a 4-second clip.
4. **Calibration:** the confidence the interface shows must roughly match accuracy (deployed-setting ECE no worse
   than the SVM's).

Then pick the **simplest** model that passes all four.

| Model | Rule 1 | Rule 2 | Rule 3 | Rule 4 | Result |
|---|---|---|---|---|---|
| SVM (RBF) | reference | **fails** (0.090, chance) | passes | reference | out |
| Random Forest | **fails** (below SVM in A and B) | 0.171 | passes | — | out |
| Log-Mel CNN | **fails** (below SVM in A, p = 0.006, and B) | 0.242 | passes | — | out |
| W2V2 frozen, last layer | ≈ SVM (0.588) | 0.218 | passes | **fails** (0.188) | out |
| **W2V2 frozen, middle layer** | **0.774** | **0.245** | **648 ms** | **0.063** | **selected** |

The simplest passing model is the frozen backbone with a linear head. No fine-tuning and no extra layers are trained,
and cutting the backbone at a middle layer makes it smaller and faster than the full model.

### The served model: `ml/artifacts/ser-w2v2-v2`

- **Layer choice (6):** layers 1–12 were each scored by leave-one-speaker-out over all 14 speakers; layer 6 was best
  (mean held-out macro-F1 0.836). Middle layers (4–8) all scored 0.80–0.84; the last layer scored 0.634. Because this
  score was used to pick the layer, it is optimistic. The honest estimate for the procedure is the nested **0.774**
  above, where the layer was chosen inside each fold.
- **Head:** trained on all 874 clips for 12 epochs, the median early-stopping epoch at layer 6.
- **Backbone:** `facebook/wav2vec2-base` (Apache-2.0), frozen, saved truncated after layer 6, so serving needs no
  network access. The export was checked to reproduce the evaluated pipeline on 5 clips, to within 1e-3.
- **Folder contents:**
  - `backbone/`
  - `head.pt`
  - `label_map.json`
  - `preprocess_config.json`: the same `ml/src` preprocessing as training (Rule 3)
  - `model_card.json`

**Backend changes**
- **Loader:** `backend/app/inference.py` reads `model_card.json` and loads either `"sklearn-joblib"` or
  `"pytorch-wav2vec2-head"`. Torch is only imported for the second format.
- **Response schema:** `/api/analyze` is unchanged.
- **Default model:** `MODEL_DIR` now points at `ser-w2v2-v2`.
- **SVM folder:** gained a `model_card.json`; its `model.joblib` and `metadata.json` are untouched. **Rollback:** set
  `MODEL_DIR` to `ml/artifacts/ser-baseline-v1`.

**Verification:** 18/18 backend tests pass, including a test that loads both formats and checks they return the same
response fields. 6/6 browser end-to-end tests pass, including the microphone and upload voice flows.

### Risks of the recommendation

- **Confidently wrong on unfamiliar recordings.** Across corpora the served model is right about 30% of the time while
  usually reporting over 90% confidence. The voice card's wording ("The model is fairly confident") can therefore
  mislead on real recordings. Two follow-ups are recommended:
  - soften or cap the confidence wording;
  - collect a small, consented set of real recordings and fit a temperature on it.
- **Heavier deployment.** About 200 MB of weights plus PyTorch, against 2.4 MB for the SVM, and about 2× its CPU
  time. Fine for a CPU-hosted demo, but the host needs roughly 1 GB of RAM.
- **Small evaluation.** 14 speakers, 2 corpora, acted speech. The gap over the SVM (+0.18 macro-F1) is large relative
  to the fold-to-fold variation (sd 0.18), but it rests on few speakers.

## Survey module

- **No model is selected.** The questionnaire result is the deterministic DASS-21 scoring formula
  (`ml/src/survey/scoring.py`): fixed item sums × 2, standard severity bands, and a documented tie rule. It
  reproduces all 1,200 stored scores, bands and labels exactly, and there is nothing to train or choose. Its output is
  a self-report screening result, not a diagnosis.
- **Lifestyle experiment (analysis only).** A separate experiment predicted the dataset's `Emotional_State` (Normal /
  Depression / Anxiety / Stress) from lifestyle answers alone, with the DASS-21 items and scores excluded. It used
  stratified 5-fold CV with tuning inside each fold (`survey/README.md`):

| Model | Macro-F1 (mean ± sd) | Balanced accuracy | Recall Normal / Depression / Anxiety / Stress |
|---|---|---|---|
| Majority baseline | 0.100 ± 0.000 | 0.250 | 0.00 / 0.00 / 1.00 / 0.00 |
| Logistic regression | 0.581 ± 0.025 | 0.583 | 0.69 / 0.54 / 0.49 / 0.61 |
| Random forest | 0.572 ± 0.038 | 0.575 | 0.69 / 0.53 / 0.49 / 0.59 |
| Gradient boosting | 0.581 ± 0.035 | 0.583 | 0.71 / 0.54 / 0.50 / 0.59 |

  The top permutation importances were mindless snacking, daily steps and blood pressure.

- **The dataset is fully synthetic.** All 1,200 respondents were artificially generated, and the lifestyle-to-state
  relationships were invented by the dataset's author. These metrics measure how well a model recovers the
  generator's rules, **not real-world accuracy**. Two snacking options ("Due to Stress", "Due to Anxity") also name
  the target classes.
- **Never served.** The lifestyle model stays analysis-only: it has no API endpoint and is never shown to users, as
  its model card states.

## Limitations

- **Acted speech.** Both corpora are actors performing emotions in studios. Everyday speech is subtler and noisier,
  and every model dropped to about 0.25 macro-F1 when moved to a recording setup it had not seen.
- **German vs English.** Emo-DB is German (10 actors) and SAVEE is English (4 male actors). Language, accent and
  recording setup are mixed up with corpus, and the Wav2Vec backbone was pretrained on English read speech only.
- **Small data.** 874 clips from 14 speakers in the deployed setting. Fold-to-fold variation is large, SAVEE has only
  4 folds, and no result here has been checked on new speakers outside these corpora.
- **Synthetic survey data.** The survey dataset is entirely generated. Nothing derived from it describes real people.
- **Neither module is a diagnostic tool.**
  - The voice check estimates the emotion expressed in one clip.
  - The questionnaire gives a self-report screening result from the user's own answers.
  - Neither one is a diagnosis or a substitute for a qualified professional, and their results are never combined.

## Source files

- **Speech:**
  - `comparison.md` (all models, A/B/C, sizes, latencies, Wav2Vec section)
  - `selection/tables.md`, `selection/metrics.csv`, `selection/calibration_*.png`,
    `selection/export_layer_selection.csv`
  - `cnn/` and `w2v2/` (per-fold results)
  - `predictions/` (out-of-fold probabilities)
  - `speech/` (Phase 2 baseline)
- **Survey:** `survey/README.md`, `survey/results.csv`, `survey/permutation_importance.csv`
- **Configs:** `ml/configs/cnn_exp*.json`, `ml/configs/w2v2_exp*.json`
- **Code:**
  - `ml/src/compare_speech.py`, `compare_wav2vec.py`, `selection_report.py`, `export_wav2vec.py`
  - `ml/src/splits.py`, `evaluate.py`
