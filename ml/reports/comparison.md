# Speech models: classical vs log-Mel CNN

Generated 2026-11-12 19:56 by `python -m ml.src.compare_speech`. Acted speech only; these are not real-world accuracy figures.

All models use the same leave-one-speaker-out folds (`ml/src/splits.py`, identical to Phase 2's `LeaveOneGroupOut`) and the same metrics (`ml/src/evaluate.py`). Classical models: Phase 2 pipelines and grids on 396 pooled MFCC/spectral statistics, tuned by speaker-grouped inner CV. CNN: 3 conv blocks on 1 x 80 x time log-Mel input, weighted cross-entropy, early stopping on a held-out training speaker's macro-F1, augmentation on training clips only.

- **Macro-F1 / UA (pooled):** over all out-of-fold predictions. UA = unweighted accuracy (mean per-class recall).
- **Per fold (mean ± sd):** the score on each held-out speaker (A, B) or held-out corpus (C); the sd is the fold-to-fold variation.
- **Size:** the serialized fold-1 model. **CPU time:** median of 20 runs for one 4-second clip, from raw waveform through preprocessing, features and prediction, on this machine's CPU (4 threads).

## Experiment A: Emo-DB, 7 classes (535 clips, 10 speakers, leave-one-speaker-out)

| Model | Macro-F1 | UA | Macro-F1 per fold (mean ± sd) | UA per fold (mean ± sd) | Size | CPU time, 4 s clip |
|---|---|---|---|---|---|---|
| Majority baseline | 0.055 | 0.143 | 0.057 ± 0.009 | 0.145 ± 0.008 | 11 KB (-) | 160 ms |
| SVM (RBF) | 0.675 | 0.673 | 0.662 ± 0.094 | 0.675 ± 0.102 | 1.2 MB (386 support vectors) | 162 ms |
| Random Forest | 0.583 | 0.583 | 0.549 ± 0.157 | 0.586 ± 0.143 | 4.5 MB (37,614 tree nodes) | 288 ms |
| Log-Mel CNN | 0.599 | 0.606 | 0.523 ± 0.082 | 0.568 ± 0.071 | 103 KB (24,311 parameters) | 12 ms |

**Experiment A: No, the CNN does not beat the SVM.** Pooled macro-F1 CNN 0.599 vs SVM 0.675 (-0.076); the CNN scores higher on 1 of 10 held-out folds (mean per-fold difference -0.139 ± 0.121); Wilcoxon signed-rank p = 0.006.

Per-class recall (pooled):

| Class | SVM (RBF) | Random Forest | Log-Mel CNN |
|---|---|---|---|
| anger | 0.83 | 0.81 | 0.57 |
| boredom | 0.73 | 0.52 | 0.69 |
| disgust | 0.54 | 0.35 | 0.61 |
| fear | 0.61 | 0.64 | 0.59 |
| happiness | 0.56 | 0.51 | 0.48 |
| neutral | 0.65 | 0.53 | 0.54 |
| sadness | 0.79 | 0.73 | 0.76 |

CNN early stopping: best epoch median 27 (range 14-37), config `ml/configs/cnn_expA_20261112-174527.json`.

## Experiment B: SAVEE, 7 classes (480 clips, 4 speakers, leave-one-speaker-out)

| Model | Macro-F1 | UA | Macro-F1 per fold (mean ± sd) | UA per fold (mean ± sd) | Size | CPU time, 4 s clip |
|---|---|---|---|---|---|---|
| Majority baseline | 0.057 | 0.143 | 0.057 ± 0.000 | 0.143 ± 0.000 | 11 KB (-) | 231 ms |
| SVM (RBF) | 0.378 | 0.375 | 0.333 ± 0.196 | 0.375 ± 0.163 | 1.1 MB (354 support vectors) | 208 ms |
| Random Forest | 0.359 | 0.354 | 0.314 ± 0.184 | 0.354 ± 0.170 | 4.5 MB (37,518 tree nodes) | 476 ms |
| Log-Mel CNN | 0.309 | 0.327 | 0.275 ± 0.117 | 0.327 ± 0.121 | 103 KB (24,311 parameters) | 18 ms |

**Experiment B: No, the CNN does not beat the SVM.** Pooled macro-F1 CNN 0.309 vs SVM 0.378 (-0.068); the CNN scores higher on 1 of 4 held-out folds (mean per-fold difference -0.058 ± 0.118); only 4 folds, too few for a meaningful significance test.

Per-class recall (pooled):

| Class | SVM (RBF) | Random Forest | Log-Mel CNN |
|---|---|---|---|
| anger | 0.58 | 0.52 | 0.35 |
| disgust | 0.30 | 0.17 | 0.25 |
| fear | 0.37 | 0.18 | 0.22 |
| happiness | 0.23 | 0.20 | 0.27 |
| neutral | 0.64 | 0.74 | 0.74 |
| sadness | 0.20 | 0.33 | 0.00 |
| surprise | 0.30 | 0.33 | 0.47 |

CNN early stopping: best epoch median 22 (range 17-41), config `ml/configs/cnn_expB_20261112-174527.json`.

## Experiment C: cross-corpus, 6 shared classes (874 clips; fold 1 trains on SAVEE and tests on Emo-DB, fold 2 the reverse)

| Model | Macro-F1 | UA | Macro-F1 per fold (mean ± sd) | UA per fold (mean ± sd) | Size | CPU time, 4 s clip |
|---|---|---|---|---|---|---|
| Majority baseline | 0.073 | 0.120 | 0.046 ± 0.005 | 0.167 ± 0.000 | as in A/B | as in A/B |
| SVM (RBF) | 0.090 | 0.184 | 0.091 ± 0.005 | 0.183 ± 0.000 | as in A/B | as in A/B |
| Random Forest | 0.171 | 0.183 | 0.128 ± 0.023 | 0.172 ± 0.057 | as in A/B | as in A/B |
| Log-Mel CNN | 0.242 | 0.257 | 0.197 ± 0.055 | 0.231 ± 0.077 | as in A/B | as in A/B |

**Experiment C: No clear winner.** Pooled macro-F1 CNN 0.242 vs SVM 0.090 (+0.152); the CNN scores higher on 2 of 2 held-out folds (mean per-fold difference +0.106 ± 0.061); only 2 folds, too few for a meaningful significance test.

Per direction (macro-F1 / UA):

- Majority baseline: test on Emo-DB: 0.049 / 0.167; test on SAVEE: 0.042 / 0.167
- SVM (RBF): test on Emo-DB: 0.094 / 0.183; test on SAVEE: 0.087 / 0.183
- Random Forest: test on Emo-DB: 0.112 / 0.132; test on SAVEE: 0.144 / 0.213
- Log-Mel CNN: test on Emo-DB: 0.158 / 0.177; test on SAVEE: 0.236 / 0.286

Per-class recall (pooled):

| Class | SVM (RBF) | Random Forest | Log-Mel CNN |
|---|---|---|---|
| anger | 0.09 | 0.20 | 0.33 |
| disgust | 0.00 | 0.23 | 0.06 |
| fear | 0.91 | 0.17 | 0.06 |
| happiness | 0.01 | 0.25 | 0.28 |
| neutral | 0.00 | 0.04 | 0.67 |
| sadness | 0.09 | 0.21 | 0.15 |

CNN early stopping: best epoch median 9 (range 5-13), config `ml/configs/cnn_expC_20261112-174527.json`.

## Caveats

- Very small data: 10 and 4 speakers. Fold-to-fold variation is large, so differences of a few points are within noise. SAVEE has only 4 folds, too few for a significance test.
- The CNN was not hyperparameter-tuned (fixed architecture and optimiser); the classical models had small grids. The CNN's early-stopping speaker is taken from the training speakers, so each CNN trains on one speaker fewer than the classical models in the same fold.
- Experiments A and B are within-corpus. Experiment C (train on one corpus, test on the other, 6 shared classes) is the only cross-corpus evidence, and it has just 2 folds. None of this shows how any model handles real, unacted speech or phone microphones.
- Out-of-fold probabilities for every model are saved in `ml/reports/predictions/` for calibration analysis.
- CNN trained on CPU; CPU timings are single clips with no batching. Library versions: torch 2.11.0+cpu, scikit-learn 1.9.1.

<!-- w2v2:start -->
## Wav2Vec 2.0, step 1: frozen backbone + head

`facebook/wav2vec2-base` (self-supervised pretraining on LibriSpeech, English read speech; not fine-tuned for emotion). Raw 16 kHz waveform after `prepare_waveform`; backbone fully frozen; embeddings extracted once per clip and cached; head = mean pooling, dropout, linear, with weighted cross-entropy and early stopping on the validation speaker's macro-F1. Same folds and metrics as above.

- **W2V2 frozen, last layer**: head on transformer layer 12, the pre-registered main result.
- **W2V2 frozen, best layer**: layer 1-12 chosen in each fold on the validation speaker only (never the test speaker); a diagnostic, because emotion information often peaks in middle layers.

### Experiment A: Emo-DB

| Model | Macro-F1 | UA | Macro-F1 per speaker (mean ± sd) | UA per speaker (mean ± sd) | Size | CPU time, 4 s clip |
|---|---|---|---|---|---|---|
| SVM (RBF), from the table above | 0.675 | 0.673 | 0.662 ± 0.094 | see above | see above | see above |
| W2V2 frozen, last layer | 0.802 | 0.799 | 0.754 ± 0.120 | 0.781 ± 0.082 | 360 MB (94.4M backbone + 5,383 head params) | 518 ms |
| W2V2 frozen, best layer | 0.878 | 0.876 | 0.862 ± 0.053 | 0.868 ± 0.051 | 360 MB (94.4M backbone + 5,383 head params) | 510 ms |

Layers chosen per fold (best-layer variant): [10, 6, 8, 3, 3, 4, 9, 10, 7, 9].

**Step-2 rule (fixed in advance: within 0.05 of the SVM):** frozen last-layer macro-F1 0.802 vs SVM 0.675 (+0.126), so step 1 is **competitive** in experiment A.

Step-2 time estimate on this CPU (4 threads), measured: 0.19 s per clip for a forward + backward pass through transformer layers 11-12 (clips of 122 frames, lower layers cached). With 428 training clips per fold, about 15 epochs and 10 folds: **about 3.9 hours**.

### Experiment B: SAVEE

| Model | Macro-F1 | UA | Macro-F1 per speaker (mean ± sd) | UA per speaker (mean ± sd) | Size | CPU time, 4 s clip |
|---|---|---|---|---|---|---|
| SVM (RBF), from the table above | 0.378 | 0.375 | 0.333 ± 0.196 | see above | see above | see above |
| W2V2 frozen, last layer | 0.377 | 0.382 | 0.343 ± 0.133 | 0.382 ± 0.103 | 360 MB (94.4M backbone + 5,383 head params) | 517 ms |
| W2V2 frozen, best layer | 0.657 | 0.664 | 0.654 ± 0.124 | 0.664 ± 0.119 | 360 MB (94.4M backbone + 5,383 head params) | 579 ms |

Layers chosen per fold (best-layer variant): [6, 7, 7, 7].

**Step-2 rule (fixed in advance: within 0.05 of the SVM):** frozen last-layer macro-F1 0.377 vs SVM 0.378 (-0.001), so step 1 is **competitive** in experiment B.

Step-2 time estimate on this CPU (4 threads), measured: 0.27 s per clip for a forward + backward pass through transformer layers 11-12 (clips of 177 frames, lower layers cached). With 240 training clips per fold, about 15 epochs and 4 folds: **about 1.2 hours**.

### Experiment C: cross-corpus, 6 shared classes (train on one corpus, test on the other)

| Model | Macro-F1 | UA | Test on Emo-DB (macro-F1 / UA) | Test on SAVEE (macro-F1 / UA) |
|---|---|---|---|---|
| SVM (RBF), from the table above | 0.090 | 0.184 | see above | see above |
| W2V2 frozen, last layer | 0.218 | 0.247 | 0.195 / 0.227 | 0.196 / 0.260 |
| W2V2 frozen, best layer | 0.245 | 0.271 | 0.283 / 0.300 | 0.160 / 0.215 |

Size and CPU time are the same as in A and B. The early-stopping speaker comes from the training corpus, so nothing from the test corpus is used for any choice.

### Deployed configuration: both corpora, 6 classes, leave-one-speaker-out (14 speakers)

| Model | Macro-F1 | UA | Macro-F1 per speaker (mean ± sd) | UA per speaker (mean ± sd) |
|---|---|---|---|---|
| W2V2 frozen, last layer | 0.588 | 0.591 | 0.634 ± 0.219 | 0.656 ± 0.216 |
| W2V2 frozen, best layer | 0.774 | 0.771 | 0.804 ± 0.181 | 0.811 ± 0.165 |

Layers chosen per fold (best-layer variant): [3, 12, 4, 6, 7, 5, 9, 5, 4, 6, 6, 7, 5, 2]. The served SVM scores macro-F1 0.596 in this same setting (see `model_selection.md`).

**Step 2 (fine-tuning the last two transformer layers) was not run.** experiment A: it would take about 3.9 hours on this CPU; experiment B: it would take about 1.2 hours on this CPU. It needs a GPU, or a decision to accept the run time.

<!-- w2v2:end -->
