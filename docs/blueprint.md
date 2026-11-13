
EMOTION DETECTIONFROM SPEECH
WITH MENTAL HEALTH INSIGHTS

A dataset-driven implementation blueprint

Prepared for: EDI Project
Scope: Dataset analysis • Model development • Evaluation • Web MVP • Responsible mental-wellness presentation
Based on the three uploaded research papers and the uploaded datasets

Implementation-first guide | November 2026

Executive summary
Build the project as a web application that accepts a short voice recording, preprocesses it, predicts one of seven speech-emotion classes, and presents a transparent explanation with optional self-reflection guidance. The prediction is an estimate of vocal emotion in that clip—not a diagnosis, a depression/anxiety detector, or a reliable measure of a person’s overall mental health.
The most practical path is a staged model strategy: (1) establish a reproducible MFCC + classical ML baseline, (2) compare a CNN or CNN–BiLSTM model on log-Mel spectrograms, and (3) attempt a frozen/fine-tuned Wav2Vec 2.0 model. Treat acoustic–semantic cross-attention as an advanced extension, not the first implementation, because the available audio corpora are small and the text CSV is not aligned to the audio.
| Recommended MVP React frontend + FastAPI backend + PyTorch/Librosa inference. Start with a compact MFCC + SVM/RBF baseline and a log-Mel CNN. Add Wav2Vec 2.0 only after the evaluation pipeline is stable. Present calibrated class probabilities, confidence/uncertainty, and a non-clinical reflection card. |


1. What you uploaded: forensic dataset analysis
1.1 Audio archive A — Emo-DB (Berlin Database of Emotional Speech)
| Property | Finding from the uploaded archive | Project implication |
| Archive contents | 535 WAV files in wav/; no accompanying label CSV or README found | Labels must be parsed from the filename convention and checked before training. |
| Emotion classes | 7 classes: anger, boredom, disgust, fear, happiness, sadness, neutral | Use the seven-class setup for the Emo-DB experiment. |
| Label codes | W=anger, L=boredom, E=disgust, A=fear, F=happiness, T=sadness, N=neutral | Write a filename parser and unit-test every mapping. |
| Class counts | Anger 127; neutral 79; boredom 81; fear 71; happiness 71; sadness 62; disgust 46 | Noticeable imbalance; use class weights and macro metrics. |
| Speakers | 10 speaker IDs inferred from filenames: 03, 08, 09, 10, 11, 12, 13, 14, 15, 16 | Use speaker-independent splits; never randomly split individual clips only. |
| Audio format | Mono, 16 kHz, 16-bit PCM in inspected examples | Already matches the 16 kHz input expected by Wav2Vec 2.0. |
| Duration | 535 clips; observed range 1.23–8.98 s; mean about 2.78 s | Use padding/truncation or masked batching; preserve natural utterance duration where possible. |
| Language / nature | German acted emotional speech, based on the well-known Emo-DB naming convention | A model trained here may not generalize to everyday English speech. |


The archive has no metadata table. The label is encoded in the filename, so a wrong character index can silently corrupt every training label. Before feature extraction, generate a manifest containing filepath, speaker_id, emotion_code, emotion_name, sample_rate, duration, channels and file hash. Validate the parser by printing a few examples from every class.
1.2 Audio archive B — SAVEE
| Property | Finding from the uploaded archive | Project implication |
| Dataset identity | Info.txt explicitly identifies the speakers and recordings as SAVEE | This is a second acted English speech corpus. |
| Unique audio | 480 unique WAV recordings | The ZIP contains a second nested copy under audiodata/AudioData; do not count it twice. |
| Speakers | DC, JE, JK and KL; all four are male speakers | Only four speakers: use leave-one-speaker-out or grouped evaluation. |
| Classes | Anger, disgust, fear, happiness, neutral, sadness, surprise | Same seven broad labels as Emo-DB, but the neutral class is larger. |
| Counts | 60 per class for anger, disgust, fear, happiness, sadness and surprise; 120 neutral | Neutral is 2× each other class; track per-class recall. |
| Audio format | Mono, 44.1 kHz, 16-bit WAV | Resample to 16 kHz for a common Wav2Vec 2.0 pipeline; preserve original files. |
| Duration | 480 clips; observed range 1.63–7.14 s; mean about 3.84 s | Use a consistent crop/padding strategy and record duration. |


SAVEE's Info.txt documents the filename labels: a=anger, d=disgust, f=fear, h=happiness, n=neutral, sa=sadness and su=surprise. There are 15 sentences per emotion category per speaker. The archive duplicates the same file content in two folder trees; select only AudioData/<speaker>/*.wav (or only the nested tree), not both.
1.3 CSV — dev_sent_emo_cleaned_processed(1).csv
| Property | Observed value | Interpretation |
| Rows / columns | 1,108 rows × 11 columns | Small text/dialogue dataset; not an audio feature table. |
| Fields | Utterance, Speaker, Emotion, Sentiment, Dialogue_ID, Utterance_ID, Season, Episode, StartTime, EndTime, Sr No. | No audio path, waveform, MFCC, spectrogram or acoustic measurement is present. |
| Emotion counts | neutral 469; joy 163; anger 153; surprise 150; sadness 111; fear 40; disgust 22 | Severely imbalanced; disgust and fear are especially underrepresented. |
| Sentiment counts | neutral 469; negative 406; positive 233 | Three sentiment labels are available in addition to seven emotion labels. |
| Speaker / dialogue | 47 speaker strings; 114 dialogue IDs; 9 seasons represented | Dialogue and speaker grouping can prevent leakage in a text-only experiment. |
| Missing values | No missing values in the inspected columns | Still check whitespace, invalid labels and duplicate rows. |
| Repeated utterance text | 66 rows have an utterance string that appears elsewhere | Do not automatically delete repeated dialogue; inspect full-row duplicates and dialogue IDs. |


| Critical data boundary This CSV appears structurally consistent with a Friends-dialogue emotion/sentiment development split (often associated with MELD-style data), but the filename alone does not prove provenance. It contains text and character names, not the speech audio. Use it only for a separate text-emotion branch or for future multimodal work if you obtain legally usable, correctly aligned audio and confirm provenance/licensing. |


1.4 Dataset decision
Main audio experiment: train and evaluate on Emo-DB and SAVEE separately first.
Shared seven-class experiment: map labels to a common schema, but remember Emo-DB has boredom while SAVEE has surprise. There is no direct one-to-one equivalent. Either use a six-class common subset only after explicitly defining it, or keep separate dataset-specific heads/experiments.
Do not combine all recordings into one training pool and report a single score without a speaker- and corpus-aware test. Dataset, language, microphone, actor style and class definition can become shortcuts.
CSV experiment: optional text-only classifier. Keep its results separate from the speech model. Never claim its text accuracy is speech emotion recognition accuracy.
2. Research-paper findings and how to use them
2.1 Paper 1 — Speech Databases, Speech Features, and Classifiers in SER: A Review (2024)
The review organizes SER around databases, features and classifiers. It describes acted, elicited and natural speech databases, and emphasizes that the kind of recording changes what a model learns. It discusses local features such as pitch, intensity and duration, global/prosodic patterns such as speaking rate, pitch contour, intonation and rhythm, and linguistic cues such as word choice and sentence structure.
Features to implement first: MFCCs, log-Mel spectrograms, pitch/F0 statistics, RMS energy, zero-crossing rate, spectral centroid/rolloff and duration/speech-rate proxies.
Classical classifiers to benchmark: SVM, Random Forest and Gaussian Mixture Models; the review also discusses HMMs and neural models.
Deep learning families to compare: CNNs for local spectral patterns; RNN/LSTM models for temporal sequences; hybrid CNN–RNN models.
Limitations highlighted by the review: acted datasets may exaggerate emotion, natural speech is harder, speakers/noise/culture vary, language coverage is uneven, and real-time/context-aware/multimodal SER remains challenging.
Project use: justify a baseline feature pipeline, compare classical and deep learning methods, and include a limitations section that explains why acted-corpus performance is not proof of real-world mental-health validity.
2.2 Paper 2 — Self-Supervised Learning with Cross-Attention Fusion (2025)
This paper proposes CAFAS: an acoustic branch and a semantic/self-supervised branch are fused with cross-attention. The reported system uses Wav2Vec 2.0 representations alongside MFCC-based acoustic features. The authors describe 40 MFCC coefficients, 25 ms windows and 10 ms stride, and use a Wav2Vec 2.0 base representation. Their experiments select the ninth transformer encoder output in their setup.
| Paper element | What it means | How to adapt it |
| Wav2Vec 2.0 | Pretrained speech representation from raw waveform at 16 kHz | Use as transfer learning; freeze initially, train a small classifier, then selectively unfreeze. |
| MFCC branch | Compact spectral/acoustic representation | Extract 40 MFCCs per frame; add a temporal encoder or statistics pooling. |
| Cross-attention | One representation attends to the other to learn complementary cues | Optional advanced model after single-branch baselines are working. |
| Reported result | 74.6% weighted accuracy on IEMOCAP; 24 ms latency on the authors' moderate-device setup | A paper result on another dataset and setup—not a performance promise for your data or laptop. |
| Ablation principle | Compare each branch and fusion to identify what adds value | Run MFCC-only, Wav2Vec-only and fused models under identical splits. |


The paper's central idea is that acoustic cues (tone, pitch, frequency and intensity) and contextual/semantic representations may complement one another. Its future directions include evaluation on contemporary datasets, robustness to noise and cross-lingual generalization. Your available corpora are small, so reproduce the idea in a lightweight form rather than attempting to copy every architectural detail.
2.3 Paper 3 — Perceptual Borderline for Balancing Multi-Class Spontaneous Emotional Data (2021)
This paper addresses severe multi-class imbalance in spontaneous emotion data. It uses a perceptual emotional space based on Valence and Arousal (and discusses the VAD dimensional model), then identifies minority-class samples near a majority-class boundary. It proposes Perceptual Borderline Oversampling (PBO) and Perceptual Borderline SMOTE (PB-SMOTE), comparing them with random oversampling and standard SMOTE.
Useful lesson: imbalance is not solved just by increasing every minority class equally; boundary samples and class overlap matter.
Practical adaptation: begin with class-weighted loss and training-only RandomOverSampler/SMOTE on handcrafted feature vectors. Compare Borderline-SMOTE only when sample counts and feature geometry make it meaningful.
Important limitation: the paper's perceptual borderline methods require valence/arousal annotations or a valid projection into that perceptual space. Your uploaded audio datasets provide categorical labels, not V/A ratings. Therefore, you cannot honestly claim to implement PBO/PB-SMOTE from the paper without obtaining or creating valid V/A labels.
Never oversample before splitting. Synthetic examples must be created inside each training fold only; validation/test data must remain untouched.
3. Recommended project scope and output contract
3.1 Define the system correctly
Suggested project description: “A speech emotion recognition web application that estimates the vocal emotion expressed in a short recording and presents non-clinical emotional-awareness insights. It is designed for learning and self-reflection, not diagnosis or treatment.”
The word “mental health” should describe the supportive presentation layer and responsible use-case—not a clinical prediction label. Emo-DB and SAVEE teach acted emotion categories; they do not contain clinically validated depression, anxiety or PTSD labels.
3.2 Seven-class label strategy
| Canonical output | Emo-DB | SAVEE | Comment |
| anger | W | a | Shared class |
| disgust | E | d | Shared class |
| fear | A | f | Shared class |
| happiness / joy | F | h | Map label naming carefully; keep original labels in metadata. |
| neutral | N | n | Shared class; SAVEE has twice the count of each other class. |
| sadness | T | sa | Shared class |
| boredom | L | — | Emo-DB-only; do not pretend SAVEE contains it. |
| surprise | — | su | SAVEE-only; do not pretend Emo-DB contains it. |


| Recommended experimental design Run (A) a seven-class Emo-DB experiment, (B) a seven-class SAVEE experiment, then (C) a six-class shared-label experiment using only anger, disgust, fear, happiness, neutral and sadness. Report all three separately. For experiment C, explicitly state that the two corpora are in different languages and recording conditions. |


4. End-to-end model training plan
4.1 Step 0 — Reproducible data manifest
Extract each ZIP into separate folders. For SAVEE, keep one copy only; compare file hashes if uncertain.
Create a CSV manifest with: filepath, corpus, speaker_id, emotion, language, sample_rate, channels, duration_seconds, file_size, file_hash.
Validate: every path exists, every label is in the expected set, no empty audio, no duplicate hash within the selected corpus, no corrupted WAV, and every speaker has expected labels.
Plot class distribution, duration histogram, speaker distribution and sample-rate distribution before training.
4.2 Step 1 — Audio preprocessing
Load WAV as mono floating-point waveform. If stereo, downmix to mono.
Resample to 16,000 Hz for the common Wav2Vec 2.0 path. Keep the original audio untouched.
Remove only obvious leading/trailing silence with conservative settings. Do not aggressively denoise or normalize away emotional intensity.
Normalize waveform amplitude consistently (for example, peak or standard normalization) while preserving relative dynamics within the clip.
For batching, pad shorter clips and use attention masks where supported. Avoid arbitrary hard truncation of longer recordings.
Optional augmentation on training data only: light background noise, small gain variation, mild time shift, limited speed perturbation and room impulse/reverb. Keep changes realistic; avoid pitch shifts that alter perceived emotion.
Use a deterministic preprocessing configuration and save it with the model. The exact sample rate, crop length, normalization and feature parameters used during training must also be used at inference.
4.3 Step 2 — Feature extraction
| Feature | Captures | Suggested implementation |
| MFCC (40 coefficients) | Compact representation of spectral envelope | 25 ms frame, 10 ms hop/stride, per-utterance sequence; paper 2 uses this setting. |
| Log-Mel spectrogram | Time-frequency energy patterns | 64 or 80 Mel bands; log amplitude; feed to 2D CNN. |
| Pitch / F0 | Vocal pitch and pitch variation | Mean, standard deviation, min/max and voiced-frame ratio; handle unvoiced frames safely. |
| RMS energy | Loudness/energy variation | Mean, standard deviation, percentiles and frame sequence. |
| Spectral centroid / rolloff | Brightness and frequency distribution | Frame-level statistics and/or sequence. |
| Zero-crossing rate | Signal sign-change rate; coarse texture cue | Mean and standard deviation. |
| Duration / speaking rate proxy | Temporal characteristics | Duration, voiced duration and syllable-rate proxy only if reliably estimated. |


Start with MFCC + pitch + energy + spectral statistics for classical ML. For CNNs, use log-Mel spectrograms. For Wav2Vec 2.0, feed the resampled raw waveform; do not feed MFCCs into the Wav2Vec model.
4.4 Step 3 — Train the models in this order
| Model | Input | Why train it | Priority |
| SVM (RBF) | Pooled MFCC/prosodic statistics | Strong, fast baseline for small tabular feature sets | Must-have |
| Random Forest | Same handcrafted feature vector | Robust nonlinear baseline; feature importance can help explain patterns | Must-have |
| CNN | Log-Mel spectrogram image/sequence | Learns local time-frequency patterns | Recommended |
| CNN–BiLSTM | Log-Mel or MFCC sequence | CNN extracts local patterns; BiLSTM models temporal evolution | Recommended if data allows |
| Wav2Vec 2.0 + MLP | 16 kHz raw waveform → pretrained embeddings | Transfer learning may help with limited labeled audio | Advanced |
| MFCC encoder + Wav2Vec 2.0 cross-attention | Two aligned feature sequences | Paper-inspired fusion; compare with each single branch | Stretch goal |


4.5 Baseline model details
SVM: StandardScaler → feature vector → SVC(kernel='rbf', class_weight='balanced', probability=True). Tune C and gamma only on validation folds. For calibrated probabilities, use a calibration procedure fitted on training/validation data, not the test set.
Random Forest: start around 300–500 trees, class_weight='balanced_subsample', tune max_depth/min_samples_leaf. Use permutation importance on held-out data; impurity importance alone can be biased.
CNN: input log-Mel tensor (e.g. 1 × 80 × time); use 2–3 convolution blocks, batch normalization, ReLU, dropout, global pooling and softmax output. Use early stopping.
CNN–BiLSTM: CNN encodes each frame/patch; BiLSTM models the sequence; attention or mean pooling creates an utterance vector; dense layer predicts emotion.
4.6 Wav2Vec 2.0 transfer-learning recipe
Use a pretrained Wav2Vec 2.0 base checkpoint compatible with 16 kHz audio.
First freeze the feature encoder/transformer backbone and train only a small classification head (mean/attention pooling + dropout + linear layer).
If validation results justify it, unfreeze only the last few transformer layers and fine-tune with a small learning rate; keep the head learning rate higher.
Use weighted cross-entropy or focal loss only if class-wise results show imbalance is hurting minority classes. Keep batch sizes small and use gradient accumulation if GPU memory is limited.
Use early stopping on validation macro-F1 or unweighted accuracy, not training accuracy.
Save the exact checkpoint, processor, label map, sample-rate and preprocessing config.
Because Emo-DB and SAVEE are small, fully fine-tuning a large speech encoder from the beginning is likely to overfit and waste compute. Freeze-first transfer learning is the safer experiment.
4.7 Cross-attention fusion — advanced extension
Paper-inspired architecture:
| Audio waveform (16 kHz) ───────────────► Wav2Vec 2.0 ─► semantic sequence           │           └─► MFCC (40, 25 ms / 10 ms) ─► acoustic encoder                                              │                      acoustic queries + semantic keys/values                                              ▼                                     Cross-attention block                                              ▼                                pooling + dense classifier                                              ▼                                 7-class emotion scores |


Keep the two branches temporally compatible. Project both to the same embedding dimension, apply layer normalization and dropout, then cross-attention and pooling. Do an ablation study: MFCC-only, Wav2Vec-only, concatenation/late fusion, and cross-attention. If the fusion model does not improve macro-F1 or cross-speaker performance consistently, keep the simpler model.
5. Splitting, imbalance, evaluation and error analysis
5.1 Split without speaker leakage
Never randomly split utterance rows if the same speaker appears in train and test.
Emo-DB: use GroupShuffleSplit/StratifiedGroupKFold with speaker ID as group. With only 10 speakers, report the fold strategy and class coverage.
SAVEE: use leave-one-speaker-out (four folds) if feasible. With only four speakers, scores will have high variance; report fold-wise scores.
For the six-class combined experiment, test cross-corpus generalization: train on one corpus and test on the other, then reverse. Do not tune on the test corpus.
Use a validation set or inner cross-validation for hyperparameter tuning. Keep the final test set untouched until model selection is complete.
5.2 Handle imbalance correctly
Start with class-weighted loss / class_weight='balanced'.
Compare class weights against RandomOverSampler or SMOTE on handcrafted feature vectors. Resampling must occur inside each training fold.
Do not apply SMOTE directly to raw waveforms or spectrogram images as if they were ordinary tabular vectors.
Use augmentation only on training audio, and avoid generating unrealistic emotional cues.
PBO/PB-SMOTE is a research extension only if you obtain valid valence/arousal annotations; the current datasets do not contain those annotations.
5.3 Metrics to report
| Metric | Why it matters | What to show |
| Macro-F1 | Treats each emotion class equally | Primary model-selection metric |
| Unweighted Accuracy (UA) / balanced accuracy | Average class recall; less dominated by majority classes | Primary SER comparison |
| Weighted Accuracy (WA) | Weights classes by their support | Report for comparison with papers, but do not use alone |
| Per-class precision / recall / F1 | Shows which emotions are missed | Table for all seven classes |
| Confusion matrix | Shows common confusions | Normalized matrix + raw counts |
| Calibration / confidence | Checks whether 0.8 confidence means roughly 80% reliability | Reliability plot or ECE if probabilities are used |
| Inference latency and model size | Important for a website MVP | CPU/GPU, device, clip length and batch size |


Accuracy alone can hide failure on rare emotions. Compare models using the same speaker-aware splits, then select the simplest model whose macro-F1, per-class recall and stability are acceptable. Include confidence intervals or fold-to-fold variation because these corpora are small.
5.4 Error analysis
Inspect confusion pairs such as sadness vs neutral, anger vs disgust, fear vs surprise.
Listen to misclassified samples only when dataset terms and privacy allow; inspect duration, clipping, silence, noise and speaker identity.
Check whether a model is learning actor identity, language, recording quality or sentence-specific artifacts instead of emotion.
Compare per-speaker performance and per-corpus performance.
Run ablations: MFCC vs prosody vs log-Mel vs Wav2Vec; class weights vs oversampling; no augmentation vs mild augmentation.
6. Website MVP: user journey and interface
6.1 Suggested product experience
Landing page: project name, one-sentence explanation, visible privacy note and a clear “Analyse a voice clip” action.
Consent and recording page: explain that the app estimates vocal emotion, not mental health condition; ask permission before microphone access.
Input panel: record 5–15 seconds or upload WAV/MP3 (convert supported formats safely); show waveform, duration, playback and re-record/delete controls.
Analysis state: show progress steps such as validating audio, extracting features and estimating vocal emotion. Do not show fake precision or imply clinical analysis.
Results page: top predicted vocal emotion, probability distribution, uncertainty indicator, optional valence/arousal only if trained with valid labels, and a short explanation.
Reflection panel: offer neutral self-reflection prompts and optional user-selected mood check-in. Do not infer a diagnosis or risk score from voice.
History: off by default. If history is offered, make saving opt-in, allow deletion and explain retention.
6.2 Example result layout
| UI section | Example content |
| Primary result | “Most likely vocal expression: Sadness” — model estimate for this recording. |
| Confidence | “Model confidence: 0.62 (moderate)” plus a note that confidence is not certainty. |
| Distribution | Sadness 62%, Neutral 18%, Fear 9%, Anger 5%, Joy 3%, Disgust 2%, Surprise 1% (illustrative values only). |
| Signal observations | “The model found patterns in pitch/energy/spectral features associated with its sadness class.” Avoid claiming a specific psychological cause. |
| Reflection | “Does this match how you feel right now?” “Would you like to take a short break or talk to someone you trust?” |
| Safety note | “This tool cannot detect or diagnose depression, anxiety or any mental-health condition. If you are concerned, contact a qualified professional.” |


| Important UI wording Say “predicted vocal emotion” or “emotion expressed in this clip,” not “your mental state.” A single short clip is not a reliable measure of a person's stable mood, mental health or personality. |


6.3 Visual design
Use a calm, accessible visual system: warm off-white background, charcoal text, muted teal/green accent and one restrained highlight color.
Avoid alarming red labels for sadness/fear/anger. Use consistent neutral icons and accessible text labels; do not communicate emotion through color alone.
Keep the result card simple: emotion name, probability bar chart, confidence explanation, audio playback and reflection prompt.
Provide a “How this works” accordion: audio preprocessing, model family, training datasets and known limitations.
Add a persistent privacy notice near upload/record controls.
7. Technical architecture and implementation stack
| Browser (React + Vite)    │  upload / microphone recording / result visualization    ▼ FastAPI backend    ├── POST /api/analyze    ├── POST /api/feedback       (optional, consent-based)    ├── GET  /api/health    └── GET  /api/model-info           │           ├── audio validation + format conversion           ├── preprocessing (16 kHz mono)           ├── feature extraction / model processor           ├── trained model inference           └── calibrated scores + safe response formatter                  │                  ▼            JSON response → React result page Training workspace (separate from production)    data manifests → preprocessing → speaker-aware splits    → baselines / deep models → evaluation → saved model artifact |


| Layer | Recommended tools | Responsibility |
| Frontend | React + Vite + TypeScript; Tailwind CSS; Recharts; MediaRecorder API | Recording/upload, waveform, results, responsive UI. |
| Backend | Python FastAPI + Pydantic | Validation, inference endpoints, response schema. |
| Audio / ML | Librosa, SoundFile, NumPy, scikit-learn, PyTorch, torchaudio, Transformers | Preprocessing, feature extraction, training and inference. |
| Storage | Temporary file storage; SQLite only if opt-in history is implemented | Avoid storing voice recordings by default. |
| Deployment | Frontend: Vercel/Netlify; API: Render/Railway or a suitable GPU host | Check memory, model download and cold-start constraints. |
| Experiment tracking | CSV/JSON logs; optionally MLflow or Weights & Biases | Record config, split IDs, seed, metrics and artifact version. |


7.1 Suggested API contract
| POST /api/analyze Request: multipart/form-data   audio: uploaded audio file   consent: true Response (illustrative schema): {   "prediction_type": "vocal_emotion",   "predicted_emotion": "sadness",   "confidence": 0.62,   "scores": {     "anger": 0.05, "disgust": 0.02, "fear": 0.09,     "happiness": 0.03, "neutral": 0.18,     "sadness": 0.62, "surprise": 0.01   },   "uncertainty": "moderate",   "model_version": "ser-baseline-v1",   "audio_duration_seconds": 7.4,   "notice": "Not a mental-health diagnosis." } |


Use a strict response schema. The values above are examples only, not actual model outputs. Return a clear error for unsupported formats, empty audio, too-short clips, oversized files and clips that fail validation.
7.2 Backend processing outline
| 1. Validate extension, MIME type, file size and decoded audio. 2. Decode safely; convert to mono; resample to 16 kHz. 3. Check duration and silence/clipping thresholds. 4. Apply the exact preprocessing used during training. 5. Run model inference in eval/no-gradient mode. 6. Convert logits to probabilities; apply calibration if validated. 7. Apply an uncertainty rule (for example, low max probability    or small top-two margin) and say “uncertain” when appropriate. 8. Return prediction + scores + model version + safety notice. 9. Delete temporary audio after inference unless the user explicitly    opted into storage. |


8. Suggested repository structure
| emotion-speech-insights/ ├── frontend/ │   ├── src/components/       # Recorder, UploadCard, ResultCard, Charts │   ├── src/pages/            # Home, Analyse, About, Privacy │   └── src/services/api.ts ├── backend/ │   ├── app/main.py           # FastAPI routes │   ├── app/schemas.py        # Request/response models │   ├── app/audio.py          # Decode, mono, resample, validate │   ├── app/inference.py      # Load model and predict │   └── app/safety.py         # Uncertainty and safe wording ├── ml/ │   ├── configs/ │   ├── data_manifest/ │   ├── notebooks/            # EDA only; move final code to scripts │   ├── src/                  # features, datasets, models, train, evaluate │   ├── reports/              # plots, metrics, confusion matrices │   └── artifacts/            # model, label map, config ├── tests/ ├── README.md └── requirements.txt |


9. Build-from-scratch implementation sequence
| Phase | Work to complete | Deliverable / exit condition |
| 1. Dataset audit | Extract archives, remove duplicate SAVEE tree, parse labels, manifest, EDA | Verified manifest; class/speaker/duration plots; no corrupted paths. |
| 2. Baseline | MFCC + prosodic features; speaker-grouped folds; SVM and Random Forest | Reproducible baseline metrics and confusion matrix. |
| 3. Deep model | Log-Mel CNN; optionally CNN–BiLSTM; training-only augmentation | Comparison against baseline using identical splits. |
| 4. Transfer learning | Frozen Wav2Vec 2.0 head; selective fine-tuning if justified | Model size, latency and fold-wise macro-F1 documented. |
| 5. Model selection | Compare class-wise recall, macro-F1, UA, calibration and latency | One selected model with limitations and model card. |
| 6. Backend | FastAPI validation, preprocessing and inference endpoint | API tested with valid and invalid audio. |
| 7. Frontend | Recorder/upload, waveform, results, chart, privacy and explanation pages | End-to-end website flow works on desktop/mobile. |
| 8. Responsible output | Uncertainty handling, no diagnosis language, consent and deletion | Safety wording and data lifecycle verified. |
| 9. Deployment | Deploy frontend/API; configure model artifact; smoke-test | Public demo URL with documented limitations. |


10. Practical training pseudocode
| # Baseline training concept (not a complete drop-in script) manifest = load_manifest("audio_manifest.csv") X, y, groups = extract_mfcc_prosody_features(manifest) for train_idx, test_idx in speaker_grouped_splits(groups, y):     X_train, X_test = X[train_idx], X[test_idx]     y_train, y_test = y[train_idx], y[test_idx]     # Fit every transform only on training data     pipeline = Pipeline([         ("scale", StandardScaler()),         ("model", SVC(kernel="rbf", class_weight="balanced"))     ])     pipeline.fit(X_train, y_train)     pred = pipeline.predict(X_test)     report_fold_metrics(y_test, pred)  # macro-F1, UA, per-class recall     save_fold_confusion_matrix(y_test, pred) # Tune hyperparameters on training/validation folds only. # Do not inspect the final test set during model selection. |


The pseudocode illustrates the correct evaluation boundary. In a real implementation, use StratifiedGroupKFold where feasible, inspect class coverage in each fold, and keep all feature scaling/resampling inside the training pipeline.
11. Testing checklist
☐ All WAV files parse into the expected labels; SAVEE duplicate directory is excluded.
☐ Train/test speakers are disjoint in every speaker-independent evaluation.
☐ Audio inference uses the same sample rate and preprocessing as training.
☐ Model performance includes macro-F1, UA/balanced accuracy, per-class recall and confusion matrix.
☐ Model is compared against majority-class and simple SVM baselines.
☐ Upload validation rejects empty, corrupt, too-large and unsupported files.
☐ Recording permission is requested explicitly; microphone access can be stopped.
☐ Audio is deleted after inference by default; no hidden storage or sharing.
☐ Low-confidence outputs are labelled uncertain rather than over-interpreted.
☐ Website never states or implies depression/anxiety diagnosis.
☐ Mobile and desktop layouts, keyboard navigation and readable chart labels are checked.
☐ Model version, dataset provenance, limitations and evaluation protocol are documented.
12. Risks, limitations and responsible mental-health framing
| Risk | Why it matters | Mitigation |
| Acted speech bias | Actors deliberately perform emotions; real spontaneous speech differs | Describe this as an academic prototype; test on consented natural speech before real-world claims. |
| Language/domain shift | Emo-DB is German; SAVEE is English; speakers and recording setups differ | Report separate and cross-corpus experiments; do not imply universal accuracy. |
| Small data | 535 + 480 clips are limited for deep models | Use transfer learning, grouped CV, regularization, early stopping and simple baselines. |
| Label imbalance | Emo-DB and CSV have uneven class counts; SAVEE neutral is larger | Class weights, training-only sampling and macro metrics. |
| Emotion ≠ mental illness | A vocal emotion label is not a diagnosis or a stable mood measure | No depression/anxiety prediction unless separately validated with ethical clinical data and qualified oversight. |
| Privacy | Voice recordings can identify people | Consent, transient processing, encryption in transit, access controls and deletion. |
| Bias / fairness | Accent, gender, age, culture, disability and speaking style affect voice | Document coverage gaps; do not make consequential decisions; test subgroup performance only with appropriate consent and data. |


Do not add an automated “depression score”, “anxiety level”, “risk percentage” or treatment recommendation based on the current datasets. The papers discuss possible healthcare applications, but that does not validate this particular prototype for clinical screening. If the user expresses distress, the interface should encourage them to contact a trusted person or qualified professional rather than interpreting the model score as evidence.
13. What to show in the final EDI demonstration
Problem statement and motivation: vocal emotion cues can support emotion-aware interfaces, but recognition is difficult and context-dependent.
Dataset audit: show Emo-DB 535, SAVEE 480 unique, CSV 1,108 text-only; show class counts and speaker counts.
Research basis: MFCC/classical models from the review; Wav2Vec 2.0 + cross-attention from the 2025 paper; imbalance strategies from the 2021 paper.
Methodology: audio preprocessing → features/model → prediction → uncertainty-aware results.
Model comparison: SVM, Random Forest, CNN and Wav2Vec 2.0 using identical speaker-aware evaluation.
Results: macro-F1, UA, per-class scores, confusion matrix, latency and limitations.
Live MVP: upload or record a clip, see predicted vocal emotion and explanation, listen to the clip, and view privacy/safety notice.
Future work: natural speech dataset with consent, language diversity, valid valence/arousal labels, cross-attention fusion, robust noise evaluation and professional clinical collaboration.
14. Recommended final scope: what to build now vs later
| Build for the EDI MVP | Keep as future enhancement |
| Upload/record a short audio clip | Continuous real-time emotion monitoring |
| Seven-class speech emotion model, with corpus-aware evaluation | Clinical depression/anxiety screening |
| MFCC + SVM/RF baseline and log-Mel CNN comparison | Full CAFAS-style cross-attention fusion if compute/data permit |
| Probability distribution + uncertainty + reflection prompts | Personalized longitudinal mental-health trends |
| Temporary audio processing and clear consent | Cloud history / user accounts (only with strong privacy design) |
| Separate optional text-only experiment using the CSV | Audio-text multimodal fusion after obtaining aligned, licensed audio |


| Best practical recommendation Do not begin by trying to build the most complex neural architecture. First make the data pipeline correct, prove that an SVM baseline works under speaker-independent evaluation, and build the complete website around that stable inference pipeline. Then replace the model behind the same API with CNN or Wav2Vec 2.0 when the evidence supports it. |


15. Research papers and source notes
The following conclusions in this blueprint are derived from the three user-provided papers and the uploaded dataset contents. Numerical results quoted from papers are their reported results on their own datasets—not results achieved by this project.
G. H. Mohmad Dar and R. Delhibabu, “Speech Databases, Speech Features, and Classifiers in Speech Emotion Recognition: A Review,” IEEE Access, 2024, DOI: 10.1109/ACCESS.2024.3476960.
B. M. Deeb, A. V. Savchenko and I. Makarov, “Enhancing Emotion Recognition in Speech Based on Self-Supervised Learning: Cross-Attention Fusion of Acoustic and Semantic Features,” IEEE Access, 2025, DOI: 10.1109/ACCESS.2025.3554454.
L. Ben Letaifa and M. Inés Torres, “Perceptual Borderline for Balancing Multi-Class Spontaneous Emotional Data,” IEEE Access, 2021, DOI: 10.1109/ACCESS.2021.3071485.
Dataset notes: Emo-DB identity and label mapping are inferred from the well-known filename convention and inspected file pattern; SAVEE identity and label mapping are explicitly stated in the archive's Info.txt. The CSV is described only from its observed columns, labels and rows; confirm its original source and license before redistribution or publication.
Final takeaway
A credible EDI project is not just a model with a high accuracy number. It is a complete, reproducible system: correctly labelled audio, speaker-independent testing, fair class-wise evaluation, an understandable website, uncertainty-aware outputs, and honest limits on what the model can say about mental health. With your current files, this is achievable as a strong academic prototype—provided the text CSV is kept separate from the speech pipeline and the mental-health claims remain non-clinical.

16. Scope addendum (November 2026)
This addendum adds to the original text above; it does not change it.
- Voice: the ban on inferring depression, anxiety or risk scores from voice still stands in full. The speech module predicts vocal emotion in a clip and nothing more.
- Survey module: a separate DASS-21 self-report questionnaire module has been added. It scores the 21 items into Depression, Anxiety and Stress subscales and standard severity bands.
- Its output is a screening result computed from the user's own answers. It is not a diagnosis and is no substitute for a qualified professional.
- Separation: the two modules' results are never combined. Voice predictions do not feed survey results, and survey results do not feed voice predictions.
- Data: the survey dataset used for development (survey_dataset_1200.xlsx) is fully synthetic. No metric derived from it is real-world accuracy.
