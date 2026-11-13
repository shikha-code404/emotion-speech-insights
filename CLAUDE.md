# Project Rules & Guidelines

These foundational rules govern all work across ML development, backend services, frontend application, and testing.

---

### Rule 1: Speaker-Disjoint Evaluation
- **Never split train/test randomly by clip; always split by speaker.**
- Rationale: Speech emotion models easily overfit to individual speaker acoustic characteristics (timbre, pitch base, recording room acoustics). Random clip splits cause severe data leakage, resulting in overly optimistic evaluation metrics that collapse on unseen speakers.
- GroupKFold / StratifiedGroupKFold or explicit speaker hold-out splits must be strictly enforced.

### Rule 2: Strict Data Hygiene & Leakage Prevention
- **Fit scalers, resampling, and calibration on training folds only.**
- Rationale: Applying standardizers, normalizers, class-resamplers (e.g. SMOTE), or probability calibrators on the combined dataset leaks validation statistics into the model.
- Transform validation and test folds using only parameters fitted on the training split.

### Rule 3: Single Source of Truth for Features
- **Preprocessing and feature code lives in `ml/src` and is imported by the backend, never duplicated.**
- Rationale: Discrepancies between training-time feature extraction and runtime inference feature extraction (e.g. sample rates, FFT window lengths, mel bands, delta calculations) destroy model performance in production.
- All audio loading, normalization, windowing, and feature extraction functions must reside solely in `ml/src` and be packaged/imported directly into `backend/app`.

### Rule 4: Ethical & Responsible Prediction Scope
The project has two modules with different wording rules.

**Speech module** (`ml/src` audio code, speech endpoints, speech UI)
- **It predicts "vocal emotion in this clip". No diagnosis, depression, anxiety, stress or risk-score wording anywhere in speech outputs.**
- Scope: classifies acoustic emotional characteristics present in a specific audio recording (e.g., neutral, happy, sad, angry, fearful, disgust, surprised).
- Disallowed vocabulary: speech code, API contracts, documentation and UI copy must not use terms implying clinical assessment, diagnostic status, mental-health pathology (e.g., "depression", "anxiety", "stress", "disorder"), or human "risk scores". Speech predictions must never be combined with or presented as survey results.

**Survey module** (`ml/src/survey`, survey endpoints, survey UI)
- **May report DASS-21 Depression, Anxiety and Stress screening scores and severity bands, always described as a self-report screening result, never a diagnosis.**
- Every survey output must state that it is a self-report screening result, is not a diagnosis, and is no substitute for a qualified professional.
- Never use "diagnosis", "diagnosed", "disorder", "patient" or "risk score" wording for survey results.
- The current survey dataset is fully synthetic; every survey report, model card and metric must say so and must not present metrics as real-world accuracy.

### Rule 5: Ephemeral Audio Handling & Privacy
- **Audio is deleted after inference by default.**
- Rationale: Voice biometric data is sensitive personal information.
- Raw audio buffers or temporary disk files received for inference must be purged immediately after feature extraction and inference conclude. No retention of raw voice recordings unless explicit opted-in logging is designed and approved.

