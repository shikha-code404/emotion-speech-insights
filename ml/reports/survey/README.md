# Survey module: lifestyle model report

> **SYNTHETIC DATA: all 1,200 survey respondents are artificially generated; none are real people. Results reflect the data generator's rules, not real-world patterns, and are not real-world accuracy.**
>
> **ANALYSIS-ONLY: this lifestyle model is never served to users and has no API endpoint. It must not be used to make statements about any person.**

Task: predict `Emotional_State` (Normal / Depression / Anxiety / Stress) from
lifestyle answers only. DASS-21 items, scores, severities and labels are
excluded (they would leak the label). Users' screening results come only
from their own DASS-21 answers; this model is an exploratory analysis of the
synthetic generator.

Protocol: 1,200 synthetic respondents (one row each). Outer stratified
5-fold CV; inner stratified 3-fold grid search on macro-F1; all
preprocessing inside the sklearn Pipeline. Mean ± sd over outer folds;
per-class recall from pooled out-of-fold predictions.

| Model | Macro-F1 | Balanced acc. | Recall Normal | Recall Depression | Recall Anxiety | Recall Stress |
|---|---|---|---|---|---|---|
| majority_baseline | 0.100 ± 0.000 | 0.250 ± 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| logistic_regression | 0.581 ± 0.025 | 0.583 ± 0.024 | 0.687 | 0.543 | 0.493 | 0.610 |
| random_forest | 0.572 ± 0.038 | 0.575 ± 0.037 | 0.693 | 0.530 | 0.487 | 0.590 |
| gradient_boosting | 0.581 ± 0.035 | 0.583 ± 0.034 | 0.710 | 0.537 | 0.500 | 0.587 |

Best family by mean outer macro-F1: **logistic_regression** (exported, analysis-only).

Note: logistic_regression and gradient_boosting differ by 0.0001 macro-F1, well inside the fold-to-fold spread (sd up to 0.035); treat them as equivalent. Ties go to the first model in the comparison table, which lists simpler models first.

Top permutation importances for logistic_regression (drop in macro-F1 on held-out folds):

- `mindless_snacking`: 0.105 ± 0.012
- `daily_steps`: 0.073 ± 0.011
- `systolic_bp`: 0.049 ± 0.015
- `diastolic_bp`: 0.042 ± 0.014
- `age_group`: 0.035 ± 0.017

Files: `results.csv`, `fold_metrics.csv`, `confusion_matrix_*.csv/png`,
`permutation_importance.csv`, `permutation_importance_*.png`,
`best_params.json`.

Reproduce: `python -m ml.src.survey.train_lifestyle`
