"""Analysis-only model: predict Emotional_State from lifestyle answers.

SYNTHETIC DATA. The lifestyle -> Emotional_State relationships were
simulated by the dataset generator, so these metrics measure how well a
model recovers the generator's rules. They are NOT real-world accuracy.

This model is ANALYSIS-ONLY. It appears in ml/reports/survey and the model
card, is never served to users, and has no API endpoint. Users' DASS-21
screening results come only from their own answers via scoring.py.

Leakage guard: Emotional_State is a deterministic function of q1..q21, so
the items and every score/severity/label column are excluded from X and
checked by ``assert_no_leakage``.

Evaluation: one row per synthetic respondent (Response_ID is unique and
checked in load.py), so stratified folds over rows are respondent-disjoint
(Rule 1). All encoders/scalers live inside the sklearn Pipeline and are fit
on training folds only (Rule 2). Outer stratified 5-fold CV; hyperparameters
tuned by an inner stratified 3-fold grid search on each outer training fold.

Run:  python -m ml.src.survey.train_lifestyle
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler  # noqa: E402

from ml.src.survey.load import (  # noqa: E402
    CATEGORICAL_OPTIONS,
    LABEL_COLUMNS,
    Q_COLUMNS,
    REPO_ROOT,
    SCORE_COLUMNS,
    SEVERITY_COLUMNS,
    SYNTHETIC_NOTICE,
    load_survey,
)

SEED = 42
TARGET = "emotional_state"
CLASSES = ("Normal", "Depression", "Anxiety", "Stress")

NUMERIC_FEATURES = ("bmi", "systolic_bp", "diastolic_bp", "daily_steps", "water_intake_l")
ORDINAL_FEATURES = ("age_group", "sleep_duration")
NOMINAL_FEATURES = (
    "gender", "smoking_status", "alcohol_status", "physical_activity_time",
    "mindless_snacking", "physical_discomfort", "occupation",
)
LIFESTYLE_FEATURES = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES

FORBIDDEN_COLUMNS = frozenset(Q_COLUMNS) | set(SCORE_COLUMNS) | set(SEVERITY_COLUMNS) | set(LABEL_COLUMNS)
_FORBIDDEN_PATTERN = re.compile(r"^q\d+$|score|severity|emotional_state", re.IGNORECASE)

REPORT_DIR = REPO_ROOT / "ml" / "reports" / "survey"
ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts" / "survey-v1"

ANALYSIS_ONLY_NOTICE = (
    "ANALYSIS-ONLY: this lifestyle model is never served to users and has no "
    "API endpoint. It must not be used to make statements about any person."
)

# Reference sequential ramp (blue 100 -> 700) and chart ink.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES_1 = "#2a78d6"
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def assert_no_leakage(columns) -> None:
    """Raise if any DASS item, score, severity or label column is a feature."""
    leaked = sorted(c for c in columns if c in FORBIDDEN_COLUMNS or _FORBIDDEN_PATTERN.search(c))
    if leaked:
        raise ValueError(f"label leakage: {leaked} must not be model inputs")


def make_X(df: pd.DataFrame) -> pd.DataFrame:
    """Lifestyle feature frame. BMI = weight_kg / (height_cm / 100)^2.

    Deterministic per-row transform with no fitted parameters, so computing
    it outside the Pipeline cannot leak fold statistics.
    """
    X = pd.DataFrame(index=df.index)
    X["bmi"] = df["weight_kg"] / (df["height_cm"] / 100.0) ** 2
    for col in LIFESTYLE_FEATURES:
        if col != "bmi":
            X[col] = df[col]
    X = X[list(LIFESTYLE_FEATURES)]
    assert_no_leakage(X.columns)
    return X


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), list(NUMERIC_FEATURES)),
        ("ord", Pipeline([
            ("enc", OrdinalEncoder(categories=[list(CATEGORICAL_OPTIONS[c]) for c in ORDINAL_FEATURES])),
            ("scale", StandardScaler()),
        ]), list(ORDINAL_FEATURES)),
        # Categories come from the form schema, not the data, so every fold
        # sees the same columns; "Prefer not to say" is its own category.
        ("nom", OneHotEncoder(
            categories=[list(CATEGORICAL_OPTIONS[c]) for c in NOMINAL_FEATURES],
            handle_unknown="ignore", sparse_output=False,
        ), list(NOMINAL_FEATURES)),
    ], verbose_feature_names_out=True)


def model_specs() -> dict[str, tuple[Pipeline, dict]]:
    def pipe(clf):
        return Pipeline([("prep", _preprocessor()), ("clf", clf)])

    return {
        "majority_baseline": (pipe(DummyClassifier(strategy="most_frequent")), {}),
        "logistic_regression": (
            pipe(LogisticRegression(max_iter=5000)),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            pipe(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
            {"clf__max_depth": [None, 8, 16], "clf__min_samples_leaf": [1, 5],
             "clf__max_features": ["sqrt", 0.5]},
        ),
        "gradient_boosting": (
            pipe(HistGradientBoostingClassifier(random_state=SEED, early_stopping=False)),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, None],
             "clf__max_iter": [100, 300]},
        ),
    }


def _tuned(pipeline: Pipeline, grid: dict, n_splits: int) -> GridSearchCV:
    return GridSearchCV(
        pipeline, grid or {"clf__strategy": ["most_frequent"]},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED),
        n_jobs=-1, refit=True,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def nested_cv(X: pd.DataFrame, y: pd.Series, outer_splits=5, inner_splits=3, n_repeats=10):
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=SEED)
    labels = list(CLASSES)
    results = {}
    for name, (pipeline, grid) in model_specs().items():
        t0 = time.time()
        oof = pd.Series(index=y.index, dtype=object)
        folds, params, importances = [], [], []
        for k, (tr, te) in enumerate(outer.split(X, y)):
            search = _tuned(pipeline, grid, inner_splits).fit(X.iloc[tr], y.iloc[tr])
            pred = search.predict(X.iloc[te])
            oof.iloc[te] = pred
            folds.append({
                "fold": k,
                "macro_f1": f1_score(y.iloc[te], pred, average="macro", labels=labels, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y.iloc[te], pred),
            })
            params.append(search.best_params_)
            if name != "majority_baseline":
                pi = permutation_importance(
                    search.best_estimator_, X.iloc[te], y.iloc[te], scoring="f1_macro",
                    n_repeats=n_repeats, random_state=SEED, n_jobs=-1,
                )
                importances.append(pi.importances_mean)
        fold_df = pd.DataFrame(folds)
        results[name] = {
            "folds": fold_df,
            "best_params": params,
            "oof": oof,
            "per_class_recall": dict(zip(labels, recall_score(y, oof, labels=labels, average=None, zero_division=0))),
            "confusion": confusion_matrix(y, oof, labels=labels),
            "importance": (np.vstack(importances) if importances else None),
            "seconds": time.time() - t0,
        }
        print(f"  {name:20s} macro-F1 {fold_df.macro_f1.mean():.3f} +/- {fold_df.macro_f1.std():.3f}  "
              f"bal-acc {fold_df.balanced_accuracy.mean():.3f}  ({results[name]['seconds']:.0f}s)")
    return results


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _write_csv(df: pd.DataFrame, path: Path, index=False) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"# {SYNTHETIC_NOTICE}\n")
        df.to_csv(f, index=index, float_format="%.4f")


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_facecolor(SURFACE)


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    labels = list(CLASSES)
    row_pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.6, 4.8), facecolor=SURFACE)
    cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
    im = ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{row_pct[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if row_pct[i, j] > 0.45 else INK)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted", color=INK_2)
    ax.set_ylabel("True", color=INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Share of true class (recall)", color=INK_2, fontsize=8)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_2, labelsize=8)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.text(0.01, 0.01, "SYNTHETIC DATA - not real-world accuracy. Pooled out-of-fold predictions, 5-fold CV.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_importance(imp: pd.DataFrame, title: str, path: Path) -> None:
    imp = imp.sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.4, 4.8), facecolor=SURFACE)
    ax.barh(imp.index, imp["mean"], xerr=imp["std"], color=SERIES_1, height=0.6,
            error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 2})
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    _style(ax)
    ax.set_xlabel("Drop in macro-F1 when shuffled (mean ± sd over outer folds)", color=INK_2, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.text(0.01, 0.01, "SYNTHETIC DATA - reflects the generator's simulated relationships, not real-world effects.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def write_reports(results: dict, report_dir: Path) -> pd.DataFrame:
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, r in results.items():
        f = r["folds"]
        rows.append({
            "model": name,
            "macro_f1_mean": f.macro_f1.mean(), "macro_f1_std": f.macro_f1.std(),
            "balanced_accuracy_mean": f.balanced_accuracy.mean(),
            "balanced_accuracy_std": f.balanced_accuracy.std(),
            **{f"recall_{c.lower()}": r["per_class_recall"][c] for c in CLASSES},
        })
    table = pd.DataFrame(rows)
    _write_csv(table, report_dir / "results.csv")
    _write_csv(pd.concat([r["folds"].assign(model=n) for n, r in results.items()]), report_dir / "fold_metrics.csv")

    imp_frames = []
    for name, r in results.items():
        cm = pd.DataFrame(r["confusion"], index=[f"true_{c}" for c in CLASSES],
                          columns=[f"pred_{c}" for c in CLASSES])
        _write_csv(cm, report_dir / f"confusion_matrix_{name}.csv", index=True)
        plot_confusion(r["confusion"], f"{name.replace('_', ' ').title()} - confusion matrix",
                       report_dir / f"confusion_matrix_{name}.png")
        if r["importance"] is not None:
            imp = pd.DataFrame({"mean": r["importance"].mean(axis=0), "std": r["importance"].std(axis=0)},
                               index=list(LIFESTYLE_FEATURES))
            imp_frames.append(imp.assign(model=name).rename_axis("feature").reset_index())
            plot_importance(imp, f"{name.replace('_', ' ').title()} - permutation importance",
                            report_dir / f"permutation_importance_{name}.png")
    _write_csv(pd.concat(imp_frames).sort_values(["model", "mean"], ascending=[True, False]),
               report_dir / "permutation_importance.csv")

    (report_dir / "best_params.json").write_text(json.dumps({
        "synthetic_data": True, "notice": SYNTHETIC_NOTICE,
        "best_params_per_outer_fold": {n: r["best_params"] for n, r in results.items()},
    }, indent=2, default=str), encoding="utf-8")
    return table


def _markdown_table(table: pd.DataFrame, pm: str = "±") -> str:
    head = "| Model | Macro-F1 | Balanced acc. | Recall Normal | Recall Depression | Recall Anxiety | Recall Stress |"
    lines = [head, "|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        lines.append(
            f"| {r.model} | {r.macro_f1_mean:.3f} {pm} {r.macro_f1_std:.3f} | "
            f"{r.balanced_accuracy_mean:.3f} {pm} {r.balanced_accuracy_std:.3f} | "
            f"{r.recall_normal:.3f} | {r.recall_depression:.3f} | {r.recall_anxiety:.3f} | {r.recall_stress:.3f} |"
        )
    return "\n".join(lines)


def _closeness_note(table: pd.DataFrame) -> str:
    ranked = table[table.model != "majority_baseline"].sort_values("macro_f1_mean", ascending=False, kind="stable")
    a, b = ranked.iloc[0], ranked.iloc[1]
    gap = a.macro_f1_mean - b.macro_f1_mean
    if gap < max(a.macro_f1_std, b.macro_f1_std):
        return (f"\nNote: {a.model} and {b.model} differ by {gap:.4f} macro-F1, well inside the fold-to-fold "
                f"spread (sd up to {max(a.macro_f1_std, b.macro_f1_std):.3f}); treat them as equivalent. "
                "Ties go to the first model in the comparison table, which lists simpler models first.")
    return ""


def write_report_readme(table: pd.DataFrame, best: str, top_features: pd.DataFrame, report_dir: Path) -> None:
    top = "\n".join(f"- `{f}`: {m:.3f} ± {s:.3f}" for f, m, s in top_features[["feature", "mean", "std"]].itertuples(index=False))
    (report_dir / "README.md").write_text(f"""# Survey module: lifestyle model report

> **{SYNTHETIC_NOTICE}**
>
> **{ANALYSIS_ONLY_NOTICE}**

Task: predict `Emotional_State` (Normal / Depression / Anxiety / Stress) from
lifestyle answers only. DASS-21 items, scores, severities and labels are
excluded (they would leak the label). Users' screening results come only
from their own DASS-21 answers; this model is an exploratory analysis of the
synthetic generator.

Protocol: 1,200 synthetic respondents (one row each). Outer stratified
5-fold CV; inner stratified 3-fold grid search on macro-F1; all
preprocessing inside the sklearn Pipeline. Mean ± sd over outer folds;
per-class recall from pooled out-of-fold predictions.

{_markdown_table(table)}

Best family by mean outer macro-F1: **{best}** (exported, analysis-only).
{_closeness_note(table)}

Top permutation importances for {best} (drop in macro-F1 on held-out folds):

{top}

Files: `results.csv`, `fold_metrics.csv`, `confusion_matrix_*.csv/png`,
`permutation_importance.csv`, `permutation_importance_*.png`,
`best_params.json`.

Reproduce: `python -m ml.src.survey.train_lifestyle`
""", encoding="utf-8")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

DASS21_ANSWER_SCALE = [
    {"value": 0, "label": "Did not apply to me at all"},
    {"value": 1, "label": "Applied to me to some degree, or some of the time"},
    {"value": 2, "label": "Applied to me to a considerable degree, or a good part of the time"},
    {"value": 3, "label": "Applied to me very much, or most of the time"},
]


def build_questions(raw_columns) -> dict:
    from ml.src.survey.load import NUMERIC_FIELDS, item_texts
    from ml.src.survey.scoring import BAND_LOWER_CUTOFFS, BANDS, SCREENING_NOTICE, SUBSCALE_ITEMS

    texts = item_texts(raw_columns)
    subscale_of = {i: s for s, items in SUBSCALE_ITEMS.items() for i in items}
    ordinal = {"age_group", "sleep_duration", "physical_discomfort", "physical_activity_time"}
    return {
        "version": "survey-v1",
        "synthetic_source_notice": SYNTHETIC_NOTICE,
        "screening_notice": SCREENING_NOTICE,
        "instructions": "Please read each statement and choose how much it applied to you over the past week.",
        "answer_scale": DASS21_ANSWER_SCALE,
        "answer_scale_source": "Standard DASS-21 response scale; the source form stores only the numbers 0-3.",
        "items": [
            {"id": f"q{i}", "number": i, "text": texts[f"q{i}"], "subscale": subscale_of[i]}
            for i in range(1, 22)
        ],
        "scoring": {
            "subscale_items": {s: list(v) for s, v in SUBSCALE_ITEMS.items()},
            "multiplier": 2,
            "bands": list(BANDS),
            "band_lower_cutoffs": {s: list(v) for s, v in BAND_LOWER_CUTOFFS.items()},
            "dominant_state_rule": (
                "Normal if all three subscales are Normal; otherwise the subscale with the highest band. "
                "Ties: larger margin above that band's lower cutoff, then Depression > Anxiety > Stress."
            ),
        },
        "lifestyle_fields": [
            *[{"id": c, "type": "number", **meta} for c, meta in NUMERIC_FIELDS.items()],
            *[{"id": c, "type": "choice", "ordered": c in ordinal, "options": list(opts)}
              for c, opts in CATEGORICAL_OPTIONS.items()],
        ],
        "lifestyle_fields_note": (
            "Option text keeps the source form's exact spellings. Lifestyle answers are used only for "
            "offline analysis; no model output from them is shown to users."
        ),
    }


def build_model_card(table: pd.DataFrame, best: str, results: dict, importance: pd.DataFrame) -> dict:
    row = table.set_index("model").loc[best]
    return {
        "model_name": "survey-lifestyle-emotional-state",
        "version": "survey-v1",
        "status": "analysis-only",
        "serving": {
            "served_to_users": False,
            "api_endpoint": None,
            "statement": ANALYSIS_ONLY_NOTICE,
        },
        "data": {
            "source": "ml/data/raw/survey/survey_dataset_1200.xlsx, sheet Labeled_Dataset",
            "synthetic": True,
            "notice": SYNTHETIC_NOTICE,
            "n_rows": 1200,
            "class_balance": "300 per class (Normal, Depression, Anxiety, Stress); gender and age-group mix "
                             "identical in every class by construction",
        },
        "metrics_disclaimer": (
            "These metrics measure recovery of relationships the synthetic data generator simulated with "
            "assumed strengths. They are not real-world accuracy and must not be reported as such."
        ),
        "task": "Predict the dataset's Emotional_State label (derived from DASS-21 answers) from lifestyle answers only.",
        "features": list(LIFESTYLE_FEATURES),
        "feature_notes": {
            "bmi": "weight_kg / (height_cm / 100)^2, computed by make_X before the pipeline",
            "ordinal": list(ORDINAL_FEATURES),
            "one_hot": list(NOMINAL_FEATURES),
            "prefer_not_to_say": "kept as its own category",
        },
        "excluded_columns": sorted(FORBIDDEN_COLUMNS, key=lambda c: (not c.startswith("q"), len(c), c)),
        "exclusion_reason": "Emotional_State is computed from q1..q21; items, scores and severities would leak the label.",
        "estimator": best,
        "estimator_params": results[best]["final_params"],
        "evaluation": {
            "protocol": "outer stratified 5-fold CV (one row per respondent); inner stratified 3-fold grid search "
                        "on macro-F1; preprocessing fit inside training folds only",
            "macro_f1": {"mean": round(row.macro_f1_mean, 4), "std": round(row.macro_f1_std, 4)},
            "balanced_accuracy": {"mean": round(row.balanced_accuracy_mean, 4),
                                  "std": round(row.balanced_accuracy_std, 4)},
            "per_class_recall": {c: round(results[best]["per_class_recall"][c], 4) for c in CLASSES},
            "all_models": {
                r.model: {"macro_f1": round(r.macro_f1_mean, 4), "balanced_accuracy": round(r.balanced_accuracy_mean, 4)}
                for r in table.itertuples()
            },
            "top_permutation_importance": [
                {"feature": f, "mean_drop_macro_f1": round(m, 4)}
                for f, m in importance[["feature", "mean"]].head(5).itertuples(index=False)
            ],
        },
        "limitations": [
            "Data are fully synthetic; relationships and their strengths are the generator author's assumptions.",
            "mindless_snacking options 'Due to Stress' and 'Due to Anxity' name the target classes; any signal there "
            "is self-description, not an independent lifestyle effect.",
            "Gender and age-group mixes are identical in every class by design, so neither predicts the class on "
            "its own; age group can still matter in combination with other features (e.g. under-18 rows are "
            "set to non-smoker / non-drinker), which is what its permutation importance reflects.",
            "Not validated on any real population.",
        ],
        "ethics": "No output of this model is a diagnosis, screening result or statement about a person. "
                  "DASS-21 screening results for users come only from scoring.py applied to their own answers.",
        "sklearn_version": sklearn.__version__,
        "random_seed": SEED,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    ap.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    args = ap.parse_args(argv)

    from ml.src.survey.load import load_raw, clean
    raw = load_raw()
    df = clean(raw)
    X, y = make_X(df), df[TARGET]
    print(SYNTHETIC_NOTICE)
    print(f"X: {X.shape}, classes: {dict(y.value_counts())}")

    results = nested_cv(X, y)
    table = write_reports(results, args.report_dir)

    candidates = table[table.model != "majority_baseline"]
    best = candidates.sort_values("macro_f1_mean", ascending=False, kind="stable").iloc[0].model

    imp = pd.read_csv(args.report_dir / "permutation_importance.csv", comment="#")
    best_imp = imp[imp.model == best].sort_values("mean", ascending=False)
    write_report_readme(table, best, best_imp.head(5), args.report_dir)

    # Final analysis-only model: same tuning procedure on all rows.
    pipeline, grid = model_specs()[best]
    final = _tuned(pipeline, grid, n_splits=5).fit(X, y)
    results[best]["final_params"] = final.best_params_
    assert_no_leakage(final.best_estimator_.feature_names_in_)

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final.best_estimator_, args.artifact_dir / "lifestyle_model.joblib")
    (args.artifact_dir / "questions.json").write_text(
        json.dumps(build_questions(raw.columns), indent=2, ensure_ascii=False), encoding="utf-8")
    (args.artifact_dir / "model_card.json").write_text(
        json.dumps(build_model_card(table, best, results, best_imp), indent=2, default=str), encoding="utf-8")

    print("\n" + _markdown_table(table, pm="+/-"))  # ASCII: Windows pipes default to cp1252
    print(f"\nBest: {best}  final params: {final.best_params_}")
    print(f"Reports -> {args.report_dir}\nArtifacts -> {args.artifact_dir}")


if __name__ == "__main__":
    main()
