"""Model-selection analysis for the speech module, computed from saved out-of-fold predictions.

Reads ml/reports/predictions/exp{A,B,C}_oof.npz (classical + CNN, from compare_speech)
and exp{A,B,C}_oof_w2v2.npz (Wav2Vec 2.0 heads, from compare_wav2vec) and writes to
ml/reports/selection/:
  tables.md                    metrics, calibration, per-class recall, main confusion pairs
  calibration_exp{A,B,C}.png   reliability plots
  calibration_deployed.png     reliability plot for the deployed configuration
  metrics.csv                  the same numbers in long form

Calibration: top-label reliability with 10 equal-width confidence bins. ECE is the
clip-weighted mean gap between a bin's average confidence and its accuracy.
Probabilities are the ones each model would serve: Platt-scaled SVM (fit on the
training fold), forest vote shares, softmax for the neural models.

"Deployed configuration" = the setting of ml/artifacts/ser-baseline-v1: both corpora,
6 shared classes, leave-one-speaker-out over 14 speakers. It is recomputed here for
the SVM only (the served model family) so its probabilities can be checked.

Run:  python -m ml.src.selection_report
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402

from ml.src import splits  # noqa: E402
from ml.src.evaluate import fold_scores, summarize  # noqa: E402
from ml.src.train_speech import LABELS, REPO_ROOT, SEED, build_features, load_manifest, model_specs  # noqa: E402

PRED_DIR = REPO_ROOT / "ml" / "reports" / "predictions"
OUT_DIR = REPO_ROOT / "ml" / "reports" / "selection"
N_BINS = 10
MODELS = {
    "majority_baseline": "Majority baseline",
    "svm_rbf": "SVM (RBF)",
    "random_forest": "Random Forest",
    "cnn": "Log-Mel CNN",
    "w2v2_frozen_last": "W2V2 frozen, last layer",
    "w2v2_frozen_best_layer": "W2V2 frozen, best layer",
}
EXP_TITLES = {
    "A": "Experiment A: Emo-DB, 7 classes, leave-one-speaker-out (10 folds)",
    "B": "Experiment B: SAVEE, 7 classes, leave-one-speaker-out (4 folds)",
    "C": "Experiment C: cross-corpus, 6 shared classes (2 folds: test on Emo-DB, test on SAVEE)",
    "D": "Deployed configuration: both corpora, 6 classes, leave-one-speaker-out (14 folds)",
}
SERIES, INK, INK_2, MUTED, GRID, SURFACE = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_experiment(exp: str, pred_dir: Path) -> dict | None:
    base = pred_dir / f"exp{exp}_oof.npz"
    if not base.exists():
        return None
    z = np.load(base, allow_pickle=False)
    out = {"labels": list(z["labels"]), "y": z["y"], "fold": z["fold"], "groups": z["groups"], "models": {}}
    files = [z]
    extra = pred_dir / f"exp{exp}_oof_w2v2.npz"
    if extra.exists():
        w = np.load(extra, allow_pickle=False)
        assert np.array_equal(w["file_hash"], z["file_hash"]), f"exp {exp}: prediction files are not aligned"
        files.append(w)
    for f in files:
        for key in f.files:
            if key.startswith("pred_"):
                name = key[5:]
                out["models"][name] = {"pred": f[key], "proba": f[f"proba_{name}"]}
    return out


def deployed_config_svm() -> dict:
    """Leave-one-speaker-out on the deployed setting (both corpora, 6 classes), SVM only."""
    m = load_manifest()
    X = build_features(m)
    y, groups, labels = m["emotion"].to_numpy(), splits.speaker_groups(m), list(LABELS)
    folds = splits.leave_one_speaker_out(groups)
    pipe, grid = model_specs()["svm_rbf"]
    pred = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), len(labels)))
    fold_of = np.empty(len(y), dtype=int)
    for f in folds:
        search = GridSearchCV(pipe, grid, scoring="f1_macro", n_jobs=-1,
                              cv=splits.inner_cv(len(np.unique(groups[f.train_idx]))))
        search.fit(X[f.train_idx], y[f.train_idx], groups=groups[f.train_idx])
        pred[f.test_idx] = search.predict(X[f.test_idx])
        pm = clone(search.best_estimator_).set_params(clf__probability=True, clf__random_state=SEED)
        pm.fit(X[f.train_idx], y[f.train_idx])
        p = pm.predict_proba(X[f.test_idx])
        for j, c in enumerate(pm.classes_):
            proba[f.test_idx, labels.index(c)] = p[:, j]
        fold_of[f.test_idx] = f.index
    return {"labels": labels, "y": y.astype(str), "fold": fold_of, "groups": groups,
            "file_hash": m["file_hash"].to_numpy().astype(str),
            "models": {"svm_rbf": {"pred": pred.astype(str), "proba": proba}}}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calibration(y: np.ndarray, proba: np.ndarray, labels: list[str]) -> dict:
    conf = proba.max(1)
    correct = np.array(labels)[proba.argmax(1)] == y
    edges = np.linspace(0, 1, N_BINS + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, N_BINS - 1)
    bins, ece = [], 0.0
    for b in range(N_BINS):
        sel = idx == b
        if sel.any():
            c, a, n = float(conf[sel].mean()), float(correct[sel].mean()), int(sel.sum())
            bins.append({"conf": c, "acc": a, "n": n})
            ece += n / len(y) * abs(c - a)
    return {"ece": ece, "mean_conf": float(conf.mean()), "acc_argmax": float(correct.mean()), "bins": bins}


def metrics_for(exp_data: dict, name: str) -> dict:
    y, labels, fold = exp_data["y"], exp_data["labels"], exp_data["fold"]
    pred, proba = exp_data["models"][name]["pred"], exp_data["models"][name]["proba"]
    per_fold = pd.DataFrame([fold_scores(y[fold == k], pred[fold == k], labels) for k in np.unique(fold)])
    s = summarize(y, pred, per_fold, labels)
    cm = confusion_matrix(y, pred, labels=labels)
    pairs = []
    for i, t in enumerate(labels):
        for j, p in enumerate(labels):
            if i != j and cm[i, j]:
                pairs.append((int(cm[i, j]), cm[i, j] / cm[i].sum(), t, p))
    pairs.sort(reverse=True)
    return {**s, "n_folds": len(per_fold), "calibration": calibration(y, proba, labels), "pairs": pairs[:3]}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_reliability(results: dict, title: str, path: Path) -> None:
    names = [n for n in results if n != "majority_baseline"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.7), facecolor=SURFACE, squeeze=False)
    for ax, name in zip(axes[0], names):
        cal = results[name]["calibration"]
        b = [x for x in cal["bins"] if x["n"] >= 5]
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="perfectly calibrated")
        ax.plot([x["conf"] for x in b], [x["acc"] for x in b], color=SERIES, linewidth=2, marker="o", markersize=6,
                label="model")
        for x in b:
            ax.annotate(str(x["n"]), (x["conf"], x["acc"]), textcoords="offset points", xytext=(5, -11), ha="left",
                        fontsize=7, color=INK_2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_aspect("equal")
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.set_title(f"{MODELS.get(name, name)}\nECE {cal['ece']:.3f}", fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Confidence (top probability)", fontsize=8, color=INK_2)
    axes[0][0].set_ylabel("Accuracy in bin", fontsize=8, color=INK_2)
    axes[0][0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Numbers beside points = clips in the bin (bins with fewer than 5 clips hidden). "
                         "Below the dashed line = over-confident. Acted speech only.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def tables(exp: str, data: dict, results: dict) -> list[str]:
    labels = data["labels"]
    out = [f"## {EXP_TITLES[exp]}", "", f"{len(data['y'])} clips.", "",
           "| Model | Macro-F1 | UA | Macro-F1 per fold (mean ± sd) | UA per fold (mean ± sd) | ECE | Mean confidence | Accuracy |",
           "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        c = r["calibration"]
        out.append(f"| {MODELS.get(name, name)} | {r['macro_f1_pooled']:.3f} | {r['ua_pooled']:.3f} | "
                   f"{r['macro_f1_fold_mean']:.3f} ± {r['macro_f1_fold_std']:.3f} | "
                   f"{r['ua_fold_mean']:.3f} ± {r['ua_fold_std']:.3f} | {c['ece']:.3f} | {c['mean_conf']:.3f} | "
                   f"{c['acc_argmax']:.3f} |")
    out += ["", "Accuracy here is plain accuracy of the highest-probability class (what the confidence should match); "
                "for the SVM it can differ slightly from the label metrics, which use the SVM's own decision rule.", "",
            "Per-class recall:", "", "| Class | " + " | ".join(MODELS.get(n, n) for n in results) + " |",
            "|---|" + "---|" * len(results)]
    for cls in labels:
        out.append(f"| {cls} | " + " | ".join(f"{results[n]['per_class_recall'][cls]:.2f}" for n in results) + " |")
    out += ["", "Main confusion pairs (true → predicted: clips, share of that true class):", ""]
    for name, r in results.items():
        if name == "majority_baseline":
            continue
        out.append(f"- {MODELS.get(name, name)}: " + "; ".join(f"{t} → {p}: {n} ({s:.0%})" for n, s, t, p in r["pairs"]))
    out.append("")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Speech model-selection analysis")
    ap.add_argument("--pred-dir", type=Path, default=PRED_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--skip-deployed", action="store_true")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    md = ["# Speech model selection: computed tables", "",
          "Generated by `python -m ml.src.selection_report` from saved out-of-fold predictions. "
          "Acted speech only; not real-world accuracy.", ""]
    rows = []
    experiments = {e: load_experiment(e, args.pred_dir) for e in "ABC"}
    if not args.skip_deployed:
        experiments["D"] = deployed_config_svm()
        extra = args.pred_dir / "expD_oof_w2v2.npz"
        if extra.exists():  # Wav2Vec heads scored in the same deployed setting
            w = np.load(extra, allow_pickle=False)
            assert np.array_equal(w["file_hash"], experiments["D"]["file_hash"]), "exp D files are not aligned"
            assert np.array_equal(w["fold"], experiments["D"]["fold"]), "exp D folds differ"
            for key in w.files:
                if key.startswith("pred_"):
                    experiments["D"]["models"][key[5:]] = {"pred": w[key], "proba": w[f"proba_{key[5:]}"]}
    for exp, data in experiments.items():
        if data is None:
            md += [f"## {EXP_TITLES[exp]}", "", "No prediction file found.", ""]
            continue
        order = [n for n in MODELS if n in data["models"]]
        results = {n: metrics_for(data, n) for n in order}
        md += tables(exp, data, results)
        png = args.out_dir / ("calibration_deployed.png" if exp == "D" else f"calibration_exp{exp}.png")
        plot_reliability(results, EXP_TITLES[exp], png)
        md += [f"Reliability plot: `{png.name}`", ""]
        for n, r in results.items():
            rows.append({"experiment": exp, "model": n, "macro_f1": r["macro_f1_pooled"], "ua": r["ua_pooled"],
                         "macro_f1_fold_mean": r["macro_f1_fold_mean"], "macro_f1_fold_std": r["macro_f1_fold_std"],
                         "ua_fold_mean": r["ua_fold_mean"], "ua_fold_std": r["ua_fold_std"],
                         "ece": r["calibration"]["ece"], "mean_confidence": r["calibration"]["mean_conf"],
                         "accuracy_argmax": r["calibration"]["acc_argmax"],
                         **{f"recall_{k}": v for k, v in r["per_class_recall"].items()}})
    (args.out_dir / "tables.md").write_text("\n".join(md), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.out_dir / "metrics.csv", index=False, float_format="%.4f")
    print(f"Wrote {args.out_dir / 'tables.md'}")


if __name__ == "__main__":
    main()
