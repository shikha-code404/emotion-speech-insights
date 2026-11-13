"""Train and evaluate the baseline speech-emotion model (ser-baseline-v1).

Classes: the 6 emotions present in BOTH corpora (anger, disgust, fear,
happiness, neutral, sadness). Boredom (Emo-DB only) and surprise (SAVEE only)
are excluded so that no class is tied to a single corpus.

Evaluation (Rule 1): leave-one-speaker-out over all 14 speakers (10 Emo-DB,
4 SAVEE). Hyperparameters are tuned by an inner speaker-grouped CV on each
outer training fold. Scaling lives inside the Pipeline and SVC's probability
calibration runs only on the data passed to fit (Rule 2).

Features come from ml/src/features.py after ml/src/preprocessing.py's
prepare_waveform, exactly as the backend does at inference (Rule 3).

Run:  python -m ml.src.train_speech
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from ml.src.features import FEATURE_DIM, extract_features  # noqa: E402
from ml.src.preprocessing import TRIM_TOP_DB, load_audio, prepare_waveform  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "ml" / "data_manifest" / "audio_manifest.csv"
ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts" / "ser-baseline-v1"
REPORT_DIR = REPO_ROOT / "ml" / "reports" / "speech"
CACHE = REPO_ROOT / "ml" / "artifacts" / "cache" / "speech_features_v1.npz"

VERSION = "ser-baseline-v1"
SAMPLE_RATE = 16000
LABELS = ("anger", "disgust", "fear", "happiness", "neutral", "sadness")
SEED = 42

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"

LIMITATIONS = [
    "Trained only on acted speech (Emo-DB: 10 German actors; SAVEE: 4 male English actors). "
    "Natural, everyday speech is usually much less exaggerated.",
    "Only 14 speakers in total, so results vary a lot from one speaker to another.",
    "Recording conditions (studio microphones) differ from phones and laptops; background noise lowers accuracy.",
    "It estimates the emotion expressed in the voice in one clip. It does not measure how a person feels "
    "overall, their mood over time, or their mental health.",
    "Only six emotion categories are recognised; everything else is forced into one of them.",
    "Evaluation scores come from held-out actors in the same two corpora, not from real-world use.",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_manifest(path: Path = MANIFEST) -> pd.DataFrame:
    m = pd.read_csv(path)
    m = m[m["emotion"].isin(LABELS)].reset_index(drop=True)
    m["group"] = m["corpus"] + ":" + m["speaker_id"].astype(str)
    return m


def clip_features(path: str) -> np.ndarray:
    y, sr = load_audio(REPO_ROOT / path, target_sr=SAMPLE_RATE)
    return extract_features(prepare_waveform(y), sr=sr)


def build_features(manifest: pd.DataFrame, cache: Path = CACHE) -> np.ndarray:
    hashes = manifest["file_hash"].to_numpy().astype(str)  # fixed-width strings: loadable without pickle
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if z["top_db"] == TRIM_TOP_DB and np.array_equal(z["hashes"], hashes):
            print(f"Using cached features {cache.name}")
            return z["X"]
    t0 = time.time()
    rows = Parallel(n_jobs=-1)(delayed(clip_features)(p) for p in manifest["filepath"])
    X = np.vstack(rows)
    assert X.shape == (len(manifest), FEATURE_DIM)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, hashes=hashes, top_db=TRIM_TOP_DB)
    print(f"Extracted features for {len(manifest)} clips in {time.time() - t0:.0f}s")
    return X


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def model_specs():
    return {
        "majority_baseline": (Pipeline([("scale", StandardScaler()),
                                        ("clf", DummyClassifier(strategy="most_frequent"))]),
                              {}),
        "svm_rbf": (Pipeline([("scale", StandardScaler()),
                              ("clf", SVC(kernel="rbf", class_weight="balanced"))]),
                    {"clf__C": [1, 3, 10, 30], "clf__gamma": ["scale", 0.001, 0.003]}),
        "random_forest": (Pipeline([("scale", StandardScaler()),
                                    ("clf", RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                                                   random_state=SEED, n_jobs=-1))]),
                          {"clf__max_features": ["sqrt", 0.2], "clf__min_samples_leaf": [1, 3]}),
    }


def _search(pipe, grid, n_inner):
    return GridSearchCV(pipe, grid or {"clf__strategy": ["most_frequent"]}, scoring="f1_macro",
                        cv=GroupKFold(n_splits=n_inner), n_jobs=-1)


def evaluate(X, y, groups, corpus):
    labels = list(LABELS)
    out = {}
    for name, (pipe, grid) in model_specs().items():
        t0 = time.time()
        oof = np.empty(len(y), dtype=object)
        per_speaker, params = [], []
        for tr, te in LeaveOneGroupOut().split(X, y, groups):
            search = _search(pipe, grid, n_inner=4).fit(X[tr], y[tr], groups=groups[tr])
            pred = search.predict(X[te])
            oof[te] = pred
            params.append(search.best_params_)
            per_speaker.append({
                "speaker": groups[te][0],
                "n": len(te),
                "macro_f1": f1_score(y[te], pred, labels=labels, average="macro", zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y[te], pred),
            })
        sp = pd.DataFrame(per_speaker)
        out[name] = {
            "per_speaker": sp,
            "params": params,
            "oof": oof,
            "macro_f1_pooled": f1_score(y, oof, labels=labels, average="macro"),
            "uar_pooled": balanced_accuracy_score(y, oof),
            "per_class_recall": dict(zip(labels, recall_score(y, oof, labels=labels, average=None))),
            "per_corpus": {c: {"macro_f1": f1_score(y[corpus == c], oof[corpus == c], labels=labels,
                                                    average="macro", zero_division=0),
                               "uar": balanced_accuracy_score(y[corpus == c], oof[corpus == c])}
                           for c in np.unique(corpus)},
            "confusion": confusion_matrix(y, oof, labels=labels),
        }
        print(f"  {name:18s} pooled macro-F1 {out[name]['macro_f1_pooled']:.3f}  UAR {out[name]['uar_pooled']:.3f}  "
              f"per-speaker macro-F1 {sp.macro_f1.mean():.3f} +/- {sp.macro_f1.std():.3f}  ({time.time() - t0:.0f}s)")
    return out


# ---------------------------------------------------------------------------
# Reports and export
# ---------------------------------------------------------------------------

def plot_confusion(cm, title, path):
    labels = list(LABELS)
    pct = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=SURFACE)
    im = ax.imshow(pct, cmap=LinearSegmentedColormap.from_list("b", BLUE_RAMP), vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if pct[i, j] > 0.45 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted", color=INK_2)
    ax.set_ylabel("True", color=INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Share of true class (recall)", color=INK_2, fontsize=8)
    cb.outline.set_visible(False)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    fig.text(0.01, 0.01, "Leave-one-speaker-out, pooled predictions. Acted speech only (Emo-DB + SAVEE).",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def write_reports(results, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, r in results.items():
        sp = r["per_speaker"]
        rows.append({"model": name, "macro_f1_pooled": r["macro_f1_pooled"], "uar_pooled": r["uar_pooled"],
                     "macro_f1_speaker_mean": sp.macro_f1.mean(), "macro_f1_speaker_std": sp.macro_f1.std(),
                     **{f"recall_{c}": r["per_class_recall"][c] for c in LABELS},
                     **{f"macro_f1_{c.lower().replace('-', '')}": v["macro_f1"] for c, v in r["per_corpus"].items()}})
        sp.to_csv(report_dir / f"per_speaker_{name}.csv", index=False, float_format="%.4f")
        pd.DataFrame(r["confusion"], index=[f"true_{c}" for c in LABELS],
                     columns=[f"pred_{c}" for c in LABELS]).to_csv(report_dir / f"confusion_matrix_{name}.csv")
        plot_confusion(r["confusion"], f"{name} - confusion matrix", report_dir / f"confusion_matrix_{name}.png")
    table = pd.DataFrame(rows)
    table.to_csv(report_dir / "results.csv", index=False, float_format="%.4f")
    return table


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train ser-baseline-v1")
    ap.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    ap.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = ap.parse_args(argv)

    m = load_manifest()
    print(f"{len(m)} clips, {m.group.nunique()} speakers, classes {dict(m.emotion.value_counts())}")
    X = build_features(m)
    y, groups, corpus = m["emotion"].to_numpy(), m["group"].to_numpy(), m["corpus"].to_numpy()

    results = evaluate(X, y, groups, corpus)
    table = write_reports(results, args.report_dir)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    cands = table[table.model != "majority_baseline"].sort_values("macro_f1_pooled", ascending=False, kind="stable")
    best = cands.iloc[0].model
    pipe, grid = model_specs()[best]
    final = _search(pipe, grid, n_inner=5).fit(X, y, groups=groups)
    model = final.best_estimator_
    if best == "svm_rbf":
        # Probabilities for the API: refit the chosen config with Platt scaling (fit on training data only).
        model.set_params(clf__probability=True, clf__random_state=SEED)
        model.fit(X, y)
    assert list(model.classes_) == sorted(LABELS)

    r = results[best]
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.artifact_dir / "model.joblib")
    meta = {
        "version": VERSION,
        "model_family": {"svm_rbf": "SVM (RBF kernel) on pooled MFCC/spectral statistics",
                         "random_forest": "Random forest on pooled MFCC/spectral statistics"}[best],
        "labels": list(LABELS),
        "corpora": ["Emo-DB (German, acted)", "SAVEE (English, acted)"],
        "sample_rate": SAMPLE_RATE,
        "feature_dim": FEATURE_DIM,
        "preprocessing": f"16 kHz mono; trim silence (top_db={TRIM_TOP_DB}); peak-normalise; ml/src/features.py",
        "params": final.best_params_,
        "metrics": {
            "evaluation": "leave-one-speaker-out over 14 speakers (acted speech)",
            "macro_f1": round(float(r["macro_f1_pooled"]), 4),
            "unweighted_average_recall": round(float(r["uar_pooled"]), 4),
            "macro_f1_per_speaker_mean": round(float(r["per_speaker"].macro_f1.mean()), 4),
            "macro_f1_per_speaker_std": round(float(r["per_speaker"].macro_f1.std()), 4),
            "per_class_recall": {k: round(float(v), 4) for k, v in r["per_class_recall"].items()},
            "per_corpus_macro_f1": {k: round(float(v["macro_f1"]), 4) for k, v in r["per_corpus"].items()},
        },
        "limitations": LIMITATIONS,
        "sklearn_version": sklearn.__version__,
    }
    (args.artifact_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_model_card(args.artifact_dir)
    print(f"\nBest: {best} {final.best_params_}\nArtifacts -> {args.artifact_dir}")


def write_model_card(artifact_dir: Path) -> Path:
    """model_card.json for an sklearn artifact, built from its metadata.json (read by backend/app/inference.py)."""
    meta = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    card = {
        "format": "sklearn-joblib",
        **{k: meta[k] for k in ("version", "model_family", "labels", "corpora", "sample_rate", "metrics",
                                "limitations")},
        "files": {"model": "model.joblib", "metadata": "metadata.json"},
        "prediction": "vocal emotion expressed in one audio clip",
        "features": f"ml/src/features.py ({meta['feature_dim']} pooled MFCC/spectral statistics)",
        "preprocessing": meta["preprocessing"],
    }
    path = artifact_dir / "model_card.json"
    path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
