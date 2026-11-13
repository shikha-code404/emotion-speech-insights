"""Phase 3 comparison: majority baseline, SVM, Random Forest and log-Mel CNN.

Experiments (within-corpus, all 7 classes of each corpus):
  A  Emo-DB  (10 speakers -> 10 leave-one-speaker-out folds)
  B  SAVEE   ( 4 speakers ->  4 leave-one-speaker-out folds)

Every model uses the same folds (ml/src/splits.py) and the same metrics
(ml/src/evaluate.py). Classical models use Phase 2's pipelines and grids
(ml/src/train_speech.model_specs) on the 396-dim features; the CNN uses
ml/src/models/cnn.py on log-Mel spectrograms. Both start from
ml/src/preprocessing (load_audio -> prepare_waveform).

This script never touches the deployed model (ml/artifacts/ser-baseline-v1)
or the backend.

Run:  python -m ml.src.compare_speech            (both experiments)
      python -m ml.src.compare_speech --experiments A --max-epochs 2   (smoke test)
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from ml.src import splits
from ml.src.evaluate import fold_scores, summarize
from ml.src.features import extract_features
from ml.src.models import cnn
from ml.src.preprocessing import TRIM_TOP_DB, load_audio, prepare_waveform
from ml.src.train_speech import LABELS, MANIFEST, REPO_ROOT, SEED, build_features, load_manifest, model_specs

EXPERIMENTS = {"A": "Emo-DB", "B": "SAVEE", "C": "cross-corpus"}
PRED_DIR = REPO_ROOT / "ml" / "reports" / "predictions"
REPORT_DIR = REPO_ROOT / "ml" / "reports"
DETAIL_DIR = REPORT_DIR / "cnn"
CONFIG_DIR = REPO_ROOT / "ml" / "configs"
CACHE_DIR = REPO_ROOT / "ml" / "artifacts" / "cache"
SR = cnn.SAMPLE_RATE
CLASSICAL = ("majority_baseline", "svm_rbf", "random_forest")
DISPLAY = {"majority_baseline": "Majority baseline", "svm_rbf": "SVM (RBF)",
           "random_forest": "Random Forest", "cnn": "Log-Mel CNN"}


def log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def prepared_waveforms(manifest: pd.DataFrame, tag: str) -> list[np.ndarray]:
    """load_audio -> prepare_waveform for every clip, cached as one flat array + offsets."""
    cache = CACHE_DIR / f"waveforms_{tag}.npz"
    hashes = manifest["file_hash"].to_numpy().astype(str)  # fixed-width strings: loadable without pickle
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if float(z["top_db"]) == TRIM_TOP_DB and np.array_equal(z["hashes"], hashes):
            return np.split(z["flat"], z["offsets"][1:-1])
    wavs = [prepare_waveform(load_audio(REPO_ROOT / p, target_sr=SR)[0]) for p in manifest["filepath"]]
    offsets = np.cumsum([0] + [len(w) for w in wavs])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, flat=np.concatenate(wavs), offsets=offsets, hashes=hashes, top_db=TRIM_TOP_DB)
    return wavs


def four_second_clip(manifest: pd.DataFrame) -> np.ndarray:
    row = manifest[manifest["duration_seconds"] >= 4.0].iloc[0]
    y, _ = load_audio(REPO_ROOT / row["filepath"], target_sr=SR)
    return y[: 4 * SR]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def run_classical(X, y, groups, folds, labels):
    out = {}
    for name, (pipe, grid) in model_specs().items():
        t0 = time.time()
        oof = np.empty(len(y), dtype=object)
        proba = np.zeros((len(y), len(labels)))
        rows, fold0 = [], None
        for f in folds:
            search = GridSearchCV(pipe, grid or {"clf__strategy": ["most_frequent"]}, scoring="f1_macro",
                                  cv=splits.inner_cv(len(np.unique(groups[f.train_idx]))), n_jobs=-1)
            search.fit(X[f.train_idx], y[f.train_idx], groups=groups[f.train_idx])
            pred = search.predict(X[f.test_idx])
            oof[f.test_idx] = pred
            # Probabilities the way the backend produces them: for the SVM, the chosen
            # configuration refit with Platt scaling on the training fold only.
            prob_model = search.best_estimator_
            if name == "svm_rbf":
                prob_model = clone(prob_model).set_params(clf__probability=True, clf__random_state=SEED)
                prob_model.fit(X[f.train_idx], y[f.train_idx])
            p = prob_model.predict_proba(X[f.test_idx])
            for j, c in enumerate(prob_model.classes_):
                proba[f.test_idx, labels.index(c)] = p[:, j]
            rows.append({"fold": f.index, "speaker": f.test_speaker, **fold_scores(y[f.test_idx], pred, labels),
                         "params": json.dumps(search.best_params_)})
            if f.index == 0:
                fold0 = prob_model
        per_fold = pd.DataFrame(rows)
        out[name] = {"per_fold": per_fold, "oof": oof, "proba": proba, "fold0_model": fold0,
                     "seconds": time.time() - t0,
                     **summarize(y, oof, per_fold, labels)}
        log(f"  {name:18s} macro-F1 {out[name]['macro_f1_pooled']:.3f}  UA {out[name]['ua_pooled']:.3f}  "
            f"({out[name]['seconds']:.0f}s)")
    return out


def run_cnn(wavs, y, groups, folds, labels, cfg, device, run_config):
    label_id = {l: i for i, l in enumerate(labels)}
    yi = np.array([label_id[v] for v in y])
    specs = [cnn.log_mel(w) for w in wavs]  # un-augmented, for validation/test
    oof = np.empty(len(y), dtype=object)
    oof_proba = np.zeros((len(y), len(labels)))
    rows, histories, fold0 = [], [], None
    t0 = time.time()
    for f in folds:
        val_spk = splits.validation_speaker(groups[f.train_idx], f.index)
        fit_idx = f.train_idx[groups[f.train_idx] != val_spk]
        val_idx = f.train_idx[groups[f.train_idx] == val_spk]
        seed = cfg.seed + f.index
        log(f"  fold {f.index} test={f.test_speaker} val={val_spk} train={len(fit_idx)} clips")
        res = cnn.train_fold([wavs[i] for i in fit_idx], yi[fit_idx], [specs[i] for i in val_idx], yi[val_idx],
                             len(labels), cfg, seed=seed, device=device, log=log)
        proba = cnn.predict_proba(res.model, [specs[i] for i in f.test_idx], device)
        pred = np.array(labels)[proba.argmax(1)]
        oof[f.test_idx] = pred
        oof_proba[f.test_idx] = proba
        rows.append({"fold": f.index, "speaker": f.test_speaker, **fold_scores(y[f.test_idx], pred, labels),
                     "val_speaker": val_spk, "best_epoch": res.best_epoch, "epochs_run": res.epochs_run,
                     "val_macro_f1": res.best_val_macro_f1, "seed": seed, "seconds": round(res.seconds, 1)})
        histories += [{"fold": f.index, **h} for h in res.history]
        run_config["folds"].append({k: rows[-1][k] for k in ("fold", "speaker", "val_speaker", "seed")}
                                   | {"n_train": int(len(fit_idx)), "n_val": int(len(val_idx)),
                                      "n_test": int(len(f.test_idx))})
        if f.index == 0:
            fold0 = res.model
    per_fold = pd.DataFrame(rows)
    result = {"per_fold": per_fold, "oof": oof, "proba": oof_proba, "fold0_model": fold0,
              "history": pd.DataFrame(histories),
              "seconds": time.time() - t0, **summarize(y, oof, per_fold, labels)}
    log(f"  {'cnn':18s} macro-F1 {result['macro_f1_pooled']:.3f}  UA {result['ua_pooled']:.3f}  "
        f"({result['seconds']:.0f}s)")
    return result


# ---------------------------------------------------------------------------
# Size and CPU latency (4-second clip, full pipeline from raw waveform)
# ---------------------------------------------------------------------------

def size_classical(model) -> dict:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    clf = model.named_steps["clf"]
    n = None
    if hasattr(clf, "support_vectors_"):
        n = f"{clf.support_vectors_.shape[0]} support vectors"
    elif hasattr(clf, "estimators_"):
        n = f"{sum(e.tree_.node_count for e in clf.estimators_):,} tree nodes"
    return {"bytes": buf.tell(), "detail": n or "-"}


def size_cnn(model) -> dict:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return {"bytes": buf.tell(), "detail": f"{cnn.n_parameters(model):,} parameters"}


def _median_ms(fn, runs=20, warmup=3) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def latency_classical(model, y4) -> float:
    return _median_ms(lambda: model.predict_proba(extract_features(prepare_waveform(y4), sr=SR).reshape(1, -1)))


def latency_cnn(model, y4) -> float:
    cpu_model = cnn.LogMelCNN(model.head.out_features, cnn.CNNConfig(channels=tuple(
        b.conv.out_channels for b in model.blocks)))
    cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    cpu_model.eval()

    @torch.no_grad()
    def run():
        s = cnn.log_mel(prepare_waveform(y4))
        return cpu_model(s[None, None], torch.tensor([s.shape[-1]])).argmax(1)

    return _median_ms(run)


# ---------------------------------------------------------------------------
# Comparison and report
# ---------------------------------------------------------------------------

def paired_vs_svm(res) -> dict:
    a = res["cnn"]["per_fold"].set_index("fold")["macro_f1"]
    b = res["svm_rbf"]["per_fold"].set_index("fold")["macro_f1"]
    d = (a - b).to_numpy()
    out = {"n_folds": len(d), "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()),
           "ties": int((np.abs(d) <= 1e-9).sum()), "mean_diff": float(d.mean()),
           "sd_diff": float(d.std(ddof=1)) if len(d) > 1 else 0.0, "p": None}
    if len(d) >= 6 and np.any(d != 0):
        out["p"] = float(wilcoxon(d).pvalue)
    return out


def verdict(exp: str, res: dict) -> str:
    c, s, pv = res["cnn"], res["svm_rbf"], res["paired"]
    pooled = c["macro_f1_pooled"] - s["macro_f1_pooled"]
    folds = f"{pv['wins']} of {pv['n_folds']} held-out folds"
    sig = (f"Wilcoxon signed-rank p = {pv['p']:.3f}" if pv["p"] is not None
           else f"only {pv['n_folds']} folds, too few for a meaningful significance test")
    if pooled > 0 and pv["mean_diff"] > 0 and pv["wins"] > pv["n_folds"] / 2 and (pv["p"] is not None and pv["p"] < 0.05):
        head = "Yes, the CNN beats the SVM"
    elif pooled < 0 and pv["mean_diff"] < 0:
        head = "No, the CNN does not beat the SVM"
    else:
        head = "No clear winner"
    return (f"**Experiment {exp}: {head}.** Pooled macro-F1 CNN {c['macro_f1_pooled']:.3f} vs SVM "
            f"{s['macro_f1_pooled']:.3f} ({pooled:+.3f}); the CNN scores higher on {folds} "
            f"(mean per-fold difference {pv['mean_diff']:+.3f} ± {pv['sd_diff']:.3f}); {sig}.")


def _mb(n: int) -> str:
    return f"{n / 1024:.0f} KB" if n < 1024 ** 2 else f"{n / 1024 ** 2:.1f} MB"


def write_markdown(all_res: dict, cfg: cnn.CNNConfig, device: torch.device, config_paths: dict) -> Path:
    lines = [
        "# Speech models: classical vs log-Mel CNN",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `python -m ml.src.compare_speech`. "
        "Acted speech only; these are not real-world accuracy figures.",
        "",
        "All models use the same leave-one-speaker-out folds (`ml/src/splits.py`, identical to Phase 2's "
        "`LeaveOneGroupOut`) and the same metrics (`ml/src/evaluate.py`). "
        "Classical models: Phase 2 pipelines and grids on 396 pooled MFCC/spectral statistics, tuned by speaker-grouped "
        "inner CV. CNN: 3 conv blocks on 1 x 80 x time log-Mel input, weighted cross-entropy, early stopping on a "
        "held-out training speaker's macro-F1, augmentation on training clips only.",
        "",
        "- **Macro-F1 / UA (pooled):** over all out-of-fold predictions. UA = unweighted accuracy "
        "(mean per-class recall).",
        "- **Per fold (mean ± sd):** the score on each held-out speaker (A, B) or held-out corpus (C); the sd is the fold-to-fold variation.",
        "- **Size:** the serialized fold-1 model. **CPU time:** median of 20 runs for one 4-second clip, "
        "from raw waveform through preprocessing, features and prediction, on this machine's CPU "
        f"({torch.get_num_threads()} threads).",
        "",
    ]
    for exp, res in all_res.items():
        lines += [f"## Experiment {exp}: {res['title']}", "",
                  "| Model | Macro-F1 | UA | Macro-F1 per fold (mean ± sd) | UA per fold (mean ± sd) | Size | CPU time, 4 s clip |",
                  "|---|---|---|---|---|---|---|"]
        for name in (*CLASSICAL, "cnn"):
            r = res[name]
            size = r.get("size")
            size_txt = f"{_mb(size['bytes'])} ({size['detail']})" if size else "as in A/B"
            lat = f"{r['latency_ms']:.0f} ms" if r.get("latency_ms") is not None else "as in A/B"
            lines.append(
                f"| {DISPLAY[name]} | {r['macro_f1_pooled']:.3f} | {r['ua_pooled']:.3f} | "
                f"{r['macro_f1_fold_mean']:.3f} ± {r['macro_f1_fold_std']:.3f} | "
                f"{r['ua_fold_mean']:.3f} ± {r['ua_fold_std']:.3f} | {size_txt} | {lat} |")
        lines += ["", verdict(exp, res), ""]
        if exp == "C":
            lines += ["Per direction (macro-F1 / UA):", ""]
            for n in (*CLASSICAL, "cnn"):
                parts = [f"{row.speaker}: {row.macro_f1:.3f} / {row.ua:.3f}" for row in res[n]["per_fold"].itertuples()]
                lines.append(f"- {DISPLAY[n]}: " + "; ".join(parts))
            lines.append("")
        rec = pd.DataFrame({DISPLAY[n]: res[n]["per_class_recall"] for n in ("svm_rbf", "random_forest", "cnn")})
        lines += ["Per-class recall (pooled):", "", "| Class | " + " | ".join(rec.columns) + " |",
                  "|---|" + "---|" * len(rec.columns)]
        lines += [f"| {cls} | " + " | ".join(f"{v:.2f}" for v in row) + " |" for cls, row in rec.iterrows()]
        pf = res["cnn"]["per_fold"]
        lines += ["", f"CNN early stopping: best epoch median {int(pf['best_epoch'].median())} "
                      f"(range {pf['best_epoch'].min()}-{pf['best_epoch'].max()}), config `{config_paths[exp]}`.", ""]
    lines += [
        "## Caveats",
        "",
        "- Very small data: 10 and 4 speakers. Fold-to-fold variation is large, so differences of a few points "
        "are within noise. SAVEE has only 4 folds, too few for a significance test.",
        "- The CNN was not hyperparameter-tuned (fixed architecture and optimiser); the classical models had small grids. "
        "The CNN's early-stopping speaker is taken from the training speakers, so each CNN trains on one speaker fewer "
        "than the classical models in the same fold.",
        "- Experiments A and B are within-corpus. Experiment C (train on one corpus, test on the other, 6 shared "
        "classes) is the only cross-corpus evidence, and it has just 2 folds. None of this shows how any model handles "
        "real, unacted speech or phone microphones.",
        "- Out-of-fold probabilities for every model are saved in `ml/reports/predictions/` for calibration analysis.",
        f"- CNN trained on {device.type.upper()}; CPU timings are single clips with no batching. Library versions: "
        f"torch {torch.__version__}, scikit-learn {sklearn.__version__}.",
        "",
    ]
    path = REPORT_DIR / "comparison.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Compare classical models and the log-Mel CNN")
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    ap.add_argument("--max-epochs", type=int, default=None, help="override (smoke tests only)")
    ap.add_argument("--out-root", type=Path, default=None, help="write reports/configs here instead (smoke tests)")
    args = ap.parse_args(argv)
    global REPORT_DIR, DETAIL_DIR, CONFIG_DIR, PRED_DIR
    if args.out_root:
        REPORT_DIR, CONFIG_DIR = args.out_root / "reports", args.out_root / "configs"
        DETAIL_DIR, PRED_DIR = REPORT_DIR / "cnn", REPORT_DIR / "predictions"
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    cfg = cnn.CNNConfig()
    if args.max_epochs:
        cfg.max_epochs = args.max_epochs
    device = cnn.device_auto()
    DETAIL_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    manifest_all = pd.read_csv(MANIFEST)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log(f"device: {device}  torch {torch.__version__}  threads {torch.get_num_threads()}")

    all_res, config_paths = {}, {}
    for exp in args.experiments:
        corpus = EXPERIMENTS[exp]
        if exp == "C":
            m = load_manifest()                      # both corpora, the 6 shared classes
            labels = list(LABELS)
            groups = splits.speaker_groups(m)
            folds = splits.leave_one_corpus_out(m)
            title = (f"cross-corpus, 6 shared classes ({len(m)} clips; fold 1 trains on SAVEE and tests on Emo-DB, "
                     "fold 2 the reverse)")
            feat_cache = CACHE_DIR / "speech_features_v1.npz"
        else:
            m = manifest_all[manifest_all["corpus"] == corpus].reset_index(drop=True)
            labels = sorted(m["emotion"].unique())
            assert len(labels) == 7, labels
            groups = splits.speaker_groups(m)
            folds = splits.leave_one_speaker_out(groups)
            title = f"{corpus}, 7 classes ({len(m)} clips, {len(folds)} speakers, leave-one-speaker-out)"
            feat_cache = CACHE_DIR / f"speech_features_exp{exp}.npz"
        y = m["emotion"].to_numpy()
        log(f"\n=== Experiment {exp}: {title}, labels {labels}")

        cfg_path = CONFIG_DIR / f"cnn_exp{exp}_{stamp}.json"
        run_config = {"experiment": exp, "corpus": corpus, "labels": labels, "created": stamp,
                      "command": " ".join(sys.argv), "device": str(device), "torch": torch.__version__,
                      "torchaudio": __import__("torchaudio").__version__, "sklearn": sklearn.__version__,
                      "preprocessing": f"load_audio 16 kHz mono -> prepare_waveform(top_db={TRIM_TOP_DB})",
                      "log_mel": {"n_mels": cnn.N_MELS, "n_fft": cnn.N_FFT, "hop": cnn.HOP, "f_min": 20.0,
                                  "f_max": SR / 2, "log": "ln(power + 1e-6)"},
                      "cnn": cfg.to_dict(), "seed_rule": "fold seed = cnn.seed + fold index",
                      "splits": "leave-one-speaker-out (ml/src/splits.py); early-stopping speaker = "
                                "validation_speaker(train speakers, fold index)",
                      "folds": []}
        cfg_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        config_paths[exp] = (cfg_path.relative_to(REPO_ROOT).as_posix() if cfg_path.is_relative_to(REPO_ROOT)
                             else cfg_path.as_posix())

        log("classical models:")
        X = build_features(m, cache=feat_cache)
        res = run_classical(X, y, groups, folds, labels)
        log("log-Mel CNN:")
        wavs = prepared_waveforms(m, f"exp{exp}")
        res["cnn"] = run_cnn(wavs, y, groups, folds, labels, cfg, device, run_config)

        if exp != "C":  # same architectures in C; size and latency are reported once, in A and B
            y4 = four_second_clip(m)
            for name in CLASSICAL:
                res[name]["size"] = size_classical(res[name]["fold0_model"])
                res[name]["latency_ms"] = latency_classical(res[name]["fold0_model"], y4)
            res["cnn"]["size"] = size_cnn(res["cnn"]["fold0_model"])
            res["cnn"]["latency_ms"] = latency_cnn(res["cnn"]["fold0_model"], y4)
        res["paired"] = paired_vs_svm(res)
        res["title"] = title
        all_res[exp] = res

        fold_of = np.empty(len(y), dtype=int)
        for f in folds:
            fold_of[f.test_idx] = f.index
        np.savez(PRED_DIR / f"exp{exp}_oof.npz", labels=np.array(labels), y=y.astype(str), groups=groups.astype(str),
                 corpus=m["corpus"].to_numpy().astype(str), fold=fold_of,
                 file_hash=m["file_hash"].to_numpy().astype(str),
                 **{f"pred_{n}": res[n]["oof"].astype(str) for n in (*CLASSICAL, "cnn")},
                 **{f"proba_{n}": res[n]["proba"] for n in (*CLASSICAL, "cnn")})

        pd.concat([res[n]["per_fold"].assign(model=n) for n in (*CLASSICAL, "cnn")]).to_csv(
            DETAIL_DIR / f"exp{exp}_per_fold.csv", index=False, float_format="%.4f")
        res["cnn"]["history"].to_csv(DETAIL_DIR / f"exp{exp}_cnn_history.csv", index=False, float_format="%.4f")
        for n in (*CLASSICAL, "cnn"):
            pd.DataFrame(res[n]["confusion"], index=[f"true_{l}" for l in labels],
                         columns=[f"pred_{l}" for l in labels]).to_csv(DETAIL_DIR / f"exp{exp}_confusion_{n}.csv")

        run_config["results"] = {n: {k: round(res[n][k], 4) for k in
                                     ("macro_f1_pooled", "ua_pooled", "macro_f1_fold_mean", "macro_f1_fold_std",
                                      "ua_fold_mean", "ua_fold_std")}
                                 for n in (*CLASSICAL, "cnn")}
        run_config["paired_vs_svm"] = res["paired"]
        cfg_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        log(verdict(exp, res).replace("±", "+/-"))  # ASCII for Windows consoles
        path = write_markdown(all_res, cfg, device, config_paths)  # rewritten after every experiment
        log(f"Report -> {path}")


if __name__ == "__main__":
    main()
