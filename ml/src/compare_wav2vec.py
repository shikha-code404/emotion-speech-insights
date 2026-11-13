"""Wav2Vec 2.0 experiment (step 1: frozen backbone + head) on experiments A and B.

Same leave-one-speaker-out folds (ml/src/splits.py) and metrics
(ml/src/evaluate.py) as the classical models and the CNN. Results are added
to ml/reports/comparison.md in a marked section.

Stages (default: all):
  extract  run the frozen backbone once per clip and cache embeddings
  train    train heads per fold, write per-fold results
  report   CPU latency for a 4-second clip, step-2 time estimate, comparison.md section
Run the report stage when nothing else is using the CPU, so latencies are fair.

Step-2 rule, fixed before results were seen: fine-tuning is only worth trying in
an experiment if the frozen last-layer head's pooled macro-F1 is within 0.05 of
the SVM's (or better).

Run:  python -m ml.src.compare_wav2vec [--stages extract train report]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.src import splits
from ml.src.evaluate import fold_scores, summarize
from ml.src.models import wav2vec_head as w2v
from ml.src.preprocessing import TRIM_TOP_DB, load_audio, prepare_waveform
from ml.src.train_speech import LABELS, MANIFEST, REPO_ROOT, load_manifest

EXPERIMENTS = {"A": "Emo-DB", "B": "SAVEE", "C": "cross-corpus", "D": "deployed configuration"}
PRED_DIR = REPO_ROOT / "ml" / "reports" / "predictions"
CACHE_DIR = REPO_ROOT / "ml" / "artifacts" / "cache"
OUT_DIR = REPO_ROOT / "ml" / "reports" / "w2v2"
CONFIG_DIR = REPO_ROOT / "ml" / "configs"
COMPARISON = REPO_ROOT / "ml" / "reports" / "comparison.md"
COMPETITIVE_MARGIN = 0.05
MAX_STEP2_HOURS = 1.0           # above this, stop before step 2 and report the estimate
MARK_START, MARK_END = "<!-- w2v2:start -->", "<!-- w2v2:end -->"


def log(msg: str) -> None:
    print(msg, flush=True)


def corpus_manifest(exp: str) -> pd.DataFrame:
    m = pd.read_csv(MANIFEST)
    return m[m["corpus"] == EXPERIMENTS[exp]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stage 1: extract
# ---------------------------------------------------------------------------

def cache_path(exp: str) -> Path:
    return CACHE_DIR / f"w2v2_base_exp{exp}.npz"


def extract(exp: str) -> None:
    m = corpus_manifest(exp)
    path = cache_path(exp)
    hashes = m["file_hash"].to_numpy().astype(str)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if np.array_equal(z["hashes"], hashes) and float(z["top_db"]) == TRIM_TOP_DB:
            log(f"  exp {exp}: cache up to date ({path.name})")
            return
    fe, model = w2v.load_backbone()
    pooled, frames, t0 = [], [], time.time()
    for i, p in enumerate(m["filepath"]):
        y, _ = load_audio(REPO_ROOT / p, target_sr=w2v.SAMPLE_RATE)
        pv, fr = w2v.embed(fe, model, prepare_waveform(y))
        pooled.append(pv)
        frames.append(fr)
        if (i + 1) % 100 == 0:
            log(f"    {i + 1}/{len(m)} clips  ({time.time() - t0:.0f}s)")
    offsets = np.cumsum([0] + [len(f) for f in frames])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pooled=np.stack(pooled), frames=np.concatenate(frames), offsets=offsets, hashes=hashes,
             top_db=TRIM_TOP_DB, checkpoint=w2v.CHECKPOINT,
             commit=str(getattr(model.config, "_commit_hash", "") or ""))
    log(f"  exp {exp}: extracted {len(m)} clips in {time.time() - t0:.0f}s -> {path.name} "
        f"({path.stat().st_size / 1024 ** 2:.0f} MB)")


# ---------------------------------------------------------------------------
# Stage 2: train heads
# ---------------------------------------------------------------------------

def load_experiment(exp: str):
    """(manifest, pooled embeddings (N, 13, 768), labels, groups, folds, checkpoint commit)."""
    if exp in ("C", "D"):  # both corpora, 6 shared classes; embeddings looked up in the A and B caches by hash
        # C = leave-one-corpus-out; D = the deployed setting, leave-one-speaker-out over all 14 speakers
        m = load_manifest()
        za, zb = (np.load(cache_path(e), allow_pickle=False) for e in ("A", "B"))
        row = {h: i for i, h in enumerate(np.concatenate([za["hashes"], zb["hashes"]]))}
        pooled = np.concatenate([za["pooled"], zb["pooled"]])[[row[h] for h in m["file_hash"]]]
        groups = splits.speaker_groups(m)
        folds = splits.leave_one_corpus_out(m) if exp == "C" else splits.leave_one_speaker_out(groups)
        return m, pooled, list(LABELS), groups, folds, str(za["commit"])
    m = corpus_manifest(exp)
    z = np.load(cache_path(exp), allow_pickle=False)
    assert np.array_equal(z["hashes"], m["file_hash"].to_numpy().astype(str)), "cache does not match manifest"
    groups = splits.speaker_groups(m)
    return (m, z["pooled"], sorted(m["emotion"].unique()), groups, splits.leave_one_speaker_out(groups),
            str(z["commit"]))


def train(exp: str, cfg: w2v.HeadConfig, save: bool = True) -> dict:
    m, pooled, labels, groups, folds, commit = load_experiment(exp)
    lid = {l: i for i, l in enumerate(labels)}
    y = m["emotion"].to_numpy()
    yi = np.array([lid[v] for v in y])

    variants = {"w2v2_frozen_last": [w2v.N_LAYERS], "w2v2_frozen_best_layer": list(range(1, w2v.N_LAYERS + 1))}
    results, cfg_folds = {}, []
    for name, layers in variants.items():
        oof = np.empty(len(y), dtype=object)
        proba = np.zeros((len(y), len(labels)))
        rows = []
        for f in folds:
            val_spk = splits.validation_speaker(groups[f.train_idx], f.index)
            fit = f.train_idx[groups[f.train_idx] != val_spk]
            val = f.train_idx[groups[f.train_idx] == val_spk]
            seed = cfg.seed + f.index
            best = None
            for L in layers:  # layer chosen on the validation speaker only
                r = w2v.train_head(pooled[fit, L], yi[fit], pooled[val, L], yi[val], len(labels), cfg, seed)
                if best is None or r.best_val_macro_f1 > best[1].best_val_macro_f1:
                    best = (L, r)
            L, r = best
            proba[f.test_idx] = w2v.head_proba(r.model, pooled[f.test_idx, L])
            pred = np.array(labels)[proba[f.test_idx].argmax(1)]
            oof[f.test_idx] = pred
            rows.append({"fold": f.index, "speaker": f.test_speaker, **fold_scores(y[f.test_idx], pred, labels),
                         "layer": L, "val_speaker": val_spk, "best_epoch": r.best_epoch, "seed": seed,
                         "val_macro_f1": r.best_val_macro_f1})
            if name == "w2v2_frozen_last":
                cfg_folds.append({"fold": f.index, "test_speaker": f.test_speaker, "val_speaker": val_spk,
                                  "seed": seed, "n_train": int(len(fit)), "n_val": int(len(val)),
                                  "n_test": int(len(f.test_idx))})
        per_fold = pd.DataFrame(rows)
        results[name] = {"per_fold": per_fold, "oof": oof, "proba": proba, **summarize(y, oof, per_fold, labels)}
        log(f"  exp {exp} {name:24s} macro-F1 {results[name]['macro_f1_pooled']:.3f}  "
            f"UA {results[name]['ua_pooled']:.3f}  layers chosen {sorted(per_fold['layer'].unique().tolist())}")

    if not save:
        return results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([r["per_fold"].assign(model=n) for n, r in results.items()]).to_csv(
        OUT_DIR / f"exp{exp}_per_fold.csv", index=False, float_format="%.4f")
    summary = {n: {k: (v if not isinstance(v, np.ndarray) else v.tolist()) for k, v in r.items()
                   if k not in ("per_fold", "oof", "proba")} for n, r in results.items()}
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    fold_of = np.empty(len(y), dtype=int)
    for f in folds:
        fold_of[f.test_idx] = f.index
    np.savez(PRED_DIR / f"exp{exp}_oof_w2v2.npz", labels=np.array(labels), y=y.astype(str),
             groups=groups.astype(str), corpus=m["corpus"].to_numpy().astype(str), fold=fold_of,
             file_hash=m["file_hash"].to_numpy().astype(str),
             **{f"pred_{n}": r["oof"].astype(str) for n, r in results.items()},
             **{f"proba_{n}": r["proba"] for n, r in results.items()})
    for old in CONFIG_DIR.glob(f"w2v2_exp{exp}_*.json"):   # keep one config per experiment: the latest run
        old.unlink()
    (OUT_DIR / f"exp{exp}_summary.json").write_text(json.dumps({"labels": labels, "results": summary}, indent=2))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (CONFIG_DIR / f"w2v2_exp{exp}_{stamp}.json").write_text(json.dumps({
        "experiment": exp, "corpus": EXPERIMENTS[exp], "checkpoint": w2v.CHECKPOINT,
        "checkpoint_commit": commit, "backbone": "fully frozen (step 1)",
        "input": "load_audio 16 kHz -> prepare_waveform -> Wav2Vec2FeatureExtractor (do_normalize) -> backbone",
        "pooling": "mean over frames, per clip, no padding",
        "head": cfg.to_dict(), "seed_rule": "fold seed = head.seed + fold index",
        "variants": {"w2v2_frozen_last": "transformer layer 12 (final)",
                     "w2v2_frozen_best_layer": "layer 1-12 chosen per fold on the validation speaker"},
        "folds": cfg_folds,
        "results": {n: {k: round(summary[n][k], 4) for k in ("macro_f1_pooled", "ua_pooled",
                                                             "macro_f1_fold_mean", "macro_f1_fold_std",
                                                             "ua_fold_mean", "ua_fold_std")}
                    for n in summary},
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
    }, indent=2), encoding="utf-8")
    return results


# ---------------------------------------------------------------------------
# Stage 3: report
# ---------------------------------------------------------------------------

def svm_reference(exp: str) -> dict | None:
    """SVM pooled scores from the latest CNN comparison run's config (same folds)."""
    paths = sorted(glob.glob(str(CONFIG_DIR / f"cnn_exp{exp}_*.json")))
    for p in reversed(paths):
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        if "results" in d:
            return d["results"]["svm_rbf"]
    return None


def latency_ms(fe, model, head: w2v.PooledHead, layer: int, y4: np.ndarray, runs=10, warmup=2) -> float:
    @torch.no_grad()
    def run():
        inputs = fe(prepare_waveform(y4), sampling_rate=w2v.SAMPLE_RATE, return_tensors="pt")
        hs = model(inputs.input_values, output_hidden_states=True).hidden_states[layer]
        return head(hs[0].mean(0, keepdim=True)).argmax(1)

    for _ in range(warmup):
        run()
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        run()
        t.append((time.perf_counter() - s) * 1000)
    return float(np.median(t))


def step2_estimate(model, exp: str, results: dict, batch: int = 4, steps: int = 6) -> dict:
    """Time forward+backward through transformer layers 11-12 + head on cached layer-10 frames,
    then extrapolate to a full step-2 run (lower layers frozen and cached, so they cost nothing)."""
    z = np.load(cache_path(exp), allow_pickle=False)
    frames = np.split(z["frames"], z["offsets"][1:-1])
    layers = torch.nn.ModuleList([copy_layer(model.encoder.layers[i]) for i in (10, 11)]).train()
    head = torch.nn.Linear(w2v.HIDDEN, 7)
    opt = torch.optim.Adam([{"params": layers.parameters(), "lr": 1e-5}, {"params": head.parameters(), "lr": 1e-3}])
    median_len = int(np.median([len(f) for f in frames]))
    x = torch.randn(batch, median_len, w2v.HIDDEN)
    times = []
    for _ in range(steps):
        s = time.perf_counter()
        h = x
        for layer in layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        torch.nn.functional.cross_entropy(head(h.mean(1)), torch.zeros(batch, dtype=torch.long)).backward()
        opt.step()
        opt.zero_grad()
        times.append(time.perf_counter() - s)
    per_clip = float(np.median(times[1:])) / batch
    n_folds = results["w2v2_frozen_last"]["per_fold"].shape[0]
    n_train = int(len(frames) * (n_folds - 2) / n_folds)        # training clips minus the validation speaker
    epochs = 15                                                 # assumed average with patience 5, max 30
    seconds = per_clip * n_train * epochs * n_folds * 1.15      # +15% for validation passes
    return {"per_clip_s": per_clip, "median_frames": median_len, "n_train_per_fold": n_train,
            "assumed_epochs": epochs, "folds": n_folds, "total_hours": seconds / 3600}


def copy_layer(layer):
    import copy as _c
    c = _c.deepcopy(layer)
    for p in c.parameters():
        p.requires_grad_(True)
    return c


def write_section(blocks: list[str]) -> None:
    text = COMPARISON.read_text(encoding="utf-8") if COMPARISON.exists() else "# Speech models\n"
    section = "\n".join([MARK_START, *blocks, MARK_END])
    if MARK_START in text:
        text = re.sub(re.escape(MARK_START) + ".*?" + re.escape(MARK_END), lambda _: section, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + section + "\n"
    COMPARISON.write_text(text, encoding="utf-8")


def report(all_results: dict) -> None:
    fe, model = w2v.load_backbone()
    size = w2v.backbone_size(model)
    blocks = [
        "## Wav2Vec 2.0, step 1: frozen backbone + head",
        "",
        f"`{w2v.CHECKPOINT}` (self-supervised pretraining on LibriSpeech, English read speech; not fine-tuned for "
        "emotion). Raw 16 kHz waveform after `prepare_waveform`; backbone fully frozen; embeddings extracted once per "
        "clip and cached; head = mean pooling, dropout, linear, with weighted cross-entropy and early stopping on the "
        "validation speaker's macro-F1. Same folds and metrics as above.",
        "",
        "- **W2V2 frozen, last layer**: head on transformer layer 12, the pre-registered main result.",
        "- **W2V2 frozen, best layer**: layer 1-12 chosen in each fold on the validation speaker only (never the "
        "test speaker); a diagnostic, because emotion information often peaks in middle layers.",
        "",
    ]
    for exp, res in all_results.items():
        if exp == "D":
            blocks += ["### Deployed configuration: both corpora, 6 classes, leave-one-speaker-out (14 speakers)", "",
                       "| Model | Macro-F1 | UA | Macro-F1 per speaker (mean ± sd) | UA per speaker (mean ± sd) |",
                       "|---|---|---|---|---|"]
            for name, title in (("w2v2_frozen_last", "W2V2 frozen, last layer"),
                                ("w2v2_frozen_best_layer", "W2V2 frozen, best layer")):
                r = res[name]
                blocks.append(f"| {title} | {r['macro_f1_pooled']:.3f} | {r['ua_pooled']:.3f} | "
                              f"{r['macro_f1_fold_mean']:.3f} ± {r['macro_f1_fold_std']:.3f} | "
                              f"{r['ua_fold_mean']:.3f} ± {r['ua_fold_std']:.3f} |")
            chosen = res["w2v2_frozen_best_layer"]["per_fold"]["layer"].tolist()
            blocks += ["", f"Layers chosen per fold (best-layer variant): {chosen}. The served SVM scores macro-F1 0.596 "
                           "in this same setting (see `model_selection.md`).", ""]
            continue
        if exp == "C":
            svm = svm_reference("C")
            blocks += ["### Experiment C: cross-corpus, 6 shared classes (train on one corpus, test on the other)", "",
                       "| Model | Macro-F1 | UA | Test on Emo-DB (macro-F1 / UA) | Test on SAVEE (macro-F1 / UA) |",
                       "|---|---|---|---|---|"]
            if svm:
                blocks.append(f"| SVM (RBF), from the table above | {svm['macro_f1_pooled']:.3f} | "
                              f"{svm['ua_pooled']:.3f} | see above | see above |")
            for name, title in (("w2v2_frozen_last", "W2V2 frozen, last layer"),
                                ("w2v2_frozen_best_layer", "W2V2 frozen, best layer")):
                r, pf = res[name], res[name]["per_fold"]
                d = " | ".join(f"{row.macro_f1:.3f} / {row.ua:.3f}" for row in pf.itertuples())
                blocks.append(f"| {title} | {r['macro_f1_pooled']:.3f} | {r['ua_pooled']:.3f} | {d} |")
            blocks += ["", "Size and CPU time are the same as in A and B. The early-stopping speaker comes from the "
                           "training corpus, so nothing from the test corpus is used for any choice.", ""]
            continue
        m = corpus_manifest(exp)
        y4 = load_audio(REPO_ROOT / m[m["duration_seconds"] >= 4.0].iloc[0]["filepath"], target_sr=16000)[0][:64000]
        labels = sorted(m["emotion"].unique())
        svm = svm_reference(exp)
        blocks += [f"### Experiment {exp}: {EXPERIMENTS[exp]}", "",
                   "| Model | Macro-F1 | UA | Macro-F1 per speaker (mean ± sd) | UA per speaker (mean ± sd) | Size | CPU time, 4 s clip |",
                   "|---|---|---|---|---|---|---|"]
        if svm:
            blocks.append(f"| SVM (RBF), from the table above | {svm['macro_f1_pooled']:.3f} | {svm['ua_pooled']:.3f} | "
                          f"{svm['macro_f1_fold_mean']:.3f} ± {svm['macro_f1_fold_std']:.3f} | see above | see above | see above |")
        for name, title in (("w2v2_frozen_last", "W2V2 frozen, last layer"),
                            ("w2v2_frozen_best_layer", "W2V2 frozen, best layer")):
            r = res[name]
            # Latency with a head from fold 1 (same shape in every fold).
            pf = r["per_fold"]
            layer = int(pf.iloc[0]["layer"])
            z = np.load(cache_path(exp), allow_pickle=False)
            groups = splits.speaker_groups(m)
            f0 = splits.leave_one_speaker_out(groups)[0]
            val_spk = splits.validation_speaker(groups[f0.train_idx], 0)
            fit = f0.train_idx[groups[f0.train_idx] != val_spk]
            val = f0.train_idx[groups[f0.train_idx] == val_spk]
            lid = {l: i for i, l in enumerate(labels)}
            yi = np.array([lid[v] for v in m["emotion"]])
            head = w2v.train_head(z["pooled"][fit, layer], yi[fit], z["pooled"][val, layer], yi[val], len(labels),
                                  w2v.HeadConfig(), seed=42).model
            lat = latency_ms(fe, model, head, layer, y4)
            head_params = sum(p.numel() for p in head.parameters())
            blocks.append(
                f"| {title} | {r['macro_f1_pooled']:.3f} | {r['ua_pooled']:.3f} | "
                f"{r['macro_f1_fold_mean']:.3f} ± {r['macro_f1_fold_std']:.3f} | "
                f"{r['ua_fold_mean']:.3f} ± {r['ua_fold_std']:.3f} | "
                f"{size['bytes_fp32'] / 1024 ** 2:.0f} MB ({size['params'] / 1e6:.1f}M backbone + {head_params:,} head params) | "
                f"{lat:.0f} ms |")
        last = res["w2v2_frozen_last"]
        chosen = res["w2v2_frozen_best_layer"]["per_fold"]["layer"].tolist()
        blocks += ["", f"Layers chosen per fold (best-layer variant): {chosen}.", ""]
        if svm:
            gap = last["macro_f1_pooled"] - svm["macro_f1_pooled"]
            competitive = gap >= -COMPETITIVE_MARGIN
            blocks.append(
                f"**Step-2 rule (fixed in advance: within {COMPETITIVE_MARGIN:.2f} of the SVM):** frozen last-layer "
                f"macro-F1 {last['macro_f1_pooled']:.3f} vs SVM {svm['macro_f1_pooled']:.3f} ({gap:+.3f}), so step 1 "
                f"is **{'competitive' if competitive else 'not competitive'}** in experiment {exp}.")
            res["competitive"] = competitive
        est = step2_estimate(model, exp, res)
        res["step2_estimate"] = est
        blocks += ["",
                   f"Step-2 time estimate on this CPU ({torch.get_num_threads()} threads), measured: "
                   f"{est['per_clip_s']:.2f} s per clip for a forward + backward pass through transformer layers 11-12 "
                   f"(clips of {est['median_frames']} frames, lower layers cached). With {est['n_train_per_fold']} "
                   f"training clips per fold, about {est['assumed_epochs']} epochs and {est['folds']} folds: "
                   f"**about {est['total_hours']:.1f} hours**.", ""]
    reasons = []
    for exp, res in all_results.items():
        if exp in ("C", "D"):
            continue
        why = []
        if res.get("competitive") is False:
            why.append("step 1 was not competitive with the SVM")
        if res["step2_estimate"]["total_hours"] > MAX_STEP2_HOURS:
            why.append(f"it would take about {res['step2_estimate']['total_hours']:.1f} hours on this CPU")
        reasons.append(f"experiment {exp}: " + (" and ".join(why) if why else "eligible"))
    blocks += ["**Step 2 (fine-tuning the last two transformer layers) was not run.** "
               + "; ".join(reasons) + ". It needs a GPU, or a decision to accept the run time.", ""]
    write_section(blocks)
    log(f"Section written to {COMPARISON}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Wav2Vec 2.0 step 1")
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS), choices=list(EXPERIMENTS))
    ap.add_argument("--stages", nargs="+", default=["extract", "train", "report"],
                    choices=["extract", "train", "report"])
    args = ap.parse_args(argv)
    cfg = w2v.HeadConfig()
    results = {}
    for exp in args.experiments:
        if "extract" in args.stages and exp in ("A", "B"):  # C and D reuse the A and B caches
            log(f"extract exp {exp}")
            extract(exp)
        if "train" in args.stages or "report" in args.stages:
            log(f"train exp {exp}")
            results[exp] = train(exp, cfg, save="train" in args.stages)
    if "report" in args.stages:
        report(results)


if __name__ == "__main__":
    main()
