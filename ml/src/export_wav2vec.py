"""Export the selected speech model: frozen wav2vec2-base, truncated at one layer, + linear head.

Selection is documented in ml/reports/model_selection.md. Setting = the deployed one:
both corpora, the 6 shared classes, leave-one-speaker-out over 14 speakers.

Procedure (no clip is ever scored by a model that saw its speaker):
  1. Layer choice: for every layer 1-12 and every held-out speaker, train a head on the
     other speakers with early stopping on a different training speaker (splits.py), and
     score the held-out speaker. Pick the layer with the best mean held-out macro-F1;
     ties go to the lower layer (smaller and faster backbone).
  2. Epochs: the median early-stopping epoch at that layer.
  3. Final head: trained on all 874 clips for that many epochs (no data held back, as is
     usual once the procedure has been evaluated).
  4. Backbone saved truncated at the chosen layer, so serving never runs the unused
     layers and needs no network access.

The honest performance estimate for this procedure is the nested result in
ml/reports/w2v2 (deployed setting, best-layer variant), not the layer-selection score.

Run:  python -m ml.src.export_wav2vec [--version ser-w2v2-v2]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ml.src import splits
from ml.src.compare_wav2vec import load_experiment
from ml.src.evaluate import fold_scores
from ml.src.models import wav2vec_head as w2v
from ml.src.preprocessing import TRIM_TOP_DB, load_audio, prepare_waveform
from ml.src.train_speech import LIMITATIONS, REPO_ROOT

ARTIFACTS = REPO_ROOT / "ml" / "artifacts"
REPORTS = REPO_ROOT / "ml" / "reports"


def log(msg: str) -> None:
    print(msg, flush=True)


def choose_layer(pooled, yi, labels, groups, folds, cfg):
    rows = []
    for L in range(1, w2v.N_LAYERS + 1):
        for f in folds:
            val_spk = splits.validation_speaker(groups[f.train_idx], f.index)
            fit = f.train_idx[groups[f.train_idx] != val_spk]
            val = f.train_idx[groups[f.train_idx] == val_spk]
            r = w2v.train_head(pooled[fit, L], yi[fit], pooled[val, L], yi[val], len(labels), cfg, cfg.seed + f.index)
            pred = w2v.head_predict(r.model, pooled[f.test_idx, L])
            s = fold_scores(yi[f.test_idx], pred, list(range(len(labels))))
            rows.append({"layer": L, "fold": f.index, "speaker": f.test_speaker, "macro_f1": s["macro_f1"],
                         "best_epoch": r.best_epoch})
        log(f"  layer {L:2d}: mean held-out macro-F1 "
            f"{np.mean([x['macro_f1'] for x in rows if x['layer'] == L]):.3f}")
    df = pd.DataFrame(rows)
    by_layer = df.groupby("layer")["macro_f1"].agg(["mean", "std"])
    best = by_layer["mean"].max()
    layer = int(by_layer.index[np.isclose(by_layer["mean"], best)].min())   # ties -> lower layer
    epochs = int(df[df.layer == layer]["best_epoch"].median())
    return layer, epochs, by_layer, df


def folder_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", default="ser-w2v2-v2")
    args = ap.parse_args(argv)
    out = ARTIFACTS / args.version
    cfg = w2v.HeadConfig()

    m, pooled, labels, groups, folds, commit = load_experiment("D")
    yi = np.array([labels.index(v) for v in m["emotion"]])
    log(f"{len(m)} clips, {len(folds)} speakers, labels {labels}")

    log("1. layer choice (leave-one-speaker-out, early stopping on another training speaker):")
    layer, epochs, by_layer, detail = choose_layer(pooled, yi, labels, groups, folds, cfg)
    log(f"   -> layer {layer}, {epochs} epochs")

    log("2. final head on all clips")
    head = w2v.train_head_fixed(pooled[:, layer], yi, len(labels), cfg, epochs, seed=cfg.seed)

    log("3. truncated backbone")
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    fe = Wav2Vec2FeatureExtractor.from_pretrained(w2v.CHECKPOINT)
    backbone = Wav2Vec2Model.from_pretrained(w2v.CHECKPOINT, num_hidden_layers=layer).eval()

    if out.exists():
        shutil.rmtree(out)
    (out / "backbone").mkdir(parents=True)
    backbone.save_pretrained(out / "backbone")
    fe.save_pretrained(out / "backbone")
    torch.save(head.state_dict(), out / "head.pt")
    (out / "label_map.json").write_text(json.dumps(
        {"labels": labels, "index_to_label": {str(i): l for i, l in enumerate(labels)}}, indent=2), encoding="utf-8")
    (out / "preprocess_config.json").write_text(json.dumps({
        "sample_rate": w2v.SAMPLE_RATE,
        "channels": "mono",
        "steps": [
            "ml/src/preprocessing.load_audio (resample to 16 kHz, mono)",
            f"ml/src/preprocessing.prepare_waveform (trim silence, top_db={TRIM_TOP_DB}; peak-normalise)",
            "Wav2Vec2FeatureExtractor (per-utterance zero mean, unit variance)",
            f"wav2vec2-base backbone truncated after transformer layer {layer}",
            "mean over frames",
            "head: standardise (training statistics) -> linear -> softmax",
        ],
        "trim_top_db": TRIM_TOP_DB,
        "checkpoint": w2v.CHECKPOINT,
        "checkpoint_commit": commit,
        "layer": layer,
        "pooling": "mean",
        "temperature": 1.0,
    }, indent=2), encoding="utf-8")

    log("4. checking the exported model against the evaluated pipeline")
    predictor = w2v.Wav2Vec2Predictor(out)
    rng = np.random.default_rng(0)
    for i in rng.choice(len(m), size=5, replace=False):
        y = prepare_waveform(load_audio(REPO_ROOT / m.iloc[i]["filepath"], target_sr=16000)[0])
        served = np.array([predictor.predict_proba(y)[l] for l in labels])
        cached = w2v.head_proba(head, pooled[i:i + 1, layer])[0]
        assert np.allclose(served, cached, atol=1e-3), (i, served, cached)
    log("   exported model reproduces the cached-embedding pipeline (5 clips, atol 1e-3)")

    row = m[m["duration_seconds"] >= 4.0].iloc[0]
    y4 = load_audio(REPO_ROOT / row["filepath"], target_sr=16000)[0][:64000]
    for _ in range(2):
        predictor.predict_proba(prepare_waveform(y4))
    t = []
    for _ in range(10):
        s = time.perf_counter()
        predictor.predict_proba(prepare_waveform(y4))
        t.append((time.perf_counter() - s) * 1000)
    latency = float(np.median(t))
    n_params = sum(p.numel() for p in predictor.backbone.parameters()) + sum(
        p.numel() for p in predictor.head.parameters())
    size = folder_bytes(out)
    log(f"   latency {latency:.0f} ms for a 4 s clip; {n_params / 1e6:.1f}M parameters; {size / 1024 ** 2:.0f} MB on disk")

    nested = json.loads((REPORTS / "w2v2" / "expD_summary.json").read_text(encoding="utf-8"))["results"][
        "w2v2_frozen_best_layer"]
    sel = pd.read_csv(REPORTS / "selection" / "metrics.csv")
    ece = float(sel[(sel.experiment == "D") & (sel.model == "w2v2_frozen_best_layer")]["ece"].iloc[0])
    card = {
        "format": "pytorch-wav2vec2-head",
        "version": args.version,
        "created": datetime.now().strftime("%Y-%m-%d"),
        "model_family": f"Wav2Vec 2.0 base (frozen, first {layer} transformer layers) + linear head",
        "labels": labels,
        "corpora": ["Emo-DB (German, acted)", "SAVEE (English, acted)"],
        "sample_rate": w2v.SAMPLE_RATE,
        "files": {"backbone": "backbone/", "head": "head.pt", "label_map": "label_map.json",
                  "preprocess_config": "preprocess_config.json"},
        "prediction": "vocal emotion expressed in one audio clip",
        "intended_use": "Educational demo of speech emotion recognition. Shows the emotion a short clip's voice "
                        "most resembles, with the model's uncertainty.",
        "not_for": ["any statement about a person's mental health or wellbeing",
                    "decisions about people", "monitoring people over time"],
        "metrics": {
            "evaluation": "leave-one-speaker-out over 14 speakers (acted speech); layer chosen inside each fold",
            "macro_f1": round(nested["macro_f1_pooled"], 4),
            "unweighted_average_recall": round(nested["ua_pooled"], 4),
            "macro_f1_per_speaker_mean": round(nested["macro_f1_fold_mean"], 4),
            "macro_f1_per_speaker_std": round(nested["macro_f1_fold_std"], 4),
            "per_class_recall": {k: round(v, 4) for k, v in nested["per_class_recall"].items()},
            "ece": round(ece, 4),
            "cross_corpus_macro_f1": 0.245,
            "layer_selection_score_optimistic": round(float(by_layer.loc[layer, "mean"]), 4),
            "cpu_latency_ms_4s_clip": round(latency),
        },
        "training": {"clips": int(len(m)), "speakers": int(len(folds)), "layer": layer, "head_epochs": epochs,
                     "head": cfg.to_dict(), "backbone_frozen": True, "fine_tuned": False},
        "size": {"parameters": int(n_params), "bytes_on_disk": int(size)},
        "backbone_license": "Apache-2.0 (facebook/wav2vec2-base)",
        "selection_report": "ml/reports/model_selection.md",
        "limitations": LIMITATIONS + [
            "The backbone was pretrained on English read speech (LibriSpeech); it was not trained for emotion.",
            "Trained on one corpus and tested on the other, every model scored poorly (best macro-F1 about 0.25), "
            "so expect much lower accuracy on everyday recordings than on the acted test speakers.",
        ],
    }
    (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    by_layer.to_csv(REPORTS / "selection" / "export_layer_selection.csv", float_format="%.4f")
    log(f"Exported -> {out}")


if __name__ == "__main__":
    main()
