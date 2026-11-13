"""Wav2Vec 2.0 (facebook/wav2vec2-base) for speech emotion: frozen backbone + small head.

Experimental; not deployed.

Input: raw 16 kHz waveform from ml/src/preprocessing (load_audio ->
prepare_waveform), then the checkpoint's own feature extractor (per-utterance
zero-mean / unit-variance), then the backbone. No MFCCs are used.

Step 1 (this module's main path)
  * Backbone fully frozen, eval mode. Each clip is passed through alone (no
    padding) once, and the result is cached to disk.
  * Cached per clip: mean over frames of every hidden state (13 x 768: CNN
    feature projection + 12 transformer layers) and, for a possible step 2,
    the frame-level input to the last two transformer layers.
  * Head = mean pooling -> dropout -> linear. Mean pooling has no parameters
    and the backbone is frozen, so pooling at cache time is exactly the same
    computation as pooling inside the head.
  * Per fold: embeddings standardised with training-clip statistics only
    (Rule 2), weighted cross-entropy, early stopping on the validation
    speaker's macro-F1, fixed seeds.

Step 2 (fine-tune the last two transformer layers) is deliberately not
implemented here yet; see ml/reports/comparison.md for the decision and the
measured time estimate.
"""

from __future__ import annotations

import copy
import os
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

CHECKPOINT = "facebook/wav2vec2-base"
SAMPLE_RATE = 16000
N_LAYERS = 12
HIDDEN = 768
STEP2_INPUT_LAYER = N_LAYERS - 2   # hidden_states[10] is the input to transformer layers 11 and 12


def load_backbone():
    """Frozen backbone + its feature extractor (downloaded once to the Hugging Face cache)."""
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # silences the expected "unused pretraining head weights" report
    fe = Wav2Vec2FeatureExtractor.from_pretrained(CHECKPOINT)
    model = Wav2Vec2Model.from_pretrained(CHECKPOINT).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return fe, model


@torch.no_grad()
def embed(fe, model, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One prepared waveform -> (13 x 768 mean-pooled hidden states, T x 768 step-2 input frames)."""
    inputs = fe(y, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    out = model(inputs.input_values, output_hidden_states=True)
    hs = out.hidden_states                                   # 13 x (1, T, 768)
    pooled = torch.stack([h[0].mean(dim=0) for h in hs]).numpy().astype(np.float32)
    frames = hs[STEP2_INPUT_LAYER][0].numpy().astype(np.float16)
    return pooled, frames


def backbone_size(model) -> dict:
    n = sum(p.numel() for p in model.parameters())
    return {"params": n, "bytes_fp32": n * 4}


# ---------------------------------------------------------------------------
# Head (dropout + linear on mean-pooled embeddings)
# ---------------------------------------------------------------------------

@dataclass
class HeadConfig:
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 300
    patience: int = 30
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


class PooledHead(nn.Module):
    """Input is the already mean-pooled embedding (pooling done at cache time); standardise, dropout, linear."""

    def __init__(self, n_classes: int, cfg: HeadConfig, dim: int = HIDDEN):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.drop = nn.Dropout(cfg.dropout)
        self.linear = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.drop((x - self.mean) / self.std))


@dataclass
class HeadResult:
    model: PooledHead
    best_epoch: int
    best_val_macro_f1: float
    epochs_run: int
    seconds: float


def train_head(Xtr: np.ndarray, ytr: np.ndarray, Xval: np.ndarray, yval: np.ndarray, n_classes: int,
               cfg: HeadConfig, seed: int) -> HeadResult:
    from sklearn.metrics import f1_score

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()
    xt = torch.as_tensor(Xtr, dtype=torch.float32)
    xv = torch.as_tensor(Xval, dtype=torch.float32)
    head = PooledHead(n_classes, cfg, dim=Xtr.shape[1])
    head.mean.copy_(xt.mean(0))
    head.std.copy_(torch.clamp(xt.std(0), min=1e-4))  # training-fold statistics only
    counts = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    w = torch.tensor(len(ytr) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    yt = torch.as_tensor(ytr, dtype=torch.long)

    best, best_epoch, best_state, epoch = -1.0, 0, None, 0
    for epoch in range(1, cfg.max_epochs + 1):
        head.train()
        order = rng.permutation(len(yt))
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i: i + cfg.batch_size]
            opt.zero_grad()
            loss_fn(head(xt[idx]), yt[idx]).backward()
            opt.step()
        head.eval()
        with torch.no_grad():
            pred = head(xv).argmax(1).numpy()
        f1 = float(f1_score(yval, pred, average="macro", labels=np.unique(yval), zero_division=0))
        if f1 > best:
            best, best_epoch, best_state = f1, epoch, copy.deepcopy(head.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            break
    head.load_state_dict(best_state)
    head.eval()
    return HeadResult(head, best_epoch, best, epoch, time.time() - t0)


@torch.no_grad()
def head_proba(head: PooledHead, X: np.ndarray) -> np.ndarray:
    head.eval()
    return torch.softmax(head(torch.as_tensor(X, dtype=torch.float32)), dim=1).numpy()


def head_predict(head: PooledHead, X: np.ndarray) -> np.ndarray:
    return head_proba(head, X).argmax(1)


def train_head_fixed(X: np.ndarray, y: np.ndarray, n_classes: int, cfg: HeadConfig, epochs: int,
                     seed: int) -> PooledHead:
    """Final model: train on every clip for a fixed number of epochs (chosen by cross-validation)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xt = torch.as_tensor(X, dtype=torch.float32)
    head = PooledHead(n_classes, cfg, dim=X.shape[1])
    head.mean.copy_(xt.mean(0))
    head.std.copy_(torch.clamp(xt.std(0), min=1e-4))
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = torch.tensor(len(y) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    yt = torch.as_tensor(y, dtype=torch.long)
    for _ in range(epochs):
        head.train()
        order = rng.permutation(len(yt))
        for i in range(0, len(order), cfg.batch_size):
            idx = order[i: i + cfg.batch_size]
            opt.zero_grad()
            loss_fn(head(xt[idx]), yt[idx]).backward()
            opt.step()
    return head.eval()


# ---------------------------------------------------------------------------
# Serving: an exported artifact (truncated backbone + head), used by backend/app/inference.py
# ---------------------------------------------------------------------------

class Wav2Vec2Predictor:
    """Loads an exported folder: backbone/ (truncated at the chosen layer), head.pt, label_map.json,
    preprocess_config.json. Input: a waveform already passed through ml/src/preprocessing.prepare_waveform."""

    def __init__(self, folder):
        import json
        from pathlib import Path

        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        folder = Path(folder)
        self.labels = json.loads((folder / "label_map.json").read_text(encoding="utf-8"))["labels"]
        self.config = json.loads((folder / "preprocess_config.json").read_text(encoding="utf-8"))
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(folder / "backbone")
        self.backbone = Wav2Vec2Model.from_pretrained(folder / "backbone").eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        if self.backbone.config.num_hidden_layers != self.config["layer"]:
            raise RuntimeError("backbone depth does not match preprocess_config.json layer")
        state = torch.load(folder / "head.pt", map_location="cpu", weights_only=True)
        self.head = PooledHead(len(self.labels), HeadConfig(), dim=state["linear.weight"].shape[1])
        self.head.load_state_dict(state)
        self.head.eval()
        self.temperature = float(self.config.get("temperature", 1.0))

    @torch.no_grad()
    def predict_proba(self, y: np.ndarray) -> dict[str, float]:
        inputs = self.fe(y, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        hidden = self.backbone(inputs.input_values).last_hidden_state[0]   # output of the chosen layer
        logits = self.head(hidden.mean(dim=0, keepdim=True))[0] / self.temperature
        return dict(zip(self.labels, torch.softmax(logits, dim=0).tolist()))
