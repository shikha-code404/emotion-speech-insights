"""Log-Mel CNN for speech emotion (speech module, experimental; not deployed).

Input pipeline (Rule 3: shared with the rest of ml/src):
  ml/src/preprocessing.load_audio (16 kHz mono)
  -> ml/src/preprocessing.prepare_waveform (trim silence, peak-normalise)
  -> log_mel(): 80-band log-Mel spectrogram, 25 ms window, 10 ms hop
  -> model input 1 x 80 x time

Variable length: each clip's spectrogram is computed on its own and then
zero-padded to the longest clip in the batch. Clips are never truncated.
The model receives the true lengths, zeroes padded frames after every block
and pools over valid frames only, so in eval mode a clip's output is the same
whether it is scored alone or in a padded batch (tested).

Normalisation: per-mel-band mean/std fitted on the fold's training clips only
(Rule 2) and stored as model buffers.

Augmentation (training batches only): mild time shift (silence added before/
after), small gain change, light Gaussian background noise. No pitch shift.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400       # 25 ms
HOP = 160         # 10 ms

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=N_FFT, hop_length=HOP,
    f_min=20.0, f_max=SAMPLE_RATE / 2, n_mels=N_MELS, power=2.0,
)


def log_mel(y: np.ndarray) -> torch.Tensor:
    """(80, frames) natural-log Mel power spectrogram of a prepared 16 kHz waveform."""
    with torch.no_grad():
        return torch.log(_mel(torch.as_tensor(y, dtype=torch.float32)) + 1e-6)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CNNConfig:
    n_mels: int = N_MELS
    channels: tuple[int, int, int] = (16, 32, 64)   # ~24k params; (32,64,128) was 2.2x slower on CPU
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    max_epochs: int = 60
    patience: int = 10                  # epochs without val macro-F1 improvement
    aug_prob: float = 0.5               # independent probability for each augmentation
    noise_snr_db: tuple[float, float] = (20.0, 35.0)
    gain_db: float = 3.0                # uniform in [-gain_db, +gain_db]
    max_shift_s: float = 0.1            # up to 100 ms of silence before and after
    seed: int = 42
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Augmentation (waveform domain, training only)
# ---------------------------------------------------------------------------

def augment(y: np.ndarray, rng: np.random.Generator, cfg: CNNConfig, sr: int = SAMPLE_RATE) -> np.ndarray:
    y = y.astype(np.float32, copy=True)
    if rng.random() < cfg.aug_prob:  # mild time shift; never cuts audio
        max_n = int(cfg.max_shift_s * sr)
        pre, post = int(rng.integers(0, max_n + 1)), int(rng.integers(0, max_n + 1))
        y = np.concatenate([np.zeros(pre, np.float32), y, np.zeros(post, np.float32)])
    if rng.random() < cfg.aug_prob:  # small gain change
        y *= np.float32(10 ** (rng.uniform(-cfg.gain_db, cfg.gain_db) / 20))
    if rng.random() < cfg.aug_prob:  # light background noise
        rms = float(np.sqrt(np.mean(y ** 2))) or 1e-4
        snr = rng.uniform(*cfg.noise_snr_db)
        y += rng.normal(0.0, rms / 10 ** (snr / 20), size=y.shape).astype(np.float32)
    return y


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _time_mask(lengths: torch.Tensor, t: int) -> torch.Tensor:
    return (torch.arange(t, device=lengths.device)[None, :] < lengths[:, None])  # (B, T)


class ConvBlock(nn.Module):
    """conv -> batch norm -> ReLU -> (2x2 max-pool) -> dropout, then zero padded frames."""

    def __init__(self, c_in: int, c_out: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = self.drop(self.pool(F.relu(self.bn(self.conv(x)))))
        lengths = torch.clamp(lengths // 2, min=1)
        return x * _time_mask(lengths, x.shape[-1])[:, None, None, :], lengths


class LogMelCNN(nn.Module):
    def __init__(self, n_classes: int, cfg: CNNConfig):
        super().__init__()
        c1, c2, c3 = cfg.channels
        self.register_buffer("band_mean", torch.zeros(cfg.n_mels))
        self.register_buffer("band_std", torch.ones(cfg.n_mels))
        self.blocks = nn.ModuleList([
            ConvBlock(1, c1, cfg.dropout), ConvBlock(c1, c2, cfg.dropout), ConvBlock(c2, c3, cfg.dropout),
        ])
        self.head = nn.Linear(2 * c3, n_classes)  # global mean + max pooling

    def set_normalisation(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.band_mean.copy_(mean)
        self.band_std.copy_(torch.clamp(std, min=1e-3))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, 80, T) raw log-Mel, zero-padded beyond `lengths` frames."""
        x = (x - self.band_mean[None, None, :, None]) / self.band_std[None, None, :, None]
        x = x * _time_mask(lengths, x.shape[-1])[:, None, None, :]
        for block in self.blocks:
            x, lengths = block(x, lengths)
        x = x.mean(dim=2)                                        # pool frequency -> (B, C, T')
        mask = _time_mask(lengths, x.shape[-1])[:, None, :]
        mean = (x * mask).sum(-1) / lengths[:, None]
        mx = x.masked_fill(~mask, float("-inf")).amax(-1)
        return self.head(torch.cat([mean, mx], dim=1))


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def pad_batch(specs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad (80, T_i) spectrograms to (B, 1, 80, max T) and return true lengths."""
    lengths = torch.tensor([s.shape[-1] for s in specs], dtype=torch.long)
    out = torch.zeros(len(specs), 1, specs[0].shape[0], int(lengths.max()))
    for i, s in enumerate(specs):
        out[i, 0, :, : s.shape[-1]] = s
    return out, lengths


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict_proba(model: LogMelCNN, specs: Sequence[torch.Tensor], device: torch.device,
                  batch_size: int = 32) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(specs), batch_size):
        x, lengths = pad_batch(specs[i: i + batch_size])
        out.append(F.softmax(model(x.to(device), lengths.to(device)), dim=1).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------
# Training (one outer fold)
# ---------------------------------------------------------------------------

def _bucketed_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffle, then sort within chunks of 8 batches by length so each batch pads little.
    Batch order is shuffled again, so training still sees a random mix."""
    order = rng.permutation(len(lengths))
    chunk = batch_size * 8
    batches = []
    for i in range(0, len(order), chunk):
        part = order[i: i + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[j: j + batch_size] for j in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


@dataclass
class FoldResult:
    model: LogMelCNN
    best_epoch: int
    best_val_macro_f1: float
    epochs_run: int
    history: list[dict]
    seconds: float


def train_fold(
    train_wavs: Sequence[np.ndarray],
    train_y: np.ndarray,              # int class ids
    val_specs: Sequence[torch.Tensor],
    val_y: np.ndarray,
    n_classes: int,
    cfg: CNNConfig,
    seed: int,
    device: Optional[torch.device] = None,
    log: Callable[[str], None] = print,
) -> FoldResult:
    from sklearn.metrics import f1_score

    device = device or device_auto()
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    t0 = time.time()

    # Normalisation and class weights from this fold's training clips only (Rule 2).
    clean = [log_mel(w) for w in train_wavs]
    frames = torch.cat(clean, dim=1)
    model = LogMelCNN(n_classes, cfg).to(device)
    model.set_normalisation(frames.mean(dim=1).to(device), frames.std(dim=1).to(device))
    counts = np.bincount(train_y, minlength=n_classes).astype(np.float64)
    weights = torch.tensor(len(train_y) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    lengths_s = np.array([len(w) for w in train_wavs])
    best_f1, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total = 0.0
        batches = _bucketed_batches(lengths_s, cfg.batch_size, rng)
        for idx in batches:
            if len(idx) < 2:  # batch norm needs >1 sample
                continue
            x, lengths = pad_batch([log_mel(augment(train_wavs[j], rng, cfg)) for j in idx])
            target = torch.as_tensor(train_y[idx], dtype=torch.long, device=device)
            opt.zero_grad()
            loss = loss_fn(model(x.to(device), lengths.to(device)), target)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        val_pred = predict_proba(model, val_specs, device).argmax(1)
        val_f1 = float(f1_score(val_y, val_pred, average="macro", labels=np.unique(val_y), zero_division=0))
        history.append({"epoch": epoch, "train_loss": total / len(train_wavs), "val_macro_f1": val_f1})
        if val_f1 > best_f1:
            best_f1, best_epoch, best_state = val_f1, epoch, copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= cfg.patience:
            break
    model.load_state_dict(best_state)
    log(f"    best epoch {best_epoch}/{len(history)}  val macro-F1 {best_f1:.3f}  ({time.time() - t0:.0f}s)")
    return FoldResult(model, best_epoch, best_f1, len(history), history, time.time() - t0)
