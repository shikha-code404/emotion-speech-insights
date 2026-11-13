"""Shared evaluation for every speech model.

Metrics:
  macro-F1              unweighted mean of per-class F1
  unweighted accuracy   (UA, a.k.a. unweighted average recall / balanced accuracy)
Each is reported pooled over all out-of-fold predictions and as the mean and
standard deviation of the per-fold (per held-out speaker) scores; the
standard deviation is the fold-to-fold variation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score


def _ua(y_true, y_pred, labels) -> float:
    # Recall averaged over the classes present in y_true (a held-out speaker may lack some classes).
    present = [l for l in labels if l in set(y_true)]
    return float(recall_score(y_true, y_pred, labels=present, average="macro", zero_division=0))


def fold_scores(y_true, y_pred, labels) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        present = [l for l in labels if l in set(y_true)]
        return {
            "n": int(len(y_true)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
            "ua": _ua(y_true, y_pred, labels),
        }


def summarize(y: np.ndarray, oof: np.ndarray, per_fold: pd.DataFrame, labels: list[str]) -> dict:
    """per_fold: one row per outer fold with columns macro_f1, ua (from fold_scores)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "macro_f1_pooled": float(f1_score(y, oof, labels=labels, average="macro", zero_division=0)),
            "ua_pooled": _ua(y, oof, labels),
            "macro_f1_fold_mean": float(per_fold["macro_f1"].mean()),
            "macro_f1_fold_std": float(per_fold["macro_f1"].std(ddof=1)),
            "ua_fold_mean": float(per_fold["ua"].mean()),
            "ua_fold_std": float(per_fold["ua"].std(ddof=1)),
            "per_class_recall": dict(zip(labels, map(float, recall_score(
                y, oof, labels=labels, average=None, zero_division=0)))),
            "confusion": confusion_matrix(y, oof, labels=labels),
        }
