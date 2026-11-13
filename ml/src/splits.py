"""Speaker-grouped cross-validation folds (Rule 1). Shared by every speech model.

Outer folds are leave-one-speaker-out over the sorted unique speaker groups,
exactly the folds sklearn's LeaveOneGroupOut produced in Phase 2
(ml/src/train_speech.py), so SVM / Random Forest / CNN results are directly
comparable fold by fold.

Inner splits never use the outer test speaker:
  * classical models tune with GroupKFold over the training speakers;
  * the CNN holds out one training speaker for early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


@dataclass(frozen=True)
class Fold:
    index: int
    test_speaker: str
    train_idx: np.ndarray
    test_idx: np.ndarray

    @property
    def n_train(self) -> int:
        return len(self.train_idx)


def speaker_groups(manifest: pd.DataFrame) -> np.ndarray:
    """One group id per clip: "<corpus>:<speaker_id>" (speaker ids repeat across corpora)."""
    return (manifest["corpus"].astype(str) + ":" + manifest["speaker_id"].astype(str)).to_numpy()


def leave_one_speaker_out(groups: np.ndarray) -> list[Fold]:
    groups = np.asarray(groups)
    folds = []
    for k, spk in enumerate(np.unique(groups)):  # np.unique sorts, as LeaveOneGroupOut does
        test = np.flatnonzero(groups == spk)
        train = np.flatnonzero(groups != spk)
        folds.append(Fold(k, str(spk), train, test))
    assert_speaker_disjoint(groups, folds)
    return folds


def assert_speaker_disjoint(groups: np.ndarray, folds: list[Fold]) -> None:
    for f in folds:
        overlap = set(groups[f.train_idx]) & set(groups[f.test_idx])
        if overlap:
            raise AssertionError(f"fold {f.index}: speakers in both train and test: {overlap}")
    covered = np.sort(np.concatenate([f.test_idx for f in folds]))
    if not np.array_equal(covered, np.arange(len(groups))):
        raise AssertionError("every clip must be tested exactly once")


def inner_cv(n_train_speakers: int, max_splits: int = 4) -> GroupKFold:
    """Speaker-grouped inner CV for hyperparameter tuning (training speakers only)."""
    return GroupKFold(n_splits=min(max_splits, n_train_speakers))


def validation_speaker(train_groups: np.ndarray, fold_index: int) -> str:
    """Deterministic early-stopping speaker, chosen from the fold's training speakers."""
    speakers = np.unique(train_groups)
    return str(speakers[fold_index % len(speakers)])


def leave_one_corpus_out(manifest: pd.DataFrame) -> list[Fold]:
    """Cross-corpus folds (experiment C): train on every other corpus, test on one.

    With two corpora this is "train Emo-DB -> test SAVEE" and the reverse.
    Corpora share no speakers, so the folds are speaker-disjoint as well.
    """
    corpus = manifest["corpus"].to_numpy()
    folds = []
    for k, c in enumerate(np.unique(corpus)):
        folds.append(Fold(k, f"test on {c}", np.flatnonzero(corpus != c), np.flatnonzero(corpus == c)))
    assert_speaker_disjoint(speaker_groups(manifest), folds)
    return folds
