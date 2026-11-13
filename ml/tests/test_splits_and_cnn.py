"""Tests for shared speaker folds and the log-Mel CNN."""

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.model_selection import LeaveOneGroupOut

from ml.src.evaluate import fold_scores
from ml.src.models.cnn import CNNConfig, LogMelCNN, augment, log_mel, pad_batch, predict_proba
from ml.src.splits import leave_one_speaker_out, speaker_groups, validation_speaker


def _manifest():
    rows = [("Emo-DB", s, e) for s in ("03", "08", "09") for e in ("anger", "neutral") for _ in range(2)]
    rows += [("SAVEE", "DC", e) for e in ("anger", "neutral")]
    return pd.DataFrame(rows, columns=["corpus", "speaker_id", "emotion"])


def test_folds_match_phase2_leave_one_group_out():
    groups = speaker_groups(_manifest())
    ours = leave_one_speaker_out(groups)
    ref = list(LeaveOneGroupOut().split(np.zeros(len(groups)), groups=groups))
    assert len(ours) == len(ref) == 4
    for f, (tr, te) in zip(ours, ref):
        assert np.array_equal(f.train_idx, tr) and np.array_equal(f.test_idx, te)


def test_folds_are_speaker_disjoint_and_cover_every_clip():
    groups = speaker_groups(_manifest())
    folds = leave_one_speaker_out(groups)
    for f in folds:
        assert not set(groups[f.train_idx]) & set(groups[f.test_idx])
    assert sorted(np.concatenate([f.test_idx for f in folds])) == list(range(len(groups)))


def test_same_speaker_id_in_two_corpora_is_two_speakers():
    m = pd.DataFrame({"corpus": ["Emo-DB", "SAVEE"], "speaker_id": ["03", "03"], "emotion": ["a", "a"]})
    assert len(set(speaker_groups(m))) == 2


def test_validation_speaker_is_a_training_speaker():
    groups = speaker_groups(_manifest())
    for f in leave_one_speaker_out(groups):
        v = validation_speaker(groups[f.train_idx], f.index)
        assert v in set(groups[f.train_idx]) and v != f.test_speaker


def test_log_mel_shape_is_80_by_time():
    s = log_mel(np.random.default_rng(0).normal(size=16000).astype(np.float32))
    assert s.shape == (80, 101)


def test_padding_does_not_change_eval_output():
    torch.manual_seed(0)
    model = LogMelCNN(7, CNNConfig()).eval()
    rng = np.random.default_rng(0)
    short = log_mel(rng.normal(size=16000).astype(np.float32))       # 1 s
    long = log_mel(rng.normal(size=16000 * 4).astype(np.float32))    # 4 s
    alone = predict_proba(model, [short], torch.device("cpu"))
    batched = predict_proba(model, [short, long], torch.device("cpu"))[:1]
    np.testing.assert_allclose(alone, batched, atol=1e-5)


def test_pad_batch_keeps_full_length():
    x, lengths = pad_batch([torch.ones(80, 50), torch.ones(80, 120)])
    assert x.shape == (2, 1, 80, 120) and lengths.tolist() == [50, 120]


def test_augment_never_shortens_and_is_seeded():
    y = np.sin(np.linspace(0, 100, 16000)).astype(np.float32)
    cfg = CNNConfig(aug_prob=1.0)
    a = augment(y, np.random.default_rng(1), cfg)
    b = augment(y, np.random.default_rng(1), cfg)
    assert len(a) >= len(y) and len(a) <= len(y) + 2 * int(0.1 * 16000)
    np.testing.assert_array_equal(a, b)


def test_fold_scores_ignores_classes_absent_for_that_speaker():
    s = fold_scores(["a", "a", "b"], ["a", "a", "b"], labels=["a", "b", "c"])
    assert s["macro_f1"] == pytest.approx(1.0) and s["ua"] == pytest.approx(1.0)
