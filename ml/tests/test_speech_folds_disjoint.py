"""Every speech experiment's folds are speaker-disjoint (blueprint 11, Rule 1), on the real manifest."""

import numpy as np
import pandas as pd
import pytest

from ml.src import splits
from ml.src.train_speech import MANIFEST, load_manifest

pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="audio manifest not present")


def _experiments():
    m = pd.read_csv(MANIFEST)
    emodb = m[m.corpus == "Emo-DB"].reset_index(drop=True)
    savee = m[m.corpus == "SAVEE"].reset_index(drop=True)
    both = load_manifest()
    return {
        "A": (emodb, splits.leave_one_speaker_out(splits.speaker_groups(emodb))),
        "B": (savee, splits.leave_one_speaker_out(splits.speaker_groups(savee))),
        "C": (both, splits.leave_one_corpus_out(both)),
        "deployed": (both, splits.leave_one_speaker_out(splits.speaker_groups(both))),
    }


@pytest.mark.parametrize("exp", ["A", "B", "C", "deployed"])
def test_outer_folds_disjoint_and_complete(exp):
    m, folds = _experiments()[exp]
    groups = splits.speaker_groups(m)
    for f in folds:
        assert not set(groups[f.train_idx]) & set(groups[f.test_idx]), (exp, f.index)
    assert np.array_equal(np.sort(np.concatenate([f.test_idx for f in folds])), np.arange(len(m)))


@pytest.mark.parametrize("exp", ["A", "B", "C", "deployed"])
def test_inner_splits_never_touch_the_test_speaker(exp):
    m, folds = _experiments()[exp]
    groups = splits.speaker_groups(m)
    for f in folds:
        tr_groups = groups[f.train_idx]
        test_speakers = set(groups[f.test_idx])
        val = splits.validation_speaker(tr_groups, f.index)          # CNN / Wav2Vec early stopping
        assert val not in test_speakers and val in set(tr_groups)
        cv = splits.inner_cv(len(np.unique(tr_groups)))               # classical tuning
        for itr, iva in cv.split(np.zeros(len(tr_groups)), groups=tr_groups):
            assert not set(tr_groups[itr]) & set(tr_groups[iva])


def test_savee_duplicate_tree_excluded_and_no_duplicate_audio():
    m = pd.read_csv(MANIFEST)
    savee = m[m.corpus == "SAVEE"]
    assert len(savee) == 480
    assert savee["filepath"].str.contains("/AudioData/AudioData/").all()   # one tree only
    assert m["file_hash"].is_unique
