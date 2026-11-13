"""Unit tests for DASS-21 scoring rules (no dataset needed)."""

import pytest

from ml.src.survey.scoring import (
    BAND_LOWER_CUTOFFS,
    BANDS,
    SUBSCALE_ITEMS,
    band,
    dominant_state,
    score_dass21,
    subscale_scores,
)


def _answers(default=0, **overrides):
    a = {i: default for i in range(1, 22)}
    for k, v in overrides.items():
        a[int(k[1:])] = v
    return a


def test_items_partition_1_to_21():
    all_items = sorted(i for items in SUBSCALE_ITEMS.values() for i in items)
    assert all_items == list(range(1, 22))


def test_scores_are_doubled_sums():
    a = _answers(q3=3, q5=1, q2=2, q1=1)
    assert subscale_scores(a) == {"Depression": 8, "Anxiety": 4, "Stress": 2}
    assert subscale_scores(_answers(3)) == {"Depression": 42, "Anxiety": 42, "Stress": 42}


@pytest.mark.parametrize("subscale,boundaries", [
    ("Depression", [(9, "Normal"), (10, "Mild"), (13, "Mild"), (14, "Moderate"),
                    (20, "Moderate"), (21, "Severe"), (27, "Severe"), (28, "Extremely Severe")]),
    ("Anxiety", [(7, "Normal"), (8, "Mild"), (9, "Mild"), (10, "Moderate"),
                 (14, "Moderate"), (15, "Severe"), (19, "Severe"), (20, "Extremely Severe")]),
    ("Stress", [(14, "Normal"), (15, "Mild"), (18, "Mild"), (19, "Moderate"),
                (25, "Moderate"), (26, "Severe"), (33, "Severe"), (34, "Extremely Severe")]),
])
def test_band_boundaries(subscale, boundaries):
    for score, expected in boundaries:
        assert band(subscale, score) == expected, (subscale, score)
    assert band(subscale, 0) == "Normal"
    assert band(subscale, 42) == "Extremely Severe"
    assert len(BAND_LOWER_CUTOFFS[subscale]) == len(BANDS)


@pytest.mark.parametrize("bad", [
    {i: 0 for i in range(1, 21)},                 # missing q21
    {**{i: 0 for i in range(1, 22)}, 5: 4},        # out of range
    {**{i: 0 for i in range(1, 22)}, 5: -1},
    {**{i: 0 for i in range(1, 22)}, 5: 1.5},      # not int
    {**{i: 0 for i in range(1, 22)}, 22: 0},       # unknown item
])
def test_invalid_answers_rejected(bad):
    with pytest.raises(ValueError):
        subscale_scores(bad)


def test_all_normal_is_normal():
    assert dominant_state({"Depression": 9, "Anxiety": 7, "Stress": 14}).state == "Normal"


def test_highest_band_wins():
    # Stress has the largest raw score but only Mild; Anxiety is Moderate.
    d = dominant_state({"Depression": 0, "Anxiety": 10, "Stress": 18})
    assert d.state == "Anxiety" and d.tied == ()


def test_tie_broken_by_margin_into_band():
    # Both Moderate: Depression 16 is 2 above 14; Stress 24 is 5 above 19.
    d = dominant_state({"Depression": 16, "Anxiety": 0, "Stress": 24})
    assert d.state == "Stress"
    assert d.tied == ("Depression", "Stress")


def test_tie_broken_by_fixed_order_when_margins_equal():
    # All Extremely Severe, each exactly at its cutoff.
    d = dominant_state({"Depression": 28, "Anxiety": 20, "Stress": 34})
    assert d.state == "Depression"
    assert d.tied == ("Depression", "Anxiety", "Stress")


def test_result_carries_screening_notice():
    r = score_dass21(_answers(0))
    assert "not a diagnosis" in r["notice"]
    assert r["dominant_state"] == "Normal"
