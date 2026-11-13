"""Recompute DASS-21 scoring for every row of the (synthetic) survey dataset
and require an exact match with the workbook's stored columns.

The workbook is git-ignored raw data; these tests skip if it is absent.
"""

import pytest

from ml.src.survey.load import DEFAULT_PATH, load_survey
from ml.src.survey.scoring import SUBSCALES, band, dominant_state, subscale_scores

pytestmark = pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="survey workbook not present")


@pytest.fixture(scope="module")
def recomputed():
    df = load_survey()
    rows = []
    for r in df.itertuples(index=False):
        scores = subscale_scores({i: getattr(r, f"q{i}") for i in range(1, 22)})
        dom = dominant_state(scores)
        rows.append({
            "scores": scores,
            "bands": {s: band(s, scores[s]) for s in SUBSCALES},
            "state": dom.state,
            "tied": dom.tied,
            "severity_level": "Normal" if dom.state == "Normal" else band(dom.state, scores[dom.state]),
        })
    return df, rows


def test_dataset_has_1200_rows(recomputed):
    df, _ = recomputed
    assert len(df) == 1200


@pytest.mark.parametrize("subscale", SUBSCALES)
def test_scores_match_exactly(recomputed, subscale):
    df, rows = recomputed
    stored = df[f"{subscale.lower()}_score"].tolist()
    mismatches = [i for i, (row, s) in enumerate(zip(rows, stored)) if row["scores"][subscale] != s]
    assert mismatches == [], f"{len(mismatches)} {subscale} score mismatches, first rows {mismatches[:5]}"


@pytest.mark.parametrize("subscale", SUBSCALES)
def test_severity_bands_match_exactly(recomputed, subscale):
    df, rows = recomputed
    stored = df[f"{subscale.lower()}_severity"].tolist()
    mismatches = [i for i, (row, s) in enumerate(zip(rows, stored)) if row["bands"][subscale] != s]
    assert mismatches == [], f"{len(mismatches)} {subscale} band mismatches, first rows {mismatches[:5]}"


def test_emotional_state_matches_exactly(recomputed):
    df, rows = recomputed
    mismatches = [i for i, (row, s) in enumerate(zip(rows, df["emotional_state"])) if row["state"] != s]
    assert mismatches == [], f"{len(mismatches)} Emotional_State mismatches, first rows {mismatches[:5]}"


def test_severity_level_matches_exactly(recomputed):
    df, rows = recomputed
    mismatches = [i for i, (row, s) in enumerate(zip(rows, df["severity_level"])) if row["severity_level"] != s]
    assert mismatches == [], f"{len(mismatches)} Severity_Level mismatches"


def test_no_ties_in_dataset(recomputed):
    """The Notes sheet says the generator guarantees exactly one dominant
    subscale; the tie-break rule therefore never fires on this data."""
    _, rows = recomputed
    tied = [i for i, row in enumerate(rows) if row["tied"]]
    assert tied == [], f"{len(tied)} rows share the top band, first rows {tied[:5]}"
