"""DASS-21 self-report screening: pure scoring functions, no ML.

Results are self-report screening scores and bands, never a diagnosis
(CLAUDE.md Rule 4, survey module).

Scoring
-------
Each item is answered 0-3. Each subscale sums its 7 items and multiplies
by 2, so scores are 0-42 and comparable to the full DASS-42.

Dominant state
--------------
1. If all three subscales are "Normal", the dominant state is "Normal".
2. Otherwise it is the subscale with the highest severity band.
3. Tie-break when two or three subscales share the highest band (this
   matters for real user input; see the dataset test for whether the
   synthetic labels ever hit it):
   a. the subscale whose score is furthest above the lower cutoff of
      that band, in score points (the band's lower bound is used, so this
      also works for the open-ended "Extremely Severe" band);
   b. if still tied, the fixed order Depression > Anxiety > Stress.
   ``dominant_state`` also returns every subscale that shared the top band
   so a UI can show the tie instead of hiding it.
"""

from __future__ import annotations

from dataclasses import dataclass

SUBSCALE_ITEMS: dict[str, tuple[int, ...]] = {
    "Depression": (3, 5, 10, 13, 16, 17, 21),
    "Anxiety": (2, 4, 7, 9, 15, 19, 20),
    "Stress": (1, 6, 8, 11, 12, 14, 18),
}
SUBSCALES: tuple[str, ...] = ("Depression", "Anxiety", "Stress")  # tie-break order

BANDS: tuple[str, ...] = ("Normal", "Mild", "Moderate", "Severe", "Extremely Severe")

# Lower cutoff (inclusive) of each band, on the x2 scale, in BANDS order.
BAND_LOWER_CUTOFFS: dict[str, tuple[int, ...]] = {
    "Depression": (0, 10, 14, 21, 28),
    "Anxiety": (0, 8, 10, 15, 20),
    "Stress": (0, 15, 19, 26, 34),
}

NORMAL_STATE = "Normal"
MIN_ANSWER, MAX_ANSWER = 0, 3
N_ITEMS = 21

SCREENING_NOTICE = (
    "This is a self-report screening result from the DASS-21 questionnaire. "
    "It is not a diagnosis and is no substitute for a qualified professional."
)


def _validate_answers(answers: dict[int, int]) -> None:
    missing = set(range(1, N_ITEMS + 1)) - set(answers)
    if missing:
        raise ValueError(f"missing answers for items {sorted(missing)}")
    extra = set(answers) - set(range(1, N_ITEMS + 1))
    if extra:
        raise ValueError(f"unknown items {sorted(extra)}")
    for item, value in answers.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"item {item}: answer must be an int, got {value!r}")
        if not MIN_ANSWER <= value <= MAX_ANSWER:
            raise ValueError(f"item {item}: answer {value} outside 0-3")


def subscale_scores(answers: dict[int, int]) -> dict[str, int]:
    """Return {subscale: sum(items) * 2} for answers keyed by item number 1-21."""
    _validate_answers(answers)
    return {
        name: 2 * sum(answers[i] for i in items)
        for name, items in SUBSCALE_ITEMS.items()
    }


def band_index(subscale: str, score: int) -> int:
    """Index into BANDS for a x2-scaled subscale score."""
    if score < 0:
        raise ValueError(f"score must be >= 0, got {score}")
    cutoffs = BAND_LOWER_CUTOFFS[subscale]
    return max(i for i, lower in enumerate(cutoffs) if score >= lower)


def band(subscale: str, score: int) -> str:
    return BANDS[band_index(subscale, score)]


@dataclass(frozen=True)
class DominantState:
    state: str                    # "Normal" or a subscale name
    tied: tuple[str, ...]         # subscales sharing the top band (len > 1 means a tie)


def dominant_state(scores: dict[str, int]) -> DominantState:
    """Apply the dominant-state rule documented in the module docstring."""
    idx = {s: band_index(s, scores[s]) for s in SUBSCALES}
    top = max(idx.values())
    if top == 0:
        return DominantState(NORMAL_STATE, ())
    tied = tuple(s for s in SUBSCALES if idx[s] == top)

    def margin(s: str) -> int:
        return scores[s] - BAND_LOWER_CUTOFFS[s][top]

    # max() keeps the first maximum, so SUBSCALES order is the final tie-break.
    winner = max(tied, key=margin)
    return DominantState(winner, tied if len(tied) > 1 else ())


def score_dass21(answers: dict[int, int]) -> dict:
    """Full screening result for one respondent."""
    scores = subscale_scores(answers)
    dom = dominant_state(scores)
    return {
        "scores": scores,
        "bands": {s: band(s, scores[s]) for s in SUBSCALES},
        "dominant_state": dom.state,
        "tied_subscales": list(dom.tied),
        "notice": SCREENING_NOTICE,
    }
