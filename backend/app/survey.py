"""DASS-21 self-report screening service.

Scoring is imported from ml/src/survey/scoring.py (Rule 3). Answers are
scored in memory and never stored. Results are a self-report screening
result, never a diagnosis (Rule 4, survey module).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from backend.app.errors import InvalidAnswers, SurveyNotLoaded
from backend.app.schemas import (
    AnswerOption,
    BandRange,
    SubscaleResult,
    SurveyItem,
    SurveyModuleInfo,
    SurveyQuestionsResponse,
    SurveyScoreResponse,
)
from ml.src.survey.scoring import (
    BAND_LOWER_CUTOFFS,
    BANDS,
    N_ITEMS,
    SUBSCALES,
    band_index,
    score_dass21,
)

MAX_SCORE = 42

SURVEY_LIMITATIONS = [
    "The DASS-21 is a self-report screening questionnaire. Its result reflects your own answers about the past week "
    "and is not a diagnosis.",
    "Severity bands use the standard DASS-21 cut-offs, which were developed for screening, not for deciding care.",
    "The dataset used to develop and test this module is fully synthetic (artificially generated respondents), "
    "so it says nothing about how scores are distributed in real people.",
    "Answers are scored in memory and are not stored.",
]

_questions: Optional[dict] = None


def load_questions(path: Path) -> Optional[dict]:
    global _questions
    _questions = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    return _questions


def is_loaded() -> bool:
    return _questions is not None


def survey_info() -> SurveyModuleInfo:
    return SurveyModuleInfo(
        version=(_questions or {}).get("version", "unavailable"),
        instrument="DASS-21 (Depression Anxiety Stress Scales, 21 items)",
        development_data_synthetic=True,
        limitations=SURVEY_LIMITATIONS,
    )


def questions() -> SurveyQuestionsResponse:
    if _questions is None:
        raise SurveyNotLoaded()
    q = _questions
    return SurveyQuestionsResponse(
        version=q["version"],
        instructions=q["instructions"],
        answer_scale=[AnswerOption(**o) for o in q["answer_scale"]],
        # Subscale membership is deliberately not sent to the client.
        items=[SurveyItem(id=i["id"], number=i["number"], text=i["text"]) for i in q["items"]],
    )


def _band_ranges(subscale: str) -> list[BandRange]:
    lows = BAND_LOWER_CUTOFFS[subscale]
    highs = [lo - 1 for lo in lows[1:]] + [MAX_SCORE]
    return [BandRange(band=b, min=lo, max=hi) for b, lo, hi in zip(BANDS, lows, highs)]


def _supportive_message(top_band_index: int) -> Optional[str]:
    if top_band_index == 0:
        return None
    if top_band_index == 1:
        return ("Some of your answers suggest the past week has had its harder moments. Small things like rest, "
                "time outdoors or talking with someone you trust can help.")
    return ("Your answers suggest the past week may have been difficult. You don't have to work through that alone. "
            "Talking with someone you trust, or with a doctor or counsellor, can really help. If you ever feel unsafe, "
            "please contact local emergency services or a crisis line straight away.")


def score(answers: dict[str, int]) -> SurveyScoreResponse:
    expected = {f"q{i}" for i in range(1, N_ITEMS + 1)}
    missing = sorted(expected - set(answers), key=lambda k: int(k[1:]))
    extra = sorted(set(answers) - expected)
    if missing:
        raise InvalidAnswers(f"missing {', '.join(missing)}")
    if extra:
        raise InvalidAnswers(f"unknown items {', '.join(extra)}")
    try:
        result = score_dass21({int(k[1:]): v for k, v in answers.items()})
    except ValueError as e:
        raise InvalidAnswers(str(e)) from e

    scores = result["scores"]
    idx = {s: band_index(s, scores[s]) for s in SUBSCALES}
    top = max(idx.values())
    tied = [s for s in SUBSCALES if idx[s] == top] if top > 0 else []

    return SurveyScoreResponse(
        subscales=[
            SubscaleResult(subscale=s, score=scores[s], band=result["bands"][s], band_index=idx[s],
                           bands=_band_ranges(s))
            for s in SUBSCALES
        ],
        dominant_state=result["dominant_state"],
        tied_states=tied,
        supportive_message=_supportive_message(top),
    )
