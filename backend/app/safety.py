"""Safety utilities: ethical language checks and audio ephemeral handling.

Rule 4 — two modules, two wording rules:
  * Speech module: predicts "vocal emotion in this clip". No diagnosis,
    depression, anxiety, stress or risk-score wording in speech outputs.
  * Survey module: may report DASS-21 Depression/Anxiety/Stress screening
    scores and bands, always as a self-report screening result, never a
    diagnosis. The two modules' results are never combined.
Rule 5 — Audio is deleted after inference by default.
"""

from __future__ import annotations

# Single source of truth for the survey wording lives in ml/src (Rule 3).
from ml.src.survey.scoring import SCREENING_NOTICE as SURVEY_NOTICE  # noqa: F401

SPEECH_NOTICE = (
    "This prediction reflects an estimate of the vocal emotion expressed "
    "in this audio clip. It is not a diagnosis, mental-health assessment, "
    "or measure of the speaker's overall emotional state."
)

# Backwards-compatible name for the speech notice.
NOTICE = SPEECH_NOTICE


def compute_uncertainty(scores: dict[str, float]) -> str:
    """Classify prediction uncertainty as low / moderate / high.

    Uses two signals:
    1. Max probability — the higher the more confident.
    2. Top-two margin — gap between 1st and 2nd highest scores.

    Returns:
        "low", "moderate", or "high".
    """
    sorted_scores = sorted(scores.values(), reverse=True)
    max_prob = sorted_scores[0]
    margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else max_prob

    if max_prob >= 0.7 and margin >= 0.25:
        return "low"
    elif max_prob >= 0.4 and margin >= 0.10:
        return "moderate"
    else:
        return "high"

