"""Pydantic schemas for API request validation and response serialization.

frontend/src/types/api.ts mirrors these models; keep them in sync.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from backend.app.safety import SPEECH_NOTICE, SURVEY_NOTICE


# ---------------------------------------------------------------------------
# Speech module
# ---------------------------------------------------------------------------

class EmotionScore(BaseModel):
    """Probability score for a single emotion class."""
    emotion: str
    score: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Response payload for POST /api/analyze."""
    prediction_type: str = "vocal_emotion"
    predicted_emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    scores: List[EmotionScore]
    uncertainty: Literal["low", "moderate", "high"]
    model_version: str
    audio_duration_seconds: float
    notice: str = SPEECH_NOTICE


class HealthResponse(BaseModel):
    """Response payload for GET /api/health."""
    status: str
    model_loaded: bool
    survey_loaded: bool


class SurveyModuleInfo(BaseModel):
    version: str
    instrument: str
    development_data_synthetic: bool
    limitations: List[str]


class ModelInfoResponse(BaseModel):
    """Response payload for GET /api/model-info."""
    version: str
    model_family: str
    labels: List[str]
    corpora: List[str]
    metrics: Dict[str, object]
    limitations: List[str]
    survey: SurveyModuleInfo


# ---------------------------------------------------------------------------
# Survey module (DASS-21 self-report screening)
# ---------------------------------------------------------------------------

class AnswerOption(BaseModel):
    value: int
    label: str


class SurveyItem(BaseModel):
    id: str
    number: int
    text: str


class SurveyQuestionsResponse(BaseModel):
    """Response payload for GET /api/survey/questions."""
    version: str
    instructions: str
    answer_scale: List[AnswerOption]
    items: List[SurveyItem]
    notice: str = SURVEY_NOTICE


class SurveyScoreRequest(BaseModel):
    """Request body for POST /api/survey/score. Answers are not stored."""
    consent: bool = False  # missing counts as not given -> consent_required, same as /api/analyze
    answers: Dict[str, int] = Field(..., description='{"q1": 0..3, ..., "q21": 0..3}')


class BandRange(BaseModel):
    band: str
    min: int
    max: int


class SubscaleResult(BaseModel):
    subscale: str
    score: int = Field(..., ge=0, le=42)
    band: str
    band_index: int = Field(..., ge=0, le=4)
    bands: List[BandRange]


class SurveyScoreResponse(BaseModel):
    """Response payload for POST /api/survey/score."""
    result_type: str = "dass21_self_report_screening"
    subscales: List[SubscaleResult]
    max_score: int = 42
    dominant_state: str
    tied_states: List[str] = Field(
        ..., description="Subscales sharing the highest non-Normal band. More than one entry means a tie.")
    supportive_message: Optional[str] = None
    notice: str = SURVEY_NOTICE


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standardised error envelope."""
    error_code: str
    message: str
