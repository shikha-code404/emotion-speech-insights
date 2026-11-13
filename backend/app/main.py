"""FastAPI app: speech-emotion and DASS-21 questionnaire endpoints.

The two modules are independent: no endpoint combines their results.

Run:  uvicorn backend.app.main:app --reload --port 8000   (from the repo root)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app import inference, survey
from backend.app.audio import ffmpeg_path
from backend.app.config import settings
from backend.app.errors import (
    ConsentRequired,
    FileTooLarge,
    InvalidRequest,
    ModelNotLoaded,
    SERError,
    ser_error_handler,
)
from backend.app.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    SurveyQuestionsResponse,
    SurveyScoreRequest,
    SurveyScoreResponse,
)

SURVEY_DIR = Path(os.getenv(
    "SURVEY_DIR", str(Path(__file__).resolve().parents[2] / "ml" / "artifacts" / "survey-v1")))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    inference.load_model()
    survey.load_questions(SURVEY_DIR / "questions.json")
    if ffmpeg_path() is None:
        print("WARNING: ffmpeg not found (PATH, FFMPEG_PATH or winget); only WAV uploads can be decoded.")
    else:
        print(f"ffmpeg: {ffmpeg_path()}")
    yield


app = FastAPI(title="Vocal emotion & DASS-21 screening API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_exception_handler(SERError, ser_error_handler)


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    loc = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    return await ser_error_handler(request, InvalidRequest(f"{loc}: {first.get('msg', 'invalid request')}"))


ERRORS = {400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse},
          422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}}


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=inference.get_model() is not None,
                          survey_loaded=survey.is_loaded())


@app.get("/api/model-info", response_model=ModelInfoResponse, responses=ERRORS)
def model_info() -> ModelInfoResponse:
    model = inference.get_model()
    if model is None:
        raise ModelNotLoaded()
    m = model.meta
    return ModelInfoResponse(
        version=m["version"], model_family=m["model_family"], labels=m["labels"], corpora=m["corpora"],
        metrics=m["metrics"], limitations=m["limitations"], survey=survey.survey_info(),
    )


@app.post("/api/analyze", response_model=PredictionResponse, responses=ERRORS)
async def analyze(audio: UploadFile = File(...), consent: str = Form("false")) -> PredictionResponse:
    if consent.strip().lower() != "true":
        raise ConsentRequired()
    data = await audio.read(settings.max_upload_bytes + 1)
    await audio.close()
    if len(data) > settings.max_upload_bytes:
        raise FileTooLarge(settings.MAX_UPLOAD_MB)
    try:
        return inference.analyze_bytes(data, audio.filename or "", audio.content_type)
    finally:
        del data  # Rule 5: drop the raw upload as soon as inference ends.


@app.get("/api/survey/questions", response_model=SurveyQuestionsResponse, responses=ERRORS)
def survey_questions() -> SurveyQuestionsResponse:
    return survey.questions()


@app.post("/api/survey/score", response_model=SurveyScoreResponse, responses=ERRORS)
def survey_score(body: SurveyScoreRequest) -> SurveyScoreResponse:
    if not body.consent:
        raise ConsentRequired()
    return survey.score(body.answers)
