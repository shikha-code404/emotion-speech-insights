"""Custom error types and exception handlers for the SER API."""

from __future__ import annotations

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("backend.errors")


class SERError(Exception):
    """Base exception for SER API errors."""

    def __init__(self, status_code: int, error_code: str, message: str) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.message = message
        super().__init__(message)


class ConsentRequired(SERError):
    def __init__(self) -> None:
        super().__init__(400, "consent_required", "Consent is required: send consent=true.")


class FileTooLarge(SERError):
    def __init__(self, max_mb: int) -> None:
        super().__init__(413, "file_too_large", f"Upload exceeds {max_mb} MB limit.")


class UnsupportedFormat(SERError):
    def __init__(self, ext: str) -> None:
        super().__init__(
            415,
            "unsupported_format",
            f"Audio format '{ext}' is not supported. Accepted: WAV, MP3, WebM, OGG, M4A.",
        )


class AudioTooShort(SERError):
    def __init__(self, duration: float, min_s: float) -> None:
        super().__init__(
            422,
            "audio_too_short",
            f"Audio is {duration:.2f}s — minimum is {min_s}s.",
        )


class AudioTooLong(SERError):
    def __init__(self, duration: float, max_s: float) -> None:
        super().__init__(
            422,
            "audio_too_long",
            f"Audio is {duration:.2f}s — maximum is {max_s}s.",
        )


class AudioSilent(SERError):
    def __init__(self) -> None:
        super().__init__(422, "audio_silent", "Audio appears to be silent or near-silent.")


class DecodeFailed(SERError):
    def __init__(self, detail: str = "") -> None:
        # The detail (ffmpeg stderr, temp paths) is logged server-side only, never sent to clients.
        self.detail = detail
        if detail:
            logger.warning("decode_failed: %s", detail)
        super().__init__(422, "decode_failed", "Could not decode audio file.")


class ModelNotLoaded(SERError):
    def __init__(self) -> None:
        super().__init__(503, "model_not_loaded", "Model is not available. Please try again later.")


async def ser_error_handler(_request: Request, exc: SERError) -> JSONResponse:
    """Translate SERError into a JSON response with error_code and message."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": exc.error_code, "message": exc.message},
    )



class InvalidAnswers(SERError):
    def __init__(self, detail: str) -> None:
        super().__init__(422, "invalid_answers", f"Questionnaire answers are incomplete or invalid: {detail}")


class SurveyNotLoaded(SERError):
    def __init__(self) -> None:
        super().__init__(503, "survey_not_loaded", "The questionnaire is not available. Please try again later.")


class InvalidRequest(SERError):
    def __init__(self, detail: str) -> None:
        super().__init__(422, "invalid_request", detail)
