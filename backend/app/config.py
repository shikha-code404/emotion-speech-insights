"""Configuration for the SER FastAPI backend, loaded from environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List


class Settings:
    """Application settings loaded from environment variables with sensible defaults."""

    MODEL_DIR: str = os.getenv(
        "MODEL_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "ml" / "artifacts" / "ser-w2v2-v2"),
    )
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    MIN_DURATION_S: float = float(os.getenv("MIN_DURATION_S", "1"))
    MAX_DURATION_S: float = float(os.getenv("MAX_DURATION_S", "30"))
    ALLOWED_ORIGINS: List[str] = [
        o.strip().rstrip("/")
        for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://localhost:3000").split(",")
        if o.strip()
    ]
    if "*" in ALLOWED_ORIGINS:
        # The API takes voice recordings: only named frontends may call it from a browser.
        raise ValueError("ALLOWED_ORIGINS must list exact origins, not '*'")

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024


settings = Settings()

