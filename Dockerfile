# API image (FastAPI + speech model + DASS-21 questionnaire). Build from the repo root:
#   docker build --build-arg ARTIFACTS_URL=<release tar.gz> -t voice-wellbeing-api .
# If ml/artifacts/<SER_MODEL> exists locally it is used and ARTIFACTS_URL can be omitted.
FROM python:3.14-slim

ARG SER_MODEL=ser-w2v2-v2
ARG ARTIFACTS_URL=""

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    MODEL_DIR=/app/ml/artifacts/${SER_MODEL} \
    SURVEY_DIR=/app/ml/artifacts/survey-v1 \
    PORT=8000

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 app

WORKDIR /app
# .dockerignore lets in only the API code, the shared ml/src modules and the served model files.
COPY . /app
RUN SER_MODEL="$SER_MODEL" ARTIFACTS_URL="$ARTIFACTS_URL" sh deploy/install.sh /app \
 && rm -rf /tmp/numba \
 && chown -R app:app /app

USER app
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
