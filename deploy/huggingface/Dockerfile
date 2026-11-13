# Hugging Face Space (Docker SDK) for the API. Upload this file and README.md from
# deploy/huggingface/ to the Space. The build clones the project from GitHub and downloads
# the model files from a GitHub Release, then runs the same deploy/install.sh as ./Dockerfile.
# Edit the four ARG defaults below before uploading (or set them as Space variables).
FROM python:3.14-slim

ARG REPO_URL=https://github.com/shikha-code404/emotion-speech-insights.git
ARG GIT_REF=main
ARG SER_MODEL=ser-w2v2-v2
ARG ARTIFACTS_URL=https://github.com/shikha-code404/emotion-speech-insights/releases/download/models-v2/ser-artifacts-ser-w2v2-v2.tar.gz

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NUMBA_CACHE_DIR=/tmp/numba \
    MODEL_DIR=/app/ml/artifacts/${SER_MODEL} \
    SURVEY_DIR=/app/ml/artifacts/survey-v1 \
    PORT=7860

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 user

RUN git clone --depth 1 --branch "$GIT_REF" "$REPO_URL" /app \
 && rm -rf /app/.git
WORKDIR /app
RUN SER_MODEL="$SER_MODEL" ARTIFACTS_URL="$ARTIFACTS_URL" sh deploy/install.sh /app \
 && rm -rf /tmp/numba \
 && chown -R user:user /app

USER user
EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
