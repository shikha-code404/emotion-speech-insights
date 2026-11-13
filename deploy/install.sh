#!/bin/sh
# Installs the API into an image. Shared by ./Dockerfile and deploy/huggingface/Dockerfile.
#   usage: SER_MODEL=<artifact folder> [ARTIFACTS_URL=<tar.gz url>] sh deploy/install.sh <app dir>
set -eu
cd "$1"
: "${SER_MODEL:?set SER_MODEL to the speech artifact folder, e.g. ser-w2v2-v2}"

# 1. Model files: use the build context if it has them, otherwise download the release tarball.
if [ ! -f "ml/artifacts/$SER_MODEL/model_card.json" ] || [ ! -f ml/artifacts/survey-v1/questions.json ]; then
  if [ -z "${ARTIFACTS_URL:-}" ]; then
    echo "ml/artifacts/$SER_MODEL is missing and ARTIFACTS_URL is not set" >&2
    exit 1
  fi
  echo "Downloading model files from $ARTIFACTS_URL"
  python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], 'artifacts.tar.gz')" "$ARTIFACTS_URL"
  tar -xzf artifacts.tar.gz -C .
  rm artifacts.tar.gz
fi

# 2. Keep only what is served: the chosen speech model and the questionnaire items.
#    The lifestyle model is analysis-only and must never be in the image.
rm -f ml/artifacts/survey-v1/lifestyle_model.joblib
find ml/artifacts -mindepth 1 -maxdepth 1 -type d ! -name "$SER_MODEL" ! -name survey-v1 -exec rm -rf {} +

# 3. Python packages: always the light set; CPU-only PyTorch only if the model needs it.
pip install --no-cache-dir -r backend/requirements-serve.txt
FORMAT=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['format'])" "ml/artifacts/$SER_MODEL/model_card.json")
echo "Speech model format: $FORMAT"
if [ "$FORMAT" = "pytorch-wav2vec2-head" ]; then
  pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.11.0
  pip install --no-cache-dir transformers==5.22.2
fi

# 4. Fail the build now, not at the first request, if anything is missing.
MODEL_DIR="$PWD/ml/artifacts/$SER_MODEL" SURVEY_DIR="$PWD/ml/artifacts/survey-v1" python -c "
from backend.app import inference, survey
from pathlib import Path
import os
assert inference.load_model(os.environ['MODEL_DIR']) is not None, 'speech model failed to load'
assert survey.load_questions(Path(os.environ['SURVEY_DIR']) / 'questions.json') is not None, 'questions missing'
print('model and questions load OK')
"
