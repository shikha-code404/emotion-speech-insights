"""Package the files the API serves into one .tar.gz for a GitHub Release.

Contents: ml/artifacts/<model>/ (speech model) and ml/artifacts/survey-v1/questions.json.
The analysis-only lifestyle model is deliberately left out.

Run from the repo root:  python deploy/package_artifacts.py [--model ser-w2v2-v2]
Output: ml/artifacts/release/ser-artifacts-<model>.tar.gz (git-ignored)
"""

import argparse
import hashlib
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "ml" / "artifacts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ser-w2v2-v2")
    args = ap.parse_args()
    model_dir = ARTIFACTS / args.model
    questions = ARTIFACTS / "survey-v1" / "questions.json"
    for p in (model_dir / "model_card.json", questions):
        if not p.exists():
            raise SystemExit(f"missing {p}; train/export first")
    out = ARTIFACTS / "release" / f"ser-artifacts-{args.model}.tar.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in model_dir.rglob("*") if p.is_file()) + [questions]
    with tarfile.open(out, "w:gz") as tar:
        for f in files:
            tar.add(f, arcname=f.relative_to(ROOT).as_posix())
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out}  ({out.stat().st_size / 1024 ** 2:.0f} MB, {len(files)} files)")
    print(f"sha256 {digest}")
    for f in files:
        print("  ", f.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
