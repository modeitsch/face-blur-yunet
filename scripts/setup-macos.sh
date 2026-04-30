#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 was not found. Install Python 3.9 or newer first."
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg was not found. Install it with: brew install ffmpeg"
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

read -r -p "Install local transcription backend faster-whisper? [y/N] " install_whisper
if [[ "$install_whisper" =~ ^[Yy]$ ]]; then
  python -m pip install faster-whisper
fi

read -r -p "Install local translation backend argostranslate? [y/N] " install_argos
if [[ "$install_argos" =~ ^[Yy]$ ]]; then
  python -m pip install argostranslate
fi

cat <<'EOF'

Setup complete.

Start the dashboard:
  source .venv/bin/activate
  python -m uvicorn face_blur_yunet.app:create_default_app --factory --host 127.0.0.1 --port 8000

Open:
  http://127.0.0.1:8000
EOF
