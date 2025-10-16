#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$PROJECT_ROOT"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[wa_whisper] Missing dependency: ffmpeg" >&2
  exit 1
fi

if ! command -v xdotool >/dev/null 2>&1; then
  echo "[wa_whisper] Missing dependency: xdotool" >&2
  exit 1
fi

VENV_DIR="${PROJECT_ROOT}/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt

echo "[wa_whisper] Bootstrap complete. Activate the venv with:"
echo "  source ${VENV_DIR}/bin/activate"
