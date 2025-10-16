#!/usr/bin/env python3
"""Quick smoke test for the wa_whisper stack."""

from __future__ import annotations

import argparse
from pathlib import Path

from wa_whisper.log_utils import DEFAULT_LOG_PATH, ensure_log_path, write_log
from wa_whisper.whisper_backend import WhisperBackend, WhisperConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe an audio clip using wa_whisper backend.")
    parser.add_argument("audio", type=Path, help="Path to the WAV/MP3/FLAC file to transcribe.")
    parser.add_argument("--model", default="large-v3", help="Whisper model to load.")
    parser.add_argument("--device", default=None, help="Force compute device (cuda/cpu).")
    args = parser.parse_args()

    if not args.audio.exists():
        parser.error(f"audio file not found: {args.audio}")

    log_path = ensure_log_path(DEFAULT_LOG_PATH)
    config = WhisperConfig(model_name=args.model, device=args.device)
    backend = WhisperBackend(config, log_path)
    write_log(f"Smoke test: transcribing {args.audio}", log_path)
    result = backend.transcribe(args.audio)
    print(result.text)


if __name__ == "__main__":
    main()
