"""Optional voice isolation stubs."""

from __future__ import annotations

from pathlib import Path

from .log_utils import write_log


class VoiceIsolationPipeline:
    """Placeholder voice isolation pipeline."""

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path

    def enhance(self, audio_path: Path) -> Path:
        """Return the input audio unmodified (placeholder)."""
        write_log("Voice isolation not yet implemented; skipping", self._log_path)
        return audio_path

