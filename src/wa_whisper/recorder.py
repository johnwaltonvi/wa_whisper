"""Audio capture utilities for wa_whisper."""

from __future__ import annotations

import contextlib
import math
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

from .log_utils import write_log


class RecorderStartError(Exception):
    """Raised when the recorder fails to start capturing audio."""

    def __init__(self, message: str, *, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


@dataclass(slots=True)
class RecorderStats:
    """Summary statistics for the most recent capture."""

    total_ms: float
    speech_ms: float
    silence_ms: float
    max_rms: float
    avg_silence_rms: Optional[float]
    speech_blocks: int
    silence_blocks: int
    total_blocks: int
    max_speech_streak_ms: float

    @property
    def speech_ratio(self) -> Optional[float]:
        """Ratio of speech time to total time."""
        if self.total_ms <= 0:
            return None
        return self.speech_ms / self.total_ms

    @property
    def speech_max_db(self) -> Optional[float]:
        """Maximum speech RMS expressed in dBFS."""
        if self.max_rms <= 0:
            return None
        return 20 * math.log10(self.max_rms)

    @property
    def silence_avg_db(self) -> Optional[float]:
        """Average silence RMS expressed in dBFS."""
        if not self.avg_silence_rms or self.avg_silence_rms <= 0:
            return None
        return 20 * math.log10(self.avg_silence_rms)


class Recorder:
    """Stream audio from the default microphone into a temporary WAV file."""

    def __init__(
        self,
        sample_rate: int,
        device_index: Optional[int],
        log_path: Path,
        rms_threshold: float,
        preamp: float = 1.0,
        start_retry_attempts: int = 3,
        start_retry_delay: float = 0.2,
    ) -> None:
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.log_path = log_path
        self.rms_threshold = rms_threshold
        self.preamp = preamp
        self._start_retry_attempts = max(1, start_retry_attempts)
        self._start_retry_delay = max(0.0, start_retry_delay)

        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._writer: sf.SoundFile | None = None
        self._file: Path | None = None
        self._running = False
        self._worker: threading.Thread | None = None
        self._last_audio_time = 0.0

        # Capture statistics that feed silence gating.
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0
        self._total_duration_ms = 0.0
        self._max_rms = 0.0
        self._silence_rms_sum = 0.0
        self._silence_block_count = 0
        self._speech_block_count = 0
        self._total_block_count = 0
        self._max_speech_streak_ms = 0.0
        self._current_speech_streak_ms = 0.0

        self._lock = threading.Lock()
        self._last_stats: RecorderStats | None = None

    def start(self) -> Path:
        """Begin recording and return the output WAV path."""
        with self._lock:
            if self._running and self._file:
                return self._file

        last_error: Exception | None = None

        def audio_callback(indata, _frames, _time_info, status):
            if status:
                write_log(f"Audio status: {status}", self.log_path)
            self._queue.put(indata.copy())

        for attempt in range(1, self._start_retry_attempts + 1):
            with self._lock:
                # Reset state so we always begin from a clean queue.
                self._queue = queue.Queue()
                self._prepare_writer()

            stream: sd.InputStream | None = None
            try:
                stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    device=self.device_index,
                    channels=1,
                    dtype="float32",
                    callback=audio_callback,
                )
                stream.start()
            except Exception as exc:  # pragma: no cover - exercised in tests via stub
                last_error = exc
                if stream is not None:
                    with contextlib.suppress(Exception):
                        stream.close()
                self._handle_failed_start(exc=exc, attempt=attempt)
                continue

            with self._lock:
                self._stream = stream
                self._running = True
                self._last_audio_time = time.time()
                self._reset_stats()

                self._worker = threading.Thread(target=self._drain_queue, daemon=True)
                self._worker.start()
                write_log(f"Recorder started -> {self._file}", self.log_path)
                return self._file

            # Success path shouldn't reach here, but guard anyway.
            break

        attempts = self._start_retry_attempts
        message = "Input stream failed to start"
        if last_error:
            message = f"{message}: {last_error}"
        raise RecorderStartError(message, attempts=attempts)

    def stop(self, timeout: float) -> Path | None:
        """Stop recording after `timeout` seconds of silence."""
        with self._lock:
            if not self._running:
                return None
            deadline = self._last_audio_time + timeout

        while time.time() < deadline:
            time.sleep(0.05)

        with self._lock:
            self._running = False
            self._teardown_stream()

        if self._worker:
            self._worker.join()
            self._worker = None

        self._close_writer(remove_file=False)

        stats = self._finalize_stats()
        self._last_stats = stats
        target = self._file
        write_log(f"Recorder stopped (stats={stats})", self.log_path)
        self._file = None
        return target

    def last_capture_stats(self) -> RecorderStats | None:
        """Return statistics for the most recent capture."""
        return self._last_stats

    # Internal helpers -----------------------------------------------------------------

    def _reset_stats(self) -> None:
        self._speech_duration_ms = 0.0
        self._silence_duration_ms = 0.0
        self._total_duration_ms = 0.0
        self._max_rms = 0.0
        self._silence_rms_sum = 0.0
        self._silence_block_count = 0
        self._speech_block_count = 0
        self._total_block_count = 0
        self._max_speech_streak_ms = 0.0
        self._current_speech_streak_ms = 0.0
        self._last_stats = None

    def _drain_queue(self) -> None:
        while self._running or not self._queue.empty():
            try:
                block = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            processed = self._prepare_block(block)
            if self._writer:
                self._writer.write(processed)
            self._accumulate_stats(processed)

    def _prepare_block(self, block: np.ndarray) -> np.ndarray:
        if self.preamp != 1.0:
            block = np.clip(block * self.preamp, -1.0, 1.0)
        return block

    def _accumulate_stats(self, block: np.ndarray) -> None:
        frames = int(block.shape[0]) if block.ndim > 0 else 0
        if frames <= 0:
            return
        block_ms = (frames / self.sample_rate) * 1000.0
        rms = float(np.sqrt(np.mean(np.square(block), dtype=np.float64)))

        with self._lock:
            self._total_duration_ms += block_ms
            self._total_block_count += 1
            self._max_rms = max(self._max_rms, rms)

            if rms > self.rms_threshold:
                self._last_audio_time = time.time()
                self._speech_duration_ms += block_ms
                self._speech_block_count += 1
                self._current_speech_streak_ms += block_ms
                if self._current_speech_streak_ms > self._max_speech_streak_ms:
                    self._max_speech_streak_ms = self._current_speech_streak_ms
            else:
                self._silence_duration_ms += block_ms
                self._current_speech_streak_ms = 0.0
                if rms > 0:
                    self._silence_rms_sum += rms
                    self._silence_block_count += 1

    def _finalize_stats(self) -> RecorderStats:
        avg_silence_rms: Optional[float] = None
        if self._silence_block_count > 0:
            avg_silence_rms = self._silence_rms_sum / self._silence_block_count
        return RecorderStats(
            total_ms=self._total_duration_ms,
            speech_ms=self._speech_duration_ms,
            silence_ms=self._silence_duration_ms,
            max_rms=self._max_rms,
            avg_silence_rms=avg_silence_rms,
            speech_blocks=self._speech_block_count,
            silence_blocks=self._silence_block_count,
            total_blocks=self._total_block_count,
            max_speech_streak_ms=self._max_speech_streak_ms,
        )

    # Internal setup/teardown helpers ------------------------------------------------

    def _prepare_writer(self) -> None:
        self._close_writer(remove_file=True)
        tmp_dir = Path(tempfile.gettempdir()) / "wa_whisper"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        fd, filename = tempfile.mkstemp(suffix=".wav", dir=tmp_dir)
        os.close(fd)
        self._file = Path(filename)
        self._writer = sf.SoundFile(
            str(self._file),
            mode="w",
            samplerate=self.sample_rate,
            channels=1,
            subtype="PCM_16",
        )

    def _handle_failed_start(self, *, exc: Exception, attempt: int) -> None:
        attempts = self._start_retry_attempts
        write_log(
            f"Recorder start failed (attempt {attempt}/{attempts}): {exc}",
            self.log_path,
        )
        with self._lock:
            self._running = False
            self._teardown_stream()
            self._close_writer(remove_file=True)
        if attempt < attempts and self._start_retry_delay > 0:
            time.sleep(self._start_retry_delay)

    def _teardown_stream(self) -> None:
        if not self._stream:
            return
        with contextlib.suppress(Exception):
            self._stream.stop()
        with contextlib.suppress(Exception):
            self._stream.close()
        self._stream = None

    def _close_writer(self, *, remove_file: bool) -> None:
        if self._writer:
            with contextlib.suppress(Exception):
                self._writer.close()
            self._writer = None
        if remove_file and self._file:
            with contextlib.suppress(FileNotFoundError):
                self._file.unlink()
            self._file = None
