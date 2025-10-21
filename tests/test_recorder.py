import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:  # pragma: no cover - fallback for test environments without sounddevice
    import sounddevice as sd  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    class _PortAudioError(Exception):
        pass

    class _SDModule:
        PortAudioError = _PortAudioError

    sd = _SDModule()  # type: ignore

from wa_whisper.recorder import Recorder, RecorderStartError


def _make_recorder(tmp_path: Path, monkeypatch, *, fail_attempts: int, sleep: float = 0.0) -> Recorder:
    counter = {"remaining": fail_attempts}

    class SequencedStream:
        def __init__(self, *, callback, **_kwargs):
            self.callback = callback
            self.stopped = False

        def start(self):
            if counter["remaining"] > 0:
                counter["remaining"] -= 1
                raise sd.PortAudioError("Wait timed out", -9987)

        def stop(self):
            self.stopped = True

        def close(self):
            self.stopped = True

    def stream_factory(**kwargs):
        if sleep:
            time.sleep(sleep)
        return SequencedStream(**kwargs)

    monkeypatch.setattr("wa_whisper.recorder.sd.InputStream", stream_factory)
    monkeypatch.setattr("wa_whisper.recorder.tempfile.gettempdir", lambda: str(tmp_path))
    log_path = tmp_path / "log.txt"
    return Recorder(
        sample_rate=16_000,
        device_index=None,
        log_path=log_path,
        rms_threshold=0.01,
        preamp=1.0,
    )


def test_recorder_start_raises_after_retries(tmp_path, monkeypatch):
    recorder = _make_recorder(tmp_path, monkeypatch, fail_attempts=3)

    with pytest.raises(RecorderStartError):
        recorder.start()

    temp_dir = tmp_path / "wa_whisper"
    assert not list(temp_dir.glob("*.wav"))
    assert recorder.last_capture_stats() is None


def test_recorder_start_recovers_after_transient_failure(tmp_path, monkeypatch):
    recorder = _make_recorder(tmp_path, monkeypatch, fail_attempts=1)

    path = recorder.start()

    assert path.exists()
    assert recorder.last_capture_stats() is None
