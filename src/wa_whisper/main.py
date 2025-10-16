"""CLI entrypoint for wa_whisper."""

from __future__ import annotations

import argparse
import queue
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

from .hotkeys import PushToTalkHotkey
from .log_utils import DEFAULT_LOG_PATH, ensure_log_path, write_log
from .recorder import Recorder, RecorderStats
from .text_postprocess import postprocess_text
from .voice_isolation import VoiceIsolationPipeline
from .whisper_backend import DEFAULT_MODEL_CACHE, WhisperBackend, WhisperConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Push-to-talk dictation powered by OpenAI Whisper.")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Microphone sample rate (Hz).")
    parser.add_argument("--rms-threshold", type=float, default=0.01, help="Voice activity RMS threshold.")
    parser.add_argument("--preamp", type=float, default=1.0, help="Signal gain applied before encoding.")
    parser.add_argument("--silence-timeout", type=float, default=0.5, help="Seconds of silence before stop.")
    parser.add_argument("--device-index", type=int, default=None, help="SoundDevice input index.")
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH, help="Log file path.")
    parser.add_argument("--model", default="large-v3", help="Whisper model name.")
    parser.add_argument("--beam-size", type=int, default=5, help="Beam search width.")
    parser.add_argument("--best-of", type=int, default=5, help="Number of candidate samples.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--initial-prompt", default=None, help="Optional prompt bias string.")
    parser.add_argument("--no-audio-mute", action="store_true", help="Disable automatic desktop audio mute.")
    parser.add_argument("--append-space", action="store_true", help="Add a trailing space to output.")
    parser.add_argument("--disable-number-normalization", action="store_true", help="Skip number conversions.")
    parser.add_argument("--disable-acronym-normalization", action="store_true", help="Skip acronym conversions.")
    parser.add_argument("--disable-punctuation", action="store_true", help="Do not enforce sentence punctuation.")
    parser.add_argument("--device", default=None, help="Force Whisper device (cuda/cpu).")
    parser.add_argument("--model-cache", type=Path, default=None, help="Override Whisper model cache dir.")
    parser.add_argument("--exit-on-esc", action="store_true", default=False, help="Stop listener on ESC.")
    parser.add_argument("--no-voice-isolation", action="store_true", help="Disable placeholder voice isolation.")
    parser.add_argument("--xdotool-path", type=Path, default=None, help="Override xdotool binary path.")
    parser.add_argument("--disable-complete-beep", action="store_true", help="Disable post-paste completion beep.")
    return parser


TaskItem = Optional[Tuple[Path, Optional[RecorderStats]]]
DEFAULT_BEEP_COMMAND = (
    Path("/usr/bin/paplay"),
    Path("/usr/share/sounds/freedesktop/stereo/bell.oga"),
)


def ensure_xdotool(path_override: Optional[Path]) -> Path:
    if path_override:
        return path_override
    resolved = shutil.which("xdotool")
    if not resolved:
        raise RuntimeError("xdotool not found; install it to enable text injection.")
    return Path(resolved)


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    log_path = ensure_log_path(args.log_path)
    write_log("wa_whisper starting", log_path)

    config = WhisperConfig(
        model_name=args.model,
        beam_size=args.beam_size,
        best_of=args.best_of,
        temperature=args.temperature,
        initial_prompt=args.initial_prompt,
        cache_dir=args.model_cache or DEFAULT_MODEL_CACHE,
        device=args.device,
    )

    backend = WhisperBackend(config, log_path)
    recorder = Recorder(
        sample_rate=args.sample_rate,
        device_index=args.device_index,
        log_path=log_path,
        rms_threshold=args.rms_threshold,
        preamp=args.preamp,
    )
    voice_isolation = None if args.no_voice_isolation else VoiceIsolationPipeline(log_path)
    xdotool_bin = ensure_xdotool(args.xdotool_path)

    task_queue: "queue.Queue[TaskItem]" = queue.Queue()
    stop_event = threading.Event()

    def worker_loop() -> None:
        while True:
            try:
                item = task_queue.get(timeout=0.1)
            except queue.Empty:
                if stop_event.is_set():
                    continue
                continue
            if item is None:
                task_queue.task_done()
                break
            audio_path, stats = item
            process_capture(
                audio_path=audio_path,
                stats=stats,
                backend=backend,
                voice_isolation=voice_isolation,
                log_path=log_path,
                append_space=args.append_space,
                normalize_numbers=not args.disable_number_normalization,
                normalize_acronyms=not args.disable_acronym_normalization,
                ensure_punct=not args.disable_punctuation,
                xdotool_bin=xdotool_bin,
                enable_beep=not args.disable_complete_beep,
            )
            task_queue.task_done()

    worker = threading.Thread(target=worker_loop, daemon=True)
    worker.start()

    def handle_capture(path: Optional[Path], stats: Optional[RecorderStats]) -> None:
        if stop_event.is_set():
            if path:
                path.unlink(missing_ok=True)
            return
        if path is None:
            write_log("Capture finished with no audio file", log_path)
            return
        task_queue.put((path, stats))

    hotkey = PushToTalkHotkey(
        recorder,
        silence_timeout=args.silence_timeout,
        on_capture_finished=handle_capture,
        log_path=log_path,
        enable_audio_mute=not args.no_audio_mute,
        exit_on_esc=args.exit_on_esc,
        on_exit=lambda: request_shutdown(signal.SIGTERM),
    )

    def request_shutdown(signum: int) -> None:
        if stop_event.is_set():
            return
        write_log(f"Received shutdown signal {signum}", log_path)
        stop_event.set()
        hotkey.stop()
        task_queue.put(None)

    def shutdown(signum: int, _frame) -> None:
        request_shutdown(signum)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        hotkey.start()
        write_log("Ready for push-to-talk (Right Alt)", log_path)
        stop_event.wait()
    finally:
        request_shutdown(signal.SIGTERM)
        task_queue.join()
        worker.join(timeout=2.0)
        write_log("wa_whisper stopped", log_path)


def process_capture(
    *,
    audio_path: Path,
    stats: Optional[RecorderStats],
    backend: WhisperBackend,
    voice_isolation: Optional[VoiceIsolationPipeline],
    log_path: Path,
    append_space: bool,
    normalize_numbers: bool,
    normalize_acronyms: bool,
    ensure_punct: bool,
    xdotool_bin: Path,
    enable_beep: bool,
) -> None:
    write_log(f"Processing capture {audio_path}", log_path)
    if stats:
        ratio = f"{stats.speech_ratio:.2f}" if stats.speech_ratio is not None else "n/a"
        max_db = f"{stats.speech_max_db:.1f}dB" if stats.speech_max_db is not None else "n/a"
        write_log(
            "Capture stats "
            f"total={stats.total_ms:.1f}ms speech={stats.speech_ms:.1f}ms "
            f"silence={stats.silence_ms:.1f}ms ratio={ratio} max_db={max_db}",
            log_path,
        )
    enhanced_path = audio_path
    try:
        if voice_isolation:
            enhanced_path = voice_isolation.enhance(audio_path)
        result = backend.transcribe(enhanced_path)
        text = postprocess_text(
            result.text,
            normalize_numbers_enabled=normalize_numbers,
            normalize_acronyms_enabled=normalize_acronyms,
            ensure_punctuation=ensure_punct,
            append_space=append_space,
        )
        if not text:
            write_log("No text produced from transcription", log_path)
            return
        inject_text(text, xdotool_bin, log_path, enable_beep=enable_beep)
        write_log(f"Injected text: {text}", log_path)
    except Exception as exc:  # pragma: no cover - defensive log
        write_log(f"Capture processing failed: {exc}", log_path)
    finally:
        if enhanced_path != audio_path and enhanced_path.exists():
            enhanced_path.unlink(missing_ok=True)
        audio_path.unlink(missing_ok=True)


def inject_text(text: str, xdotool_bin: Path, log_path: Path, *, enable_beep: bool) -> None:
    try:
        subprocess.run(
            [str(xdotool_bin), "type", "--clearmodifiers", text],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as exc:
        write_log(f"xdotool failed: {exc}", log_path)
    else:
        if enable_beep:
            play_completion_beep(log_path)


def play_completion_beep(log_path: Path) -> None:
    player, sound = DEFAULT_BEEP_COMMAND
    if not player.exists() or not sound.exists():
        write_log("Completion beep skipped: paplay or bell sound missing", log_path)
        return
    try:
        subprocess.run(
            [str(player), str(sound)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", "ignore") if exc.stderr else ""
        write_log(f"Completion beep failed: {exc}; stderr={stderr.strip()}", log_path)


if __name__ == "__main__":
    main(sys.argv[1:])
