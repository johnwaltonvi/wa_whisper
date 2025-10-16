# wa_whisper Prototype Plan

## Goal
Create a standalone push-to-talk dictation utility (`wa_whisper`) that mirrors the ergonomics of `wa_parakeet` (Right-Alt press to record, release to transcribe + inject text) while swapping the ASR backend to OpenAI’s `whisper-large-v3` (English only).

## Reference Behaviour from `wa_parakeet`
- **Hotkey handling** – `src/parakeet_push_to_talk.py` uses `pynput.keyboard.Listener` to detect Right Alt press/release, toggling recording via a `Recorder` helper.
- **Audio capture** – Captures 16 kHz mono PCM through `sounddevice` with an RMS gate and timeout to avoid hanging.
- **Post-processing pipeline** – Applies DTLN denoising, emotion detection, number/acronym normalization, punctuation (NeMo), grammar cleanup, and finally types the text using `xdotool`.
- **Service tooling** – Systemd unit, CLI flags, install script, logging to `~/.cache/Parakeet`.

## High-Level Approach
1. Build a minimal MVP that preserves: Right-Alt push-to-talk, `sounddevice` recorder + RMS gate, optional DTLN denoising, text injection via `xdotool`, logging.
2. Swap NeMo-specific logic for a Whisper backend wrapper:
   - Load `whisper-large-v3` with `fp16=True`, `device="cuda"` fallback to CPU.
   - Restrict `language="en"` and disable translation features.
   - Use beam search settings (e.g., `beam_size=5`, `best_of=5`) tuned for accuracy.
3. Maintain optional hooks for future parity (grammar cleanup, number formatting) by porting reusable helpers from `wa_parakeet`.
4. Package in a new repository/directory `~/Documents/wa_whisper` with independent virtualenv + CLI entry point.

## Detailed Plan

### Phase 0 – Environment & Dependencies
1. Create `~/Documents/wa_whisper` with `pyproject.toml` or `requirements.txt` listing:
   - `openai-whisper` (or `faster-whisper` for optimized decoding)
   - `torch` (CUDA build matching locally installed drivers)
   - `sounddevice`, `soundfile`, `numpy`
   - `pynput`, `xdotool` (system package), `pyyaml` (for future configs)
2. Set up a dedicated virtual environment (`python3 -m venv .venv`), add bootstrap script analogous to `wa_parakeet/scripts/bootstrap.sh`.
3. Document GPU requirements: 24 GB VRAM sufficient for FP16 `whisper-large-v3` (~2.9 GiB weights + activations).

### Phase 1 – Audio Capture Layer
1. Reuse/paraphrase `Recorder` logic:
   - Manage capture thread with `sounddevice.InputStream`.
   - Apply RMS threshold, preamp gains, and silence timeout identical to `wa_parakeet`.
   - Save recordings to `/tmp/wa_whisper/*.wav`.
2. Implement identical hotkey behaviour:
   - `pynput.keyboard.Listener` capturing Right Alt (key code detection: `keyboard.Key.alt_r`).
   - On press: start recording; optionally mute output audio (reuse `AudioMuteController` pattern).
   - On release: stop recording, enqueue transcription task on worker thread.

### Phase 2 – Whisper Backend Wrapper
1. Build `whisper_backend.py` with:
   - Lazy loader for `whisper.load_model("large-v3", device, download_root=cache_dir)`.
   - `transcribe(audio_path, *, temperature, beam_size, prompt)` wrapper returning text + optional timestamps.
2. Enforce English transcription:
   - Pass `language="en"` to `model.transcribe`.
   - Set `task="transcribe"` and disable translation.
3. Provide hooks for:
   - Prompt priming (future context/bias).
   - Timestamp retrieval for alignment.
   - Fallback to CPU with warning if CUDA unavailable.

### Phase 3 – Post-processing & Text Injection
1. Port number/acronym normalization utilities from `wa_parakeet` into shared helper module.
2. Optionally include grammar cleanup (LanguageTool) and punctuation heuristics; evaluate value once Whisper output is inspected.
3. Integrate with `xdotool type --clearmodifiers` for text injection, identical to Parakeet.
4. Ensure push-to-talk flow (Right Alt release) triggers the pipeline and logs result.

### Phase 4 – CLI & Configuration
1. Provide `wa_whisper.py` entry point with CLI args modelled after Parakeet but pared down:
   - `--sample-rate`, `--rms-threshold`, `--preamp`, `--timeout`
   - `--voice-isolation` flag (optional DTLN support if we port the wrapper)
   - `--log-path`, `--append-space`
2. Maintain logging in `~/.cache/wa_whisper/push_to_talk.log`.
3. Add config file support (YAML/JSON) for advanced options (future).

### Phase 5 – Testing & Validation
1. Manual tests:
   - Verify hotkey toggling, audio capture, transcription latency, and injection in text editor/terminal.
   - Test with background noise to ensure RMS gate works.
2. Compare sample outputs against `wa_parakeet` to evaluate accuracy differences.
3. Evaluate GPU usage (`nvidia-smi`) during transcription; adjust beam params if needed to stay within VRAM.

### Phase 6 – Packaging & Automation (Optional)
1. Systemd service or background daemon mirroring `wa_parakeet` for auto-start.
2. Installer script to set up environment, download Whisper model, and install system dependencies.
3. Documentation README capturing setup instructions, known limitations, switching between `wa_parakeet` and `wa_whisper`.

## Transferable Components from `wa_parakeet`
- Recorder + RMS logic (adapted).
- Audio mute controller.
- Number/acronym normalization, grammar cleanup, punctuation pipeline.
- Logging utilities (`write_log` style helper).
- CLI structure and hotkey management patterns.

## Future Enhancements
- Integrate `faster-whisper` backend for faster inference/CPU fallback.
- Explore context biasing via Whisper prompts or logit bias to emulate Parakeet’s hotword boosts.
- Expand to additional languages using smaller Whisper checkpoints if desired.

---

## Execution Prompt (Prototype Build Checklist)
Follow this instruction set precisely to deliver the initial `wa_whisper` prototype:

1. **Repository Bootstrap**
   - Create `~/Documents/wa_whisper` and initialize a git repo (optional but recommended for tracking).
   - Create `.venv` via `python3 -m venv .venv` and activate for all subsequent steps.
   - Generate `requirements.txt` containing:  
     `openai-whisper`, `torch` (CUDA wheel pin), `sounddevice`, `soundfile`, `numpy`, `pynput`, `PyYAML`, plus `faster-whisper` if GPU benchmarks dictate.
   - Write a `scripts/bootstrap.sh` mirroring Parakeet’s style to install dependencies and verify `ffmpeg`, `xdotool`.

2. **Core Application Skeleton**
   - Create `src/` package housing:
     - `__init__.py`
     - `main.py` (CLI entry)
     - `recorder.py` (ported/trimmed from Parakeet’s recorder)
     - `hotkeys.py` (Right Alt listener)
     - `whisper_backend.py`
     - `text_postprocess.py` (number/acronym/grammar utilities)
     - `log_utils.py`
   - Add a console entry point (`wa-whisper`) via `pyproject.toml` or `setup.cfg`.

3. **Recorder & Audio Muting**
   - Port Parakeet’s `Recorder` class adapting imports to remain dependency-light.
   - Preserve RMS threshold, silence timeout, microphone selection flag.
   - Port `AudioMuteController` (PulseAudio sink detection + mute/unmute).

4. **Hotkey Loop**
   - Implement `RightAltListener` using `pynput`:  
     - On key press: call `recorder.start()` and optionally mute audio.  
     - On key release: stop recording, enqueue transcription on worker thread.
   - Ensure ESC fallback present (optional CLI flag).

5. **Whisper Backend**
   - Implement loader that downloads `whisper-large-v3` to cache (`~/.cache/wa_whisper/models`).
   - Force `language="en"`, `task="transcribe"`, `fp16=True` (with CPU fallback message).
   - Provide adjustable inference params (`beam_size`, `best_of`, `temperature`, `compression_ratio_threshold`).
   - Return a structured result object (`text`, `segments`, `avg_logprob`).

6. **Processing Pipeline**
   - Integrate DTLN voice isolation optionally (reuse `voice_isolation.py` from Parakeet when available).
   - Apply number/acronym normalization, grammar cleanup, punctuation (reuse existing helper functions).
   - Append optional trailing space, then output via `xdotool type --clearmodifiers`.
   - Log every stage to `~/.cache/wa_whisper/push_to_talk.log`.

7. **CLI & Config**
   - Use `argparse` to expose: sample rate, RMS threshold, timeout, log path, append space flag, disable normalization toggles, backend parameters (beam size etc.).
   - Provide YAML config override (optional) via `--config`.

8. **Testing & Validation**
   - Create `tests/fixtures/` with sample WAV clips.
   - Write smoke test script `scripts/run_smoke_tests.py` to transcribe fixtures and compare to stored reference outputs.
   - Manually verify GPU usage and inference latency; tune parameters if >2s for short utterances.

9. **Docs & Ops**
   - Author `README.md` detailing setup, hotkey usage, and switch-over steps from `wa_parakeet`.
   - Draft optional systemd unit (`systemd/wa-whisper.service`) to auto-start on login.
   - Document troubleshooting (missing CUDA, xdotool, PulseAudio mute issues).

10. **Final Verification**
    - Test full workflow in text editor, terminal, and browser text fields.
    - Confirm fallback path when GPU unavailable (CPU decode warning, performance expectations).
    - Tag initial release and capture install instructions for future automation.

Deliverables: functioning `wa_whisper` command, activation instructions (`source .venv/bin/activate && wa-whisper`), and comprehensive README/plan artifacts.
