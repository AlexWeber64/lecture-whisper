"""
transcribe_openvino_gpu.py

Standalone EXPERIMENTAL transcription script. Parallel to, and does NOT
touch or replace, transcribe_once.py / lecture_watcher.py (the CTranslate2
CPU/int8 daily driver).

Pipeline:
  audio file (any format) -> faster_whisper.audio.decode_audio (mono,
  16kHz, float32, same decode path faster-whisper itself uses)
  -> Silero VAD (faster_whisper.vad -- the exact same ONNX model and
     get_speech_timestamps() call the CPU pipeline uses, so silence/dead-air
     stripping behaves identically) to find speech chunks and strip silence
  -> speech chunks concatenated into a trimmed buffer, fed to OpenVINO
     GenAI's WhisperPipeline running on the Arc iGPU (device="GPU")
  -> segment timestamps mapped from trimmed-audio time back to
     original-file time via SpeechTimestampsMap -- the same restore-
     timestamps mechanism faster-whisper itself uses internally
     (see faster_whisper.transcribe.restore_speech_timestamps)
  -> transcript written as "[start.XXs -> end.XXs] text" lines, matching
     the existing pipeline's output format for direct diffing

Requires the .venv in this folder (openvino, openvino-genai, faster-whisper
for its VAD module only, etc. -- see requirements.txt).

Usage:
    .venv\\Scripts\\python.exe transcribe_openvino_gpu.py "C:\\path\\to\\audio.m4a"
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as ovg
from faster_whisper.audio import decode_audio
from faster_whisper.vad import (
    SpeechTimestampsMap,
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
)

DEFAULT_MODEL_DIR = r"C:\WhisperModels\large-v3-turbo-openvino"
SAMPLING_RATE = 16000


def require_gpu(core: ov.Core) -> None:
    """Confirm OpenVINO actually sees the Arc iGPU. Refuses to silently
    fall back to CPU -- if GPU isn't available, this is a hard stop with
    an explanation, not a quiet downgrade."""
    devices = core.available_devices
    print(f"OpenVINO available devices: {devices}")
    for d in devices:
        try:
            name = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            name = "(name unavailable)"
        print(f"  {d}: {name}")

    if "GPU" not in devices:
        print(
            "\nERROR: No OpenVINO GPU device detected on this system.\n"
            "This script requires the Intel Arc iGPU via OpenVINO's GPU\n"
            "plugin and will not silently fall back to CPU. Check:\n"
            "  - Intel Arc/Graphics driver is installed and up to date\n"
            "    (Settings > Windows Update > Optional updates, or\n"
            "    intel.com/content/www/us/en/download-center)\n"
            "  - No other process has exclusive hold of the GPU\n"
            "  - This venv's `openvino` package version is compatible\n"
            "    with the installed driver\n",
            file=sys.stderr,
        )
        sys.exit(1)


def load_audio(path: Path) -> np.ndarray:
    print(f"  Decoding audio: {path.name}")
    return decode_audio(str(path), sampling_rate=SAMPLING_RATE)


def strip_silence(audio: np.ndarray):
    """Runs Silero VAD standalone (faster_whisper's bundled ONNX model,
    identical to what vad_filter=True uses in the CPU pipeline) and
    returns (trimmed_audio, ts_map). ts_map is None if no speech was
    found anywhere in the file."""
    vad_options = VadOptions(min_silence_duration_ms=500)
    speech_chunks = get_speech_timestamps(audio, vad_options)

    if not speech_chunks:
        print("  Warning: VAD found no speech in this file.")
        return audio, None

    audio_chunks, _ = collect_chunks(audio, speech_chunks)
    trimmed = np.concatenate(audio_chunks, axis=0)
    ts_map = SpeechTimestampsMap(speech_chunks, SAMPLING_RATE)

    removed_s = (len(audio) - len(trimmed)) / SAMPLING_RATE
    print(
        f"  VAD: kept {len(speech_chunks)} speech chunk(s), "
        f"stripped {removed_s:.1f}s of silence "
        f"({len(audio) / SAMPLING_RATE:.1f}s -> {len(trimmed) / SAMPLING_RATE:.1f}s)"
    )
    return trimmed, ts_map


def transcribe(audio_path: Path, out_txt: Path, model_dir: str, language: str) -> None:
    core = ov.Core()
    require_gpu(core)

    print("  Loading local OpenVINO model from disk (no network fetch)...")
    pipeline = ovg.WhisperPipeline(model_dir, device="GPU")

    audio = load_audio(audio_path)

    print("  Running VAD...")
    trimmed_audio, ts_map = strip_silence(audio)

    print("  Transcribing on GPU...")
    gen_kwargs = dict(return_timestamps=True, task="transcribe")
    if language:
        gen_kwargs["language"] = f"<|{language}|>"

    result = pipeline.generate(trimmed_audio.tolist(), **gen_kwargs)

    with open(out_txt, "w", encoding="utf-8") as f:
        if not result.chunks:
            # Short-form fallback: no per-segment chunks were returned,
            # just the plain concatenated text.
            text = result.texts[0] if result.texts else ""
            f.write(f"[0.00s -> {len(trimmed_audio) / SAMPLING_RATE:.2f}s] {text}\n")
        else:
            for chunk in result.chunks:
                if ts_map is not None:
                    start = ts_map.get_original_time(chunk.start_ts)
                    end = ts_map.get_original_time(chunk.end_ts, is_end=True)
                else:
                    start, end = chunk.start_ts, chunk.end_ts
                f.write(f"[{start:.2f}s -> {end:.2f}s] {chunk.text}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Experimental transcription via OpenVINO GenAI on the Arc iGPU."
    )
    parser.add_argument("audio_path", help="Path to the audio file (wav, m4a, mp3, etc.)")
    parser.add_argument("--out", default=None, help="Output .txt path (default: alongside input file)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Path to converted OpenVINO IR model dir")
    parser.add_argument("--language", default="en", help="Language code, e.g. en (default: en; pass '' to auto-detect)")
    args = parser.parse_args()

    src = Path(args.audio_path)
    if not src.exists():
        print(f"File not found: {src}")
        sys.exit(1)

    if not Path(args.model_dir).exists():
        print(f"Model dir not found: {args.model_dir}\nRun the optimum-cli export step first.")
        sys.exit(1)

    out_path = Path(args.out) if args.out else src.with_suffix(".ov.txt")

    start = time.time()
    transcribe(src, out_path, args.model_dir, args.language)
    elapsed = (time.time() - start) / 60
    print(f"Done in {elapsed:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
