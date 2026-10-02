r"""
lecture_watcher_gpu.py

EXPERIMENTAL variant of lecture_watcher.py: same watch/reconcile/archive
orchestration, but transcribes via OpenVINO GenAI on the Arc 140V iGPU
(transcribe_openvino_gpu.py's engine) instead of faster-whisper/CPU.

Does NOT touch lecture_watcher.py.

IMPORTANT -- to run this manually, you MUST use the project venv's
interpreter, not a bare `python`/`pythonw` command. From inside this
folder, run:
    .venv\Scripts\python.exe lecture_watcher_gpu.py
A bare `python` on this machine's PATH resolves to the system-wide
Python 3.14 install, which does NOT have openvino_genai (or anything
else this file needs) installed -- only .venv does. This is exactly
what the Task Scheduler entry ("lecture watcher gpu") already points
at, so the scheduled/automatic runs are unaffected either way.

Benchmarked against the CPU pipeline (2026-08-17): faster AND cooler on
this machine (9-min clip 26.6s vs 3m03s CPU; 38-min lecture 1m19-1m54s vs
12m45s CPU; CPU package stayed cooler and PL1 didn't get pulled down the
way it did during CPU-only runs; transcript quality was indistinguishable
between the two engines on both test files). Also confirmed switching to
the Windows "Best Performance" power overlay makes no measurable
difference to GPU-bound run time (83.6s vs. a 79-115s un-boosted range),
so this file intentionally has no power-overlay or P-core-pinning logic
-- see the judgment-call notes below.

Judgment calls vs. lecture_watcher.py:
  - Power-overlay boost to "Best Performance": DROPPED. It existed to raise
    the CPU package's power ceiling for sustained CPU-bound compute. The
    GPU pipeline isn't CPU-bound, and an explicit A/B test (same 38-min
    file, with vs. without the boost) showed no meaningful time difference
    -- confirmed empirically, not just assumed.
  - P-core pinning: DROPPED. It kept heavy compute threads off the weak
    E-cores. Inference now happens on the iGPU; the only CPU-side work
    left (audio decode, VAD, Python glue) is light and brief. Restricting
    the whole process to 4 logical cores could, if anything, constrain
    whatever driver/submission threads OpenVINO's GPU runtime spins up,
    for no compute benefit.

Logging / failure behavior (added once this moved from manual testing
toward being the Task Scheduler entry point):
  - This runs headless under pythonw.exe (no console window), so all
    output goes through the `logging` module to a rotating log file
    (LOG_FILE below) instead of relying on print(). Running it manually
    in a terminal still shows the same messages live (console handler).
  - "Die loudly" on failure: if the GPU isn't available at startup, or if
    anything goes wrong processing a specific recording (GPU or
    otherwise), this does NOT quietly skip the file and keep going --
    since a real GPU failure would likely affect every subsequent file
    too, silently continuing would just produce a pile of silent
    failures. Instead it (1) logs the full error, (2) tries a Windows
    toast notification, (3) UNCONDITIONALLY also writes a
    "FAILED - <name>.txt" marker file to both the Desktop and Downloads
    folders (not only as a fallback if the toast fails -- toast delivery
    isn't guaranteed when Task Scheduler runs this without an interactive
    session, so the marker file is the one channel guaranteed to work),
    and (4) hard-exits the whole process so the failure is impossible to
    miss and nothing is silently skipped.
"""

import gc
import json
import logging
import os
import subprocess
import shutil
import sys
import time
import traceback
import uuid
import zlib
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

import numpy as np
import openvino as ov
import openvino_genai as ovg
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from faster_whisper.audio import decode_audio
from faster_whisper.vad import (
    SpeechTimestampsMap,
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
)

import config
import keep_awake
import notes_pipeline

# ----------------------------------------------------------------------
# Fixed settings, and the files this watcher keeps next to itself.
# Everything machine-specific (folders, model locations, classes) comes
# from config.json instead -- see the CONFIG block further down.
# ----------------------------------------------------------------------

# large-v3 is the default (more accurate). large-v3-turbo is only used when
# the dashboard's "Switch to large-v3-turbo" button is clicked for the
# recording currently being processed -- see SWITCH_REQUEST_FILE.
DEFAULT_MODEL = "large-v3"
FAST_MODEL = "large-v3-turbo"

LOG_FILE = Path(__file__).parent / "lecture_watcher_gpu.log"
# Read by dashboard_server.py to show live status -- see that file. Written
# defensively (never allowed to raise) so a dashboard-only feature can never
# affect the actual transcription pipeline.
STATUS_FILE = Path(__file__).parent / "status.json"
# One JSON object per line, one line per completed file -- the dashboard's
# History tab reads this. Append-only; never rewritten in place, so a
# crash mid-write can corrupt at most the last (incomplete) line.
HISTORY_FILE = Path(__file__).parent / "history.jsonl"
# Written by dashboard_server.py when the switch-model button is clicked.
# Names the recording it applies to, so it can never leak onto a later
# file; polled once per ~30s Whisper window while transcribing, and
# cleared when each recording starts and finishes.
SWITCH_REQUEST_FILE = Path(__file__).parent / "model_switch_request.json"
# Replaced by config.json's failure_notice_folders once it loads. This
# fallback only matters if config.json itself can't be read, so that
# failure still leaves a marker file somewhere.
FAILURE_NOTICE_FOLDERS = [Path.home() / "Downloads"]

AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".aac", ".ogg"}
SAMPLING_RATE = 16000
LANGUAGE = "en"

# ----------------------------------------------------------------------

logger = logging.getLogger("lecture_watcher_gpu")
logger.setLevel(logging.INFO)
_file_handler = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
_file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_file_handler)
if sys.stdout is not None:
    # Only useful when run interactively (python.exe in a terminal); under
    # pythonw.exe there's no console and stdout is None, so skip this handler.
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_console_handler)


def write_status(state: str, **fields) -> None:
    """Best-effort status snapshot for dashboard_server.py. Never raises --
    this is a nice-to-have, not something that should ever affect the
    actual transcription pipeline. Written atomically (temp file + replace)
    so the dashboard never reads a half-written file mid-update."""
    try:
        payload = {"state": state, "updated_at": datetime.now().isoformat(timespec="seconds"), **fields}
        tmp_path = STATUS_FILE.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, STATUS_FILE)
    except Exception:
        pass


def notify_failure(name: str, error: Exception) -> None:
    """'Die loudly': best-effort Windows toast, PLUS an unconditional
    marker file on the Desktop and Downloads (toast delivery isn't
    guaranteed when running headless under Task Scheduler, so the marker
    file -- not the toast -- is the guaranteed channel)."""
    message = f"{type(error).__name__}: {error}"
    logger.error("FAILURE (%s): %s\n%s", name, message, traceback.format_exc())

    try:
        ps_script = (
            "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, "
            "ContentType = WindowsRuntime] > $null\n"
            "$template = [Windows.UI.Notifications.ToastNotificationManager]::"
            "GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)\n"
            "$text = $template.GetElementsByTagName('text')\n"
            f"$text.Item(0).AppendChild($template.CreateTextNode('Lecture Whisper GPU watcher failed')) > $null\n"
            f"$text.Item(1).AppendChild($template.CreateTextNode({message[:180]!r})) > $null\n"
            "$toast = [Windows.UI.Notifications.ToastNotification]::new($template)\n"
            "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier("
            "'{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\\WindowsPowerShell\\v1.0\\powershell.exe'"
            ").Show($toast)\n"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_script],
            capture_output=True, timeout=10,
        )
    except Exception as notify_err:
        logger.warning("Toast notification attempt failed: %s", notify_err)

    marker_name = f"FAILED - {name}.txt"
    marker_body = (
        f"Lecture Whisper GPU watcher failed at {datetime.now().isoformat(timespec='seconds')}\n"
        f"Recording/stage: {name}\n"
        f"Error: {message}\n\n"
        f"Full traceback:\n{traceback.format_exc()}\n"
        f"Full log: {LOG_FILE}\n"
    )
    for folder in FAILURE_NOTICE_FOLDERS:
        try:
            (folder / marker_name).write_text(marker_body, encoding="utf-8")
        except OSError as write_err:
            logger.warning("Could not write failure marker to %s: %s", folder, write_err)


# ----------------------------------------------------------------------
# CONFIG - read from config.json (git-ignored; start from
# config.example.json). Loaded only now, once logging and
# notify_failure() exist, so a missing or broken config.json dies loudly
# like any other failure instead of exiting silently under pythonw.exe.
# ----------------------------------------------------------------------

try:
    _config = config.load()
    WATCH_FOLDER = Path(_config["watch_folder"])
    ARCHIVE_FOLDER = Path(_config["archive_folder"])
    # Can be in a different OneDrive account than the recordings -- e.g. the
    # one the scheduled Claude routine reads transcripts from.
    TRANSCRIPT_FOLDER = Path(_config["transcript_folder"])
    # Must NOT be inside WATCH_FOLDER (or any OneDrive-synced folder): a
    # converted .wav written there would look like a new recording.
    SCRATCH_FOLDER = Path(_config["scratch_folder"])
    FAILURE_NOTICE_FOLDERS = [Path(p) for p in _config["failure_notice_folders"]]
    MODEL_DIRS = {model: _config["models"][model] for model in (DEFAULT_MODEL, FAST_MODEL)}
    CLASS_ALIASES = {label: c["aliases"] for label, c in _config["classes"].items()}
    CLASS_HOTWORDS = {label: c["hotwords"] for label, c in _config["classes"].items() if c.get("hotwords")}
except Exception as e:
    write_status("error", error=str(e), note="Couldn't read config.json -- watcher exited")
    notify_failure("config", e)
    os._exit(1)  # die loudly -- see module docstring

for folder in (WATCH_FOLDER, ARCHIVE_FOLDER, TRANSCRIPT_FOLDER, SCRATCH_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------

# Each class in config.json maps a label to its filename "aliases" and,
# optionally, its "hotwords".
#
# Aliases drive the dashboard's History tab. They're matched against the
# ORIGINAL recording's filename only (case-insensitive substring match on
# any alias; the first class listed wins) -- this is a naming-convention
# heuristic, NOT the real Notion classification, which happens later in
# the separate Claude Cowork routine this script has no visibility into.
# A filename matching none of them falls back to "Unknown". Update the
# list in config.json when the classes change each semester.
#
# An alias prefixed with "^" only matches at the very START of the
# filename (e.g. "^art" matches "Art history.m4a" but not "Start...m4a"
# or "Smart...m4a"); all other aliases match as a substring anywhere.
#
# Hotwords are per-class vocabulary passed to Whisper -- OpenVINO feeds
# them to EVERY ~30s window as if they were the preceding transcript,
# nudging Whisper toward these spellings. A recording whose name matches
# no class gets none. Build each list from words the recognizer actually
# mishears in that class's lectures, plus rarer terms from upcoming units.
#
# Keep each entry under ~190 tokens. The hotwords share a 448-token window
# with the text Whisper writes for each 30s of audio, and nothing stops an
# oversized list from crowding that text out: in testing a 414-token list
# silently dropped two-thirds of a clip's words (a 207-token one lost
# nothing). Check with the tokenizer in the large-v3 model folder after
# editing. Hotwords cost ~16-19% more transcription time. Swap terms in
# and out as the semester moves on; skip ones Whisper already gets right
# (common words, famous names). Leave out bare "AND"/"OR"/"NOT" so
# ordinary "and"/"or" don't start coming out capitalized.


def classify_filename(original_name: str) -> str:
    lowered = original_name.lower()
    for label, aliases in CLASS_ALIASES.items():
        for alias in aliases:
            if alias.startswith("^"):
                if lowered.startswith(alias[1:]):
                    return label
            elif alias in lowered:
                return label
    return "Unknown"


def switch_requested(file_name: str) -> bool:
    """True if the dashboard asked to finish `file_name` on FAST_MODEL.
    Never raises -- it's called from inside the OpenVINO streamer callback,
    where an exception would abort the transcription itself."""
    try:
        return json.loads(SWITCH_REQUEST_FILE.read_text(encoding="utf-8")).get("file") == file_name
    except Exception:
        return False


def clear_switch_request() -> None:
    try:
        SWITCH_REQUEST_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def append_history(
    original_name: str, transcript_filename: str, started_at: str,
    elapsed_min: float, audio_duration_min: float, model: str,
) -> None:
    """Best-effort append to history.jsonl for the dashboard's History tab.
    Never raises, same rationale as write_status(). `class_guess` comes
    from classify_filename() -- see CLASS_ALIASES above -- a
    naming-convention heuristic, NOT the real Notion classification (that
    happens later, in the separate Claude Cowork routine, which this
    script has no visibility into).

    Records two distinct durations: `elapsed_min` (how long the GPU
    actually took to transcribe) and `audio_duration_min` (how long the
    source recording itself is) -- a 90-minute lecture done in 3 minutes
    and a 5-minute clip done in 3 minutes are very different signals, and
    collapsing them into one number would hide that."""
    try:
        class_guess = classify_filename(original_name)
        record = {
            "id": uuid.uuid4().hex,
            "file": original_name,
            "transcript_filename": transcript_filename,
            "class_guess": class_guess,
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_min": round(elapsed_min, 1),
            "audio_duration_min": round(audio_duration_min, 1),
            "model": model,
        }
        with open(HISTORY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


# Pipeline is loaded LAZILY (same rationale as lecture_watcher.py's
# get_model()): idle RAM/GPU-memory footprint stays near zero between
# lectures.
_pipeline = None
_pipeline_model = None


def require_gpu() -> None:
    core = ov.Core()
    devices = core.available_devices
    logger.info("OpenVINO available devices: %s", devices)
    if "GPU" not in devices:
        raise RuntimeError(
            "No OpenVINO GPU device detected on this system. This watcher "
            "requires the Arc iGPU via OpenVINO's GPU plugin and will not "
            "silently fall back to CPU. Check the Arc/Graphics driver, "
            "then re-run."
        )


def get_pipeline(model: str) -> ovg.WhisperPipeline:
    global _pipeline, _pipeline_model
    if _pipeline is not None and _pipeline_model != model:
        release_pipeline()  # free the old model's GPU memory before loading the new one
    if _pipeline is None:
        logger.info("  Loading local OpenVINO model %s from disk (no network fetch)...", model)
        _pipeline = ovg.WhisperPipeline(MODEL_DIRS[model], device="GPU")
        _pipeline_model = model
    return _pipeline


def release_pipeline() -> None:
    global _pipeline, _pipeline_model
    _pipeline = None
    _pipeline_model = None
    gc.collect()


def wait_for_stable_file(path: Path, checks: int = 3, interval: float = 2.0, timeout: float = 600.0) -> bool:
    """Block until the file's size stops changing across `checks` consecutive
    polls (protects against OneDrive still syncing / a cloud placeholder)."""
    last_size = -1
    stable_count = 0
    elapsed = 0.0
    while stable_count < checks:
        if elapsed > timeout:
            return False
        if not path.exists():
            time.sleep(interval)
            elapsed += interval
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        if size > 0 and size == last_size:
            stable_count += 1
        else:
            stable_count = 0
        last_size = size
        time.sleep(interval)
        elapsed += interval
    return True


def convert_to_wav(src: Path) -> Path:
    """Mono, 16kHz, 16-bit PCM. Written to SCRATCH_FOLDER (outside the
    watched folder) so it can never be mistaken for a new incoming
    recording -- same self-triggering-loop fix as lecture_watcher.py."""
    dst = SCRATCH_FOLDER / (src.stem + ".wav")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-ac", "1", "-ar", str(SAMPLING_RATE), "-c:a", "pcm_s16le",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )
    return dst


def strip_silence(audio: np.ndarray):
    """Standalone Silero VAD (faster_whisper's bundled ONNX model -- the
    same model and parameters vad_filter=True uses in the CPU pipeline)."""
    vad_options = VadOptions(min_silence_duration_ms=500)
    speech_chunks = get_speech_timestamps(audio, vad_options)
    if not speech_chunks:
        return audio, None
    audio_chunks, _ = collect_chunks(audio, speech_chunks)
    trimmed = np.concatenate(audio_chunks, axis=0)
    ts_map = SpeechTimestampsMap(speech_chunks, SAMPLING_RATE)
    return trimmed, ts_map


# If a switch lands during large-v3's final window, it has already finished
# and all that's left past its last chunk is a sliver of tail -- don't hand
# that to turbo (Whisper tends to hallucinate on near-empty input).
MIN_FAST_REMAINDER_S = 1.0
# How much of large-v3's last output turbo gets as its "previous
# transcription" prompt when taking over mid-file. Without it, turbo tends
# to drop a clipped first word at the seam and lose punctuation style
# (checked on 9 cut points across two lectures). Capped well under
# Whisper's 224-token prompt limit.
FAST_PROMPT_SEGMENTS = 3
FAST_PROMPT_MAX_CHARS = 500


# Seconds of speech (silence already trimmed) one ~30s Whisper window moves
# through on average. Each window ends at its last complete segment and the
# next one picks up there, so it falls a little short of 30s.
# Measured on four lectures (2026-10-02): 25.09-25.68s on both models.
# Rounded down, so the bar runs a little behind rather than ahead.
SPEECH_S_PER_WINDOW = 25.0
# The window count is only an estimate, so the bar holds here until the
# transcript is actually done instead of claiming 100% early.
MAX_ESTIMATED_PROGRESS = 0.99
# How many windows the pace is timed over before an ETA is shown. Timing
# starts once the first window is done, since that one also pays for model
# loading and encoder warm-up.
ETA_MIN_WINDOWS = 3


class Progress:
    """Live transcription progress for the dashboard, written to status.json
    after every ~30s window. OpenVINO's streamer hands over each window's
    text but not where in the audio it ended (timestamp tokens are stripped
    before it's called), so position is estimated as windows done x
    SPEECH_S_PER_WINDOW, starting from a known point: 0, or the exact seam
    of a mid-run switch to FAST_MODEL. The ETA comes from this run's own
    pace per window, since total run time varies a lot from one lecture to
    the next (2-13 min for similar lengths in history.jsonl)."""

    def __init__(self, file_name: str, started_at: str, total_s: float):
        self.file_name = file_name
        self.started_at = started_at
        self.total_s = total_s

    def start(self, model: str, from_s: float = 0.0) -> None:
        # Pace is per model (turbo is much faster), so it restarts here.
        self.model = model
        self.from_s = from_s
        self.windows = 0
        self.first_window_at = None
        self._write(eta_s=None)

    def window_done(self) -> None:
        """Called from inside the OpenVINO streamer callback, so it never
        raises (same rationale as switch_requested())."""
        try:
            now = time.time()
            self.windows += 1
            if self.first_window_at is None:
                self.first_window_at = now
            eta_s = None
            if self.windows > ETA_MIN_WINDOWS:
                s_per_window = (now - self.first_window_at) / (self.windows - 1)
                windows_left = max(0.0, self.total_s - self._done_s()) / SPEECH_S_PER_WINDOW
                eta_s = round(windows_left * s_per_window)
            self._write(eta_s)
        except Exception:
            pass

    def _done_s(self) -> float:
        return self.from_s + self.windows * SPEECH_S_PER_WINDOW

    def _write(self, eta_s) -> None:
        fraction = min(self._done_s() / self.total_s, MAX_ESTIMATED_PROGRESS) if self.total_s else 0.0
        write_status(
            "processing", file=self.file_name, stage="transcribing", started_at=self.started_at,
            model=self.model, progress=round(fraction, 3), eta_s=eta_s,
        )


def _generate(model: str, audio: np.ndarray, streamer=None, **kwargs):
    return get_pipeline(model).generate(
        audio.tolist(),
        streamer=streamer,
        return_timestamps=True,
        task="transcribe",
        language=f"<|{LANGUAGE}|>",
        **kwargs,
    )


def _segments(result, audio_len_s: float, offset_s: float = 0.0) -> list:
    """(start, end, text) tuples in trimmed-audio seconds, shifted by offset_s."""
    if not result.chunks:
        text = result.texts[0] if result.texts else ""
        return [(offset_s, offset_s + audio_len_s, text)]
    return [(offset_s + c.start_ts, offset_s + c.end_ts, c.text) for c in result.chunks]


# Two ways a ~30s Whisper window goes wrong, both checked after transcription
# (thresholds calibrated on every transcript + the hotword test runs as of
# 2026-09-29):
#
#  - Runaway: stuck repeating itself ("yeah yeah yeah..." x444) and never
#    closing the segment, which OpenVINO reports as an end timestamp of -1.
#    openai-whisper's own repetition check is a compression ratio above 2.4,
#    but that's too low here: Discrete lectures read binary strings aloud
#    ("0, 0, 0, 0, 0, 1, 1, 0...") that score up to 2.46, while the real
#    loops seen so far scored 28.9 and 71.6.
#  - Skipped: Whisper writes a scrap ("Thank you.", "Thank you for
#    watching.") and jumps to the end of the window, silently dropping the
#    speech in between -- one segment covering 30+ seconds with a handful of
#    words. Hotwords make this likelier (3 lost passages in Micro sept 28),
#    but plain large-v3 does it too. Genuine pauses ("I'll give you a minute
#    to think") look the same, so a retry is only kept if it recovers
#    clearly more words.
LOOP_COMPRESSION_RATIO = 3.0
SKIP_MIN_S = 15.0  # a segment at least this long (seconds of speech, silence already trimmed)...
SKIP_MAX_WORDS_PER_S = 0.5  # ...with fewer words per second than this (normal speech is ~2-3)


def _compression_ratio(text: str) -> float:
    data = text.encode("utf-8")
    return len(data) / len(zlib.compress(data)) if data else 0.0


def _problem(segment):
    """"runaway", "skipped", or None."""
    start, end, text = segment
    if end < start or _compression_ratio(text) > LOOP_COMPRESSION_RATIO:
        return "runaway"
    if end - start >= SKIP_MIN_S and len(text.split()) < SKIP_MAX_WORDS_PER_S * (end - start):
        return "skipped"
    return None


def _is_better(problem: str, original, candidate: list) -> bool:
    if any(_problem(c) == "runaway" for c in candidate):
        return False
    if problem == "runaway":
        return True
    old_words = len(original[2].split())
    new_words = sum(len(c[2].split()) for c in candidate)
    return new_words >= max(2 * old_words, old_words + 10)


def repair_segments(segments: list, trimmed_audio: np.ndarray, attempts: dict) -> list:
    """Re-transcribe the audio under each runaway or skipped segment,
    trying each (model, generate kwargs) in attempts[problem] in turn and
    keeping the first result that's actually better.

    If none is: a runaway with a broken timestamp is pure loop and is
    dropped (a gap beats 444 "yeah"s); anything else is kept as-is."""
    repaired = []
    for i, segment in enumerate(segments):
        problem = _problem(segment)
        if problem is None:
            repaired.append(segment)
            continue
        start, end, text = segment
        # A runaway's end timestamp is missing, and a skip may have jumped
        # past audio, so the span runs to wherever the next segment picks up.
        span_end = segments[i + 1][0] if i + 1 < len(segments) else len(trimmed_audio) / SAMPLING_RATE
        span = trimmed_audio[int(start * SAMPLING_RATE):int(span_end * SAMPLING_RATE)]
        logger.warning(
            "  %s segment at %.0fs of speech (%d words over %.0fs: %r) -- re-transcribing that span.",
            problem.capitalize(), start, len(text.split()), len(span) / SAMPLING_RATE, text[:40],
        )
        replacement = None
        if len(span) >= SAMPLING_RATE:  # under a second of audio isn't worth retrying
            for model, kwargs in attempts[problem]:
                result = _generate(model, span, **kwargs)
                candidate = _segments(result, len(span) / SAMPLING_RATE, offset_s=start)
                if _is_better(problem, segment, candidate):
                    replacement = candidate
                    logger.info(
                        "    Fixed with %s%s: %r", model, " + hotwords" if kwargs else "",
                        " ".join(c[2].strip() for c in candidate)[:80],
                    )
                    break
        if replacement is not None:
            repaired.extend(replacement)
        elif end < start:
            logger.warning("    Couldn't fix it; dropping the looped text.")
        else:
            logger.info("    No retry was better; keeping the original.")
            repaired.append(segment)
    return repaired


def transcribe(wav_path: Path, out_txt: Path, file_name: str, started_at: str) -> tuple[float, str]:
    """Returns (original pre-VAD-trim audio duration in minutes, model(s)
    used). The duration lets callers record it alongside processing time --
    e.g. a 90-minute lecture that took 3 minutes to transcribe vs. a
    5-minute clip that took 3 minutes are very different signals.

    Runs DEFAULT_MODEL unless the dashboard requests FAST_MODEL for this
    recording. A request that arrives mid-run stops large-v3 at the next
    ~30s window boundary, keeps everything it finished, and has turbo
    transcribe only the rest."""
    audio = decode_audio(str(wav_path), sampling_rate=SAMPLING_RATE)
    audio_duration_min = len(audio) / SAMPLING_RATE / 60
    trimmed_audio, ts_map = strip_silence(audio)
    trimmed_s = len(trimmed_audio) / SAMPLING_RATE

    class_label = classify_filename(file_name)
    hotword_kwargs = {"hotwords": CLASS_HOTWORDS[class_label]} if class_label in CLASS_HOTWORDS else {}
    logger.info("  Class guess: %s (%s)", class_label, "using its hotwords" if hotword_kwargs else "no hotwords")

    segments = []
    models_used = []
    progress = Progress(file_name, started_at, trimmed_s)
    fast_from_s = 0.0 if switch_requested(file_name) else None  # None = turbo not needed

    if fast_from_s is None:
        progress.start(DEFAULT_MODEL)
        switched = False

        def check_for_switch(_text):
            # Called by OpenVINO once per decoded ~30s window. STOP makes
            # generate() return early with the chunks finished so far.
            nonlocal switched
            progress.window_done()
            if switch_requested(file_name):
                switched = True
                return ovg.StreamingStatus.STOP
            return ovg.StreamingStatus.RUNNING

        result = _generate(DEFAULT_MODEL, trimmed_audio, streamer=check_for_switch, **hotword_kwargs)
        if switched:
            # Resume at the end of large-v3's last chunk -- the same point
            # Whisper's own long-form loop would have continued from.
            segments = [(c.start_ts, c.end_ts, c.text) for c in result.chunks]
            if segments and segments[-1][1] < segments[-1][0]:
                segments.pop()  # a runaway has no usable end -- let turbo redo that stretch
            fast_from_s = segments[-1][1] if segments else 0.0
            logger.info(
                "  Switch to %s requested -- keeping %s output for the first %.1f of %.1f min of speech.",
                FAST_MODEL, DEFAULT_MODEL, fast_from_s / 60, trimmed_s / 60,
            )
        else:
            segments = _segments(result, trimmed_s)
        if segments:
            models_used.append(DEFAULT_MODEL)

    if fast_from_s is not None:
        remainder = trimmed_audio[int(fast_from_s * SAMPLING_RATE):]
        remainder_s = len(remainder) / SAMPLING_RATE
        if fast_from_s == 0.0 or remainder_s >= MIN_FAST_REMAINDER_S:
            progress.start(FAST_MODEL, from_s=fast_from_s)
            logger.info("  Transcribing %.1f min of speech on %s...", remainder_s / 60, FAST_MODEL)
            prompt_kwargs = {}
            if segments:
                prompt = " ".join(text.strip() for _, _, text in segments[-FAST_PROMPT_SEGMENTS:])
                prompt_kwargs["initial_prompt"] = prompt[-FAST_PROMPT_MAX_CHARS:]

            def track_progress(_text):
                progress.window_done()
                return ovg.StreamingStatus.RUNNING

            result = _generate(FAST_MODEL, remainder, streamer=track_progress, **prompt_kwargs, **hotword_kwargs)
            segments += _segments(result, remainder_s, offset_s=fast_from_s)
            models_used.append(FAST_MODEL)

    # Retrying a runaway with the same settings just loops again -- the audio
    # itself is repetitive (a class chanting binary digits, a room answering
    # "yes, yes") -- so each retry changes the model and/or the prompt. Order
    # from testing on the two real loops: turbo without hotwords fixed both,
    # large-v3 with hotwords fixed one, the other two fixed neither. Skips
    # are retried without hotwords, their likelier cause.
    attempts = {
        "runaway": [(FAST_MODEL, {}), (DEFAULT_MODEL, hotword_kwargs), (DEFAULT_MODEL, {}), (FAST_MODEL, hotword_kwargs)],
        "skipped": [(DEFAULT_MODEL, {}), (FAST_MODEL, {})],
    }
    for problem, tries in attempts.items():  # without hotwords, some of these are duplicates
        attempts[problem] = [t for n, t in enumerate(tries) if t not in tries[:n]]
    if any(_problem(s) for s in segments):
        # Its own stage, so the dashboard doesn't sit at 99% through the retries.
        write_status("processing", file=file_name, stage="repairing_segments", started_at=started_at, model=" -> ".join(models_used))
    segments = repair_segments(segments, trimmed_audio, attempts)

    with open(out_txt, "w", encoding="utf-8") as f:
        for start, end, text in segments:
            if ts_map is not None:
                start = ts_map.get_original_time(start)
                end = ts_map.get_original_time(end, is_end=True)
            f.write(f"[{start:.2f}s -> {end:.2f}s] {text}\n")

    return audio_duration_min, " -> ".join(models_used)


def process_file(path: Path) -> None:
    logger.info("New recording detected: %s", path.name)
    clear_switch_request()  # e.g. one left behind by a crash (os._exit skips `finally`)
    keep_awake.start(path.name)  # closing the lid / going idle won't sleep until this recording is done
    write_status("processing", file=path.name, stage="waiting_for_stable_file", started_at=datetime.now().isoformat(timespec="seconds"))

    if not wait_for_stable_file(path):
        logger.warning("Timed out waiting for %s to finish syncing. Skipping for now.", path.name)
        write_status("idle", note=f"Timed out waiting for {path.name} to sync; skipped")
        keep_awake.stop()
        return

    try:
        try:
            write_status("processing", file=path.name, stage="converting_audio")
            wav = convert_to_wav(path)
        except subprocess.CalledProcessError as e:
            write_status("error", file=path.name, error=str(e))
            notify_failure(path.stem, e)
            keep_awake.stop()  # os._exit skips `finally`
            os._exit(1)  # die loudly -- see module docstring

        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        out_txt = TRANSCRIPT_FOLDER / f"{path.stem}_{stamp}.txt"

        logger.info("  Transcribing on GPU...")
        started_at_iso = datetime.now().isoformat(timespec="seconds")
        start = time.time()
        audio_duration_min, model_used = transcribe(wav, out_txt, path.name, started_at_iso)
        elapsed = (time.time() - start) / 60
        logger.info("  Done in %.1f min (%s) -> %s", elapsed, model_used, out_txt.name)

        wav.unlink(missing_ok=True)
        shutil.move(str(path), str(ARCHIVE_FOLDER / path.name))

        # Post-processing: Claude notes + Notion push. Additive -- the local
        # transcript above is already safely written regardless of what
        # happens here. process_transcript() catches everything internally
        # and only ever logs, so a Claude/Notion failure here is NOT treated
        # as a die-loudly GPU failure -- it's a separate, lower-stakes step.
        write_status("processing", file=path.name, stage="notes_pipeline")
        notes_pipeline.process_transcript(out_txt)

        append_history(path.name, out_txt.name, started_at_iso, elapsed, audio_duration_min, model_used)
        write_status(
            "idle",
            last_file=path.name,
            last_completed_at=datetime.now().isoformat(timespec="seconds"),
            last_elapsed_min=round(elapsed, 1),
        )
    except Exception as e:
        write_status("error", file=path.name, error=str(e))
        notify_failure(path.stem, e)
        keep_awake.stop()  # os._exit skips `finally`
        os._exit(1)  # die loudly -- see module docstring
    finally:
        release_pipeline()  # free GPU/host memory once this lecture is done
        clear_switch_request()
        keep_awake.stop()


def reconcile_on_startup() -> None:
    """Catch anything that arrived while the watcher wasn't running."""
    pending = sorted(
        p for p in WATCH_FOLDER.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not pending:
        logger.info("Startup scan: nothing pending.")
        return
    logger.info("Startup scan: found %d unprocessed file(s), catching up...", len(pending))
    for p in pending:
        process_file(p)


class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix.lower() in AUDIO_EXTENSIONS:
            process_file(p)


if __name__ == "__main__":
    # Put back a lid-close setting a crash/kill left on "Do nothing" -- see keep_awake.py.
    keep_awake.stop()

    try:
        require_gpu()  # fail fast and clearly if the iGPU isn't available -- no silent CPU fallback
    except Exception as e:
        write_status("error", error=str(e), note="GPU startup check failed -- watcher exited")
        notify_failure("GPU_startup_check", e)
        os._exit(1)  # die loudly -- see module docstring

    write_status("idle", note="Startup scan running")
    reconcile_on_startup()

    observer = Observer()
    observer.schedule(Handler(), str(WATCH_FOLDER), recursive=False)
    observer.start()
    logger.info("Watching: %s", WATCH_FOLDER)
    logger.info("Idle between lectures -- model only loads when a file shows up.")
    logger.info("Leave this process running. Ctrl+C to stop.")
    write_status("idle", note="Watching for new recordings")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
