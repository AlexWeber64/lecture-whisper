"""
config.py

Loads config.json (next to this file): the machine-specific folders,
model locations, and per-class settings that lecture_watcher_gpu.py and
dashboard_server.py read. config.json is git-ignored because it holds
personal paths; copy config.example.json to config.json to start one.

Pure standard library, so dashboard_server.py can still run on any Python.
"""

import json
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "config.json"

REQUIRED_KEYS = (
    "watch_folder", "archive_folder", "transcript_folder", "scratch_folder",
    "failure_notice_folders", "models", "classes",
)


class ConfigError(Exception):
    pass


def load() -> dict:
    """Read and sanity-check config.json. Raises ConfigError with a
    message that says what to fix -- callers decide how loudly to fail."""
    if not CONFIG_FILE.exists():
        raise ConfigError(
            f"{CONFIG_FILE} not found. Copy config.example.json to config.json "
            f"and fill in your folders."
        )
    try:
        cfg = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigError(
            f"{CONFIG_FILE.name} isn't valid JSON (line {e.lineno}, column {e.colno}: {e.msg}). "
            f"Remember that backslashes in Windows paths must be doubled."
        ) from e
    missing = [key for key in REQUIRED_KEYS if key not in cfg]
    if missing:
        raise ConfigError(f"{CONFIG_FILE.name} is missing: {', '.join(missing)}. See config.example.json.")
    return cfg
