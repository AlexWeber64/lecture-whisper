"""
Makes the watcher and dashboard importable anywhere, including a Linux CI
runner with no GPU.

  - Packages that only work on the real laptop (OpenVINO, faster-whisper,
    watchdog, the Windows lid-close code in keep_awake.py, and the
    Claude/Notion notes step) are replaced with empty stand-ins. The tests
    only cover pure logic, so nothing ever calls into them.
  - config.py is pointed at a throwaway config.json whose folders all live
    in a temporary directory, so importing the watcher (which creates its
    folders on import) can't touch OneDrive or anything else real.
  - The watcher's log handler is removed so a local test run never writes
    to the real lecture_watcher_gpu.log.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

for name in (
    "openvino", "openvino_genai",
    "watchdog", "watchdog.observers", "watchdog.events",
    "faster_whisper", "faster_whisper.audio", "faster_whisper.vad",
    "keep_awake", "notes_pipeline",
):
    sys.modules[name] = mock.MagicMock(name=name)
sys.modules["watchdog.events"].FileSystemEventHandler = object  # the watcher subclasses it

TMP = Path(tempfile.mkdtemp(prefix="lecture-whisper-tests-"))
_config = json.loads((REPO / "config.example.json").read_text(encoding="utf-8"))
for key in ("watch_folder", "archive_folder", "transcript_folder", "scratch_folder"):
    _config[key] = str(TMP / key)
_config["failure_notice_folders"] = [str(TMP / "notices")]
# Fixed test classes, so the tests don't depend on the example's sample courses.
_config["classes"] = {
    "ECON101 - Microeconomics": {"aliases": ["econ101", "micro", "^econ"], "hotwords": "ceteris paribus"},
    "ART110 - Art History": {"aliases": ["^art"]},
}
(TMP / "config.json").write_text(json.dumps(_config), encoding="utf-8")

import config  # noqa: E402

config.CONFIG_FILE = TMP / "config.json"

import lecture_watcher_gpu  # noqa: E402

for _handler in list(lecture_watcher_gpu.logger.handlers):
    if isinstance(_handler, logging.FileHandler):
        lecture_watcher_gpu.logger.removeHandler(_handler)
        _handler.close()
