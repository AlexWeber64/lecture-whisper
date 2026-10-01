"""
keep_awake.py

Keeps the laptop from going into standby while lecture_watcher_gpu.py is
processing a recording, including when the lid is closed. Called from
process_file(): start() when a recording is picked up, stop() when it's
done (or failed).

Without this, closing the lid mid-transcription puts this Modern Standby
(S0 low-power idle) laptop into standby, which freezes the watcher until
the lid is reopened -- confirmed from the Kernel-Power event log against
history.jsonl: runs that hit a lid close or idle timeout paused and then
finished normally after wake, they didn't fail. It just meant the
transcript wasn't ready until the laptop was opened again.

Two separate mechanisms, because Windows treats them separately:
  - Idle timeout: SetThreadExecutionState(ES_SYSTEM_REQUIRED). Released
    automatically by Windows if this process dies, so it can't leak.
  - Lid close: power requests can't override the lid (it's an explicit
    user action), so start() temporarily sets the active power scheme's
    "When I close the lid" action to "Do nothing" (AC and DC) and stop()
    puts the original values back. Closing the lid then only turns the
    screen off.

The lid change is a real system setting, so a crash must never leave it
stuck on "Do nothing". The original values are saved to LID_BACKUP_FILE
*before* anything is changed, and stop() restores from that file -- the
watcher also calls stop() on startup, so a crash/kill is repaired the next
time it launches (at logon, via Task Scheduler).

Works against whatever scheme Windows reports as active (PowerGetActiveScheme)
-- Balanced on this machine, whose own lid action is Sleep. (Until 2026-09-29
the "active scheme" was wrongly the Best Performance overlay GUID; this
handles either.)

Nothing in here ever raises: failing to keep the laptop awake must never
fail a transcription. Problems are logged as warnings instead.
"""

import ctypes
import json
import logging
import os
import uuid
from ctypes import wintypes
from pathlib import Path

logger = logging.getLogger("lecture_watcher_gpu.keep_awake")

LID_BACKUP_FILE = Path(__file__).parent / "lid_action_backup.json"

SUB_BUTTONS = "4f971e89-eebd-4455-a8de-9e59040e7347"
LIDACTION = "5ca83367-6e45-459f-a27b-476b1d01c936"
LID_DO_NOTHING = 0

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.SetThreadExecutionState.restype = wintypes.DWORD
_powrprof = ctypes.WinDLL("powrprof")


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]


def _guid(s: str) -> _GUID:
    return _GUID.from_buffer_copy(uuid.UUID(s).bytes_le)


def _check(rc: int, what: str) -> None:
    if rc != 0:
        raise OSError(rc, f"{what} failed (Win32 error {rc})")


def _active_scheme() -> str:
    ptr = ctypes.POINTER(_GUID)()
    _check(_powrprof.PowerGetActiveScheme(None, ctypes.byref(ptr)), "PowerGetActiveScheme")
    try:
        return str(uuid.UUID(bytes_le=bytes(ptr.contents)))
    finally:
        _kernel32.LocalFree(ptr)


def _read_lid(scheme: str) -> tuple[int, int]:
    values = []
    for fn in (_powrprof.PowerReadACValueIndex, _powrprof.PowerReadDCValueIndex):
        value = wintypes.DWORD()
        _check(
            fn(None, ctypes.byref(_guid(scheme)), ctypes.byref(_guid(SUB_BUTTONS)), ctypes.byref(_guid(LIDACTION)), ctypes.byref(value)),
            fn.__name__,
        )
        values.append(value.value)
    return values[0], values[1]


def _write_lid(scheme: str, ac: int, dc: int) -> None:
    for fn, value in ((_powrprof.PowerWriteACValueIndex, ac), (_powrprof.PowerWriteDCValueIndex, dc)):
        _check(
            fn(None, ctypes.byref(_guid(scheme)), ctypes.byref(_guid(SUB_BUTTONS)), ctypes.byref(_guid(LIDACTION)), wintypes.DWORD(value)),
            fn.__name__,
        )
    # Edits to the active scheme only take effect once it's re-applied. Skip
    # that if the user switched plans mid-run, so we never flip them back.
    if scheme == _active_scheme():
        _check(_powrprof.PowerSetActiveScheme(None, ctypes.byref(_guid(scheme))), "PowerSetActiveScheme")


def start(reason: str) -> None:
    """Block idle standby and lid-close standby until stop()."""
    if not _kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
        logger.warning("  Couldn't block idle sleep (SetThreadExecutionState failed).")

    try:
        if LID_BACKUP_FILE.exists():
            # Left over from a run that never reached stop(). The values in
            # it are the real originals -- the live ones may already be our
            # "Do nothing" -- so keep it rather than overwrite it.
            logger.warning("  Lid backup from an earlier run still present; keeping its original values.")
        else:
            scheme = _active_scheme()
            ac, dc = _read_lid(scheme)
            tmp_path = LID_BACKUP_FILE.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps({"scheme": scheme, "ac": ac, "dc": dc, "reason": reason}), encoding="utf-8")
            os.replace(tmp_path, LID_BACKUP_FILE)  # saved BEFORE changing anything
        backup = json.loads(LID_BACKUP_FILE.read_text(encoding="utf-8"))
        _write_lid(backup["scheme"], LID_DO_NOTHING, LID_DO_NOTHING)
        logger.info("  Lid close set to 'Do nothing' until this recording is done.")
    except Exception as e:
        logger.warning("  Couldn't override lid-close action -- closing the lid will still sleep: %s", e)


def stop() -> None:
    """Undo start(). Safe to call any time, including at startup to repair
    a lid setting left behind by a crash."""
    _kernel32.SetThreadExecutionState(ES_CONTINUOUS)

    if not LID_BACKUP_FILE.exists():
        return
    try:
        backup = json.loads(LID_BACKUP_FILE.read_text(encoding="utf-8"))
        _write_lid(backup["scheme"], backup["ac"], backup["dc"])
        LID_BACKUP_FILE.unlink()
        logger.info("  Lid close action restored (AC=%s, DC=%s).", backup["ac"], backup["dc"])
    except Exception as e:
        # Backup file is kept, so the next stop() (e.g. at next startup) retries.
        logger.warning("  Couldn't restore lid-close action from %s: %s", LID_BACKUP_FILE.name, e)
