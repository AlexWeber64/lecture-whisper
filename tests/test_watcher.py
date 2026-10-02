"""Pure logic in lecture_watcher_gpu.py: course matching, the repair pass's
problem detection, and dashboard progress. See conftest.py for how it's
imported without a GPU."""

from types import SimpleNamespace

import pytest

import lecture_watcher_gpu as w

ECON = "ECON101 - Microeconomics"
ART = "ART110 - Art History"


# --- Course matching from the filename ------------------------------------

@pytest.mark.parametrize("name", ["Micro sept 24.m4a", "ECON101 review.m4a", "econ midterm.m4a"])
def test_aliases_map_different_names_to_one_course(name):
    assert w.classify_filename(name) == ECON


def test_caret_alias_only_matches_at_the_start():
    assert w.classify_filename("Art history.m4a") == ART
    assert w.classify_filename("Smart phones.m4a") == "Unknown"
    assert w.classify_filename("Start of term.m4a") == "Unknown"


def test_unmatched_name_is_unknown():
    assert w.classify_filename("Voice memo 12.m4a") == "Unknown"


# --- Repair pass: spotting runaway loops and skipped passages -------------

NORMAL_TEXT = (
    " So the demand curve shifts out, but the quantity supplied moves along the curve,"
    " which is exactly why we separate a change in supply from a change in quantity supplied."
)


def test_normal_segment_is_fine():
    assert w._problem((0.0, 12.0, NORMAL_TEXT)) is None


def test_broken_end_timestamp_is_a_runaway():
    assert w._problem((40.0, -1.0, " yeah")) == "runaway"


def test_repetitive_text_is_a_runaway():
    assert w._problem((0.0, 30.0, " yeah" * 200)) == "runaway"


def test_long_segment_with_almost_no_words_is_a_skip():
    assert w._problem((0.0, 32.0, " Thank you.")) == "skipped"


def test_short_pause_is_not_a_skip():
    # Under SKIP_MIN_S, so a brief "Okay." is just a pause.
    assert w._problem((0.0, 10.0, " Okay.")) is None


def test_any_clean_retry_beats_a_runaway():
    assert w._is_better("runaway", (0.0, -1.0, " yeah" * 50), [(0.0, 5.0, " Right.")])


def test_a_retry_that_loops_is_never_better():
    assert not w._is_better("skipped", (0.0, 32.0, " Thank you."), [(0.0, 32.0, " yeah" * 200)])


def words(n):
    """n distinct words (repeating one word would read as a runaway loop)."""
    return "".join(f" w{i}" for i in range(n))


def test_skip_retry_must_recover_clearly_more_words():
    # Needs at least twice the words and at least 10 more.
    original = (0.0, 32.0, " Thank you.")  # 2 words -> needs 12
    assert not w._is_better("skipped", original, [(0.0, 32.0, words(11))])
    assert w._is_better("skipped", original, [(0.0, 32.0, words(12))])

    original = (0.0, 40.0, words(15))  # 15 words -> needs 30
    assert not w._is_better("skipped", original, [(0.0, 40.0, words(29))])
    assert w._is_better("skipped", original, [(0.0, 40.0, words(30))])


def test_segments_are_shifted_by_the_offset():
    result = SimpleNamespace(
        chunks=[SimpleNamespace(start_ts=0.0, end_ts=4.0, text=" a"), SimpleNamespace(start_ts=4.0, end_ts=9.5, text=" b")],
        texts=[],
    )
    assert w._segments(result, 9.5, offset_s=100.0) == [(100.0, 104.0, " a"), (104.0, 109.5, " b")]


def test_result_without_chunks_becomes_one_segment():
    result = SimpleNamespace(chunks=[], texts=[" all of it"])
    assert w._segments(result, 30.0, offset_s=10.0) == [(10.0, 40.0, " all of it")]


# --- Dashboard progress ----------------------------------------------------

@pytest.fixture
def progress_env(monkeypatch):
    """Captures status writes and lets each test set the clock."""
    writes = []
    clock = [1000.0]
    monkeypatch.setattr(w, "write_status", lambda state, **fields: writes.append({"state": state, **fields}))
    monkeypatch.setattr(w.time, "time", lambda: clock[0])
    return writes, clock


def test_progress_starts_at_zero_without_an_eta(progress_env):
    writes, _ = progress_env
    w.Progress("a.m4a", "2026-10-02T10:00:00", total_s=1000.0).start(w.DEFAULT_MODEL)
    assert writes[-1] == {
        "state": "processing", "file": "a.m4a", "stage": "transcribing",
        "started_at": "2026-10-02T10:00:00", "model": w.DEFAULT_MODEL, "progress": 0.0, "eta_s": None,
    }


def test_each_window_advances_by_the_measured_average(progress_env):
    writes, _ = progress_env
    p = w.Progress("a.m4a", "t", total_s=1000.0)
    p.start(w.DEFAULT_MODEL)
    p.window_done()
    p.window_done()
    assert writes[-1]["progress"] == round(2 * w.SPEECH_S_PER_WINDOW / 1000.0, 3)


def test_eta_waits_for_enough_windows_then_uses_this_runs_pace(progress_env):
    writes, clock = progress_env
    p = w.Progress("a.m4a", "t", total_s=1000.0)
    p.start(w.DEFAULT_MODEL)
    for _ in range(w.ETA_MIN_WINDOWS):
        p.window_done()
        assert writes[-1]["eta_s"] is None
        clock[0] += 3.0  # 3 seconds per window
    p.window_done()
    windows_done = w.ETA_MIN_WINDOWS + 1
    windows_left = (1000.0 - windows_done * w.SPEECH_S_PER_WINDOW) / w.SPEECH_S_PER_WINDOW
    assert writes[-1]["eta_s"] == round(windows_left * 3.0)


def test_progress_never_claims_to_be_finished(progress_env):
    writes, _ = progress_env
    p = w.Progress("a.m4a", "t", total_s=50.0)
    p.start(w.DEFAULT_MODEL)
    for _ in range(10):
        p.window_done()
    assert writes[-1]["progress"] == w.MAX_ESTIMATED_PROGRESS
    assert writes[-1]["eta_s"] in (None, 0)


def test_model_switch_resumes_from_the_exact_seam(progress_env):
    writes, _ = progress_env
    p = w.Progress("a.m4a", "t", total_s=1000.0)
    p.start(w.DEFAULT_MODEL)
    for _ in range(6):
        p.window_done()
    p.start(w.FAST_MODEL, from_s=400.0)
    assert writes[-1]["model"] == w.FAST_MODEL
    assert writes[-1]["progress"] == 0.4
    assert writes[-1]["eta_s"] is None  # turbo's pace is timed from scratch
    p.window_done()
    assert writes[-1]["progress"] == round((400.0 + w.SPEECH_S_PER_WINDOW) / 1000.0, 3)


def test_window_done_never_raises(monkeypatch):
    # It runs inside OpenVINO's streamer callback, where an exception would
    # abort the transcription itself.
    def broken_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(w, "write_status", broken_write)
    p = w.Progress("a.m4a", "t", total_s=1000.0)
    p.model, p.from_s, p.windows, p.first_window_at = w.DEFAULT_MODEL, 0.0, 0, None
    p.window_done()
