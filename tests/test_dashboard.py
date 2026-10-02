"""dashboard_server.py: when switching is allowed, and the safety rules on its
HTTP endpoints. The server runs for real on a random local port, with every
file it reads redirected into a temporary folder."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

import dashboard_server as d


@pytest.fixture
def files(tmp_path, monkeypatch):
    """Points the dashboard's status, history, and request files at tmp_path."""
    paths = {
        "status": tmp_path / "status.json",
        "history": tmp_path / "history.jsonl",
        "switch": tmp_path / "model_switch_request.json",
        "lid": tmp_path / "lid_action_backup.json",
    }
    monkeypatch.setattr(d, "STATUS_FILE", paths["status"])
    monkeypatch.setattr(d, "HISTORY_FILE", paths["history"])
    monkeypatch.setattr(d, "SWITCH_REQUEST_FILE", paths["switch"])
    monkeypatch.setattr(d, "LID_BACKUP_FILE", paths["lid"])
    d.TRANSCRIPT_FOLDER.mkdir(parents=True, exist_ok=True)
    return paths


def write_status(files, **status):
    files["status"].write_text(json.dumps(status), encoding="utf-8")


# --- When the switch button is offered ------------------------------------

def test_missing_status_file_is_unknown(files):
    assert d._load_status()["state"] == "unknown"


@pytest.mark.parametrize("stage", d.SWITCHABLE_STAGES)
def test_switch_is_offered_while_large_v3_can_still_be_swapped(files, stage):
    write_status(files, state="processing", file="a.m4a", stage=stage, model="large-v3")
    assert d._load_status()["can_switch"]


def test_no_switch_once_on_turbo(files):
    write_status(files, state="processing", file="a.m4a", stage="transcribing", model=d.FAST_MODEL)
    assert not d._load_status()["can_switch"]


@pytest.mark.parametrize("stage", ["repairing_segments", "notes_pipeline"])
def test_no_switch_after_transcription(files, stage):
    write_status(files, state="processing", file="a.m4a", stage=stage, model="large-v3")
    assert not d._load_status()["can_switch"]


def test_switch_request_only_counts_for_the_same_recording(files):
    write_status(files, state="processing", file="a.m4a", stage="transcribing", model="large-v3")
    files["switch"].write_text(json.dumps({"file": "older.m4a"}), encoding="utf-8")
    assert not d._load_status()["switch_requested"]
    files["switch"].write_text(json.dumps({"file": "a.m4a"}), encoding="utf-8")
    assert d._load_status()["switch_requested"]


# --- HTTP endpoints ---------------------------------------------------------

@pytest.fixture
def server(files):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), d.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def get(url):
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8")


def post(url, body, content_type="application/json"):
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def add_history(files, *records):
    with open(files["history"], "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def test_page_and_status_are_served(server, files):
    write_status(files, state="idle", note="Watching for new recordings")
    assert get(server + "/")[0] == 200
    status, body = get(server + "/status")
    assert status == 200 and json.loads(body)["state"] == "idle"


def test_transcript_text_is_html_escaped(server, files):
    (d.TRANSCRIPT_FOLDER / "a.txt").write_text("[0.00s -> 1.00s] <script>alert(1)</script>", encoding="utf-8")
    add_history(files, {"id": "abc", "file": "<b>a.m4a</b>", "transcript_filename": "a.txt"})
    status, body = get(server + "/transcript/abc")
    assert status == 200
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in body
    assert "<b>a.m4a</b>" not in body


def test_transcript_paths_cannot_escape_the_transcripts_folder(server, files):
    (d.TRANSCRIPT_FOLDER.parent / "outside.txt").write_text("secret", encoding="utf-8")
    add_history(files, {"id": "evil", "file": "x", "transcript_filename": "../outside.txt"})
    status, body = get(server + "/transcript/evil")
    assert status == 403
    assert "secret" not in body


def test_unknown_transcript_id_is_404(server, files):
    assert get(server + "/transcript/nope")[0] == 404


def test_damaged_history_line_is_skipped(server, files):
    add_history(files, {"id": "1", "file": "a.m4a"})
    with open(files["history"], "a", encoding="utf-8") as f:
        f.write('{"id": "2", "fi')  # a crash mid-write
    status, body = get(server + "/history")
    assert status == 200 and [r["id"] for r in json.loads(body)] == ["1"]


def test_switch_requires_a_json_body(server, files):
    write_status(files, state="processing", file="a.m4a", stage="transcribing", model="large-v3")
    status, _ = post(server + "/switch-model", {"file": "a.m4a"}, content_type="text/plain")
    assert status == 415
    assert not files["switch"].exists()


def test_switch_refused_when_nothing_is_switchable(server, files):
    write_status(files, state="idle")
    status, result = post(server + "/switch-model", {"file": "a.m4a"})
    assert status == 409 and not result["ok"]
    assert not files["switch"].exists()


def test_switch_refused_for_a_different_recording(server, files):
    write_status(files, state="processing", file="new.m4a", stage="transcribing", model="large-v3")
    status, _ = post(server + "/switch-model", {"file": "old.m4a"})
    assert status == 409
    assert not files["switch"].exists()


def test_switch_writes_a_request_naming_the_recording(server, files):
    write_status(files, state="processing", file="a.m4a", stage="transcribing", model="large-v3")
    status, result = post(server + "/switch-model", {"file": "a.m4a"})
    assert status == 200 and result["ok"]
    assert json.loads(files["switch"].read_text(encoding="utf-8"))["file"] == "a.m4a"
