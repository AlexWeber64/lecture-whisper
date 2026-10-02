"""
dashboard_server.py

Standalone status dashboard for lecture_watcher_gpu.py. Pure standard
library (http.server) -- no extra dependencies, so it can be run with any
Python, not just the project venv.

Two tabs:
  - Live: current watcher state (idle/processing/error), polled from
    status.json, plus a progress bar with time left and a "Switch to
    large-v3-turbo" button while a recording is being transcribed.
  - History: every completed run, read from history.jsonl (append-only,
    one JSON record per file), with client-side filtering by class/time,
    and a link to view each transcript's full text.

Both source files are written by lecture_watcher_gpu.py, defensively (a
failure writing either can never affect the actual transcription). This
server only reads them. Its one control, the switch button, just drops a
model_switch_request.json file that the watcher polls -- it never talks to
the watcher process directly, and can be started/stopped independently at
any time.

"Class" in the History tab is a NAMING-CONVENTION HEURISTIC -- matched
against a fixed list of known course aliases (see CLASS_ALIASES in
lecture_watcher_gpu.py), falling back to "Unknown" -- not the real Notion
classification, which happens later in the separate Claude Cowork
routine this script has no visibility into.

Usage:
    .venv\\Scripts\\python.exe dashboard_server.py
Then open http://localhost:8787 -- refresh is automatic, no need to reload.
"""

import json
import os
import sys
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import config

PORT = 8787
PROJECT_DIR = Path(__file__).parent
STATUS_FILE = PROJECT_DIR / "status.json"
HISTORY_FILE = PROJECT_DIR / "history.jsonl"
# Must match lecture_watcher_gpu.py's SWITCH_REQUEST_FILE / FAST_MODEL.
SWITCH_REQUEST_FILE = PROJECT_DIR / "model_switch_request.json"
FAST_MODEL = "large-v3-turbo"
# Must match keep_awake.py's LID_BACKUP_FILE -- it exists exactly while the
# watcher has lid-close set to "Do nothing".
LID_BACKUP_FILE = PROJECT_DIR / "lid_action_backup.json"
# Stages where switching still changes anything -- by "notes_pipeline" the
# transcript is already written.
SWITCHABLE_STAGES = ("waiting_for_stable_file", "converting_audio", "transcribing")
# Same folder lecture_watcher_gpu.py writes transcripts to, from the shared
# config.json (config.py is standard library only, like this file).
TRANSCRIPT_FOLDER = Path(config.load()["transcript_folder"])

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Lecture Whisper -- Status</title>
<style>
  :root { color-scheme: dark; }
  body {
    font-family: -apple-system, Segoe UI, Roboto, sans-serif;
    background: #14161a; color: #e8e8ea; margin: 0; padding: 2.5rem 1.5rem;
    display: flex; flex-direction: column; align-items: center;
  }
  .card {
    background: #1c1f26; border: 1px solid #2a2e37; border-radius: 12px;
    padding: 2rem 2.5rem; max-width: 860px; width: 100%; box-shadow: 0 4px 24px rgba(0,0,0,0.3);
  }
  h1 { font-size: 1.1rem; font-weight: 600; color: #9aa4b2; margin: 0 0 1.25rem 0; letter-spacing: 0.02em; }
  .tabs { display: flex; gap: 0.5rem; margin-bottom: 1.5rem; border-bottom: 1px solid #2a2e37; }
  .tab-btn {
    background: none; border: none; color: #8890a0; font-size: 0.95rem; font-weight: 600;
    padding: 0.6rem 1.1rem; cursor: pointer; border-bottom: 2px solid transparent; margin-bottom: -1px;
  }
  .tab-btn.active { color: #e8e8ea; border-bottom-color: #6ea8f2; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  .badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.35rem 0.9rem; border-radius: 999px; font-weight: 600; font-size: 0.95rem;
    margin-bottom: 1.25rem;
  }
  .dot { width: 0.6rem; height: 0.6rem; border-radius: 50%; }
  .badge.idle { background: #1a3a2a; color: #6ee7a8; }
  .badge.idle .dot { background: #6ee7a8; }
  .badge.processing { background: #2a2a10; color: #f2d24b; }
  .badge.processing .dot { background: #f2d24b; animation: pulse 1.2s infinite ease-in-out; }
  .badge.error { background: #3a1a1a; color: #f26b6b; }
  .badge.error .dot { background: #f26b6b; }
  .badge.unknown { background: #262626; color: #999; }
  .badge.unknown .dot { background: #999; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
  .row { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #262a33; font-size: 0.95rem; }
  .row:last-child { border-bottom: none; }
  .label { color: #8890a0; }
  .value { color: #e8e8ea; font-weight: 500; text-align: right; max-width: 60%; overflow-wrap: anywhere; }
  .elapsed { font-variant-numeric: tabular-nums; }
  .progress { margin-bottom: 1rem; }
  .progress[hidden] { display: none; }
  .progress-track { height: 0.55rem; background: #262a33; border-radius: 999px; overflow: hidden; }
  .progress-fill { height: 100%; width: 0; background: #6ea8f2; border-radius: 999px; transition: width 0.8s ease-out; }
  .progress-text { display: flex; justify-content: space-between; margin-top: 0.45rem; font-size: 0.85rem; color: #8890a0; font-variant-numeric: tabular-nums; }
  .progress-text strong { color: #e8e8ea; font-weight: 600; }
  .updated { margin-top: 1.5rem; font-size: 0.8rem; color: #5a6070; text-align: right; }
  .controls { margin-top: 1.25rem; display: flex; align-items: center; gap: 0.9rem; flex-wrap: wrap; }
  .controls button {
    background: #1f3350; border: 1px solid #2f4d75; color: #cfe2fb; border-radius: 6px;
    padding: 0.5rem 0.95rem; font-size: 0.88rem; font-weight: 600; cursor: pointer;
  }
  .controls button:hover { background: #26406a; }
  .controls button:disabled { opacity: 0.6; cursor: default; }
  .hint { font-size: 0.8rem; color: #8890a0; }

  .filters { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-bottom: 1.25rem; align-items: flex-end; }
  .filters label { display: flex; flex-direction: column; gap: 0.3rem; font-size: 0.78rem; color: #8890a0; }
  .filters select, .filters input {
    background: #14161a; border: 1px solid #2a2e37; color: #e8e8ea; border-radius: 6px;
    padding: 0.4rem 0.6rem; font-size: 0.85rem;
  }
  .filters button {
    background: #262a33; border: 1px solid #2a2e37; color: #c8ccd4; border-radius: 6px;
    padding: 0.45rem 0.8rem; font-size: 0.82rem; cursor: pointer;
  }
  .filters button:hover { background: #2e333d; }
  table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
  th, td { text-align: left; padding: 0.55rem 0.6rem; border-bottom: 1px solid #262a33; }
  th { color: #8890a0; font-weight: 600; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.03em; }
  tr:hover td { background: #20232b; }
  a.tlink { color: #6ea8f2; text-decoration: none; }
  a.tlink:hover { text-decoration: underline; }
  .empty { color: #5a6070; padding: 1.5rem 0; text-align: center; }
  .class-tag {
    background: #262a33; color: #a8b0c0; padding: 0.15rem 0.5rem; border-radius: 4px; font-size: 0.78rem;
  }
</style>
</head>
<body>
  <div class="card">
    <h1>LECTURE WHISPER</h1>
    <div class="tabs">
      <button class="tab-btn active" data-tab="live" onclick="showTab('live')">Live</button>
      <button class="tab-btn" data-tab="history" onclick="showTab('history')">History</button>
    </div>

    <div id="tab-live" class="tab-panel active">
      <div id="badge" class="badge unknown"><span class="dot"></span><span id="badge-text">Loading...</span></div>
      <div id="progress" class="progress" hidden>
        <div class="progress-track"><div class="progress-fill" id="progress-fill"></div></div>
        <div class="progress-text"><strong id="progress-pct"></strong><span id="progress-eta"></span></div>
      </div>
      <div id="rows"></div>
      <div class="controls" id="controls"></div>
      <div class="updated" id="updated"></div>
    </div>

    <div id="tab-history" class="tab-panel">
      <div class="filters">
        <label>Class
          <select id="filter-class"><option value="">All</option></select>
        </label>
        <label>From
          <input type="date" id="filter-from">
        </label>
        <label>To
          <input type="date" id="filter-to">
        </label>
        <button onclick="clearFilters()">Clear filters</button>
      </div>
      <table>
        <thead>
          <tr><th>Completed</th><th>File</th><th>Class</th><th>Lecture length</th><th>Processing time</th><th>Transcript</th></tr>
        </thead>
        <tbody id="history-body"></tbody>
      </table>
      <div id="history-empty" class="empty" style="display:none;">No transcripts yet.</div>
    </div>
  </div>

<script>
let currentStartedAt = null;
let currentFile = null;
let historyData = [];

function showTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.getElementById('tab-live').classList.toggle('active', name === 'live');
  document.getElementById('tab-history').classList.toggle('active', name === 'history');
}

function fmtElapsed(iso) {
  if (!iso) return null;
  const started = new Date(iso).getTime();
  const secs = Math.max(0, Math.floor((Date.now() - started) / 1000));
  const m = Math.floor(secs / 60), s = secs % 60;
  return m + "m " + String(s).padStart(2, "0") + "s";
}

function fmtEta(secs) {
  if (secs == null) return 'Estimating time left...';
  if (secs < 60) return 'Under a minute left';
  return 'About ' + Math.round(secs / 60) + ' min left';
}

// Updates the bar in place (not via rows.innerHTML) so its width animates
// between polls instead of jumping.
function renderProgress(data) {
  const box = document.getElementById('progress');
  const fill = document.getElementById('progress-fill');
  const show = data.state === 'processing' && data.stage === 'transcribing' && data.progress != null;
  box.hidden = !show;
  if (!show) {
    fill.style.width = '0';  // so the next recording's bar starts empty
    return;
  }
  const pct = Math.floor(data.progress * 100);
  fill.style.width = pct + '%';
  document.getElementById('progress-pct').textContent = pct + '%';
  document.getElementById('progress-eta').textContent = fmtEta(data.eta_s);
}

function row(label, value) {
  return '<div class="row"><span class="label">' + label + '</span><span class="value">' + value + '</span></div>';
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function pollLive() {
  let data;
  try {
    const resp = await fetch('/status', {cache: 'no-store'});
    data = await resp.json();
  } catch (e) {
    data = {state: 'unknown'};
  }

  const badge = document.getElementById('badge');
  const badgeText = document.getElementById('badge-text');
  const rows = document.getElementById('rows');
  const controls = document.getElementById('controls');
  const updated = document.getElementById('updated');

  const state = data.state || 'unknown';
  badge.className = 'badge ' + state;
  renderProgress(data);

  let html = '';
  let controlsHtml = '';
  if (state === 'processing') {
    badgeText.textContent = 'Transcribing';
    currentStartedAt = data.started_at || currentStartedAt;
    currentFile = data.file;
    html += row('File', escapeHtml(data.file || '?'));
    html += row('Stage', escapeHtml((data.stage || '?').replaceAll('_', ' ')));
    if (data.model) html += row('Model', escapeHtml(data.model));
    html += row('Elapsed', '<span class="elapsed" id="elapsed">' + (fmtElapsed(currentStartedAt) || '--') + '</span>');
    html += row('Lid close', data.lid_stays_awake ? 'Safe -- keeps running until this is done' : 'Will sleep and pause this (see log)');
    if (data.can_switch && data.switch_requested) {
      controlsHtml = '<span class="hint">' + (data.stage === 'transcribing'
        ? 'Switching to large-v3-turbo -- large-v3 finishes its current 30-second window, then turbo takes over.'
        : 'large-v3-turbo will be used once transcription starts.') + '</span>';
    } else if (data.can_switch) {
      controlsHtml = '<button id="switch-btn" onclick="switchToFast()">Switch to large-v3-turbo</button>' +
        '<span class="hint">Faster, slightly less accurate. Anything already transcribed is kept.</span>';
    }
  } else if (state === 'idle') {
    badgeText.textContent = 'Idle -- watching for new recordings';
    currentStartedAt = null;
    if (data.last_file) {
      html += row('Last processed', escapeHtml(data.last_file));
      html += row('Completed at', data.last_completed_at ? new Date(data.last_completed_at).toLocaleString() : '?');
      html += row('Took', (data.last_elapsed_min != null ? data.last_elapsed_min + ' min' : '?'));
    }
    if (data.note) html += row('Note', escapeHtml(data.note));
  } else if (state === 'error') {
    badgeText.textContent = 'Error';
    currentStartedAt = null;
    if (data.file) html += row('File', escapeHtml(data.file));
    html += row('Error', escapeHtml(data.error || data.note || 'unknown error'));
  } else {
    badgeText.textContent = 'No status yet -- is the watcher running?';
  }
  rows.innerHTML = html;
  // Only re-render the controls when they actually change, so the 2s poll
  // doesn't swap the button out from under a click.
  if (controls.dataset.html !== controlsHtml) {
    controls.innerHTML = controlsHtml;
    controls.dataset.html = controlsHtml;
  }
  updated.textContent = data.updated_at ? ('Last update: ' + new Date(data.updated_at).toLocaleTimeString()) : '';
}

async function switchToFast() {
  const btn = document.getElementById('switch-btn');
  btn.disabled = true;
  btn.textContent = 'Requesting...';
  try {
    const resp = await fetch('/switch-model', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({file: currentFile}),
    });
    const result = await resp.json();
    if (!result.ok) alert(result.error || 'Switch request failed.');
  } catch (e) {
    alert('Could not reach the dashboard server.');
  }
  document.getElementById('controls').dataset.html = '';  // force a re-render either way
  pollLive();
}

function populateClassFilter() {
  const sel = document.getElementById('filter-class');
  const current = sel.value;
  const classes = [...new Set(historyData.map(r => r.class_guess || 'Unknown'))].sort();
  sel.innerHTML = '<option value="">All</option>' + classes.map(c => '<option value="' + escapeHtml(c) + '">' + escapeHtml(c) + '</option>').join('');
  sel.value = current;
}

function clearFilters() {
  document.getElementById('filter-class').value = '';
  document.getElementById('filter-from').value = '';
  document.getElementById('filter-to').value = '';
  renderHistory();
}

function renderHistory() {
  const cls = document.getElementById('filter-class').value;
  const from = document.getElementById('filter-from').value;
  const to = document.getElementById('filter-to').value;

  let rows = historyData;
  if (cls) rows = rows.filter(r => (r.class_guess || 'Unknown') === cls);
  if (from) rows = rows.filter(r => r.completed_at && r.completed_at.slice(0, 10) >= from);
  if (to) rows = rows.filter(r => r.completed_at && r.completed_at.slice(0, 10) <= to);

  const body = document.getElementById('history-body');
  const empty = document.getElementById('history-empty');
  if (rows.length === 0) {
    body.innerHTML = '';
    empty.style.display = 'block';
    return;
  }
  empty.style.display = 'none';
  body.innerHTML = rows.map(r => (
    '<tr>' +
    '<td>' + (r.completed_at ? new Date(r.completed_at).toLocaleString() : '?') + '</td>' +
    '<td>' + escapeHtml(r.file || '?') + '</td>' +
    '<td><span class="class-tag">' + escapeHtml(r.class_guess || 'Unknown') + '</span></td>' +
    '<td>' + (r.audio_duration_min != null ? r.audio_duration_min + ' min' : '?') + '</td>' +
    '<td>' + (r.elapsed_min != null ? r.elapsed_min + ' min' : '?') + '</td>' +
    '<td><a class="tlink" href="/transcript/' + encodeURIComponent(r.id) + '" target="_blank">View</a></td>' +
    '</tr>'
  )).join('');
}

async function loadHistory() {
  try {
    const resp = await fetch('/history', {cache: 'no-store'});
    historyData = await resp.json();
  } catch (e) {
    historyData = [];
  }
  populateClassFilter();
  renderHistory();
}

document.getElementById('filter-class').addEventListener('change', renderHistory);
document.getElementById('filter-from').addEventListener('change', renderHistory);
document.getElementById('filter-to').addEventListener('change', renderHistory);

setInterval(pollLive, 2000);
setInterval(() => {
  const el = document.getElementById('elapsed');
  if (el && currentStartedAt) el.textContent = fmtElapsed(currentStartedAt);
}, 1000);
setInterval(loadHistory, 15000);
pollLive();
loadHistory();
</script>
</body>
</html>
"""

TRANSCRIPT_PAGE = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>{title}</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: #14161a; color: #e8e8ea; margin: 0; padding: 2rem; }}
  h1 {{ font-size: 1rem; color: #9aa4b2; margin: 0 0 1rem 0; }}
  pre {{ white-space: pre-wrap; word-wrap: break-word; font-family: Consolas, monospace; font-size: 0.9rem; line-height: 1.6; }}
</style>
</head>
<body>
<h1>{title}</h1>
<pre>{content}</pre>
</body>
</html>
"""


def _load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    records = []
    for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a corrupted last line from a mid-write crash
    records.reverse()  # newest first
    return records


def _load_status() -> dict:
    if not STATUS_FILE.exists():
        return {"state": "unknown", "note": "status.json not found -- has the watcher been started?"}
    try:
        status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"state": "unknown"}
    status["can_switch"] = (
        status.get("state") == "processing"
        and status.get("stage") in SWITCHABLE_STAGES
        and status.get("model") != FAST_MODEL
    )
    try:
        requested_file = json.loads(SWITCH_REQUEST_FILE.read_text(encoding="utf-8")).get("file")
    except (json.JSONDecodeError, OSError, AttributeError):
        requested_file = None
    status["switch_requested"] = requested_file is not None and requested_file == status.get("file")
    status["lid_stays_awake"] = LID_BACKUP_FILE.exists()
    return status


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # keep the console quiet; this is a status page, not a debug tool

    def _send(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))

        elif path == "/status":
            self._send(200, "application/json", json.dumps(_load_status()).encode("utf-8"))

        elif path == "/history":
            self._send(200, "application/json", json.dumps(_load_history()).encode("utf-8"))

        elif path.startswith("/transcript/"):
            record_id = urllib.parse.unquote(path[len("/transcript/"):])
            record = next((r for r in _load_history() if r.get("id") == record_id), None)
            if record is None:
                self._send(404, "text/plain", b"No such transcript record.")
                return
            # Resolve against TRANSCRIPT_FOLDER only -- never trust a raw
            # client-supplied path. record["transcript_filename"] is a bare
            # filename the watcher itself wrote, but re-validate anyway.
            candidate = (TRANSCRIPT_FOLDER / record["transcript_filename"]).resolve()
            try:
                candidate.relative_to(TRANSCRIPT_FOLDER.resolve())
            except ValueError:
                self._send(403, "text/plain", b"Refused: path escapes the transcripts folder.")
                return
            if not candidate.is_file():
                self._send(404, "text/plain", f"Transcript file not found on disk: {candidate.name}".encode("utf-8"))
                return
            text = candidate.read_text(encoding="utf-8", errors="replace")
            page = TRANSCRIPT_PAGE.format(
                title=record.get("file", candidate.name).replace("<", "&lt;").replace(">", "&gt;"),
                content=text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
            )
            self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))

        else:
            self._send(404, "text/plain", b"Not found.")

    def _send_json(self, status: int, payload: dict):
        self._send(status, "application/json", json.dumps(payload).encode("utf-8"))

    def do_POST(self):
        if urllib.parse.urlsplit(self.path).path != "/switch-model":
            self._send(404, "text/plain", b"Not found.")
            return
        # A browser can't send a cross-origin application/json POST without
        # a CORS preflight, which this server never approves -- so requiring
        # it keeps any other website open in the browser from hitting this.
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self._send_json(415, {"ok": False, "error": "Expected application/json."})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            file_name = body["file"]
        except (ValueError, KeyError, TypeError):
            self._send_json(400, {"ok": False, "error": "Expected a JSON body with a 'file' field."})
            return

        # Requiring the file name (not just "whatever's running") means a
        # click from a stale page can't switch a different recording that
        # started in the meantime.
        status = _load_status()
        if not status.get("can_switch"):
            self._send_json(409, {"ok": False, "error": "Nothing is being transcribed right now that can be switched."})
            return
        if status.get("file") != file_name:
            self._send_json(409, {"ok": False, "error": f"{file_name} is no longer the recording being transcribed."})
            return

        request = {"file": file_name, "requested_at": datetime.now().isoformat(timespec="seconds")}
        tmp_path = SWITCH_REQUEST_FILE.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(request), encoding="utf-8")
        os.replace(tmp_path, SWITCH_REQUEST_FILE)
        self._send_json(200, {"ok": True})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    if sys.stdout is not None:
        # Under pythonw.exe (no console, e.g. launched via Task Scheduler)
        # sys.stdout is None, and a bare print() would raise AttributeError
        # before the server ever starts -- this is exactly what broke the
        # scheduled task on the first attempt: pythonw silently discards
        # both the exception's traceback AND stdout/stderr, so it looked
        # like a clean, instant, unexplained exit.
        print(f"Dashboard running at http://localhost:{PORT}")
        print("Reads status.json + history.jsonl in this folder -- written by lecture_watcher_gpu.py.")
        print("Independent of the watcher (the switch-model button only writes a request file); Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
