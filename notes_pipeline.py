"""
notes_pipeline.py

Post-processing step run after each transcript is written by
lecture_watcher_gpu.py: sends the transcript to Claude for structured
study notes + course classification (content-based, not filename-based),
then creates a linked entry in the Notion "Notes" database.

ADDITIVE ONLY. By the time this runs, the local .txt transcript already
exists -- that is the one thing this file must never put at risk. Every
public entry point here (process_transcript) catches everything and only
ever logs; it never raises, and it never blocks or crashes the watcher.

Credentials -- environment variables only, never written to any file:
  ANTHROPIC_API_KEY   (console.anthropic.com)
  NOTION_API_KEY      (notion.so/my-integrations -- an internal integration)

The Notion integration must also be explicitly shared with BOTH the
Courses and Notes databases (database "..." menu -> Connections), or the
API can't see them regardless of a valid token.

Optional overrides, only needed if database-title search below is
ambiguous (e.g. you have two databases with "Courses" in the name):
  NOTION_COURSES_DB_ID
  NOTION_NOTES_DB_ID

Design notes / judgment calls:
  - Course matching is content-only: Claude is given the transcript text
    and the live list of course names (fetched from Notion at call time,
    never hardcoded) and picks one or reports AMBIGUOUS_NO_MATCH. The
    JSON schema's `enum` is built from that live list, so Claude is
    structurally incapable of returning a course name that doesn't exist.
  - Notion property names (the Notes database's relation-to-Courses
    property, its title property) are discovered from the live schema at
    call time -- never hardcoded, since the exact property name wasn't
    known in advance.
  - Model: claude-opus-5 with thinking disabled (this task has no tool
    use, so the "tool call leaks into visible text" failure mode doesn't
    apply) and effort=medium -- a reasonable cost/quality balance for a
    task that runs automatically on every lecture. Swap CLAUDE_MODEL to
    "claude-sonnet-5" for a cheaper alternative if desired.
  - Optional: if either API key isn't set, this step is skipped with a
    single log line, so the watcher works fine without it.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

import anthropic
import requests

logger = logging.getLogger("lecture_watcher_gpu.notes_pipeline")

NOTION_VERSION = "2022-06-28"
NOTION_API_BASE = "https://api.notion.com/v1"
CLAUDE_MODEL = "claude-opus-5"


def _notion_headers() -> dict:
    token = os.environ["NOTION_API_KEY"]
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _notion_find_database(name_contains: str) -> dict:
    """Search databases shared with this integration, matching by title.
    Only ever sees databases explicitly shared via Connections."""
    resp = requests.post(
        f"{NOTION_API_BASE}/search",
        headers=_notion_headers(),
        json={"filter": {"value": "database", "property": "object"}},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    matches = []
    for db in results:
        title = "".join(t.get("plain_text", "") for t in db.get("title", []))
        if name_contains.lower() in title.lower():
            matches.append((title, db))
    if not matches:
        raise RuntimeError(
            f"No Notion database shared with this integration has a title containing "
            f"'{name_contains}'. Share it via the database's '...' menu -> Connections."
        )
    if len(matches) > 1:
        titles = ", ".join(t for t, _ in matches)
        raise RuntimeError(
            f"Multiple Notion databases match '{name_contains}': {titles}. "
            f"Set an explicit NOTION_{name_contains.upper()}_DB_ID env var to disambiguate."
        )
    return matches[0][1]


def _get_database(env_var: str, title_contains: str) -> dict:
    db_id = os.environ.get(env_var)
    if db_id:
        resp = requests.get(f"{NOTION_API_BASE}/databases/{db_id}", headers=_notion_headers(), timeout=30)
        resp.raise_for_status()
        return resp.json()
    return _notion_find_database(title_contains)


def get_courses_database() -> dict:
    return _get_database("NOTION_COURSES_DB_ID", "Courses")


def get_notes_database() -> dict:
    return _get_database("NOTION_NOTES_DB_ID", "Notes")


def list_course_entries(courses_db: dict) -> list:
    """Every entry in the Courses database, as [{name, page_id}]."""
    db_id = courses_db["id"]
    title_prop_name = next(
        name for name, prop in courses_db["properties"].items() if prop["type"] == "title"
    )
    courses = []
    cursor = None
    while True:
        body = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        resp = requests.post(
            f"{NOTION_API_BASE}/databases/{db_id}/query",
            headers=_notion_headers(), json=body, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        for page in data.get("results", []):
            title_parts = page["properties"].get(title_prop_name, {}).get("title", [])
            name = "".join(t.get("plain_text", "") for t in title_parts)
            if name:
                courses.append({"name": name, "page_id": page["id"]})
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return courses


def find_notes_relation_property(notes_db: dict, courses_db_id: str) -> str:
    """The Notes database's relation property pointing at Courses --
    discovered from the live schema, never hardcoded (the exact property
    name -- 'Course', 'Class', 'Related Course', etc. -- wasn't known)."""
    normalized_courses_id = courses_db_id.replace("-", "")
    relation_props = []
    for name, prop in notes_db["properties"].items():
        if prop["type"] != "relation":
            continue
        relation_props.append(name)
        target = prop.get("relation", {}).get("database_id", "")
        if target.replace("-", "") == normalized_courses_id:
            return name
    if len(relation_props) == 1:
        return relation_props[0]
    raise RuntimeError(
        f"Could not find a relation property on the Notes database pointing to Courses. "
        f"Relation properties found: {relation_props or 'none'}."
    )


def find_notes_title_property(notes_db: dict) -> str:
    return next(name for name, prop in notes_db["properties"].items() if prop["type"] == "title")


def _text_block(text: str) -> dict:
    return {
        "object": "block", "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": text[:2000]}}]},
    }


def _heading(text: str) -> dict:
    return {
        "object": "block", "type": "heading_2",
        "heading_2": {"rich_text": [{"type": "text", "text": {"content": text[:2000]}}]},
    }


def _bulleted(text: str) -> dict:
    return {
        "object": "block", "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": text[:2000]}}]},
    }


def create_notes_page(
    notes_db_id: str, title_prop: str, relation_prop: str, course_page_id: str,
    title: str, summary: str, key_concepts: list, review_questions: list,
) -> str:
    children = [_heading("Summary"), _text_block(summary), _heading("Key Concepts")]
    children += [_bulleted(c) for c in key_concepts]
    children.append(_heading("Exam Review Questions"))
    children += [_bulleted(q) for q in review_questions]

    body = {
        "parent": {"database_id": notes_db_id},
        "properties": {
            title_prop: {"title": [{"text": {"content": title[:2000]}}]},
            relation_prop: {"relation": [{"id": course_page_id}]},
        },
        "children": children[:100],  # Notion's per-request children cap
    }
    resp = requests.post(f"{NOTION_API_BASE}/pages", headers=_notion_headers(), json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()["id"]


def generate_notes_and_classification(transcript_text: str, course_names: list) -> Optional[dict]:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    schema = {
        "type": "object",
        "properties": {
            "course_name": {
                "type": "string",
                "enum": course_names + ["AMBIGUOUS_NO_MATCH"],
                "description": (
                    "The course this lecture belongs to, based purely on the transcript's "
                    "content. Use AMBIGUOUS_NO_MATCH if the content doesn't clearly match "
                    "any listed course -- do not guess."
                ),
            },
            "match_reasoning": {
                "type": "string",
                "description": "One or two sentences on why this course was chosen (or why it's ambiguous).",
            },
            "title": {
                "type": "string",
                "description": "A short, descriptive title for this lecture's notes (course + topic).",
            },
            "summary": {"type": "string", "description": "A few paragraphs summarizing the lecture."},
            "key_concepts": {
                "type": "array", "items": {"type": "string"},
                "description": "Key concepts/terms covered, each a short standalone bullet.",
            },
            "review_questions": {
                "type": "array", "items": {"type": "string"},
                "description": "Exam-style review questions covering the lecture content.",
            },
        },
        "required": ["course_name", "match_reasoning", "title", "summary", "key_concepts", "review_questions"],
        "additionalProperties": False,
    }

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=8000,
        thinking={"type": "disabled"},
        output_config={
            "effort": "medium",
            "format": {"type": "json_schema", "schema": schema},
        },
        system=(
            "You are a study-notes assistant for a college student. You will be given a raw "
            "lecture transcript (auto-generated, may contain minor transcription errors) and "
            "the student's actual current course names. Produce structured study notes and "
            "classify which course this lecture belongs to, based purely on the transcript's "
            "content -- not on filenames or any other metadata, which you are not given."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Valid course names:\n{json.dumps(course_names)}\n\n"
                f"Transcript:\n{transcript_text}"
            ),
        }],
    )

    if response.stop_reason == "refusal":
        logger.error("Claude declined to process this transcript (stop_reason=refusal).")
        return None

    text = next((b.text for b in response.content if b.type == "text"), None)
    if text is None:
        logger.error("Claude response had no text block (stop_reason=%s).", response.stop_reason)
        return None

    return json.loads(text)


def process_transcript(transcript_path: Path) -> None:
    """Best-effort: transcript -> Claude notes+classification -> Notion page.
    Never raises. Any failure is logged and swallowed -- the local
    transcript this runs after is already safely written."""
    missing = [name for name in ("ANTHROPIC_API_KEY", "NOTION_API_KEY") if not os.environ.get(name)]
    if missing:
        logger.info("Notes pipeline: %s not set -- skipping (optional step).", " and ".join(missing))
        return
    try:
        transcript_text = transcript_path.read_text(encoding="utf-8")
        if not transcript_text.strip():
            logger.warning("Notes pipeline: %s is empty, skipping.", transcript_path.name)
            return

        courses_db = get_courses_database()
        notes_db = get_notes_database()

        courses = list_course_entries(courses_db)
        if not courses:
            logger.error("Notes pipeline: Courses database has no entries -- skipping.")
            return
        course_names = [c["name"] for c in courses]

        result = generate_notes_and_classification(transcript_text, course_names)
        if result is None:
            logger.error("Notes pipeline: no usable Claude output for %s -- skipping Notion push.", transcript_path.name)
            return

        if result["course_name"] == "AMBIGUOUS_NO_MATCH":
            logger.warning(
                "Notes pipeline: could not confidently match %s to any course (%s) -- skipping Notion push.",
                transcript_path.name, result.get("match_reasoning", ""),
            )
            return

        course_page_id = next((c["page_id"] for c in courses if c["name"] == result["course_name"]), None)
        if course_page_id is None:
            logger.error(
                "Notes pipeline: Claude returned course '%s' not in the live course list -- skipping.",
                result["course_name"],
            )
            return

        title_prop = find_notes_title_property(notes_db)
        relation_prop = find_notes_relation_property(notes_db, courses_db["id"])

        page_id = create_notes_page(
            notes_db["id"], title_prop, relation_prop, course_page_id,
            result["title"], result["summary"], result["key_concepts"], result["review_questions"],
        )
        logger.info(
            "Notes pipeline: created Notion page %s for '%s' (course: %s).",
            page_id, transcript_path.name, result["course_name"],
        )

    except Exception as e:
        logger.error("Notes pipeline failed for %s: %s", transcript_path.name, e, exc_info=True)
