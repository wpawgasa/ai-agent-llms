#!/usr/bin/env python3
"""`claude -p` driver that authors the missing Task A conversation turns.

1,465 of the 5,549 Task A SFT conversations cannot be brought onto the
tool-call stay convention by `plan_repair` alone: they need 1-14 short
messages written (3,902 inserts in total -- 2,229 assistant "hand-off bridges"
carrying a displaced `[STATE: X -> Y]` advance, and 1,673 `user`
acknowledgements with no marker at all). This script drives the
`corpus-remediator` agent over those inserts in batches and collects a
*decision ledger*.

The agent NEVER writes corpus rows -- it writes one JSON line per requested
insert, which `scripts/remediate_task_a_states.py apply` later replays
deterministically. That keeps `dvc repro` free of LLM spend, makes every
authored sentence PR-reviewable as a diff, and makes a rejected entry degrade
to `drop` in isolation instead of poisoning a batch.

`validate_entry` is the trust boundary. Everything it accepts is baked
permanently into a fine-tuning corpus, and no downstream stage looks at the
prose again. It is therefore deliberately strict: a rejected entry costs one
dropped conversation out of 5,549, an accepted-but-wrong entry costs a
corrupted training signal that nobody notices. Every rule it enforces is
documented in `.claude/agents/corpus-remediator.md` ("Hard formatting rules")
and in the playbook's Layer-1 gate table -- the three must not drift.

Mirrors `scripts/generate_sft_until_target.py::verify_batch_with_agent`, this
repo's only other `claude -p` call site: preflight the CLI with
`shutil.which`, catch every subprocess failure mode, unwrap the
`--output-format json` envelope, and never raise.

Usage:
    python scripts/build_remediation_ledger.py \\
        --input-dir data/output/sft/task_a \\
        --triage-report data/interim/task_a_state_triage/report.json \\
        --ledger-dir data/interim/task_a_remediation_ledger \\
        [--batch-size 10] [--max-workers 4] [--timeout 900] [--limit N] [--dry-run]

Re-running the identical command after an interruption is the supported
recovery path: `accepted.jsonl` is the checkpoint and completed inserts are
skipped. `rejected.jsonl` is *not* consulted, so a rejected insert is retried.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1

# The exact field set written to accepted.jsonl. Load-bearing:
# remediate_task_a_states.py::_load_ledger keys on `insert_id` and apply_plan
# reads `entry["content"]`. Accepted entries are rebuilt from these keys only,
# so an agent cannot smuggle extra keys (e.g. a fake `annotations`) into the
# ledger that `apply` would then carry into the corpus.
LEDGER_FIELDS = (
    "insert_id", "conversation_id", "role", "content",
    "rationale", "agent_model", "schema_version",
)

# Entry fields that must be present, non-empty strings.
_REQUIRED_STR_FIELDS = (
    "insert_id", "conversation_id", "role", "content", "rationale", "agent_model",
)

MIN_CONTENT_LEN = 20
MAX_CONTENT_LEN = 600
# Minimum *visible* prose, measured with invisible characters removed and
# whitespace runs collapsed (see `_visible_prose_len`). Applied to both roles:
#   * assistant -- measured after the marker. Markers run 25-60 chars, so a
#     marker plus a newline clears MIN_CONTENT_LEN while carrying no message.
#   * user -- measured on the whole content. MIN_CONTENT_LEN counts padding, so
#     "x" plus 19 spaces cleared it with no content at all.
# 10 is the length of "ok, thanks" -- the shortest acknowledgement anyone would
# plausibly write -- and half the content floor, so a real ack that clears
# MIN_CONTENT_LEN honestly clears this with room to spare. Thai acks are shorter
# still per character ("ขอบคุณค่ะ รับทราบนะคะ" is 21 chars / 20 visible), and are
# bounded by MIN_CONTENT_LEN long before this floor binds.
MIN_PROSE_LEN = 10
# Only a long verbatim copy of a neighbouring turn is treated as a copy: short
# acknowledgements ("ขอบคุณค่ะ") legitimately repeat.
COPY_GUARD_MIN_LEN = 40

VALID_LANGUAGES = ("en", "th", "code_switch")
VALID_ROLES = ("assistant", "user")

# Thai *letters*: consonants (U+0E01-U+0E2E), the spacing vowels U+0E30/32/33
# and the leading vowels U+0E40-U+0E45. Presence/absence is the wrong-language
# signal; see the report for why data_validator.detect_thai_corruption is not
# the right tool here.
#
# Deliberately NOT the whole U+0E00-U+0E7F block. That block also holds the baht
# sign (฿ U+0E3F), the repetition/abbreviation marks (ๆ ฯ ๏ ๚ ๛) and the Thai
# digits ๐-๙, none of which is evidence that a sentence was written in Thai.  # noqa: RUF003
# Matching the block defeated this rule in both directions: a single `฿` in an
# all-English answer satisfied a `th`/`code_switch` request (measured: it let all
# 2,853 th/code_switch rows through), while `en` rows were rejected for a price
# -- the one legitimate English use of a Thai-block character.
#
# Combining marks (U+0E31, U+0E34-U+0E3A, U+0E47-U+0E4E) are excluded on the same
# reasoning: they cannot occur without a base consonant, so they add no signal
# while reinstating the same hole for a stray tone mark.
_THAI_RE = re.compile(r"[ก-ฮะาำเ-ๅ]")
_TOOL_CALL_RE = re.compile(r"</?tool_call>")
# Invisible characters that survive `str.strip()`: Unicode format characters
# (U+200B ZWSP, U+200C/D ZWNJ/ZWJ, U+FEFF BOM, U+2060 word joiner, the bidi
# overrides, U+00AD soft hyphen) and C0/C1 controls other than \n and \t.
# U+200B is not whitespace to Python -- '\u200b'.strip() is non-empty -- which
# reinstated the empty-turn hole the prose floor exists to close. None of these
# has a legitimate use in a corpus turn, so they are both stripped before any
# length is measured AND rejected outright.
_INVISIBLE_RE = re.compile(
    "["
    "\u0000-\u0008\u000b-\u001f\u007f-\u009f"  # C0/C1 controls except \n and \t
    "\u00ad\u061c\u180e"                          # soft hyphen, ALM, Mongolian vowel sep
    "\u200b-\u200f\u202a-\u202e"                 # zero-width, bidi embedding/override
    "\u2060-\u2064\u2066-\u206f"                 # word joiner, invisible ops, isolates
    "\ufeff\ufff9-\ufffb"                         # BOM / ZWNBSP, interlinear annotation
    "]"
)
# Chat-template sentinels. Anything here baked into `content` is re-read as a
# turn boundary or a control token when the corpus is templated, corrupting
# tokenisation for every model that owns the token. Gated by *form*, not by one
# literal, because Task A trains across several families:
#   <|...|>            ChatML/Qwen (<|im_end|>), Llama 3 (<|eot_id|>), GLM (<|user|>)
#   <start_of_turn>    Gemma 3/4
#   <s> </s> <bos> ... SentencePiece / Mistral / Llama
#   [INST] [gMASK]     Mistral instruct, GLM
#   <extra_id_N>       T5 / Nemotron sentinel range
_SPECIAL_TOKEN_RE = re.compile(
    r"<\|[^|<>]{0,64}\|>"
    r"|</?(?:s|bos|eos|pad|unk|mask|sop|eop|start_of_turn|end_of_turn"
    r"|begin_of_text|end_of_text|extra_id_\d+)>"
    r"|\[/?(?:INST|SYS|gMASK|sMASK|MASK|CLS|SEP|PAD)\]",
    re.IGNORECASE,
)
_STATE_TOKEN = "[STATE:"
# Mirrors llm_workflow_agents.data._workflow_script._STATE_RE. Duplicated
# rather than imported to keep this driver stdlib-only; a unit test asserts the
# two patterns stay identical.
_STATE_MARKER_RE = re.compile(
    r"\[STATE:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:→|->)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]"
)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_REPORTED_COUNT_RE = re.compile(r"^\s*LEDGER:\s+\S+\s+(\d+)\s*$", re.MULTILINE)

_PROMPT_TEMPLATE = (
    "Author the missing conversation turns described in the batch request file "
    "at {batch_path} ({count} request(s)). Activate the project venv first "
    "(source .venv/bin/activate) before running any Python tools. Append your "
    "ledger entries, one JSON object per line, to {ledger_path} as you go -- "
    "that exact path and no other. When you are done, reply with a single line "
    "exactly of the form 'LEDGER: {ledger_path} <count>', then at most 3 "
    "sentences of summary. Never edit any file under data/output/."
)

_print_lock = threading.Lock()


def _log(message: str) -> None:
    with _print_lock:
        print(message, flush=True)


# ---------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------


def _validate_request(request: object) -> list[str]:
    """Sanity-check the *driver's own* request before judging the agent.

    A malformed request means this script (or the triage report) is broken, and
    silently skipping the checks that depend on the missing field would make
    the gate quietly weaker. Reporting it as a violation fails loudly in
    `rejected.jsonl` and safely (the conversation drops) instead.
    """
    if not isinstance(request, dict):
        return ["request is not a JSON object"]
    problems: list[str] = []
    if not isinstance(request.get("insert_id"), str) or not request["insert_id"]:
        problems.append("request has no insert_id")
    if not isinstance(request.get("conversation_id"), str) or not request["conversation_id"]:
        problems.append("request has no conversation_id")
    language = request.get("language")
    if language not in VALID_LANGUAGES:
        problems.append(f"request has invalid language {language!r}")
    role = request.get("role")
    if role not in VALID_ROLES:
        problems.append(f"request has invalid role {role!r}")
    marker = request.get("required_marker")
    if not isinstance(marker, str):
        problems.append("request required_marker is not a string")
    elif role == "assistant" and not marker:
        problems.append("assistant request has an empty required_marker")
    elif role == "user" and marker:
        problems.append("user request carries a non-empty required_marker")
    return problems


def _strip_invisible(text: str) -> str:
    """Drop characters that occupy no width but survive ``str.strip()``."""
    return _INVISIBLE_RE.sub("", text)


def _visible_prose_len(text: str) -> int:
    """Length of `text` with invisible characters dropped and whitespace runs
    collapsed to a single space.

    This is what the prose floors are measured on, so neither space padding nor
    a run of U+200B can buy length. Whitespace is collapsed rather than removed
    so that "ok, thanks" measures its true 10 characters.
    """
    return len(" ".join(_strip_invisible(text).split()))


def _strip_structure(text: str) -> str:
    """Prose only: markers and tool-call blocks removed, whitespace collapsed.

    Invisible characters are removed first so a ZWSP cannot make two identical
    sentences compare unequal and slip past the copy / duplicate guards.
    """
    text = _TOOL_CALL_BLOCK_RE.sub(" ", text)
    text = _STATE_MARKER_RE.sub(" ", text)
    return " ".join(_strip_invisible(text).split())


def _copies_a_context_turn(prose: str, request: dict) -> bool:
    normalised = " ".join(prose.split())
    if len(normalised) < COPY_GUARD_MIN_LEN:
        return False
    for message in request.get("context_window") or []:
        if not isinstance(message, dict):
            continue
        other = message.get("content")
        if not isinstance(other, str):
            continue
        other = _strip_structure(other)
        if len(other) >= COPY_GUARD_MIN_LEN and other == normalised:
            return True
    return False


def validate_entry(entry: object, request: object) -> list[str]:
    """Deterministic entry-level gate. Empty list == accept. Never raises.

    `entry` is untrusted LLM output: it may be any JSON value, and its fields
    may be of any type. Every access is type-guarded.
    """
    request_problems = _validate_request(request)
    if request_problems:
        return request_problems
    # _validate_request has established that `request` is a dict carrying
    # insert_id / conversation_id / language / role / required_marker.

    if not isinstance(entry, dict):
        return ["ledger entry is not a JSON object"]
    if entry.get("refuse"):
        return ["agent refused this insert"]

    violations: list[str] = []
    for field in _REQUIRED_STR_FIELDS:
        value = entry.get(field)
        if value is None:
            violations.append(f"missing field '{field}'")
        elif not isinstance(value, str):
            violations.append(f"field '{field}' is {type(value).__name__}, expected string")
        elif not value.strip():
            violations.append(f"field '{field}' is empty")
    if violations:
        return violations

    if entry["insert_id"] != request["insert_id"]:
        violations.append(
            f"insert_id {entry['insert_id']!r} does not match the outstanding "
            f"request {request['insert_id']!r}"
        )
    if entry["conversation_id"] != request["conversation_id"]:
        violations.append(
            f"conversation_id {entry['conversation_id']!r} does not match the "
            f"request {request['conversation_id']!r}"
        )
    if entry["role"] != request["role"]:
        violations.append(
            f"role {entry['role']!r} does not match requested role {request['role']!r}"
        )
    schema_version = entry.get("schema_version", SCHEMA_VERSION)
    if schema_version != SCHEMA_VERSION:
        violations.append(f"schema_version {schema_version!r} != {SCHEMA_VERSION}")

    content: str = entry["content"]
    if _TOOL_CALL_RE.search(content):
        violations.append("content must not contain <tool_call> or </tool_call>")
    special = _SPECIAL_TOKEN_RE.search(content)
    if special is not None:
        violations.append(
            f"content contains the chat-template special token {special.group(0)!r}"
        )
    if _INVISIBLE_RE.search(content):
        violations.append("content contains zero-width or control characters")

    required_marker: str = request["required_marker"]
    if required_marker:
        if not content.startswith(required_marker):
            violations.append(
                f"content does not start with required marker {required_marker!r} "
                "(byte for byte, arrow glyph included)"
            )
            prose = content
        else:
            remainder = content[len(required_marker):]
            if not remainder.startswith("\n"):
                violations.append("required marker must be followed by a newline")
            prose = remainder.lstrip("\n")
            if _STATE_TOKEN in remainder:
                violations.append("content contains a second [STATE:] marker")
    else:
        prose = content
        if _STATE_TOKEN in content:
            violations.append("user-role insert must not contain a [STATE:] marker")

    # Both roles. A `user` ack carries no marker, so MIN_CONTENT_LEN was its only
    # length rule -- and that counts padding, so "x" plus 19 spaces passed with
    # no content at all. 1,673 of the 3,902 inserts are acks.
    prose_len = _visible_prose_len(prose)
    if prose_len < MIN_PROSE_LEN:
        where = "after the marker" if required_marker else "in the insert"
        violations.append(
            f"only {prose_len} characters of visible prose {where} "
            f"(minimum {MIN_PROSE_LEN})"
        )

    if not (MIN_CONTENT_LEN <= len(content) <= MAX_CONTENT_LEN):
        violations.append(
            f"content length {len(content)} outside [{MIN_CONTENT_LEN}, {MAX_CONTENT_LEN}]"
        )

    language: str = request["language"]
    has_thai = bool(_THAI_RE.search(prose))
    if language in ("th", "code_switch") and not has_thai:
        violations.append(
            f"language '{language}' entry contains no Thai letters (a Thai digit, "
            "the baht sign or a repetition mark does not count)"
        )
    elif language == "en" and has_thai:
        violations.append("language 'en' entry contains Thai letters")

    if _copies_a_context_turn(prose, request):
        violations.append("content copies a message already in the context window")

    return violations


# ---------------------------------------------------------------------------
# append-only ledger IO
# ---------------------------------------------------------------------------


def _ends_without_newline(path: Path) -> bool:
    """True if `path` exists, is non-empty, and its last byte is not a newline."""
    try:
        with open(path, "rb") as handle:
            if handle.seek(0, os.SEEK_END) == 0:
                return False
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) != b"\n"
    except OSError:
        return False


def append_jsonl(path: Path, records: list[dict]) -> int:
    """Append `records` as one `write()` of one blob, then fsync.

    Crash-safety: a single buffered write followed by an explicit fsync is the
    narrowest possible window in which a kill can leave a torn final line, and
    `load_accepted_ids` repairs exactly that case on the next start. Called
    only from the main thread, so concurrent batches never interleave lines.
    """
    if not records:
        return 0
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    # Never glue a record onto an unterminated final line. That tail is
    # reachable two ways: `_repair_torn_tail` deliberately PRESERVES a
    # complete-but-unterminated line (a kill that landed exactly on an object
    # boundary), and the ledger is designed to be hand-edited and PR-reviewed,
    # so an editor that drops the trailing newline leaves the same state.
    # Gluing destroys BOTH entries -- the concatenated line parses as neither,
    # so both vanish from the resume set and
    # `remediate_task_a_states.py::_load_ledger` then fails on it.
    if _ends_without_newline(path):
        blob = "\n" + blob
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(blob)
        handle.flush()
        os.fsync(handle.fileno())
    return len(records)


def _repair_torn_tail(path: Path) -> None:
    """Truncate a trailing partial JSON line left by a crash mid-append.

    Only truncates when the file does not end in a newline AND the trailing
    fragment does not parse. A complete last line that merely lacks its newline
    keeps its entry -- but the newline is written, so the checkpoint on disk is
    always well-formed for the next reader and the next appender. (`append_jsonl`
    guards the same case independently, because `rejected.jsonl` is appended
    without ever being repaired and either file may be hand-edited.)
    """
    try:
        data = path.read_bytes()
    except OSError as exc:  # pragma: no cover - unreadable checkpoint
        _log(f"warning: cannot read {path}: {exc}")
        return
    if not data or data.endswith(b"\n"):
        return
    cut = data.rfind(b"\n") + 1
    tail = data[cut:]
    try:
        json.loads(tail.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass
    else:
        # Complete, just unterminated. Keep the entry, terminate the line.
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:  # pragma: no cover - read-only checkpoint
            _log(f"warning: cannot terminate the final line of {path}: {exc}")
        return
    try:
        os.truncate(path, cut)
    except OSError as exc:  # pragma: no cover - read-only checkpoint
        _log(f"warning: cannot repair torn tail of {path}: {exc}")
        return
    _log(f"warning: repaired a torn final line in {path} ({len(tail)} bytes dropped)")


def load_accepted_ids(ledger_dir: Path) -> set[str]:
    """insert_ids already accepted. The resume checkpoint. Never raises."""
    path = Path(ledger_dir) / "accepted.jsonl"
    if not path.exists():
        return set()
    _repair_torn_tail(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - unreadable checkpoint
        _log(f"warning: cannot read {path}: {exc}")
        return set()
    seen: set[str] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            _log(f"warning: {path}:{number} is not valid JSON; ignoring for resume")
            continue
        if isinstance(entry, dict) and isinstance(entry.get("insert_id"), str):
            seen.add(entry["insert_id"])
        else:
            _log(f"warning: {path}:{number} has no insert_id; ignoring for resume")
    return seen


def _write_progress(path: Path, payload: dict) -> None:
    """Atomic replace so a crash can never leave a half-written progress file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# the agent call
# ---------------------------------------------------------------------------


def render_prompt(batch_path: Path, ledger_path: Path, count: int) -> str:
    return _PROMPT_TEMPLATE.format(
        batch_path=batch_path, ledger_path=ledger_path, count=count
    )


def build_command(prompt: str) -> list[str]:
    return ["claude", "-p", prompt, "--agent", "corpus-remediator",
            "--output-format", "json"]


def _count_requests(batch_path: Path) -> int:
    """How many requests the batch file holds, for the prompt's own statement."""
    try:
        payload = json.loads(batch_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return len(payload) if isinstance(payload, list) else 0


def _unwrap_result(stdout: str) -> str:
    """`--output-format json` wraps the final assistant text in an envelope."""
    text = (stdout or "").strip()
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(envelope, dict):
        return str(envelope.get("result") or envelope.get("text") or text)
    return text


def run_agent_batch(batch_path: Path, scratch_dir: Path, timeout: int) -> dict:
    """Invoke the agent on one batch. Never raises.

    Returns ``{"error": str|None, "ledger_path": Path|None, "reported_count":
    int|None}``. The ledger is located **by the path this function assigned**,
    never by parsing the reply -- a chatty or wrong reply can neither break
    parsing nor redirect a read. The reply's count is extracted for logging
    only.
    """
    batch_path = Path(batch_path).resolve()
    ledger_path = (Path(scratch_dir).resolve() / f"{batch_path.stem}.ledger.jsonl")
    # A ledger left behind by an earlier crashed attempt in this same scratch
    # dir must never be mistaken for this run's output.
    try:
        if ledger_path.exists():
            ledger_path.unlink()
    except OSError as exc:
        return {"error": f"cannot clear stale ledger: {exc}", "ledger_path": None,
                "reported_count": None}

    if shutil.which("claude") is None:
        return {"error": "claude CLI not found", "ledger_path": None, "reported_count": None}

    prompt = render_prompt(batch_path, ledger_path, _count_requests(batch_path))
    cmd = build_command(prompt)
    try:
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout}s", "ledger_path": None,
                "reported_count": None}
    except FileNotFoundError:
        return {"error": "claude CLI not found", "ledger_path": None, "reported_count": None}
    except OSError as exc:
        return {"error": f"could not launch claude: {exc}", "ledger_path": None,
                "reported_count": None}

    if proc.returncode != 0:
        return {"error": f"claude exited {proc.returncode}: {(proc.stderr or '').strip()[:200]}",
                "ledger_path": None, "reported_count": None}

    reported = _REPORTED_COUNT_RE.search(_unwrap_result(proc.stdout))
    reported_count = int(reported.group(1)) if reported else None
    if not ledger_path.exists():
        return {"error": f"claude exited 0 but wrote no ledger file at {ledger_path}",
                "ledger_path": None, "reported_count": reported_count}
    return {"error": None, "ledger_path": ledger_path, "reported_count": reported_count}


def _request_id(request: object) -> str | None:
    """The request's `insert_id`, or None for any shape that cannot carry one."""
    if isinstance(request, dict):
        value = request.get("insert_id")
        if isinstance(value, str) and value:
            return value
    return None


def rejection_stub(request: object, index: int, reasons: list[str]) -> dict:
    """A rejection record for a request of *any* shape.

    Indexing `request["insert_id"]` was the one place the never-raise guarantee
    leaked: a request without the key raised `KeyError` out of `process_batch`,
    and main's `except Exception` net re-raised while building its own fallback
    record from the same key. A request this broken means the triage report or
    this driver is wrong, so it degrades the way every other failure does --
    a rejection with a reason, and the conversation drops.
    """
    return {
        "insert_id": _request_id(request) or f"<request {index} without insert_id>",
        "conversation_id": (request.get("conversation_id", "")
                            if isinstance(request, dict) else ""),
        "reasons": reasons,
    }


def process_batch(requests: list[dict], scratch_dir: Path, timeout: int) -> dict:
    """Author one batch and gate every entry. Never raises.

    Returns ``{"accepted": [...], "rejected": [...], "error": str|None,
    "reported_count": int|None}``. Every request is accounted for exactly once:
    accepted, or rejected with reasons.
    """
    scratch_dir = Path(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    batch_path = scratch_dir / "batch.json"

    # A request with no usable insert_id can never be answered -- the agent has
    # nothing to echo back and the ledger line could not be matched to it. It is
    # rejected here and kept out of the batch file rather than confusing the
    # agent, but it still appears in `rejected` so the count reconciles.
    by_id: dict[str, dict] = {}
    authorable: list[dict] = []
    unusable: list[dict] = []
    for index, request in enumerate(requests):
        insert_id = _request_id(request)
        if insert_id is None:
            unusable.append(rejection_stub(
                request, index,
                ["request has no insert_id; it cannot be authored or matched"],
            ))
            continue
        by_id[insert_id] = request
        authorable.append(request)

    def _reject_all(reason: str) -> dict:
        return {
            "accepted": [],
            "rejected": unusable + [rejection_stub(r, i, [reason])
                                    for i, r in enumerate(authorable)],
            "error": reason,
            "reported_count": None,
        }

    if not authorable:
        reason = "batch contains no request with a usable insert_id"
        return {"accepted": [], "rejected": unusable, "error": reason,
                "reported_count": None}

    try:
        batch_path.write_text(
            json.dumps(authorable, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        return _reject_all(f"cannot write batch file: {exc}")

    outcome = run_agent_batch(batch_path, scratch_dir, timeout)
    if outcome["error"] or outcome["ledger_path"] is None:
        return _reject_all(outcome["error"] or "no ledger produced")

    try:
        text = Path(outcome["ledger_path"]).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return _reject_all(f"cannot read agent ledger: {exc}")

    accepted: list[dict] = []
    rejected: list[dict] = []
    decided: set[str] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            rejected.append({"insert_id": f"<line {number}>", "conversation_id": "",
                             "reasons": [f"malformed JSON on ledger line {number}: {exc.msg}"]})
            continue
        insert_id = entry.get("insert_id") if isinstance(entry, dict) else None
        request = by_id.get(insert_id) if isinstance(insert_id, str) else None
        if request is None:
            rejected.append({"insert_id": str(insert_id), "conversation_id": "",
                             "reasons": [f"insert_id {insert_id!r} is not in this batch"]})
            continue
        if insert_id in decided:
            rejected.append({"insert_id": insert_id,
                             "conversation_id": request.get("conversation_id", ""),
                             "reasons": [f"duplicate ledger entry on line {number}; "
                                         "the first decision stands"]})
            continue
        decided.add(insert_id)
        violations = validate_entry(entry, request)
        if violations:
            rejected.append({"insert_id": insert_id,
                             "conversation_id": request.get("conversation_id", ""),
                             "reasons": violations})
            continue
        accepted.append({
            **{field: entry[field] for field in LEDGER_FIELDS if field != "schema_version"},
            "schema_version": SCHEMA_VERSION,
        })

    accepted, duplicates = _reject_duplicate_content(accepted, by_id)
    rejected.extend(duplicates)

    for insert_id, request in by_id.items():
        if insert_id not in decided:
            rejected.append({"insert_id": insert_id,
                             "conversation_id": request.get("conversation_id", ""),
                             "reasons": ["no ledger entry produced"]})

    return {"accepted": accepted, "rejected": unusable + rejected, "error": None,
            "reported_count": outcome["reported_count"]}


def _reject_duplicate_content(accepted: list[dict], by_id: dict) -> tuple[list[dict], list[dict]]:
    """Drop entries that reuse another entry's prose verbatim within a batch.

    An agent under load pastes the same sentence into several inserts -- the
    "generic 'Thank you!' repeated across rows" defect the playbook tells
    reviewers to hunt for, and the one shape that is obviously wrong inside a
    closing pair (the `user` ack and the terminal turn must read as one
    exchange, not as one sentence twice). The first occurrence stands; later
    ones are rejected. Short prose is exempt on the same reasoning as the
    context copy-guard: brief acknowledgements legitimately repeat.
    """
    kept: list[dict] = []
    rejected: list[dict] = []
    first_seen: dict[str, str] = {}
    for entry in accepted:
        prose = _strip_structure(entry["content"])
        if len(prose) >= COPY_GUARD_MIN_LEN and prose in first_seen:
            rejected.append({
                "insert_id": entry["insert_id"],
                "conversation_id": entry.get("conversation_id", ""),
                "reasons": [f"content duplicates the prose already authored for "
                            f"{first_seen[prose]} in this batch"],
            })
            continue
        first_seen.setdefault(prose, entry["insert_id"])
        kept.append(entry)
    return kept, rejected


def check_ledger_file(batch_path: Path, ledger_path: Path) -> list[str]:
    """Run the full gate over a written ledger. Empty list == every entry accepted.

    This is what the agent's own self-check calls, so the agent is judged by
    the exact code the driver will judge it by -- there is no second
    implementation to drift. A deliberate refusal is not reported as a problem.
    Never raises.
    """
    problems: list[str] = []
    try:
        requests = json.loads(Path(batch_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read batch file: {exc}"]
    if not isinstance(requests, list):
        return ["batch file is not a JSON array"]
    by_id = {r.get("insert_id"): r for r in requests if isinstance(r, dict)}
    try:
        text = Path(ledger_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"cannot read ledger file: {exc}"]

    entries: list[dict] = []
    seen: set[str] = set()
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"line {number}: malformed JSON ({exc.msg})")
            continue
        insert_id = entry.get("insert_id") if isinstance(entry, dict) else None
        request = by_id.get(insert_id)
        if request is None:
            problems.append(f"line {number}: insert_id {insert_id!r} is not in this batch")
            continue
        seen.add(insert_id)
        # A deliberate refusal is a valid outcome, not a problem to fix.
        violations = [v for v in validate_entry(entry, request)
                      if v != "agent refused this insert"]
        problems.extend(f"{insert_id}: {v}" for v in violations)
        if not violations and not entry.get("refuse"):
            entries.append(entry)
    _, duplicates = _reject_duplicate_content(entries, by_id)
    problems.extend(f"{d['insert_id']}: {d['reasons'][0]}" for d in duplicates)
    problems.extend(f"{insert_id}: no ledger entry written yet"
                    for insert_id in by_id if insert_id not in seen)
    return problems


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def build_batches(pending: list[dict], batch_size: int) -> list[list[dict]]:
    """Pack requests into batches **on conversation boundaries**.

    Flat slicing splits 249 of the 1,465 conversations at the default size of
    10, which breaks the two cases that most need to be authored together: a
    closing pair (the `user` ack and the terminal turn must read as one
    exchange) and the 100+ conversations needing 6-14 bridges. A conversation
    larger than `batch_size` gets a batch to itself rather than being split.
    """
    batches: list[list[dict]] = []
    current: list[dict] = []
    for group in _group_by_conversation(pending):
        if current and len(current) + len(group) > batch_size:
            batches.append(current)
            current = []
        current.extend(group)
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)
    return batches


def limit_pending(pending: list[dict], limit: int | None) -> list[dict]:
    """Trim to at most `limit` inserts **without ever splitting a conversation**.

    A flat slice took, say, 2 of a conversation's 4 inserts, so the smoke run
    authored a conversation it could not complete -- `apply_plan` drops a
    conversation with a missing entry, so the smoke output misrepresented
    exactly the thing the smoke run exists to check.

    Whole conversations are taken, in report order, until the next one would
    exceed the limit. The first conversation is always taken, so a limit smaller
    than one conversation still produces something reviewable instead of
    nothing; that is the only case where the result exceeds `limit`.
    """
    if limit is None:
        return pending
    if limit <= 0:
        return []
    trimmed: list[dict] = []
    for group in _group_by_conversation(pending):
        if trimmed and len(trimmed) + len(group) > limit:
            break
        trimmed.extend(group)
    return trimmed


def _group_by_conversation(pending: list[dict]):
    """Consecutive runs sharing a conversation key, in report order.

    Keys on ``key`` (``[file_stem, line_index]``) rather than
    ``conversation_id``: two records in the corpus share a conversation_id.
    """
    group: list[dict] = []
    current_key = None
    for request in pending:
        key = tuple(request.get("key") or [request.get("conversation_id")])
        if group and key != current_key:
            yield group
            group = []
        current_key = key
        group.append(request)
    if group:
        yield group


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_requests(report_path: Path, input_dir: str) -> tuple[list[dict], str | None]:
    """Flatten the triage report into authoring requests. Returns (reqs, error)."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return [], f"cannot read triage report: {exc}"
    except json.JSONDecodeError as exc:
        return [], f"triage report is not valid JSON: {exc}"
    if not isinstance(report, dict) or not isinstance(report.get("records"), list):
        return [], "triage report has no 'records' list"

    reported_dir = report.get("input_dir")
    if reported_dir and Path(reported_dir) != Path(input_dir):
        # insert_ids are "<file_stem>:<line_index>:<ordinal>", so a ledger built
        # against a different directory would splice prose into the wrong rows.
        return [], (
            f"--input-dir {input_dir!r} does not match the triage report's "
            f"input_dir {reported_dir!r}; insert_ids are file-stem and "
            "line-index keyed, so mixing them would splice prose into the "
            "wrong conversations. Re-run `remediate_task_a_states.py triage` "
            "against the directory you mean to remediate."
        )

    requests: list[dict] = []
    for record in report["records"]:
        if not isinstance(record, dict):
            continue
        for insert in record.get("inserts") or []:
            requests.append({
                **insert,
                "conversation_id": record.get("conversation_id", ""),
                "language": record.get("language", ""),
                "key": record.get("key"),
            })
    if not requests:
        return [], "triage report contains no inserts to author"
    return requests, None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Drive `claude -p --agent corpus-remediator` over the Task A "
                    "conversations that need authored turns, and collect a ledger."
    )
    parser.add_argument("--input-dir", required=True,
                        help="Corpus directory the triage report was built from "
                             "(cross-checked against the report; not read).")
    parser.add_argument("--triage-report", required=True)
    parser.add_argument("--ledger-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Max inserts per agent call; never splits a conversation.")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--limit", type=int, default=None,
                        help="Author at most N pending inserts (smoke runs). "
                             "Trims on conversation boundaries, so every "
                             "conversation it starts is authored in full; a "
                             "single conversation larger than N is still taken "
                             "whole.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Render the first batch's prompt and exit. No subprocess "
                             "calls, no files written.")
    args = parser.parse_args(argv)

    all_requests, error = _load_requests(Path(args.triage_report), args.input_dir)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    ledger_dir = Path(args.ledger_dir)
    already_done = load_accepted_ids(ledger_dir)
    pending = [r for r in all_requests if r.get("insert_id") not in already_done]
    pending = limit_pending(pending, args.limit)
    batches = build_batches(pending, max(1, args.batch_size))

    conversations = sum(1 for _ in _group_by_conversation(pending))
    limited = "" if args.limit is None else f" (--limit {args.limit}, trimmed whole conversations)"
    print(f"Inserts: {len(all_requests)} total, {len(already_done)} already accepted, "
          f"{len(pending)} pending across {conversations} conversation(s) "
          f"in {len(batches)} batch(es){limited}")

    if args.dry_run:
        if batches:
            scratch = ledger_dir / "batches" / "<run>" / "batch_00000"
            prompt = render_prompt(scratch / "batch.json",
                                   scratch / "batch.ledger.jsonl", len(batches[0]))
            print("\n--- command ---")
            print(" ".join(part if part != prompt else "<prompt>"
                           for part in build_command(prompt)))
            print("\n--- prompt ---")
            print(prompt)
            print("\n--- batch request file ---")
            print(json.dumps(batches[0], indent=2, ensure_ascii=False)[:4000])
        return 0

    if not batches:
        print("Nothing to do.")
        return 0

    ledger_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    scratch_root = ledger_dir / "batches" / run_id
    accepted_path = ledger_dir / "accepted.jsonl"
    rejected_path = ledger_dir / "rejected.jsonl"
    progress_path = ledger_dir / "progress.json"

    progress = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "triage_report": str(args.triage_report),
        "input_dir": args.input_dir,
        "batch_size": args.batch_size,
        "max_workers": args.max_workers,
        "timeout": args.timeout,
        "inserts_total": len(all_requests),
        "already_accepted_at_start": len(already_done),
        "pending_at_start": len(pending),
        "batches_total": len(batches),
        "batches_done": 0,
        "accepted": 0,
        "rejected": 0,
        "batch_errors": [],
        "finished_at": None,
    }
    _write_progress(progress_path, progress)

    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
        futures = {
            pool.submit(process_batch, batch, scratch_root / f"batch_{i:05d}", args.timeout): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                outcome = future.result()
            except Exception as exc:  # process_batch is not supposed to raise
                # Built with `rejection_stub`, not `r["insert_id"]`: the net
                # itself must not raise while recording a failure.
                outcome = {"accepted": [], "error": f"unexpected: {exc!r}",
                           "reported_count": None,
                           "rejected": [rejection_stub(r, i, [f"driver error: {exc!r}"])
                                        for i, r in enumerate(batches[index])]}
            # accepted.jsonl is the checkpoint and is written FIRST: a crash
            # between the two appends loses only a rejection log line, and a
            # rejected insert is retried on the next run anyway.
            append_jsonl(accepted_path, outcome["accepted"])
            append_jsonl(rejected_path, outcome["rejected"])
            progress["batches_done"] += 1
            progress["accepted"] += len(outcome["accepted"])
            progress["rejected"] += len(outcome["rejected"])
            if outcome["error"]:
                progress["batch_errors"].append({"batch": index, "error": outcome["error"]})
            _write_progress(progress_path, progress)
            expected = len(outcome["accepted"]) + len(outcome["rejected"])
            mismatch = ""
            if outcome["reported_count"] not in (None, expected):
                mismatch = f"  [agent reported {outcome['reported_count']} entries]"
            _log(f"batch {progress['batches_done']}/{len(batches)} (#{index}): "
                 f"+{len(outcome['accepted'])} accepted, "
                 f"+{len(outcome['rejected'])} rejected{mismatch}")

    progress["finished_at"] = datetime.now(UTC).isoformat()
    _write_progress(progress_path, progress)
    print(f"Done: {progress['accepted']} accepted, {progress['rejected']} rejected. "
          f"See {accepted_path}, {rejected_path}, {progress_path}.")
    if progress["accepted"] == 0:
        print("error: no entries were accepted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
