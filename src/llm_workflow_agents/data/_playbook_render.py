"""Task C playbook rendering: gold graph -> natural-language playbook.

Renders a gold workflow graph as a business playbook in one of six registers and
three languages (see docs/data_generation_recipes_task_c.md, §Playbook Registers).
The `state_script` register is rendered programmatically via build_workflow_script
(zero teacher cost); the other five are teacher-authored.

The teacher must obey the anchor contract (every canonical state name appears
verbatim at least once); verification of that contract lives in _playbook_verify.
"""

from __future__ import annotations

import json
import random
from enum import StrEnum
from typing import Any, Iterable, Sequence

import structlog

from llm_workflow_agents.data._teacher_client import call_teacher_model
from llm_workflow_agents.data._workflow_script import build_workflow_script

logger = structlog.get_logger(__name__)


class Register(StrEnum):
    """The six playbook registers. Values match the dataset `register` field."""

    SOP_DOCUMENT = "sop_document"
    PROSE_NARRATIVE = "prose_narrative"
    BULLET_QUICK_REFERENCE = "bullet_quick_reference"
    WIKI_TRAINING_NOTES = "wiki_training_notes"
    STATE_SCRIPT = "state_script"
    MANAGER_TRANSCRIPT = "manager_transcript"


TEACHER_REGISTERS: frozenset[Register] = frozenset(Register) - {Register.STATE_SCRIPT}

# Distractor paragraphs: generic policy boilerplate. Kept free of SCREAMING_SNAKE
# state names and snake_case tool names so they never create label ambiguity.
DISTRACTOR_LIBRARY: dict[str, tuple[str, ...]] = {
    "en": (
        "Remember that all interactions are recorded for quality assurance and may be "
        "reviewed by a supervisor at any time.",
        "Maintain a courteous and professional tone throughout; empathy statements should "
        "precede any request for sensitive information.",
        "Our published service level agreement commits to a first response within one "
        "business day and full resolution within five.",
        "Never disclose internal system identifiers or pricing formulas to the customer; "
        "these are confidential business information.",
        "If the caller becomes abusive, follow the disengagement guidance in the staff "
        "handbook rather than continuing the conversation.",
        "Company policy requires that personal data be handled in line with the applicable "
        "privacy regulations for the customer's region.",
        "Breaks are scheduled by the workforce management team; do not leave a session "
        "unattended without transferring it first.",
        "Seasonal promotions are announced through the internal bulletin; check it at the "
        "start of every shift for the latest offers.",
    ),
    "th": (
        "โปรดจำไว้ว่าการสนทนาทั้งหมดจะถูกบันทึกไว้เพื่อการควบคุมคุณภาพ และหัวหน้างานอาจตรวจสอบได้ทุกเมื่อ",
        "รักษาน้ำเสียงที่สุภาพและเป็นมืออาชีพตลอดการสนทนา และควรแสดงความเห็นอกเห็นใจก่อนขอข้อมูลที่ละเอียดอ่อน",
        "ข้อตกลงระดับการให้บริการของเรากำหนดให้ตอบกลับครั้งแรกภายในหนึ่งวันทำการ และแก้ไขให้เสร็จภายในห้าวัน",
        "ห้ามเปิดเผยรหัสภายในระบบหรือสูตรการคำนวณราคาแก่ลูกค้า เพราะถือเป็นข้อมูลลับทางธุรกิจ",
        "หากผู้โทรใช้ถ้อยคำรุนแรง ให้ปฏิบัติตามแนวทางการยุติการสนทนาในคู่มือพนักงานแทนการสนทนาต่อ",
        "นโยบายบริษัทกำหนดให้จัดการข้อมูลส่วนบุคคลตามระเบียบความเป็นส่วนตัวที่บังคับใช้ในภูมิภาคของลูกค้า",
    ),
}


PLAYBOOK_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "en": "Language: English — write the entire playbook in English.",
    "th": (
        "Language: Thai (th-TH) — write the entire playbook in Thai. State names "
        "(SCREAMING_SNAKE), tool names, and any JSON must stay in English/ASCII."
    ),
    "code_switch": (
        "Language: Thai-English code-switching — write mostly in Thai with embedded English "
        "business terms, as in a Thai company's internal document. State names "
        "(SCREAMING_SNAKE), tool names, and any JSON must stay in English/ASCII."
    ),
}

_REGISTER_STYLE: dict[Register, str] = {
    Register.SOP_DOCUMENT: (
        "Register: FORMAL STANDARD OPERATING PROCEDURE. Use numbered sections (1.0, 2.0, ...), "
        "an authoritative tone ('the representative shall ...'), and a document heading."
    ),
    Register.PROSE_NARRATIVE: (
        "Register: FLOWING PROSE. Write in connected paragraphs with no headings or bullet lists — "
        "describe the process as continuous narrative ('When a customer calls, first ...')."
    ),
    Register.BULLET_QUICK_REFERENCE: (
        "Register: BULLET QUICK REFERENCE. Use terse bullet points and short fragments with arrows, "
        "as a one-page cheat sheet an agent glances at."
    ),
    Register.WIKI_TRAINING_NOTES: (
        "Register: INFORMAL WIKI / TRAINING NOTES. Conversational team notes with asides and the "
        "occasional emoji, as if written on an internal wiki page for new hires."
    ),
    Register.MANAGER_TRANSCRIPT: (
        "Register: MANAGER-EXPLAINING TRANSCRIPT. A spoken monologue of a manager explaining the "
        "process to a new agent, with natural digressions ('so basically what you do is ...')."
    ),
}

_PARAPHRASE_INSTRUCTIONS: dict[str, str] = {
    "low": "Paraphrase density LOW: reuse each canonical state name throughout its description.",
    "medium": "Paraphrase density MEDIUM: use each canonical state name once, then paraphrase it afterward.",
    "high": (
        "Paraphrase density HIGH: use each canonical state name exactly once, then refer to it with "
        "aggressive paraphrase and pronouns."
    ),
}

_CONDITION_INSTRUCTIONS: dict[str, str] = {
    "explicit": "Conditions EXPLICIT: state each transition condition directly ('if X fails, go to Y').",
    "narrative_order": (
        "Conditions NARRATIVE ORDER: imply the default (priority-0) path through the order in which "
        "you describe steps, rather than naming every condition."
    ),
    "listing_order": (
        "Conditions LISTING ORDER: when a step has alternatives, describe them in the order they are "
        "listed — the first is the default, later ones are lower-priority fallbacks."
    ),
}

_RENDER_SYSTEM = (
    "You author internal business playbooks. Given a workflow graph, write a playbook that a "
    "human agent could follow. Return a single JSON object of the form {\"playbook\": \"<text>\"} "
    "and nothing else."
)


def draw_distractors(
    count: int,
    language: str,
    rng: random.Random,
    forbidden_terms: Iterable[str],
) -> list[str]:
    """Draw `count` distractor paragraphs for the language, skipping any that contain a forbidden term."""
    if count <= 0:
        return []
    lang_key = "th" if language in ("th", "code_switch") else "en"
    pool = [
        p for p in DISTRACTOR_LIBRARY[lang_key]
        if not any(term and term in p.split() for term in forbidden_terms)
    ]
    rng.shuffle(pool)
    return pool[:count]


def build_render_prompts(
    graph_dict: dict[str, Any],
    tool_schemas: list[dict[str, Any]],
    register: Register,
    language: str,
    knobs: dict[str, Any],
) -> tuple[str, str]:
    """Build (system, user) prompts for a teacher rendering of the graph."""
    graph_block = json.dumps({"graph": graph_dict, "tools": tool_schemas}, ensure_ascii=False, indent=2)
    tool_names = sorted({t for sd in graph_dict["state_details"] for t in sd["tools"]})

    parts = [
        _REGISTER_STYLE[register],
        "",
        "## GOLD WORKFLOW GRAPH",
        "```json",
        graph_block,
        "```",
        "",
        "## HARD REQUIREMENTS",
        "- Each state's canonical name (e.g. " + ", ".join(graph_dict["states"][:3])
        + ") must appear verbatim at least once — as a heading, a parenthetical, or a quoted label.",
        "- Introduce the initial state (" + graph_dict["initial"] + ") before any other state.",
        "- The document must explicitly signal where the workflow ends (its terminal state(s): "
        + ", ".join(graph_dict["terminal"]) + ").",
        "- Every tool granted to a state must be named verbatim in that state's description"
        + (" (tools: " + ", ".join(tool_names) + ")." if tool_names else "."),
        "- For every transition, mention the target state's canonical name within or immediately "
        "after the source state's description; always describe branch (alternative) transitions "
        "alongside their source state.",
        "",
        "## STYLE KNOBS",
        _PARAPHRASE_INSTRUCTIONS[knobs["paraphrase_density"]],
        _CONDITION_INSTRUCTIONS[knobs["condition_explicitness"]],
    ]

    distractors = knobs.get("_distractors") or []
    if distractors:
        parts.append("")
        parts.append(
            "Insert the following paragraphs verbatim as standalone paragraphs at natural points "
            "(they are unrelated policy notes — do not tie them to any state):"
        )
        parts.extend(f"- {d}" for d in distractors)

    parts.append("")
    parts.append(PLAYBOOK_LANGUAGE_INSTRUCTIONS.get(language, PLAYBOOK_LANGUAGE_INSTRUCTIONS["en"]))

    corrections = knobs.get("_corrections")
    if corrections:
        parts.append("")
        parts.append("## CORRECTIONS REQUIRED")
        parts.append(
            "Your previous attempt was rejected by an automated check for the issues below. "
            "Rewrite the ENTIRE playbook, fixing every one of them:"
        )
        parts.extend(f"- {c}" for c in corrections)

    return _RENDER_SYSTEM, "\n".join(parts)


def _interleave_distractors(script: str, distractors: Sequence[str], rng: random.Random) -> str:
    """Insert distractor paragraphs between `### [` sections of a programmatic script."""
    if not distractors:
        return script
    blocks = script.split("\n\n")
    for d in distractors:
        pos = rng.randint(0, len(blocks))
        blocks.insert(pos, d)
    return "\n\n".join(blocks)


def render_playbook(
    graph_dict: dict[str, Any],
    tool_schemas: list[dict[str, Any]],
    register: Register,
    language: str,
    knobs: dict[str, Any],
    teacher_model: str,
    rng: random.Random,
    corrections: list[str] | None = None,
) -> str:
    """Render the gold graph as a playbook. STATE_SCRIPT is programmatic; others call the teacher."""
    register = Register(register)
    if register is Register.STATE_SCRIPT:
        script_lang = "en" if language == "code_switch" else language
        script = build_workflow_script(graph_dict, tool_schemas, language=script_lang)
        return _interleave_distractors(script, knobs.get("_distractors") or [], rng)

    knobs_with = dict(knobs)
    if corrections:
        knobs_with["_corrections"] = corrections
    system, user = build_render_prompts(graph_dict, tool_schemas, register, language, knobs_with)
    raw = call_teacher_model(teacher_model, system, user)
    data = json.loads(raw)
    text = str(data.get("playbook", "")).strip()
    if not text:
        raise ValueError("empty playbook in teacher response")
    return text
