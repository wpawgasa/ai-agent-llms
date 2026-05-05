"""Shared system-prompt enrichment for Task A workflow datasets.

Single source of truth for FORMAT_RULES and the enrichment helper used by
both the data-generation pipeline and the benchmark harness.
"""

from __future__ import annotations

import json
from typing import Any

from llm_workflow_agents.data._workflow_script import build_workflow_script

FORMAT_RULES = """\
Rules:

1. Turn template — EVERY assistant turn MUST start with a state annotation, including
   tool-only turns and the terminal turn. Use exactly this format on the first line:
       [STATE: CURRENT → NEXT]
   If the state does not change, write the same name on both sides:
       [STATE: QUALIFY_PROSPECT → QUALIFY_PROSPECT]
   Never omit this line. A turn that is "just a tool call" still needs the STATE line
   above the <tool_call> tag.

2. Tool-call format — when you call a tool, emit it on its own line(s) as:
       <tool_call>{"name": "<tool_name>", "arguments": {<arg_key>: <arg_value>, ...}}</tool_call>
   The two top-level keys are exactly "name" and "arguments". Do NOT flatten arguments
   into the top level. Worked example for a schema with required=[patient_id, specialty]
   and optional reason:
       [STATE: VERIFY_PATIENT → TERMINAL]
       <tool_call>{"name": "request_referral", "arguments": {"patient_id": "P12345", "specialty": "cardiology"}}</tool_call>

3. Tool authority — the "Tool schemas" section is the ONLY authoritative source for which
   tools exist and which parameters they accept. The "Workflow script" hints at conversation
   flow but its per-state tool listings are UNRELIABLE; if it conflicts with a tool schema,
   trust the schema. Note that the schema uses "parameters" (OpenAI tools format) while
   your <tool_call> emits "arguments" — these refer to the same thing, do not confuse them.

4. Argument discipline (strict):
   a. Pass ONLY parameters listed in the schema's "required" array, plus any optional
      parameter for which the user has EXPLICITLY stated a value in the conversation.
   b. Do NOT invent values for optional parameters. If the user has not said anything
      about `reason`, `description`, `offer_details`, `notes`, etc., omit those fields.
   c. Use parameter values verbatim from the user. Do not paraphrase, expand abbreviations,
      or reformat (e.g. user says "premium" → pass exactly "premium", not "premium package";
      user says "competitor.com" → ask for the full URL before calling; do not fabricate one).

5. Tool-call necessity — do not call a tool unless the workflow requires it. Greetings,
   acknowledgements, clarifying questions, and terminal closings are text-only turns; do
   not append a tool call just to "wrap up" the conversation.

6. Multi-turn negotiation — if a required argument is missing from what the user has said
   so far, ask the user for it BEFORE calling the tool. Do not synthesize plausible values.

7. If a tool returns an error, attempt recovery before escalating.
8. Reach a terminal state to complete the workflow.
9. Never skip states or make invalid transitions."""


def build_enriched_system_prompt(
    sample: dict[str, Any],
    original_content: str,
    force_rebuild: bool = False,
) -> str:
    """Prepend workflow script, structured reference, and format rules to a system prompt.

    Idempotent: if *original_content* already contains the enrichment marker
    ``"Workflow script"`` the function returns it unchanged. This makes the
    benchmark harness safe to call on both legacy (bare) and new (pre-enriched)
    JSONL files without double-enriching.

    Args:
        sample: A dataset sample dict with optional keys ``workflow_script``,
            ``workflow_graph``, and ``tool_schemas``.
        original_content: The existing system-message content (may be bare or
            already enriched).
        force_rebuild: If True, strip any existing enrichment from
            *original_content* and rebuild from current code. Use this when
            loading old training data where the baked-in prompt is stale
            (e.g. broken workflow_script, old FORMAT_RULES).

    Returns:
        Enriched system-message string.
    """
    if force_rebuild:
        # Strip the existing enrichment back to the bare persona line so the
        # rebuild path uses current FORMAT_RULES and a fresh workflow_script.
        for marker in ("\n\nWorkflow script", "\nWorkflow script"):
            idx = original_content.find(marker)
            if idx != -1:
                original_content = original_content[:idx].rstrip()
                break
    elif "Workflow script" in original_content:
        return original_content

    parts: list[str] = [original_content]

    # Regenerate the workflow script from the actual GT conversation rather than
    # trusting sample["workflow_script"] (which is frozen at data-generation time
    # against a randomly-populated state.tools field and is wrong in ~60% of
    # samples — see audit in docs/task_a_data_quality_review.md).
    workflow_graph = sample.get("workflow_graph") or {}
    messages = sample.get("messages") or []
    tool_schemas = sample.get("tool_schemas") or []
    language = sample.get("language") or "en"
    if workflow_graph.get("state_details"):
        script = build_workflow_script(
            workflow_graph,
            tool_schemas=tool_schemas,
            language=language,
            messages=messages,
        )
    else:
        script = sample.get("workflow_script") or ""
    if script:
        parts.append(f"\nWorkflow script (follow this for conversation flow):\n{script}")

    graph = sample.get("workflow_graph", {})
    initial = graph.get("initial", "")
    terminal = graph.get("terminal", [])
    tool_schemas = sample.get("tool_schemas") or []

    if initial or terminal or tool_schemas:
        ref_parts = [
            "\nStructured reference:",
            f"  Initial state: {initial}",
            f"  Terminal states: {', '.join(terminal)}",
        ]
        if tool_schemas:
            ref_parts.append(
                "\nTool schemas (authoritative — call only these tools, with only these "
                "parameters; honour required vs optional):\n"
                + json.dumps(tool_schemas, indent=2, ensure_ascii=False)
            )
        else:
            ref_parts.append("\nTool schemas: none — this workflow does not call any tools.")
        parts.append("\n".join(ref_parts))

    parts.append(f"\n{FORMAT_RULES}")
    return "\n".join(parts)
