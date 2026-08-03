"""Shared system-prompt enrichment for Task A workflow datasets.

Single source of truth for FORMAT_RULES and the enrichment helper used by
both the data-generation pipeline and the benchmark harness.
"""

from __future__ import annotations

import json
import os
import warnings
from functools import lru_cache
from typing import Any

from llm_workflow_agents.config.schema import COMPLEXITY_SPECS
from llm_workflow_agents.data._workflow_script import build_workflow_script

# Default-on since 2026-07-31 (task-a-sft-v2). Opt out with TASK_A_STAY_RULE=0
# to reproduce the exact v1 prompt every existing checkpoint (ckpt-500,
# ckpt-1770) was trained against — see tests/unit/_v1_format_rules_golden.py
# and docs/superpowers/specs/2026-07-31-task-a-tool-stay-convention-design.md.
#
# FORMAT_RULES rule 1 gives the SYNTAX for a self-loop but never the POLICY.
# Meanwhile the workflow script said only "on success: proceed to [Y]", which a
# policy follows literally: it advances on the same turn it issues the tool
# call, before the result exists. Measured on ckpt-500 / task-a-sft-v1: gold
# expects a self-loop on 41.6% of turns, the policy emits one on 4.8%, and
# state accuracy on stay+tool-expected turns is 0.055.
# See docs/cat_a_state_annotation_convention_review.md §5.
#
# NOTE: this flag selects the whole prompt REVISION, not just the stay rule.
# v1 (disabled) is a frozen historical artifact and differs from v2 in three
# places: no stay rule, the rule-2 worked example shows an advancing transition
# (the defect this revision fixes), and the tool-error rule is the vague
# one-liner with no retry budget. Anything that changes v1's rendered bytes
# breaks checkpoint comparability and is caught by the golden test.
def _resolve_stay_rule_flag() -> bool:
    """Read TASK_A_STAY_RULE. Only the exact string "0" opts out.

    Parsing is deliberately exact rather than lenient: the flag selects which
    prompt revision gets frozen into a corpus, so "close enough" matching is the
    wrong trade. But `!= "0"` fails OPEN toward v2 — a typo like "false", "off"
    or a stray trailing space silently yields the v2 prompt when v1 was meant,
    and that is the direction that breaks checkpoint comparability. So an
    unrecognised value still resolves to enabled (no behaviour change, v1 bytes
    untouched) but warns, so the typo surfaces instead of quietly producing the
    wrong corpus.
    """
    raw = os.environ.get("TASK_A_STAY_RULE")
    if raw is None:
        return True
    if raw not in ("0", "1"):
        warnings.warn(
            f"TASK_A_STAY_RULE={raw!r} is not a recognised value; expected "
            '"1" (default — the v2 prompt with the stay rule) or "0" (opt out to '
            "the frozen v1 prompt). Only the exact string \"0\" disables it, so "
            "this value resolves to ENABLED (v2). If you meant to opt out, set "
            'TASK_A_STAY_RULE=0 exactly.',
            RuntimeWarning,
            stacklevel=2,
        )
    return raw != "0"


_STAY_RULE_ENABLED = _resolve_stay_rule_flag()

STAY_RULE = """\
2. Tool-execution turns do NOT advance. When your turn issues a <tool_call>, you have not
   yet seen the result, so you MUST remain in the current state and write the same name on
   both sides:
       [STATE: SEARCH_OPTIONS → SEARCH_OPTIONS]
       <tool_call>{"name": "search_flights", "arguments": {...}}</tool_call>
   Advance only on a LATER turn, after the tool result has come back and you have used it.
   The workflow script's "- on success: proceed to [Y]" describes where to go once the
   result confirms success — it does NOT mean advance on the turn that issues the call.
   Never announce a state you do not yet have the evidence to be in."""

# v2 tool-error rule, in two forms. The budget counts TOTAL attempts including the
# first, so budget 1 means "no retry at all" — phrasing that as "retry at most 1
# time(s)" is both self-contradictory and ungrammatical. No shipped complexity
# level uses budget 1 any more (L1-L4 = 2, L5 = 3), but budget 1 is still the
# DEFAULT_RETRY_BUDGET degradation path for a sample with a missing or unknown
# complexity_level, so both forms must stay grammatical. The two forms state the
# same policy; pick with _retry_rule().
_RETRY_RULE_NO_RETRY = """\
{n}. If a tool returns an error, do NOT retry it — this workflow allows one attempt per
   call. Stay in the current state, keep annotating [STATE: X → X], and never advance on a
   failure: follow the workflow's error path if the script names one, otherwise say plainly
   that this step cannot be completed right now and that you will hand off / follow up,
   while remaining in [STATE: X → X]. Do not invent a transition just to move past a
   failure."""

_RETRY_RULE_WITH_RETRIES = """\
{n}. If a tool returns an error, stay in the current state and you may retry the SAME call,
   annotating [STATE: X → X] on every attempt — never advance while retrying. You get
   {retry_budget} attempts at that call in total, counting the first; once they are used up, stop
   retrying: follow the workflow's error path if the script names one, otherwise say plainly
   that this step cannot be completed right now and that you will hand off / follow up,
   while remaining in [STATE: X → X]. Do not invent a transition just to move past a
   failure."""

# v1 tool-error rule, frozen. Note v1 renders its final three rules separated by
# SINGLE newlines (every other rule is separated by a blank line); that spacing
# is part of the bytes the checkpoints saw, so it is reproduced exactly here.
_RETRY_RULE_V1 = "{n}. If a tool returns an error, attempt recovery before escalating."


def _retry_rule(n: int, retry_budget: int) -> str:
    """Render the v2 tool-error rule for *retry_budget* total attempts.

    ``retry_budget`` counts the first attempt, so a budget of 1 permits no retry
    and selects the no-retry wording. Budgets below 1 are treated the same way
    (there is no coherent "fewer than one attempt" policy to state, and refusing
    to render would fail a corpus build rather than a config check).
    """
    if retry_budget <= 1:
        return _RETRY_RULE_NO_RETRY.format(n=n)
    return _RETRY_RULE_WITH_RETRIES.format(n=n, retry_budget=retry_budget)


def _format_rules(retry_budget: int = 1, stay_rule: bool | None = None) -> str:
    """Render FORMAT_RULES.

    Args:
        retry_budget: Max TOTAL attempts for a failing tool call (the first
            attempt counts), rendered into the v2 tool-error rule. A budget of 1
            therefore permits no retry and selects different wording — see
            :func:`_retry_rule`. Ignored when *stay_rule* is False, because the
            v1 rule states no budget and its bytes are frozen.
        stay_rule: Select the prompt revision. ``True`` → v2 (stay rule, fixed
            worked example, retry budget). ``False`` → the frozen v1 text.
            ``None`` (default) → follow the ``TASK_A_STAY_RULE`` env flag.
    """
    if stay_rule is None:
        stay_rule = _STAY_RULE_ENABLED

    # v1's rule-2 worked example demonstrated an ADVANCING transition on a
    # tool-call turn — the exact defect this revision exists to fix. v2 shows a
    # self-loop, matching rule 1's syntax example.
    example_transition = (
        "VERIFY_PATIENT → VERIFY_PATIENT" if stay_rule else "VERIFY_PATIENT → TERMINAL"
    )

    rules = [
        """1. Turn template — EVERY assistant turn MUST start with a state annotation, including
   tool-only turns and the terminal turn. Use exactly this format on the first line:
       [STATE: CURRENT → NEXT]
   If the state does not change, write the same name on both sides:
       [STATE: QUALIFY_PROSPECT → QUALIFY_PROSPECT]
   Never omit this line. A turn that is "just a tool call" still needs the STATE line
   above the <tool_call> tag.""",
    ]

    next_num = 2
    if stay_rule:
        rules.append(STAY_RULE)
        next_num = 3

    rules.append(
        f"""{next_num}. Tool-call format — when you call a tool, emit it on its own line(s) as:
       <tool_call>{{"name": "<tool_name>", "arguments": {{<arg_key>: <arg_value>, ...}}}}</tool_call>
   The two top-level keys are exactly "name" and "arguments". Do NOT flatten arguments
   into the top level. Worked example for a schema with required=[patient_id, specialty]
   and optional reason:
       [STATE: {example_transition}]
       <tool_call>{{"name": "request_referral", "arguments": {{"patient_id": "P12345", "specialty": "cardiology"}}}}</tool_call>"""
    )

    next_num += 1
    rules.append(
        f"""{next_num}. Tool authority — the "Tool schemas" section is the ONLY authoritative source for which
   tools exist and which parameters they accept. The "Workflow script" hints at conversation
   flow but its per-state tool listings are UNRELIABLE; if it conflicts with a tool schema,
   trust the schema. Note that the schema uses "parameters" (OpenAI tools format) while
   your <tool_call> emits "arguments" — these refer to the same thing, do not confuse them."""
    )

    next_num += 1
    rules.append(
        f"""{next_num}. Argument discipline (strict):
   a. Pass ONLY parameters listed in the schema's "required" array, plus any optional
      parameter for which the user has EXPLICITLY stated a value in the conversation.
   b. Do NOT invent values for optional parameters. If the user has not said anything
      about `reason`, `description`, `offer_details`, `notes`, etc., omit those fields.
   c. Use parameter values verbatim from the user. Do not paraphrase, expand abbreviations,
      or reformat (e.g. user says "premium" → pass exactly "premium", not "premium package";
      user says "competitor.com" → ask for the full URL before calling; do not fabricate one)."""
    )

    next_num += 1
    rules.append(
        f"""{next_num}. Tool-call necessity — do not call a tool unless the workflow requires it. Greetings,
   acknowledgements, clarifying questions, and terminal closings are text-only turns; do
   not append a tool call just to "wrap up" the conversation."""
    )

    next_num += 1
    rules.append(
        f"""{next_num}. Multi-turn negotiation — if a required argument is missing from what the user has said
   so far, ask the user for it BEFORE calling the tool. Do not synthesize plausible values."""
    )

    next_num += 1
    terminal_rule = f"{next_num + 1}. Reach a terminal state to complete the workflow."
    never_skip_rule = f"{next_num + 2}. Never skip states or make invalid transitions."

    if stay_rule:
        rules.append(_retry_rule(next_num, retry_budget))
        rules.append(terminal_rule)
        rules.append(never_skip_rule)
    else:
        # Frozen v1 spacing: the last three rules form one single-newline block.
        rules.append(
            "\n".join(
                [_RETRY_RULE_V1.format(n=next_num), terminal_rule, never_skip_rule]
            )
        )

    return "Rules:\n\n" + "\n\n".join(rules)


# Module-level rendering at the DEFAULT budget, kept as a stable import for
# callers that want "the rules" without a sample in hand (tests, docs, ad-hoc
# probes). Per-sample prompts must NOT use this constant — they go through
# :func:`format_rules_for_sample`, which renders the rule at the sample's own
# retry budget. Keeping the constant at budget 1 preserves its historical
# meaning for every existing importer.
FORMAT_RULES = _format_rules()

#: Public name for :func:`_format_rules`, for callers outside this module.
#: ``generate_workflows.py`` needs the rules at an arbitrary budget when building
#: a teacher prompt — neither the frozen :data:`FORMAT_RULES` constant nor the
#: sample-keyed :func:`format_rules_for_sample` fits — and reaching across
#: modules for an underscored name made this module's private surface part of
#: its contract by accident. Same object, so the two can never drift.
render_format_rules = _format_rules

#: Budget used when a sample carries no usable ``complexity_level``. Matches the
#: pre-Task-9 hardcoded behaviour (one attempt, no retry), so an unlabelled
#: sample renders exactly the prompt it rendered before this became per-sample.
DEFAULT_RETRY_BUDGET = 1


@lru_cache(maxsize=32)
def _format_rules_cached(retry_budget: int, stay_rule: bool) -> str:
    """Memoised :func:`_format_rules`.

    ``build_enriched_system_prompt`` is called once per row at SFT/GRPO data-load
    time (tens of thousands of calls), but there are only ~3 distinct budgets, so
    the rendering collapses to a handful of cache entries. The cache lives on the
    module object, so the module-reload trick the ``TASK_A_STAY_RULE`` tests use
    (see ``tests/unit/test_stay_rule_flag.py``) gets a fresh cache and cannot
    observe a stale revision.
    """
    return _format_rules(retry_budget=retry_budget, stay_rule=stay_rule)


def retry_budget_for_sample(sample: Any) -> int:
    """Resolve a sample's tool-error retry budget from its complexity level.

    Task 9 made ``retry_budget`` a per-level property of
    :data:`~llm_workflow_agents.config.schema.COMPLEXITY_SPECS` (L1-L4: 2,
    L5: 3) and the generator bakes conversations that actually retry
    that many times. This resolves the same number at train/eval load time so the
    rebuilt prompt states the policy the row's data was generated under.

    Degrades to :data:`DEFAULT_RETRY_BUDGET` — never raises — for a sample with a
    missing, non-string, or unknown ``complexity_level``: a prompt rebuild must
    not be able to fail a training run over a metadata gap.
    """
    if not isinstance(sample, dict):
        return DEFAULT_RETRY_BUDGET
    level = sample.get("complexity_level")
    if not isinstance(level, str):
        return DEFAULT_RETRY_BUDGET
    # ComplexityLevel is a StrEnum, so a plain "L3" hashes/compares equal to the
    # enum member and indexes COMPLEXITY_SPECS directly.
    spec = COMPLEXITY_SPECS.get(level.strip().upper())
    if spec is None:
        return DEFAULT_RETRY_BUDGET
    return max(1, int(spec.retry_budget))


def format_rules_for_sample(sample: Any) -> str:
    """Render FORMAT_RULES at *sample*'s own retry budget.

    Same text as :data:`FORMAT_RULES` except for the tool-error rule, which
    states the sample's budget instead of a hardcoded 1. Under
    ``TASK_A_STAY_RULE=0`` the budget is ignored entirely (the frozen v1 rule
    states no budget), so v1 bytes are identical for every sample.
    """
    return _format_rules_cached(retry_budget_for_sample(sample), _STAY_RULE_ENABLED)


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
            ``workflow_graph``, ``tool_schemas``, and ``complexity_level``. The
            last selects the retry budget stated by the prompt's two tool-error
            passages (see :func:`retry_budget_for_sample`); a sample without it
            renders at :data:`DEFAULT_RETRY_BUDGET`.
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

    # One budget for the whole prompt. Both halves that mention retries — the
    # workflow script's per-state tool-error note and FORMAT_RULES' tool-error
    # rule — are rendered from THIS value, so a sample can never carry two
    # contradictory retry policies a few hundred characters apart.
    retry_budget = retry_budget_for_sample(sample)

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
            tool_turn_semantics=True,
            retry_budget=retry_budget,
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

    # STAY_RULE is rendered inline as rule 2 of FORMAT_RULES (default-on since
    # 2026-07-31); it must NOT also be appended here or TASK_A_STAY_RULE=1 would
    # emit it twice with inconsistent numbering.
    #
    # Rendered per-sample rather than using the module-level FORMAT_RULES
    # constant, which is frozen at budget 1: an L5 row whose data shows three
    # attempts must not be shipped under a "do NOT retry it" rule.
    parts.append(f"\n{_format_rules_cached(retry_budget, _STAY_RULE_ENABLED)}")
    return "\n".join(parts)
