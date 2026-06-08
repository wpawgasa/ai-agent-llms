"""Experiment A: Generate multi-turn workflow conversation datasets.

Generates datasets at 5 complexity levels (L1-L5) with tool-calling
annotations and state-machine ground truth. Supports 18 call center
domains (decoupled from complexity level).

Teacher model generation:
  Set ``teacher_model`` in ``generate_workflow_dataset`` to a model name to
  call a live API instead of the placeholder generator.
  Supported prefixes:
    - ``gemini-*``  → Google GenAI  (requires GEMINI_API_KEY env var)
    - ``gpt-*``     → OpenAI        (requires OPENAI_API_KEY env var)
    - ``claude-*``  → Anthropic     (requires ANTHROPIC_API_KEY env var)
  Falls back to placeholder on API error.
"""

from __future__ import annotations

import json
import re
import random
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog

from llm_workflow_agents.data._teacher_client import call_teacher_model
from llm_workflow_agents.data._workflow_script import find_tool_placement_violations
from llm_workflow_agents.data.system_prompt import FORMAT_RULES as _FORMAT_RULES
from llm_workflow_agents.config.schema import (
    COMPLEXITY_SPECS,
    TOOL_ERROR_RATE,
    USER_BEHAVIOR_DISTRIBUTION,
    ComplexityLevel,
    ComplexitySpec,
)
from llm_workflow_agents.data.domain_registry import (
    ALL_DOMAIN_NAMES,
    CROSS_CUTTING_INTENTS,
    CROSS_CUTTING_TOOLS,
    DOMAIN_REGISTRY,
    DomainSpec,
    classify_intent,
)

logger = structlog.get_logger(__name__)


def _find_transition_violations(
    valid_edges: set[tuple[str, str]],
    messages: list[dict[str, Any]],
) -> list[str]:
    """Return violation descriptions for [STATE: X→Y] annotations not in valid_edges."""
    violations = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        ann = (msg.get("annotations") or {}).get("state_transition") or {}
        src = ann.get("from")
        dst = ann.get("to")
        if src and dst and src != dst:  # skip in-state (X→X) annotations
            if (src, dst) not in valid_edges:
                violations.append(f"invalid transition [{src}→{dst}]")
    return violations


BEHAVIOR_PRESETS: dict[str, dict[str, float]] = {
    "default": {
        "cooperative": 0.60,
        "adversarial_probing": 0.15,
        "digressing": 0.10,
        "invalid_tool_inputs": 0.15,
    },
    "adversarial": {
        "cooperative": 0.45,
        "adversarial_probing": 0.25,
        "digressing": 0.15,
        "invalid_tool_inputs": 0.15,
    },
    "balanced": {
        "cooperative": 0.25,
        "adversarial_probing": 0.25,
        "digressing": 0.25,
        "invalid_tool_inputs": 0.25,
    },
    "cooperative_only": {
        "cooperative": 1.00,
        "adversarial_probing": 0.00,
        "digressing": 0.00,
        "invalid_tool_inputs": 0.00,
    },
}

INTENT_CATEGORY_PRESETS: dict[str, dict[str, float]] = {
    "default":      {"service": 0.70, "upsell_promo": 0.30},
    "service_only": {"service": 1.00, "upsell_promo": 0.00},
    "upsell_heavy": {"service": 0.50, "upsell_promo": 0.50},
}

INITIATION_PRESETS: dict[str, dict[str, float]] = {
    "default":         {"user": 1.00, "agent": 0.00},  # 100% inbound (back-compat)
    "balanced":        {"user": 0.70, "agent": 0.30},
    "outbound_heavy":  {"user": 0.40, "agent": 0.60},
}


@dataclass
class DatasetMetadata:
    """Metadata returned after dataset generation."""

    output_dir: Path
    complexity_level: str
    num_samples: int
    output_files: list[Path] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowState:
    """A single state in a workflow state machine."""

    id: str
    name: str
    tools: list[str] = field(default_factory=list)
    entry_actions: list[str] = field(default_factory=list)
    instruction: str = ""


@dataclass
class WorkflowTransition:
    """A transition between workflow states."""

    from_state: str
    to_state: str
    condition: str          # legacy field; rendered from label at to_dict()
    priority: int = 0
    label: str = ""         # authored human-readable label
    trigger: str = "always" # one of _VALID_TRIGGERS
    optional: bool = False
    intent_category: str | None = None


@dataclass
class WorkflowGraph:
    """Complete workflow graph with states and transitions."""

    states: list[WorkflowState]
    transitions: list[WorkflowTransition]
    initial_state: str
    terminal_states: list[str]

    def to_dict(self) -> dict[str, Any]:
        name_of: dict[str, str] = {s.id: s.name for s in self.states}
        return {
            "states": [s.name for s in self.states],
            "state_details": [
                {
                    "name": s.name,
                    "tools": s.tools,
                    "entry_actions": s.entry_actions,
                    "instruction": s.instruction,
                }
                for s in self.states
            ],
            "transitions": [
                {
                    "from": name_of.get(t.from_state, t.from_state),
                    "to": name_of.get(t.to_state, t.to_state),
                    "condition": t.label if t.label else t.condition,
                    "priority": t.priority,
                }
                for t in self.transitions
            ],
            "initial": name_of.get(self.initial_state, self.initial_state),
            "terminal": [name_of.get(t, t) for t in self.terminal_states],
        }


def select_subgraph(
    domain: "DomainSpec",
    spec: "ComplexitySpec",
    rng: random.Random,
    intent_category: str = "service",
) -> WorkflowGraph:
    """Build a semantically-valid subgraph of domain's canonical edge graph.

    Algorithm:
    1. Walk the spine (optional=False edges) from initial, reserving one slot for the
       terminal so the total state count stays within target_path_len.
    2. Add num_branches optional edges (+ their dst states if not yet included).
    3. Add num_loops back-edges to earlier distinct states.
    4. Add tool_error recovery arcs if include_recovery.
    5. Add upsell arc when intent_category == 'upsell_promo' and domain has one.
    """
    # Build lookup structures
    state_map = {s.name: s for s in domain.states}
    spine_edges: dict[str, list] = {}
    branch_edges: dict[str, list] = {}
    for e in domain.edges:
        if not e.optional:
            spine_edges.setdefault(e.src, []).append(e)
        else:
            branch_edges.setdefault(e.src, []).append(e)

    terminal_set = set(domain.terminals)

    # Step 1: walk spine for target_len - 1 non-terminal states, then add terminal.
    # The last slot is reserved for the terminal so total count == target_len.
    target_len = rng.randint(*spec.target_path_len)
    spine_states: list[str] = [domain.initial]
    current = domain.initial

    while len(spine_states) < target_len - 1 and current not in terminal_set:
        candidates = spine_edges.get(current, [])
        if not candidates:
            break
        nxt = candidates[0].dst
        if nxt in set(spine_states):
            break
        spine_states.append(nxt)
        current = nxt
        if current in terminal_set:
            break

    # Ensure the last state is a terminal (may be a shortcut over intermediate states)
    if spine_states[-1] not in terminal_set:
        spine_states.append(domain.terminals[0])

    included_names: set[str] = set(spine_states)
    selected_states: list[str] = list(spine_states)

    # Collect spine transitions; use a synthetic edge where we jumped over spine states
    selected_transitions: list[WorkflowTransition] = []
    for i in range(len(spine_states) - 1):
        src, dst = spine_states[i], spine_states[i + 1]
        edge = next(
            (e for e in spine_edges.get(src, []) if e.dst == dst),
            None,
        )
        if edge:
            selected_transitions.append(WorkflowTransition(
                from_state=src, to_state=dst,
                condition=edge.label,
                label=edge.label,
                trigger=edge.trigger,
                optional=False,
                priority=0,
            ))
        else:
            # Shortcut edge — jumped over intermediate spine states to reach terminal
            selected_transitions.append(WorkflowTransition(
                from_state=src, to_state=dst,
                condition="proceed to resolution",
                label="proceed to resolution",
                trigger="always",
                optional=False,
                priority=0,
            ))

    # Step 2: add num_branches optional edges
    num_branches_target = rng.randint(*spec.num_branches)
    candidate_branch_edges = [
        e for src in included_names
        for e in branch_edges.get(src, [])
        if e.intent_category != "upsell_promo"
    ]
    rng.shuffle(candidate_branch_edges)
    branches_added = 0
    for e in candidate_branch_edges:
        if branches_added >= num_branches_target:
            break
        if e.dst not in included_names:
            included_names.add(e.dst)
            selected_states.append(e.dst)
        selected_transitions.append(WorkflowTransition(
            from_state=e.src, to_state=e.dst,
            condition=e.label, label=e.label,
            trigger=e.trigger, optional=True, priority=1,
        ))
        branches_added += 1

    # Step 3: add num_loops back-edges
    num_loops_target = rng.randint(*spec.num_loops)
    loops_added = 0
    all_loop_candidates = [
        e for src in included_names
        for e in branch_edges.get(src, [])
        if e.dst in included_names
        and e.dst != src
        and e.src in selected_states
        and selected_states.index(e.dst) < selected_states.index(e.src)
    ]
    rng.shuffle(all_loop_candidates)
    for e in all_loop_candidates:
        if loops_added >= num_loops_target:
            break
        if not any(t.from_state == e.src and t.to_state == e.dst for t in selected_transitions):
            selected_transitions.append(WorkflowTransition(
                from_state=e.src, to_state=e.dst,
                condition=e.label, label=e.label,
                trigger=e.trigger, optional=True, priority=1,
            ))
            loops_added += 1

    # Step 4: recovery arcs
    if spec.include_recovery:
        recovery_candidates = [
            e for src in included_names
            for e in branch_edges.get(src, [])
            if e.trigger == "tool_error" and e.dst in included_names
        ]
        for e in recovery_candidates:
            if not any(t.from_state == e.src and t.trigger == "tool_error" for t in selected_transitions):
                selected_transitions.append(WorkflowTransition(
                    from_state=e.src, to_state=e.dst,
                    condition=e.label, label=e.label,
                    trigger=e.trigger, optional=True, priority=1,
                ))

    # Step 5: upsell arc
    if intent_category == "upsell_promo":
        upsell_candidates = [
            e for src in included_names
            for e in branch_edges.get(src, [])
            if e.intent_category == "upsell_promo" and e.dst in included_names
        ]
        for e in upsell_candidates[:1]:
            selected_transitions.append(WorkflowTransition(
                from_state=e.src, to_state=e.dst,
                condition=e.label, label=e.label,
                trigger=e.trigger, optional=True, priority=1,
                intent_category="upsell_promo",
            ))

    # Step 6: close the subgraph — every non-terminal state must have an outgoing edge.
    # Branch destination states may lack outgoing edges if they weren't on the spine.
    changed = True
    while changed:
        changed = False
        states_with_out = {t.from_state for t in selected_transitions}
        for name in list(selected_states):
            if name in terminal_set or name in states_with_out:
                continue
            s_edges = spine_edges.get(name, [])
            if s_edges:
                e = s_edges[0]
                if e.dst not in included_names:
                    included_names.add(e.dst)
                    selected_states.append(e.dst)
                    changed = True
                selected_transitions.append(WorkflowTransition(
                    from_state=name, to_state=e.dst,
                    condition=e.label, label=e.label,
                    trigger=e.trigger, optional=False, priority=0,
                ))
            else:
                # No spine edge — add synthetic shortcut to terminal
                terminal_name = domain.terminals[0]
                if terminal_name not in included_names:
                    included_names.add(terminal_name)
                    selected_states.append(terminal_name)
                    changed = True
                selected_transitions.append(WorkflowTransition(
                    from_state=name, to_state=terminal_name,
                    condition="proceed to resolution",
                    label="proceed to resolution",
                    trigger="always", optional=False, priority=0,
                ))

    # Build WorkflowState list (spine order first, then appended branch states)
    def to_workflow_state(idx: int, name: str) -> WorkflowState:
        node = state_map[name]
        return WorkflowState(
            id=f"S{idx + 1}",
            name=name,
            tools=list(node.tools),
            instruction=node.instruction,
        )

    wf_states = [to_workflow_state(i, n) for i, n in enumerate(selected_states)]
    id_map = {s.name: s.id for s in wf_states}

    wf_transitions = [
        WorkflowTransition(
            from_state=id_map.get(t.from_state, t.from_state),
            to_state=id_map.get(t.to_state, t.to_state),
            condition=t.label,
            label=t.label,
            trigger=t.trigger,
            optional=t.optional,
            priority=t.priority,
            intent_category=t.intent_category,
        )
        for t in selected_transitions
        if t.from_state in id_map and t.to_state in id_map
    ]

    terminal_ids = [
        id_map[n] for n in selected_states
        if state_map[n].kind == "terminal"
    ]

    return WorkflowGraph(
        states=wf_states,
        transitions=wf_transitions,
        initial_state=id_map[domain.initial],
        terminal_states=terminal_ids,
    )


_VALID_TRIGGERS_SET = frozenset({
    "always", "tool_success", "tool_error",
    "intent_match", "slot_present", "user_declines",
})


def walk_path(
    subgraph: WorkflowGraph,
    domain: "DomainSpec",
    behavior: str,
    intent_category: str,
    rng: random.Random,
) -> list[WorkflowTransition]:
    """Traverse subgraph from initial, picking edges by simulated trigger.

    Returns the sequence of WorkflowTransition objects walked.
    Always terminates at a terminal state.
    """
    id_to_name = {s.id: s.name for s in subgraph.states}
    name_to_state = {s.name: s for s in subgraph.states}
    terminal_ids = set(subgraph.terminal_states)

    # Build outgoing edge map keyed by state id
    outgoing: dict[str, list[WorkflowTransition]] = {}
    for t in subgraph.transitions:
        outgoing.setdefault(t.from_state, []).append(t)

    path: list[WorkflowTransition] = []
    current_id = subgraph.initial_state
    max_steps = len(subgraph.states) * 3 + 5  # safety cap

    for _ in range(max_steps):
        if current_id in terminal_ids:
            break

        edges = outgoing.get(current_id, [])
        if not edges:
            break

        # Simulate trigger outcomes for this state
        current_state = name_to_state.get(id_to_name.get(current_id, ""))
        has_tools = bool(current_state and current_state.tools)

        fired: set[str] = {"always"}
        if has_tools:
            if rng.random() < TOOL_ERROR_RATE:
                fired.add("tool_error")
            else:
                fired.add("tool_success")
        if intent_category == "upsell_promo" and rng.random() < 0.4:
            fired.add("intent_match")
        if behavior in ("adversarial_probing",) and rng.random() < 0.3:
            fired.add("user_declines")
        if behavior in ("cooperative",) and rng.random() < 0.5:
            fired.add("slot_present")

        # Priority: upsell arc > optional with fired trigger > spine > anything else
        def edge_priority(e: WorkflowTransition) -> int:
            if e.intent_category == "upsell_promo" and "intent_match" in fired:
                return 0
            if e.optional and e.trigger in fired:
                return 1
            if not e.optional:
                return 2
            return 3

        sorted_edges = sorted(edges, key=edge_priority)
        chosen = sorted_edges[0]

        path.append(chosen)
        current_id = chosen.to_state

    return path


# _FORMAT_RULES is imported from data.system_prompt (single source of truth)

_SCRIPT_TEMPLATES: dict[str, dict[str, str]] = {
    "en": {
        "header": "### [{section}]",
        "initial_marker": "(initial state)",
        "terminal_marker": "This is the terminal state — end the conversation here.",
        "tools_intro": "Available tools: {tools}",
        "no_tools": "No tools available in this state.",
        "primary_branch": "- On success: proceed to [{to}]",
        "alt_branch": "- If {condition}: go to [{to}]",
        "condition_fallback": "alternative condition met",
    },
    "th": {
        "header": "### [{section}]",
        "initial_marker": "(สถานะเริ่มต้น)",
        "terminal_marker": "นี่คือสถานะสิ้นสุด — จบการสนทนาที่นี่",
        "tools_intro": "เครื่องมือที่ใช้ได้: {tools}",
        "no_tools": "ไม่มีเครื่องมือในสถานะนี้",
        "primary_branch": "- เมื่อสำเร็จ: ดำเนินการต่อที่ [{to}]",
        "alt_branch": "- หาก{condition}: ไปที่ [{to}]",
        "condition_fallback": "เงื่อนไขอื่น",
    },
}


def _humanise_condition(condition: str) -> str:
    """Convert a snake_case condition name to a readable phrase."""
    import re
    cleaned = condition.replace("proceed_from_", "").replace("branch_", "")
    cleaned = cleaned.replace("_", " ")                      # underscores → spaces first
    cleaned = re.sub(r"\b[Ss]\d+\b", "", cleaned)           # remove S1, S2 … (now word-bounded)
    cleaned = re.sub(r"\bto\b", "", cleaned)                 # remove bare "to" connector
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _graph_to_script(
    workflow: "WorkflowGraph",
    tool_schemas: list[dict[str, Any]],
    language: str = "en",
    messages: list[dict[str, Any]] | None = None,
) -> str:
    """Convert a WorkflowGraph to a natural language script.

    Delegates to :func:`build_workflow_script` (single source of truth in
    ``data/_workflow_script.py``). When ``messages`` is provided, per-state
    tools are inferred from actual GT tool calls and override the graph's
    ``state.tools`` field — fixes the data-generation bug where ``state.tools``
    was set randomly by ``rng.randint(0, 2)`` independent of conversation
    content (~60% of samples have script-vs-GT mismatches).
    """
    from llm_workflow_agents.data._workflow_script import build_workflow_script

    return build_workflow_script(
        workflow.to_dict(),
        tool_schemas=tool_schemas,
        language=language,
        messages=messages,
    )


_STATE_ANNOTATION_RE = re.compile(
    r"\[STATE:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:→|->)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\]"
)
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)


def _extract_ground_truth(
    messages: list[dict[str, Any]],
    workflow: "WorkflowGraph",
) -> dict[str, Any]:
    """Extract ground-truth labels from annotated messages.

    Reads from the ``annotations`` metadata dict if present (placeholder
    generator). Falls back to parsing ``[STATE: X → Y]`` and
    ``<tool_call>`` from message content (teacher-model generator).

    Returns a dict with:
    - ``state_sequence``: list of {from, to} transition dicts in order
    - ``tool_calls``: flat list of all tool-call dicts across the conversation
    - ``tool_chain_dependencies``: per-turn list of tool calls (preserves turn order)
    - ``terminal_state``: the last terminal state reached, or empty string
    """
    # Build a set of both IDs and names that count as terminal
    terminal_ids = set(workflow.terminal_states)
    terminal_names = {s.name for s in workflow.states if s.id in terminal_ids}
    terminal_set = terminal_ids | terminal_names

    state_sequence: list[dict[str, str]] = []
    tool_calls: list[dict[str, Any]] = []
    tool_chain_dependencies: list[list[dict[str, Any]]] = []

    for msg in messages:
        if msg.get("role") != "assistant":
            continue

        annotations = msg.get("annotations") or {}
        content = msg.get("content", "") or ""

        # --- State transitions ---
        transition = annotations.get("state_transition") or {}
        if transition.get("from") and transition.get("to"):
            state_sequence.append({"from": transition["from"], "to": transition["to"]})
        else:
            # Fallback: parse [STATE: X → Y] from content
            m = _STATE_ANNOTATION_RE.search(content)
            if m:
                state_sequence.append({"from": m.group(1), "to": m.group(2)})

        # --- Tool calls ---
        turn_tools = annotations.get("tool_calls") or []
        if not turn_tools:
            # Fallback: parse <tool_call>{JSON}</tool_call> from content
            for tc_match in _TOOL_CALL_RE.finditer(content):
                try:
                    parsed = json.loads(tc_match.group(1))
                    name = parsed.get("name", "")
                    args = parsed.get("arguments", {})
                    if name:
                        turn_tools.append({"name": name, "arguments": args})
                except json.JSONDecodeError:
                    continue
        tool_calls.extend(turn_tools)
        if turn_tools:
            tool_chain_dependencies.append(turn_tools)

    terminal_state = ""
    if state_sequence:
        last_to = state_sequence[-1]["to"]
        if last_to in terminal_set:
            terminal_state = last_to

    return {
        "state_sequence": state_sequence,
        "tool_calls": tool_calls,
        "tool_chain_dependencies": tool_chain_dependencies,
        "terminal_state": terminal_state,
    }


@dataclass
class ConversationSample:
    """A single generated conversation sample."""

    conversation_id: str
    complexity_level: str
    domain: str
    num_states: int
    num_tools: int
    chain_depth: int
    workflow_graph: dict[str, Any]
    workflow_script: str
    tool_schemas: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    user_behavior: str
    language: str = "en"
    ground_truth: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "complexity_level": self.complexity_level,
            "domain": self.domain,
            "num_states": self.num_states,
            "num_tools": self.num_tools,
            "chain_depth": self.chain_depth,
            "workflow_graph": self.workflow_graph,
            "workflow_script": self.workflow_script,
            "tool_schemas": self.tool_schemas,
            "messages": self.messages,
            "user_behavior": self.user_behavior,
            "language": self.language,
            "ground_truth": self.ground_truth,
        }


def _select_user_behavior(
    rng: random.Random,
    distribution: dict[str, float] | None = None,
) -> str:
    """Select a user behavior type based on the configured distribution."""
    dist = distribution if distribution is not None else USER_BEHAVIOR_DISTRIBUTION
    behaviors = list(dist.keys())
    weights = list(dist.values())
    return rng.choices(behaviors, weights=weights, k=1)[0]


def _select_intent_category(
    rng: random.Random,
    distribution: dict[str, float],
) -> str:
    """Select an intent category ('service' or 'upsell_promo') from a weighted distribution."""
    cats = list(distribution.keys())
    weights = list(distribution.values())
    return rng.choices(cats, weights=weights, k=1)[0]


def _select_initiator(
    rng: random.Random,
    distribution: dict[str, float],
) -> str:
    """Select who opens the conversation ('user' inbound | 'agent' outbound)."""
    cats = list(distribution.keys())
    weights = list(distribution.values())
    return rng.choices(cats, weights=weights, k=1)[0]


def _pick_intent_by_category(
    rng: random.Random,
    domain_intents: tuple[str, ...] | list[str],
    target_category: str,
) -> str:
    """Pick a domain intent matching target_category; fall back to any intent if none match."""
    matching = [i for i in domain_intents if classify_intent(i) == target_category]
    pool = matching if matching else list(domain_intents)
    return rng.choice(pool) if pool else ""


def _select_domain(
    rng: random.Random,
    domain: str | None = None,
    spec: "ComplexitySpec | None" = None,
) -> tuple[str, DomainSpec]:
    """Select a domain, filtering to those eligible for the requested complexity level."""
    _LEGACY_MAP: dict[str, str] = {
        "faq_lookup": "product_info",
        "order_status_cancel": "order_management",
        "booking_payment": "travel",
        "it_troubleshoot": "technical_support",
        "it_troubleshoot_escalation": "technical_support",
        "multi_dept_workflow": "complaints",
    }

    if domain and domain in DOMAIN_REGISTRY:
        return domain, DOMAIN_REGISTRY[domain]
    if domain and domain in _LEGACY_MAP:
        key = _LEGACY_MAP[domain]
        return key, DOMAIN_REGISTRY[key]

    # Level-aware: domain eligible iff canonical state count >= target_path_len min
    min_states = spec.target_path_len[0] if spec else 0
    eligible = [
        k for k, d in DOMAIN_REGISTRY.items()
        if len(d.states) >= min_states
    ]
    if not eligible:
        eligible = ALL_DOMAIN_NAMES  # fallback: all domains

    key = rng.choice(eligible)
    return key, DOMAIN_REGISTRY[key]



def _generate_placeholder_conversation(
    workflow: WorkflowGraph,
    tool_schemas: list[dict[str, Any]],
    behavior: str,
    spec: ComplexitySpec,
    rng: random.Random,
    domain_spec: DomainSpec | None = None,
    language: str = "en",
    intent_category: str = "service",
) -> list[dict[str, Any]]:
    """Generate a placeholder conversation following the workflow graph.

    In production, this would call the teacher model (GPT-4o / Claude).
    For now, generates structurally valid placeholder conversations using
    walk_path to traverse the subgraph with trigger-based edge selection.
    """
    messages: list[dict[str, Any]] = []
    domain_name = domain_spec.name if domain_spec else spec.level
    domain_intents = list(domain_spec.intents) if domain_spec else []

    messages.append({"role": "system", "content": f"You are a customer service agent handling {domain_name} workflows."})

    id_to_name = {s.id: s.name for s in workflow.states}
    name_to_state = {s.name: s for s in workflow.states}
    terminal_ids = set(workflow.terminal_states)

    # walk_path requires domain_spec for trigger simulation
    if domain_spec:
        path = walk_path(workflow, domain_spec, behavior, intent_category, rng)
    else:
        # Minimal fallback: walk states in spine order
        path = [
            WorkflowTransition(
                from_state=workflow.states[i].id,
                to_state=workflow.states[i + 1].id,
                condition="proceed",
                label="proceed",
                trigger="always",
            )
            for i in range(len(workflow.states) - 1)
        ]

    _user_templates: dict[str, dict[str, str]] = {
        "en": {
            "cooperative": "[Turn {t}] I need help with {intent}",
            "adversarial_probing": "[Turn {t}] Can you skip {state} and just do {intent} directly?",
            "digressing": "[Turn {t}] Actually, before we continue with {intent}, unrelated question about something else",
            "invalid_tool_inputs": "[Turn {t}] Process {intent} for ###invalid_id###",
        },
        "th": {
            "cooperative": "[ตา {t}] ฉันต้องการความช่วยเหลือเรื่อง{intent}",
            "adversarial_probing": "[ตา {t}] ข้ามขั้นตอน {state} แล้วทำ{intent}เลยได้ไหม?",
            "digressing": "[ตา {t}] จริงๆ ก่อนจะไปต่อเรื่อง{intent} ขอถามเรื่องอื่นก่อน",
            "invalid_tool_inputs": "[ตา {t}] ดำเนินการ{intent}สำหรับ ###invalid_id###",
        },
        "code_switch": {
            "cooperative": "[ตา {t}] ขอ help เรื่อง {intent} หน่อยนะคะ",
            "adversarial_probing": "[ตา {t}] ข้าม {state} step แล้ว proceed กับ {intent} เลยได้ไหมคะ?",
            "digressing": "[ตา {t}] ก่อนจะ continue เรื่อง {intent} ขอถาม unrelated เรื่องนึงก่อนนะคะ",
            "invalid_tool_inputs": "[ตา {t}] ช่วย process {intent} for ###invalid_id### ด้วยนะคะ",
        },
    }

    lang_templates = _user_templates.get(language or "en", _user_templates["en"])

    visited_state_ids: set[str] = set()
    turn_idx = 0

    for step in path:
        from_name = id_to_name.get(step.from_state, step.from_state)
        to_name = id_to_name.get(step.to_state, step.to_state)
        current_state = name_to_state.get(from_name)

        intent = _pick_intent_by_category(rng, domain_intents, intent_category) if domain_intents else domain_name
        intent_text = intent.replace("_", " ")
        tmpl = lang_templates.get(behavior, lang_templates["cooperative"])
        user_msg = tmpl.format(t=turn_idx + 1, intent=intent_text, state=from_name)
        messages.append({"role": "user", "content": user_msg})

        # In-state tool turn (emitted once per state visit)
        if current_state and current_state.tools and step.from_state not in visited_state_ids:
            tool_name = current_state.tools[0]
            tool_call = {"name": tool_name, "arguments": {"placeholder": "value"}}
            in_state_content = f"[STATE: {from_name} → {from_name}]\n<tool_call>{json.dumps(tool_call)}</tool_call>"
            messages.append({
                "role": "assistant",
                "content": in_state_content,
                "annotations": {
                    "state_transition": {"from": from_name, "to": from_name},
                    "tool_calls": [tool_call],
                },
            })
            if rng.random() < TOOL_ERROR_RATE:
                tool_response = json.dumps({"error": "Service temporarily unavailable"})
            else:
                tool_response = json.dumps({"status": "success", "data": {"result": "ok"}})
            messages.append({"role": "tool", "content": tool_response})
            visited_state_ids.add(step.from_state)

        # Transition turn
        transition_content = f"[STATE: {from_name} → {to_name}]"
        if not (current_state and current_state.tools):
            transition_content += f"\nHandling {domain_name} in state {from_name}."
        messages.append({
            "role": "assistant",
            "content": transition_content,
            "annotations": {"state_transition": {"from": from_name, "to": to_name}},
        })

        turn_idx += 1

    return messages


_TEACHER_SYSTEM_PROMPT = """\
You are a dataset generation expert creating training data for LLM workflow agents.
Generate a realistic multi-turn customer service conversation that strictly follows
the provided workflow graph, tool schemas, and user behavior pattern.

OUTPUT FORMAT — return a JSON object with a single key "messages" containing an array:
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {
      "role": "assistant",
      "content": "[STATE: CURRENT → NEXT]\\n<tool_call>{...}</tool_call>",
      "annotations": {
        "state_transition": {"from": "CURRENT", "to": "NEXT"},
        "tool_calls": [{"name": "...", "arguments": {...}}]
      }
    },
    {"role": "tool", "content": "{...}"},
    ...
  ]
}

RULES:
- Every assistant message MUST include a [STATE: X → Y] annotation in the content.
- When invoking a tool include <tool_call>{"name": "...", "arguments": {...}}</tool_call>.
- Follow the user behavior pattern exactly (cooperative / adversarial_probing / digressing / invalid_tool_inputs).
- The conversation MUST reach one of the terminal states before ending.
- ~20 % of tool responses should be errors: {"error": "Service temporarily unavailable"}.
- Output ONLY the JSON object — no markdown fences, no extra keys.
"""


_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "en": "Language: English — generate the entire conversation in English.",
    "th": (
        "Language: Thai (th-TH) — generate the entire conversation in Thai. "
        "Keep [STATE: X → Y] annotations, tool call JSON, and tool response JSON in English/ASCII. "
        "All user and assistant dialogue must be in Thai."
    ),
    "code_switch": (
        "Language: Thai-English code-switching (th-TH / en) — the conversation naturally mixes "
        "Thai and English within and across turns, as is common in Thai call-centre interactions. "
        "User messages may start in Thai and embed English terms (e.g. product names, status words, "
        "technical jargon) or switch to English mid-sentence. Assistant responses should mirror this "
        "register. Keep [STATE: X → Y] annotations, tool call JSON, and tool response JSON in "
        "English/ASCII."
    ),
}


def _build_teacher_prompt(
    workflow: WorkflowGraph,
    tool_schemas: list[dict[str, Any]],
    behavior: str,
    spec: ComplexitySpec,
    domain_spec: DomainSpec | None,
    language: str = "en",
    intent_category: str = "service",
) -> str:
    domain_name = domain_spec.name if domain_spec else spec.domain
    tool_names = [t["function"]["name"] for t in tool_schemas]
    lang_instruction = _LANGUAGE_INSTRUCTIONS.get(language, _LANGUAGE_INSTRUCTIONS["en"])
    script = _graph_to_script(workflow, tool_schemas, language)
    promo_line = (
        "Conversation focus: naturally weave in promotion, cross-sell, or upsell "
        "opportunities relevant to this domain. The workflow must still reach a "
        "terminal state; the upsell is a secondary arc, not a hijack.\n"
        if intent_category == "upsell_promo"
        else ""
    )
    transition_key = (
        "Transition trigger types: "
        "'always'=unconditional spine, 'tool_success'=after successful tool call, "
        "'tool_error'=after failed tool call, 'intent_match'=customer intent matches, "
        "'user_declines'=customer refuses, 'slot_present'=required slot provided.\n"
    )
    return (
        f"Domain: {domain_name}\n"
        f"Complexity level: {spec.level} "
        f"(path_len={spec.target_path_len[0]}–{spec.target_path_len[1]}, chain_depth={spec.chain_depth})\n"
        f"User behavior: {behavior}\n"
        f"{promo_line}"
        f"{lang_instruction}\n\n"
        f"Workflow script (natural language — follow this for conversation flow):\n{script}\n\n"
        f"{transition_key}\n"
        f"Workflow graph (structured reference — use for state annotations):\n{json.dumps(workflow.to_dict(), indent=2)}\n\n"
        f"Available tools ({len(tool_schemas)}):\n{json.dumps(tool_schemas, indent=2)}\n\n"
        f"Tool names in scope: {tool_names}\n\n"
        f"{_FORMAT_RULES}\n\n"
        "Generate the conversation now."
    )


_RICH_PROMPT_SYSTEM = """\
You are an expert at writing realistic voicebot and chatbot system prompts for
customer-service workflows. Given a workflow graph, tool schemas, and a target
language, author a rich natural-language system prompt that would guide an agent
through the conversation end-to-end.

OUTPUT FORMAT — return a JSON object with exactly one key "system_prompt":
{"system_prompt": "<the authored rich prompt text>"}

The authored prompt MUST contain:
1. A one-sentence persona / role line for the agent (mention the domain).
2. A "## GOAL" section stating the purpose of the call in plain language.
3. One "### [state_name]" section per state in the workflow graph, using the
   exact state names from the graph.
4. Inside each section:
   - One or more suggested dialogue lines in quotes that the agent should say.
   - Intent-based branching as bullets — "If the customer confirms, say '...'
     -> follow the [next_section] path"; "If the customer asks to be called
     back, proceed to [call_later] section"; etc.
   - Tool-call instructions when a state has tools: "Call <tool>(args) to
     <purpose>. Then confirm: '...'".
5. Cross-references between sections using [state_name] that match the
   transitions in the workflow graph.

RULES:
- Write the prompt in the requested language (English / Thai / Thai-English mix).
- Do NOT include TTS or serving markers such as <S>, <F>, [END_CONVERSATION],
  or [TRANSFER] — those are deployment concerns, not training signal.
- Use customer-intent language for branches, not schematic state-machine phrasing.
- Keep tool names and state names verbatim from the inputs.
- Output ONLY the JSON object — no markdown fences, no extra keys.
"""


def _build_rich_prompt_request(
    workflow: WorkflowGraph,
    tool_schemas: list[dict[str, Any]],
    domain_spec: DomainSpec | None,
    language: str,
) -> str:
    domain_name = domain_spec.name if domain_spec else "customer service"
    lang_instruction = _LANGUAGE_INSTRUCTIONS.get(language, _LANGUAGE_INSTRUCTIONS["en"])
    return (
        f"Domain: {domain_name}\n"
        f"{lang_instruction}\n\n"
        f"Workflow graph (authoritative — authored sections must match the state names):\n"
        f"{json.dumps(workflow.to_dict(), indent=2)}\n\n"
        f"Available tools:\n{json.dumps(tool_schemas, indent=2)}\n\n"
        f"Author the rich system prompt now."
    )


def _generate_rich_system_prompt(
    workflow: WorkflowGraph,
    tool_schemas: list[dict[str, Any]],
    domain_spec: DomainSpec | None,
    language: str,
    teacher_model: str,
) -> str:
    """Ask the teacher model to author a natural-language system prompt.

    Returns empty string on any failure; the caller falls back to the bare
    role line in that case.
    """
    user_prompt = _build_rich_prompt_request(workflow, tool_schemas, domain_spec, language)
    try:
        raw = call_teacher_model(teacher_model, _RICH_PROMPT_SYSTEM, user_prompt)
        data = json.loads(raw)
        text = str(data.get("system_prompt", "")).strip()
        if not text:
            raise ValueError("empty system_prompt in teacher response")
        return text
    except Exception as exc:
        logger.warning(
            "rich_system_prompt_fallback",
            teacher_model=teacher_model,
            error=str(exc),
        )
        return ""


def _parse_messages_response(raw: str) -> list[dict[str, Any]]:
    """Parse a JSON response into a list of messages.

    Filters out any items that are not dicts with a ``role`` key so that
    malformed teacher responses don't silently propagate invalid messages.
    Raises ValueError if nothing valid remains (triggers placeholder fallback).
    """
    data = json.loads(raw)
    items: list[Any] = data["messages"] if isinstance(data, dict) and "messages" in data else data
    valid = [m for m in items if isinstance(m, dict) and "role" in m]
    if not valid:
        raise ValueError("Teacher model returned no valid messages (missing 'role' key)")
    return valid


def _generate_teacher_conversation(
    workflow: WorkflowGraph,
    tool_schemas: list[dict[str, Any]],
    behavior: str,
    spec: ComplexitySpec,
    rng: random.Random,
    domain_spec: DomainSpec | None,
    teacher_model: str,
    language: str = "en",
    intent_category: str = "service",
) -> list[dict[str, Any]]:
    """Call a teacher model API to generate a conversation.

    Falls back to placeholder generation if the API call fails.
    """
    user_prompt = _build_teacher_prompt(
        workflow, tool_schemas, behavior, spec, domain_spec, language, intent_category
    )
    try:
        raw = call_teacher_model(teacher_model, _TEACHER_SYSTEM_PROMPT, user_prompt)
        messages = _parse_messages_response(raw)
        if not messages:
            raise ValueError("Teacher model returned empty messages")
        return messages
    except Exception as exc:
        logger.warning(
            "teacher_model_fallback",
            teacher_model=teacher_model,
            error=str(exc),
        )
        return _generate_placeholder_conversation(
            workflow, tool_schemas, behavior, spec, rng, domain_spec, language, intent_category
        )


def generate_workflow_dataset(
    complexity_level: Literal["L1", "L2", "L3", "L4", "L5"],
    num_samples: int = 200,
    teacher_model: str | None = None,
    output_dir: Path = Path("data/output/task_a"),
    seed: int = 42,
    domain: str | None = None,
    language: Literal["en", "th", "code_switch"] | None = None,
    behavior_preset: str = "default",
    rich_prompt_rate: float = 0.30,
    intent_category_preset: str = "default",
    repair_incoherent: bool = True,
    max_repair_retries: int = 2,
) -> DatasetMetadata:
    """Generate multi-turn conversation dataset for a single complexity level.

    Domains are decoupled from complexity levels. If ``domain`` is None,
    each sample picks a random domain from the 17-domain registry, producing
    diverse training data across all call center verticals.

    Args:
        complexity_level: One of L1-L5.
        num_samples: Number of conversations to generate.
        teacher_model: Teacher model name for live API generation.
            Supported prefixes: ``gemini-*`` (GEMINI_API_KEY),
            ``gpt-*`` (OPENAI_API_KEY), ``claude-*`` (ANTHROPIC_API_KEY).
            If ``None``, uses the local placeholder generator.
        output_dir: Directory for output JSONL files.
        seed: Random seed for reproducibility.
        domain: Optional domain key (e.g., "banking", "healthcare").
            If None, randomly samples from all 17 domains per conversation.
        language: Conversation language. ``"en"`` for English, ``"th"`` for Thai,
            ``"code_switch"`` for Thai-English code-switching within each conversation,
            or ``None`` to randomly assign ``"en"`` or ``"th"`` per sample (50/50).
        behavior_preset: User behavior distribution preset. One of
            ``"default"`` (60/15/10/15), ``"adversarial"`` (45/25/15/15),
            or ``"balanced"`` (25/25/25/25).
        rich_prompt_rate: Fraction of samples whose ``messages[0]`` is a
            teacher-authored rich natural-language system prompt (persona +
            named sections with dialogue hints and intent-based branches).
            Applied only when ``teacher_model`` is set; otherwise ignored.
            The workflow script from ``_graph_to_script`` is still appended
            to every sample regardless of this setting. Default 0.30.
        intent_category_preset: Controls the share of promo/upsell-focused
            conversations. ``"default"`` targets 70% service / 30% upsell_promo.
            ``"service_only"`` disables upsell biasing entirely.
            ``"upsell_heavy"`` uses a 50/50 split.
        repair_incoherent: When True (default), teacher-generated conversations
            that call a tool in a state the curated map disallows are retried and,
            if still incoherent, replaced by the deterministic placeholder
            conversation so every emitted sample respects the curated tool
            placement. No-op for placeholder generation (already coherent).
        max_repair_retries: Maximum teacher regenerations per sample before the
            placeholder fallback is used. Reported via the ``repair_retries`` and
            ``repair_fallbacks`` stats.

    Returns:
        DatasetMetadata with paths to generated JSONL files and statistics.
    """
    if behavior_preset not in BEHAVIOR_PRESETS:
        raise ValueError(
            f"Unknown behavior_preset {behavior_preset!r}. "
            f"Valid options: {list(BEHAVIOR_PRESETS)}"
        )
    if intent_category_preset not in INTENT_CATEGORY_PRESETS:
        raise ValueError(
            f"Unknown intent_category_preset {intent_category_preset!r}. "
            f"Valid options: {list(INTENT_CATEGORY_PRESETS)}"
        )
    active_distribution = BEHAVIOR_PRESETS[behavior_preset]
    active_intent_dist = INTENT_CATEGORY_PRESETS[intent_category_preset]

    level = ComplexityLevel(complexity_level)
    spec = COMPLEXITY_SPECS[level]
    rng = random.Random(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lang_tag = language or "mixed"
    model_tag = teacher_model.replace("/", "-").replace(".", "-") if teacher_model else "placeholder"
    preset_tag = f"_{behavior_preset}" if behavior_preset != "default" else ""
    output_file = output_dir / f"{complexity_level.lower()}_conversations_{lang_tag}_{model_tag}{preset_tag}_{timestamp}.jsonl"

    logger.info(
        "generating_workflow_dataset",
        level=complexity_level,
        num_samples=num_samples,
        domain=domain or "random (18 domains)",
        teacher_model=teacher_model or "placeholder",
        language=language or "mixed (en/th)",
        behavior_preset=behavior_preset,
        intent_category_preset=intent_category_preset,
    )

    samples: list[ConversationSample] = []
    behavior_counts: dict[str, int] = {b: 0 for b in active_distribution}
    domain_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    intent_category_counts: dict[str, int] = {c: 0 for c in active_intent_dist}
    tool_error_count = 0
    total_tool_calls = 0
    rich_prompt_count = 0
    repair_retries = 0
    repair_fallbacks = 0

    for i in range(num_samples):
        behavior = _select_user_behavior(rng, active_distribution)
        behavior_counts[behavior] += 1

        # Select language (fixed or 50/50 mix per sample)
        sample_language = language if language is not None else rng.choice(["en", "th"])
        language_counts[sample_language] = language_counts.get(sample_language, 0) + 1

        # Select domain (fixed or random per sample, filtered by complexity level)
        domain_key, domain_spec = _select_domain(rng, domain, spec)
        domain_counts[domain_key] = domain_counts.get(domain_key, 0) + 1

        # Select intent category (service vs upsell_promo)
        intent_category = _select_intent_category(rng, active_intent_dist)
        intent_category_counts[intent_category] = intent_category_counts.get(intent_category, 0) + 1

        workflow = select_subgraph(domain_spec, spec, rng, intent_category)

        # Collect tools referenced by the selected subgraph states
        subgraph_tool_names = {tool for s in workflow.states for tool in s.tools}
        tool_schemas = [
            t for t in domain_spec.tools
            if t["function"]["name"] in subgraph_tool_names
        ]
        # Supplement to reach spec.num_tools if needed; never truncate subgraph tools
        if len(tool_schemas) < spec.num_tools:
            extra = [t for t in CROSS_CUTTING_TOOLS if t not in tool_schemas]
            tool_schemas.extend(extra[: spec.num_tools - len(tool_schemas)])

        def _placeholder() -> list[dict[str, Any]]:
            return _generate_placeholder_conversation(
                workflow, tool_schemas, behavior, spec, rng, domain_spec, sample_language,
                intent_category,
            )

        if teacher_model:
            messages = _generate_teacher_conversation(
                workflow, tool_schemas, behavior, spec, rng, domain_spec, teacher_model,
                sample_language, intent_category,
            )
            # Post-generation repair: the teacher is only *prompted* with the
            # curated tool placement, so it may still call a tool in a state that
            # does not allow it. Retry, then fall back to the deterministic
            # placeholder conversation (always coherent) so every emitted sample
            # respects the curated map without shrinking the dataset.
            if repair_incoherent:
                allowed = {s.name: set(s.tools) for s in workflow.states}
                schema_names = {t["function"]["name"] for t in tool_schemas}
                id_to_name = {s.id: s.name for s in workflow.states}
                valid_edge_pairs = {
                    (id_to_name.get(t.from_state, t.from_state),
                     id_to_name.get(t.to_state, t.to_state))
                    for t in workflow.transitions
                } | {(id_to_name.get(sid, sid), id_to_name.get(sid, sid))
                     for sid in {t.from_state for t in workflow.transitions}}

                def _has_violations(msgs: list[dict[str, Any]]) -> bool:
                    return bool(
                        find_tool_placement_violations(allowed, msgs, schema_names)
                        or _find_transition_violations(valid_edge_pairs, msgs)
                    )

                tries = 0
                while _has_violations(messages):
                    if tries >= max_repair_retries:
                        messages = _placeholder()
                        repair_fallbacks += 1
                        break
                    tries += 1
                    repair_retries += 1
                    messages = _generate_teacher_conversation(
                        workflow, tool_schemas, behavior, spec, rng, domain_spec,
                        teacher_model, sample_language, intent_category,
                    )
        else:
            messages = _placeholder()

        # Optionally replace the bare role line with a teacher-authored rich
        # natural-language system prompt (persona + named sections + dialogue
        # + intent-based branches). Applied before enrichment so the workflow
        # script still gets appended.
        use_rich_prompt = bool(teacher_model) and rng.random() < rich_prompt_rate
        if use_rich_prompt:
            rich_prompt = _generate_rich_system_prompt(
                workflow, tool_schemas, domain_spec, sample_language, teacher_model
            )
            if rich_prompt:
                rich_prompt_count += 1
                if messages and messages[0].get("role") == "system":
                    messages[0] = {"role": "system", "content": rich_prompt}
                else:
                    messages.insert(0, {"role": "system", "content": rich_prompt})

        # Keep messages[0] a BARE system message. The workflow script, tool
        # schemas, and format rules are deliberately NOT embedded here: the
        # training (sft.py, grpo.py) and eval (agent_benchmark.py) pipelines
        # always inject them at load time via build_enriched_system_prompt,
        # which derives a fresh script from workflow_graph + messages. Baking
        # the enrichment into the stored data would only duplicate that and go
        # stale (hence force_rebuild=True in the loaders). The authoritative
        # rendered script is preserved separately in the top-level
        # `workflow_script` field below. The teacher path returns no system
        # turn, so ensure one exists for the loaders' messages[0] check.
        if not (messages and messages[0].get("role") == "system"):
            domain_name = domain_spec.name if domain_spec else spec.domain
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": f"You are a customer service agent handling {domain_name} workflows.",
                },
            )

        # Count tool calls and errors
        for msg in messages:
            if msg.get("role") == "tool":
                total_tool_calls += 1
                try:
                    content = json.loads(msg["content"])
                    if "error" in content:
                        tool_error_count += 1
                except (json.JSONDecodeError, TypeError):
                    pass

        sample = ConversationSample(
            conversation_id=f"{complexity_level}_{i + 1:03d}",
            complexity_level=complexity_level,
            domain=domain_key,
            num_states=len(workflow.states),
            num_tools=spec.num_tools,
            chain_depth=spec.chain_depth,
            workflow_graph=workflow.to_dict(),
            workflow_script=_graph_to_script(workflow, tool_schemas, sample_language, messages=messages),
            tool_schemas=tool_schemas,
            messages=messages,
            user_behavior=behavior,
            language=sample_language,
            ground_truth=_extract_ground_truth(messages, workflow),
        )
        samples.append(sample)

    # Write JSONL
    with open(output_file, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    stats = {
        "behavior_distribution": behavior_counts,
        "domain_distribution": domain_counts,
        "language_distribution": language_counts,
        "intent_category_distribution": intent_category_counts,
        "tool_error_rate": tool_error_count / max(total_tool_calls, 1),
        "total_tool_calls": total_tool_calls,
        "avg_states": sum(s.num_states for s in samples) / len(samples),
        "num_domains": len(domain_counts),
        "rich_prompt_count": rich_prompt_count,
        "rich_prompt_rate_effective": rich_prompt_count / max(len(samples), 1),
        "repair_retries": repair_retries,
        "repair_fallbacks": repair_fallbacks,
    }

    logger.info("dataset_generated", output_file=str(output_file), **stats)

    return DatasetMetadata(
        output_dir=output_dir,
        complexity_level=complexity_level,
        num_samples=num_samples,
        output_files=[output_file],
        stats=stats,
    )
