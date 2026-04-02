"""Experiment A: Generate multi-turn workflow conversation datasets.

Generates datasets at 5 complexity levels (L1-L5) with tool-calling
annotations and state-machine ground truth. Supports 17 call center
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
import random
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog

from llm_workflow_agents.data._teacher_client import call_teacher_model
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
)

logger = structlog.get_logger(__name__)


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


@dataclass
class WorkflowTransition:
    """A transition between workflow states."""

    from_state: str
    to_state: str
    condition: str
    priority: int = 0


@dataclass
class WorkflowGraph:
    """Complete workflow graph with states and transitions."""

    states: list[WorkflowState]
    transitions: list[WorkflowTransition]
    initial_state: str
    terminal_states: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "states": [s.id for s in self.states],
            "state_details": [
                {"id": s.id, "name": s.name, "tools": s.tools, "entry_actions": s.entry_actions}
                for s in self.states
            ],
            "transitions": [
                {
                    "from": t.from_state,
                    "to": t.to_state,
                    "condition": t.condition,
                    "priority": t.priority,
                }
                for t in self.transitions
            ],
            "initial": self.initial_state,
            "terminal": self.terminal_states,
        }


def _extract_ground_truth(
    messages: list[dict[str, Any]],
    workflow: "WorkflowGraph",
) -> dict[str, Any]:
    """Extract ground-truth labels from annotated messages.

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
        annotations = msg.get("annotations", {})

        transition = annotations.get("state_transition", {})
        if transition.get("from") and transition.get("to"):
            state_sequence.append({"from": transition["from"], "to": transition["to"]})

        turn_tools = annotations.get("tool_calls", [])
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
            "tool_schemas": self.tool_schemas,
            "messages": self.messages,
            "user_behavior": self.user_behavior,
            "language": self.language,
            "ground_truth": self.ground_truth,
        }


def _select_user_behavior(rng: random.Random) -> str:
    """Select a user behavior type based on the configured distribution."""
    behaviors = list(USER_BEHAVIOR_DISTRIBUTION.keys())
    weights = list(USER_BEHAVIOR_DISTRIBUTION.values())
    return rng.choices(behaviors, weights=weights, k=1)[0]


def _select_domain(
    rng: random.Random, domain: str | None = None
) -> tuple[str, DomainSpec]:
    """Select a domain from the registry.

    If ``domain`` is provided and exists in the registry, use it.
    If ``domain`` matches a legacy name, map it to the closest registry entry.
    Otherwise, pick a random domain.
    """
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

    key = rng.choice(ALL_DOMAIN_NAMES)
    return key, DOMAIN_REGISTRY[key]


def _generate_tool_schemas(
    spec: ComplexitySpec,
    domain_spec: DomainSpec,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Select tool schemas from the domain registry for the given complexity.

    Picks ``spec.num_tools`` tools from the domain's tool list. If the domain
    has fewer tools than needed, supplements with cross-cutting tools.
    """
    available = list(domain_spec.tools)
    if len(available) < spec.num_tools:
        available.extend(CROSS_CUTTING_TOOLS)
    rng.shuffle(available)
    return available[: spec.num_tools]


def _get_tool_templates_for_domain(domain: str) -> list[dict[str, Any]]:
    """Return tool schema templates for a given domain."""
    domain_tools: dict[str, list[dict[str, Any]]] = {
        "faq_lookup": [
            {
                "type": "function",
                "function": {
                    "name": "search_faq",
                    "description": "Search the FAQ database for answers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "category": {
                                "type": "string",
                                "enum": ["billing", "technical", "general"],
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ],
        "order_status_cancel": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "description": "Look up order details by order ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string", "description": "The order ID"},
                        },
                        "required": ["order_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_order",
                    "description": "Cancel an active order",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["order_id", "reason"],
                    },
                },
            },
        ],
        "booking_payment": [
            {
                "type": "function",
                "function": {
                    "name": "search_availability",
                    "description": "Search for available booking slots",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date": {"type": "string", "format": "date"},
                            "service_type": {"type": "string"},
                            "location": {"type": "string"},
                        },
                        "required": ["date", "service_type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_booking",
                    "description": "Create a new booking",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "slot_id": {"type": "string"},
                            "customer_name": {"type": "string"},
                            "customer_email": {"type": "string", "format": "email"},
                        },
                        "required": ["slot_id", "customer_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "process_payment",
                    "description": "Process payment for a booking",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "booking_id": {"type": "string"},
                            "amount": {"type": "number"},
                            "payment_method": {
                                "type": "string",
                                "enum": ["credit_card", "debit_card", "bank_transfer"],
                            },
                        },
                        "required": ["booking_id", "amount", "payment_method"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "send_confirmation",
                    "description": "Send booking confirmation to customer",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "booking_id": {"type": "string"},
                            "channel": {
                                "type": "string",
                                "enum": ["email", "sms", "both"],
                            },
                        },
                        "required": ["booking_id"],
                    },
                },
            },
        ],
        "it_troubleshoot_escalation": [
            {
                "type": "function",
                "function": {
                    "name": "check_system_status",
                    "description": "Check status of an IT system or service",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system_name": {"type": "string"},
                            "check_type": {
                                "type": "string",
                                "enum": ["health", "connectivity", "performance"],
                            },
                        },
                        "required": ["system_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_diagnostic",
                    "description": "Run a diagnostic test on a system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system_name": {"type": "string"},
                            "diagnostic_type": {"type": "string"},
                            "verbose": {"type": "boolean", "default": False},
                        },
                        "required": ["system_name", "diagnostic_type"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "apply_fix",
                    "description": "Apply a known fix to a system issue",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "system_name": {"type": "string"},
                            "fix_id": {"type": "string"},
                            "force": {"type": "boolean", "default": False},
                        },
                        "required": ["system_name", "fix_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_ticket",
                    "description": "Create an escalation ticket",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "priority": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"],
                            },
                            "assignee_group": {"type": "string"},
                        },
                        "required": ["title", "priority"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "restart_service",
                    "description": "Restart a system service",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_name": {"type": "string"},
                            "graceful": {"type": "boolean", "default": True},
                        },
                        "required": ["service_name"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "notify_team",
                    "description": "Send notification to an operations team",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "team": {"type": "string"},
                            "message": {"type": "string"},
                            "urgency": {
                                "type": "string",
                                "enum": ["info", "warning", "critical"],
                            },
                        },
                        "required": ["team", "message"],
                    },
                },
            },
        ],
        "multi_dept_workflow": [
            {
                "type": "function",
                "function": {
                    "name": "route_to_department",
                    "description": "Route request to appropriate department",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "department": {"type": "string"},
                            "request_summary": {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                        },
                        "required": ["department", "request_summary"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "lookup_customer",
                    "description": "Look up customer information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                            "lookup_type": {
                                "type": "string",
                                "enum": ["full", "summary", "contact"],
                            },
                        },
                        "required": ["customer_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "check_policy",
                    "description": "Check applicable policies for a request",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "policy_area": {"type": "string"},
                            "customer_tier": {"type": "string"},
                            "request_type": {"type": "string"},
                        },
                        "required": ["policy_area"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "process_refund",
                    "description": "Process a customer refund",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                            "amount": {"type": "number"},
                            "reason": {"type": "string"},
                            "refund_method": {"type": "string"},
                        },
                        "required": ["order_id", "amount", "reason"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "schedule_callback",
                    "description": "Schedule a callback from a specialist",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                            "department": {"type": "string"},
                            "preferred_time": {"type": "string"},
                            "notes": {"type": "string"},
                        },
                        "required": ["customer_id", "department"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "update_account",
                    "description": "Update customer account information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                            "field": {"type": "string"},
                            "new_value": {"type": "string"},
                            "verification_code": {"type": "string"},
                        },
                        "required": ["customer_id", "field", "new_value"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "generate_report",
                    "description": "Generate a summary report of customer interactions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "customer_id": {"type": "string"},
                            "report_type": {
                                "type": "string",
                                "enum": ["interaction", "billing", "service"],
                            },
                            "date_range_days": {"type": "integer", "default": 30},
                        },
                        "required": ["customer_id", "report_type"],
                    },
                },
            },
        ],
    }
    return domain_tools.get(domain, [])


def _generate_workflow_graph(
    spec: ComplexitySpec,
    rng: random.Random,
    domain_spec: DomainSpec | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
) -> WorkflowGraph:
    """Generate a random workflow graph conforming to the complexity spec."""
    num_states = rng.randint(*spec.num_states)

    if tool_schemas is None:
        tool_schemas = _get_tool_templates_for_domain(spec.domain)
    tool_names = [t["function"]["name"] for t in tool_schemas][: spec.num_tools]

    # Use domain state templates if available, otherwise fall back to generic
    templates = list(domain_spec.state_templates) if domain_spec else []

    states: list[WorkflowState] = []
    for i in range(num_states):
        state_id = f"S{i + 1}"
        if i == 0:
            name = templates[0] if templates else "GREETING"
        elif i == num_states - 1:
            name = templates[-1] if templates else "TERMINAL"
        else:
            # Pick from middle templates, cycling if needed
            middle_templates = templates[1:-1] if len(templates) > 2 else [f"STATE_{i + 1}"]
            name = middle_templates[(i - 1) % len(middle_templates)] if middle_templates else f"STATE_{i + 1}"

        # Assign tools to non-terminal states
        state_tools: list[str] = []
        if i > 0 and i < num_states - 1 and tool_names:
            n_tools = rng.randint(0, min(2, len(tool_names)))
            state_tools = rng.sample(tool_names, n_tools) if n_tools > 0 else []

        states.append(WorkflowState(id=state_id, name=name, tools=state_tools))

    # Generate transitions ensuring connectivity
    transitions: list[WorkflowTransition] = []
    for i in range(num_states - 1):
        transitions.append(
            WorkflowTransition(
                from_state=states[i].id,
                to_state=states[i + 1].id,
                condition=f"proceed_from_{states[i].name.lower()}",
            )
        )

    # Add branching transitions
    max_branching = rng.randint(*spec.branching_factor)
    for _ in range(min(max_branching, num_states - 2)):
        src = rng.randint(0, num_states - 3)
        dst = rng.randint(src + 2, num_states - 1)
        transitions.append(
            WorkflowTransition(
                from_state=states[src].id,
                to_state=states[dst].id,
                condition=f"branch_{states[src].id}_to_{states[dst].id}",
                priority=1,
            )
        )

    terminal_states = [states[-1].id]
    # Possibly add an alternative terminal
    if num_states > 4 and rng.random() < 0.3:
        alt_terminal = states[-2].id
        terminal_states.append(alt_terminal)

    return WorkflowGraph(
        states=states,
        transitions=transitions,
        initial_state=states[0].id,
        terminal_states=terminal_states,
    )


def _generate_placeholder_conversation(
    workflow: WorkflowGraph,
    tool_schemas: list[dict[str, Any]],
    behavior: str,
    spec: ComplexitySpec,
    rng: random.Random,
    domain_spec: DomainSpec | None = None,
    language: str = "en",
) -> list[dict[str, Any]]:
    """Generate a placeholder conversation following the workflow graph.

    In production, this would call the teacher model (GPT-4o / Claude).
    For now, generates structurally valid placeholder conversations.
    """
    messages: list[dict[str, Any]] = []
    domain_name = domain_spec.name if domain_spec else spec.domain
    domain_intents = list(domain_spec.intents) if domain_spec else [spec.domain]

    # System message with workflow prompt
    system_content = (
        f"You are a customer service agent handling {domain_name} workflows.\n"
        f"Follow this state machine:\n"
        f"Initial state: {workflow.initial_state}\n"
        f"Terminal states: {', '.join(workflow.terminal_states)}\n"
        f"Available tools: {json.dumps([t['function']['name'] for t in tool_schemas])}\n"
    )
    messages.append({"role": "system", "content": system_content})

    # Placeholder text templates per language
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
    }

    # Generate turns following the workflow
    current_state_idx = 0
    for turn_idx in range(min(len(workflow.states) * 2, 20)):
        if current_state_idx >= len(workflow.states):
            break

        current_state = workflow.states[current_state_idx]

        # User message — use domain intents for realistic context
        intent = rng.choice(domain_intents) if domain_intents else spec.domain
        intent_text = intent.replace("_", " ")
        templates = _user_templates.get(language or "en", _user_templates["en"])
        tmpl = templates.get(behavior, templates["cooperative"])
        user_msg = tmpl.format(t=turn_idx + 1, intent=intent_text, state=current_state.name)

        messages.append({"role": "user", "content": user_msg})

        # Assistant response with state annotation
        next_state_idx = min(current_state_idx + 1, len(workflow.states) - 1)
        next_state = workflow.states[next_state_idx]

        assistant_content = f"[STATE: {current_state.name} → {next_state.name}]"
        annotations: dict[str, Any] = {
            "state_transition": {"from": current_state.name, "to": next_state.name},
        }

        # Add tool call if the state has tools
        if current_state.tools:
            tool_name = current_state.tools[0]
            tool_call = {"name": tool_name, "arguments": {"placeholder": "value"}}
            assistant_content += f"\n<tool_call>{json.dumps(tool_call)}</tool_call>"
            annotations["tool_calls"] = [tool_call]

            messages.append(
                {"role": "assistant", "content": assistant_content, "annotations": annotations}
            )

            # Tool response (with error rate)
            if rng.random() < TOOL_ERROR_RATE:
                tool_response = json.dumps({"error": "Service temporarily unavailable"})
            else:
                tool_response = json.dumps({"status": "success", "data": {"result": "ok"}})

            messages.append({"role": "tool", "content": tool_response})
        else:
            assistant_content += f"\nHandling {spec.domain} in state {current_state.name}."
            messages.append(
                {"role": "assistant", "content": assistant_content, "annotations": annotations}
            )

        current_state_idx = next_state_idx

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
}


def _build_teacher_prompt(
    workflow: WorkflowGraph,
    tool_schemas: list[dict[str, Any]],
    behavior: str,
    spec: ComplexitySpec,
    domain_spec: DomainSpec | None,
    language: str = "en",
) -> str:
    domain_name = domain_spec.name if domain_spec else spec.domain
    tool_names = [t["function"]["name"] for t in tool_schemas]
    lang_instruction = _LANGUAGE_INSTRUCTIONS.get(language, _LANGUAGE_INSTRUCTIONS["en"])
    return (
        f"Domain: {domain_name}\n"
        f"Complexity level: {spec.level} "
        f"({spec.num_states[0]}–{spec.num_states[1]} states, chain_depth={spec.chain_depth})\n"
        f"User behavior: {behavior}\n"
        f"{lang_instruction}\n\n"
        f"Workflow graph:\n{json.dumps(workflow.to_dict(), indent=2)}\n\n"
        f"Available tools ({len(tool_schemas)}):\n{json.dumps(tool_schemas, indent=2)}\n\n"
        f"Tool names in scope: {tool_names}\n\n"
        "Generate the conversation now."
    )


def _parse_messages_response(raw: str) -> list[dict[str, Any]]:
    """Parse a JSON response into a list of messages."""
    data = json.loads(raw)
    if isinstance(data, dict) and "messages" in data:
        return data["messages"]
    return data  # assume it's already a list


def _generate_teacher_conversation(
    workflow: WorkflowGraph,
    tool_schemas: list[dict[str, Any]],
    behavior: str,
    spec: ComplexitySpec,
    rng: random.Random,
    domain_spec: DomainSpec | None,
    teacher_model: str,
    language: str = "en",
) -> list[dict[str, Any]]:
    """Call a teacher model API to generate a conversation.

    Falls back to placeholder generation if the API call fails.
    """
    user_prompt = _build_teacher_prompt(workflow, tool_schemas, behavior, spec, domain_spec, language)
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
            workflow, tool_schemas, behavior, spec, rng, domain_spec, language
        )


def generate_workflow_dataset(
    complexity_level: Literal["L1", "L2", "L3", "L4", "L5"],
    num_samples: int = 200,
    teacher_model: str | None = None,
    output_dir: Path = Path("data/output/task_a"),
    seed: int = 42,
    domain: str | None = None,
    language: Literal["en", "th"] | None = None,
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
            or ``None`` to mix 50 % English / 50 % Thai per sample.

    Returns:
        DatasetMetadata with paths to generated JSONL files and statistics.
    """
    level = ComplexityLevel(complexity_level)
    spec = COMPLEXITY_SPECS[level]
    rng = random.Random(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    lang_tag = language or "mixed"
    model_tag = teacher_model.replace("/", "-").replace(".", "-") if teacher_model else "placeholder"
    output_file = output_dir / f"{complexity_level.lower()}_conversations_{lang_tag}_{model_tag}_{timestamp}.jsonl"

    logger.info(
        "generating_workflow_dataset",
        level=complexity_level,
        num_samples=num_samples,
        domain=domain or "random (17 domains)",
        teacher_model=teacher_model or "placeholder",
        language=language or "mixed (en/th)",
    )

    samples: list[ConversationSample] = []
    behavior_counts: dict[str, int] = {b: 0 for b in USER_BEHAVIOR_DISTRIBUTION}
    domain_counts: dict[str, int] = {}
    language_counts: dict[str, int] = {}
    tool_error_count = 0
    total_tool_calls = 0

    for i in range(num_samples):
        behavior = _select_user_behavior(rng)
        behavior_counts[behavior] += 1

        # Select language (fixed or 50/50 mix per sample)
        sample_language = language if language is not None else rng.choice(["en", "th"])
        language_counts[sample_language] = language_counts.get(sample_language, 0) + 1

        # Select domain (fixed or random per sample)
        domain_key, domain_spec = _select_domain(rng, domain)
        domain_counts[domain_key] = domain_counts.get(domain_key, 0) + 1

        tool_schemas = _generate_tool_schemas(spec, domain_spec, rng)
        workflow = _generate_workflow_graph(spec, rng, domain_spec, tool_schemas)

        if teacher_model:
            messages = _generate_teacher_conversation(
                workflow, tool_schemas, behavior, spec, rng, domain_spec, teacher_model, sample_language
            )
        else:
            messages = _generate_placeholder_conversation(
                workflow, tool_schemas, behavior, spec, rng, domain_spec, sample_language
            )

        # Count tool calls and errors
        for msg in messages:
            if msg["role"] == "tool":
                total_tool_calls += 1
                content = json.loads(msg["content"])
                if "error" in content:
                    tool_error_count += 1

        sample = ConversationSample(
            conversation_id=f"{complexity_level}_{i + 1:03d}",
            complexity_level=complexity_level,
            domain=domain_key,
            num_states=len(workflow.states),
            num_tools=spec.num_tools,
            chain_depth=spec.chain_depth,
            workflow_graph=workflow.to_dict(),
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
            f.write(json.dumps(sample.to_dict()) + "\n")

    stats = {
        "behavior_distribution": behavior_counts,
        "domain_distribution": domain_counts,
        "language_distribution": language_counts,
        "tool_error_rate": tool_error_count / max(total_tool_calls, 1),
        "total_tool_calls": total_tool_calls,
        "avg_states": sum(s.num_states for s in samples) / len(samples),
        "num_domains": len(domain_counts),
    }

    logger.info("dataset_generated", output_file=str(output_file), **stats)

    return DatasetMetadata(
        output_dir=output_dir,
        complexity_level=complexity_level,
        num_samples=num_samples,
        output_files=[output_file],
        stats=stats,
    )
