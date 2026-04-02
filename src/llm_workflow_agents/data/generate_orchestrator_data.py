"""Generate orchestrator routing training data for Cat A multi-agent fine-tuning.

Produces multi-turn conversations where the orchestrator must:
  1. Classify user intent → select routing target (Cat B / Cat C / self-handle)
  2. Delegate to the correct specialist with a well-formed request
  3. Synthesize the specialist's response and continue the workflow

The generated data teaches Cat A to work as an orchestrator in Phase 4
multi-agent deployment, complementing the Task A solo workflow data.
"""

from __future__ import annotations

import json
import random
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import structlog

from llm_workflow_agents.data.domain_registry import (
    ALL_DOMAIN_NAMES,
    CROSS_CUTTING_INTENTS,
    DOMAIN_REGISTRY,
    DomainSpec,
)

logger = structlog.get_logger(__name__)

# Routing targets
ROUTING_TARGETS = ("tool_execution", "graph_extraction", "self_handle")

# Routing distribution (matches eval benchmark default)
DEFAULT_ROUTING_DISTRIBUTION: dict[str, float] = {
    "tool_execution": 0.60,
    "graph_extraction": 0.20,
    "self_handle": 0.20,
}

# Specialist response templates
_SPECIALIST_TOOL_RESPONSES = [
    '{{"status": "success", "result": {{"id": "{ref_id}", "message": "Completed {action}"}}}}',
    '{{"status": "success", "data": {{"confirmation": "CONF-{ref_id}", "action": "{action}"}}}}',
    '{{"status": "error", "error": "Service temporarily unavailable for {action}"}}',
    '{{"status": "success", "result": {{"items": [{{"id": "{ref_id}", "status": "processed"}}]}}}}',
]

_SPECIALIST_GRAPH_RESPONSES = [
    '{{"nodes": [{{"id": "S1", "name": "{state1}"}}, {{"id": "S2", "name": "{state2}"}}, {{"id": "S3", "name": "{state3}"}}], "edges": [{{"from": "S1", "to": "S2", "condition": "proceed"}}, {{"from": "S2", "to": "S3", "condition": "complete"}}], "initial_state": "S1", "terminal_states": ["S3"]}}',
]

# User message templates by routing scenario
_USER_TEMPLATES: dict[str, list[str]] = {
    "tool_execution": [
        "I need to {action} for my {entity}",
        "Can you help me {action}? My {entity} is {value}",
        "Please {action}. Here are the details: {entity} = {value}",
        "I want to {action} on my account",
        "I'd like to request {action} for {entity}",
        "Hi, I'm calling about {action}. My {entity} is {value}",
        "I have an issue with {action}, can you process it?",
    ],
    "graph_extraction": [
        "Can you show me the workflow for {action}?",
        "I need a visual diagram of the {domain_name} process",
        "Generate a flowchart for how {action} works",
        "What does the {action} workflow look like step by step?",
        "Map out the process for {action} in {domain_name}",
        "I need to understand the {action} flow — can you diagram it?",
    ],
    "self_handle": [
        "Hello, I need some help",
        "Hi there",
        "Can you repeat what you just said?",
        "I didn't understand that, can you explain differently?",
        "Actually, never mind my previous question",
        "Thank you, that's all I needed",
        "What services do you offer?",
        "Can I speak to a human agent?",
    ],
}

# Orchestrator reasoning templates — what the orchestrator "thinks" before routing
_ROUTING_REASONING: dict[str, list[str]] = {
    "tool_execution": [
        "This request requires executing {action}. I'll delegate to the tool execution specialist.",
        "The user needs {action} — routing to specialist for tool execution.",
        "This is a {domain_name} request requiring {action}. Delegating to Cat B specialist.",
    ],
    "graph_extraction": [
        "The user wants a workflow visualization. Routing to the graph extraction specialist.",
        "This is a diagram/flowchart request — delegating to Cat C specialist.",
        "Visualization request for {domain_name}. Routing to graph extraction.",
    ],
    "self_handle": [
        "This is a general inquiry I can handle directly.",
        "No specialist needed — I'll respond directly.",
        "Simple conversational turn — handling without delegation.",
    ],
}


@dataclass
class OrchestratorSample:
    """A single orchestrator training conversation."""

    conversation_id: str
    domain: str
    routing_target: str
    intent: str
    messages: list[dict[str, Any]]
    num_turns: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "domain": self.domain,
            "routing_target": self.routing_target,
            "intent": self.intent,
            "messages": self.messages,
            "num_turns": self.num_turns,
        }


@dataclass
class OrchestratorDatasetMetadata:
    """Metadata for generated orchestrator dataset."""

    output_dir: Path
    num_samples: int
    output_files: list[Path] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def _generate_tool_execution_conversation(
    domain_key: str,
    domain_spec: DomainSpec,
    rng: random.Random,
    conv_id: str,
) -> OrchestratorSample:
    """Generate a multi-turn orchestrator → Cat B specialist conversation."""
    intent = rng.choice(list(domain_spec.intents))
    action = intent.replace("_", " ")
    entity = rng.choice(list(domain_spec.entity_slots)) if domain_spec.entity_slots else "account"
    value = f"{rng.randint(100000, 999999)}"
    ref_id = f"REF-{rng.randint(1000, 9999)}"

    # Select a tool the specialist would call
    tool = rng.choice(list(domain_spec.tools))
    tool_name = tool["function"]["name"]
    tool_params = list(tool["function"]["parameters"].get("required", []))

    messages: list[dict[str, Any]] = []

    # System prompt
    messages.append({
        "role": "system",
        "content": (
            "You are a workflow orchestrator managing a team of specialists.\n"
            "When a user request requires tool execution, delegate to the tool specialist.\n"
            "When a user request requires workflow visualization, delegate to the graph specialist.\n"
            "For simple inquiries (greetings, clarifications), handle directly.\n\n"
            "Format routing decisions as:\n"
            "[ROUTE: <target>] <reasoning>\n"
            "Where target is: tool_execution | graph_extraction | self_handle\n\n"
            "After receiving specialist results, synthesize and respond to the user."
        ),
    })

    # Turn 1: User request
    template = rng.choice(_USER_TEMPLATES["tool_execution"])
    user_msg = template.format(action=action, entity=entity, value=value, domain_name=domain_spec.name)
    messages.append({"role": "user", "content": user_msg})

    # Turn 2: Orchestrator routes to tool specialist
    reasoning = rng.choice(_ROUTING_REASONING["tool_execution"]).format(
        action=action, domain_name=domain_spec.name
    )
    delegate_request = json.dumps({
        "tool": tool_name,
        "arguments": {p: f"<{p}_value>" for p in tool_params[:3]},
    })
    messages.append({
        "role": "assistant",
        "content": (
            f"[ROUTE: tool_execution] {reasoning}\n\n"
            f"<delegate_to_specialist>\n{delegate_request}\n</delegate_to_specialist>"
        ),
        "annotations": {
            "routing_target": "tool_execution",
            "specialist": "cat_b",
            "tool_name": tool_name,
        },
    })

    # Turn 3: Specialist response
    resp_template = rng.choice(_SPECIALIST_TOOL_RESPONSES)
    specialist_response = resp_template.format(ref_id=ref_id, action=action)
    messages.append({
        "role": "tool",
        "content": f"<specialist_response>{specialist_response}</specialist_response>",
    })

    # Turn 4: Orchestrator synthesizes
    is_error = '"error"' in specialist_response
    if is_error:
        synthesis = (
            f"I apologize, but I encountered an issue while processing your {action} request. "
            f"Let me try an alternative approach or escalate this to a senior specialist."
        )
    else:
        synthesis = (
            f"I've completed your {action} request. "
            f"Your reference number is {ref_id}. Is there anything else I can help with?"
        )
    messages.append({"role": "assistant", "content": synthesis})

    # Optional Turn 5-6: Follow-up
    if rng.random() < 0.4:
        messages.append({"role": "user", "content": "Yes, can you also check the status?"})
        messages.append({
            "role": "assistant",
            "content": (
                f"[ROUTE: tool_execution] Follow-up request for status check.\n\n"
                f'<delegate_to_specialist>\n{{"tool": "check_status", "arguments": {{"ref_id": "{ref_id}"}}}}\n</delegate_to_specialist>'
            ),
            "annotations": {"routing_target": "tool_execution", "specialist": "cat_b"},
        })

    return OrchestratorSample(
        conversation_id=conv_id,
        domain=domain_key,
        routing_target="tool_execution",
        intent=intent,
        messages=messages,
        num_turns=len([m for m in messages if m["role"] in ("user", "assistant")]),
    )


def _generate_graph_extraction_conversation(
    domain_key: str,
    domain_spec: DomainSpec,
    rng: random.Random,
    conv_id: str,
) -> OrchestratorSample:
    """Generate an orchestrator → Cat C specialist conversation."""
    intent = rng.choice(list(domain_spec.intents))
    action = intent.replace("_", " ")

    # Pick state templates for the graph response
    states = list(domain_spec.state_templates)
    s1 = states[0] if states else "START"
    s2 = states[len(states) // 2] if len(states) > 1 else "PROCESS"
    s3 = states[-1] if len(states) > 2 else "END"

    messages: list[dict[str, Any]] = []

    # System prompt (same as tool execution)
    messages.append({
        "role": "system",
        "content": (
            "You are a workflow orchestrator managing a team of specialists.\n"
            "When a user request requires tool execution, delegate to the tool specialist.\n"
            "When a user request requires workflow visualization, delegate to the graph specialist.\n"
            "For simple inquiries (greetings, clarifications), handle directly.\n\n"
            "Format routing decisions as:\n"
            "[ROUTE: <target>] <reasoning>\n"
            "Where target is: tool_execution | graph_extraction | self_handle\n\n"
            "After receiving specialist results, synthesize and respond to the user."
        ),
    })

    # User requests visualization
    template = rng.choice(_USER_TEMPLATES["graph_extraction"])
    user_msg = template.format(action=action, domain_name=domain_spec.name)
    messages.append({"role": "user", "content": user_msg})

    # Orchestrator routes to graph specialist
    reasoning = rng.choice(_ROUTING_REASONING["graph_extraction"]).format(
        domain_name=domain_spec.name
    )
    messages.append({
        "role": "assistant",
        "content": (
            f"[ROUTE: graph_extraction] {reasoning}\n\n"
            f"<delegate_to_specialist>\n"
            f'{{"prompt": "Generate workflow graph for {action} in {domain_spec.name}"}}\n'
            f"</delegate_to_specialist>"
        ),
        "annotations": {"routing_target": "graph_extraction", "specialist": "cat_c"},
    })

    # Specialist returns graph
    graph_response = rng.choice(_SPECIALIST_GRAPH_RESPONSES).format(
        state1=s1, state2=s2, state3=s3
    )
    messages.append({
        "role": "tool",
        "content": f"<specialist_response>{graph_response}</specialist_response>",
    })

    # Orchestrator synthesizes
    messages.append({
        "role": "assistant",
        "content": (
            f"Here's the workflow for {action}:\n\n"
            f"The process starts at **{s1}**, moves through **{s2}**, "
            f"and completes at **{s3}**.\n\n"
            f"Would you like me to explain any step in detail?"
        ),
    })

    return OrchestratorSample(
        conversation_id=conv_id,
        domain=domain_key,
        routing_target="graph_extraction",
        intent="workflow_visualization",
        messages=messages,
        num_turns=len([m for m in messages if m["role"] in ("user", "assistant")]),
    )


def _generate_self_handle_conversation(
    rng: random.Random,
    conv_id: str,
) -> OrchestratorSample:
    """Generate a conversation the orchestrator handles directly."""
    messages: list[dict[str, Any]] = []

    # System prompt
    messages.append({
        "role": "system",
        "content": (
            "You are a workflow orchestrator managing a team of specialists.\n"
            "When a user request requires tool execution, delegate to the tool specialist.\n"
            "When a user request requires workflow visualization, delegate to the graph specialist.\n"
            "For simple inquiries (greetings, clarifications), handle directly.\n\n"
            "Format routing decisions as:\n"
            "[ROUTE: <target>] <reasoning>\n"
            "Where target is: tool_execution | graph_extraction | self_handle"
        ),
    })

    # Pick a cross-cutting intent
    intent = rng.choice(list(CROSS_CUTTING_INTENTS))
    user_msg = rng.choice(_USER_TEMPLATES["self_handle"])
    messages.append({"role": "user", "content": user_msg})

    reasoning = rng.choice(_ROUTING_REASONING["self_handle"])
    if intent in ("greeting", "closing"):
        response = "Hello! I'm your workflow assistant. How can I help you today?"
    elif intent == "human_handoff":
        response = (
            "[ROUTE: self_handle] The user wants to speak to a human agent.\n\n"
            "I understand you'd like to speak with a human agent. "
            "Let me transfer you now. Please hold for a moment."
        )
    elif intent in ("clarification", "repeat_rephrase"):
        response = (
            "[ROUTE: self_handle] Clarification request — no specialist needed.\n\n"
            "Of course! Could you tell me more about what you need help with? "
            "I can assist with account management, billing, technical support, and more."
        )
    else:
        response = (
            f"[ROUTE: self_handle] {reasoning}\n\n"
            "I'm here to help! Please let me know what you need and I'll "
            "connect you with the right specialist or handle it directly."
        )

    messages.append({
        "role": "assistant",
        "content": response,
        "annotations": {"routing_target": "self_handle", "specialist": None},
    })

    return OrchestratorSample(
        conversation_id=conv_id,
        domain="general",
        routing_target="self_handle",
        intent=intent,
        messages=messages,
        num_turns=2,
    )


def generate_orchestrator_dataset(
    num_samples: int = 1000,
    output_dir: Path = Path("data/output/task_orchestrator"),
    seed: int = 42,
    routing_distribution: dict[str, float] | None = None,
    teacher_model: str = "gpt-4o",
) -> OrchestratorDatasetMetadata:
    """Generate orchestrator routing training data.

    Produces multi-turn conversations where the orchestrator classifies
    user intent, delegates to specialists (Cat B/C), and synthesizes
    responses. Data is split into train/val/test (85/10/5).

    Args:
        num_samples: Total number of conversations to generate.
        output_dir: Output directory for JSONL files.
        seed: Random seed.
        routing_distribution: Target distribution for routing classes.
            Defaults to 60% tool_execution, 20% graph_extraction, 20% self_handle.
        teacher_model: Teacher model for generation (future use).

    Returns:
        OrchestratorDatasetMetadata with output paths and stats.
    """
    if routing_distribution is None:
        routing_distribution = DEFAULT_ROUTING_DISTRIBUTION

    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "generating_orchestrator_dataset",
        num_samples=num_samples,
        routing_distribution=routing_distribution,
        teacher_model=teacher_model,
    )

    samples: list[OrchestratorSample] = []
    routing_counts: dict[str, int] = {t: 0 for t in ROUTING_TARGETS}
    domain_counts: dict[str, int] = {}

    for i in range(num_samples):
        conv_id = f"orch_{i + 1:04d}"

        # Select routing target
        target = rng.choices(
            list(routing_distribution.keys()),
            weights=list(routing_distribution.values()),
            k=1,
        )[0]
        routing_counts[target] += 1

        if target == "tool_execution":
            domain_key = rng.choice(ALL_DOMAIN_NAMES)
            domain_spec = DOMAIN_REGISTRY[domain_key]
            sample = _generate_tool_execution_conversation(
                domain_key, domain_spec, rng, conv_id
            )
        elif target == "graph_extraction":
            domain_key = rng.choice(ALL_DOMAIN_NAMES)
            domain_spec = DOMAIN_REGISTRY[domain_key]
            sample = _generate_graph_extraction_conversation(
                domain_key, domain_spec, rng, conv_id
            )
        else:
            sample = _generate_self_handle_conversation(rng, conv_id)
            domain_key = "general"

        samples.append(sample)
        domain_counts[domain_key] = domain_counts.get(domain_key, 0) + 1

    # Split: 85% train, 10% val, 5% test
    rng.shuffle(samples)
    n_train = int(num_samples * 0.85)
    n_val = int(num_samples * 0.10)

    train_samples = samples[:n_train]
    val_samples = samples[n_train : n_train + n_val]
    test_samples = samples[n_train + n_val :]

    # Write JSONL files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_files: list[Path] = []
    for split_name, split_samples in [
        ("train", train_samples),
        ("val", val_samples),
        ("test", test_samples),
    ]:
        path = output_dir / f"{split_name}_{timestamp}.jsonl"
        with open(path, "w") as f:
            for sample in split_samples:
                f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
        output_files.append(path)

    stats = {
        "routing_distribution": routing_counts,
        "domain_distribution": domain_counts,
        "num_domains": len(domain_counts),
        "splits": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "avg_turns": sum(s.num_turns for s in samples) / len(samples) if samples else 0,
    }

    logger.info("orchestrator_dataset_generated", **stats)

    return OrchestratorDatasetMetadata(
        output_dir=output_dir,
        num_samples=num_samples,
        output_files=output_files,
        stats=stats,
    )
