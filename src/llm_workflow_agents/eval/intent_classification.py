"""Intent classification benchmark for orchestrator routing evaluation.

Tests whether the Cat A orchestrator model correctly classifies user intent
to route to the appropriate specialist (Cat B for tool execution, Cat C for
graph extraction) across the 17-domain taxonomy.

Two levels of classification:
  1. Routing target: "tool_execution" (Cat B) vs "graph_extraction" (Cat C)
     vs "self_handle" (Cat A handles directly)
  2. Domain classification: which of the 17 domains the request belongs to

Supports both zero-shot prompting and few-shot with exemplars.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


# --- Routing Targets ---

ROUTING_TARGETS = ("tool_execution", "graph_extraction", "self_handle")


# --- Dataclasses ---


@dataclass
class IntentSample:
    """A single intent classification sample."""

    text: str
    domain: str
    intent: str
    routing_target: str  # "tool_execution", "graph_extraction", "self_handle"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentClassificationMetrics:
    """Results from intent classification evaluation."""

    # Routing-level metrics (3-class: tool_execution, graph_extraction, self_handle)
    routing_accuracy: float = 0.0
    routing_precision_per_class: dict[str, float] = field(default_factory=dict)
    routing_recall_per_class: dict[str, float] = field(default_factory=dict)
    routing_f1_per_class: dict[str, float] = field(default_factory=dict)
    routing_confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    # Domain-level metrics (17-class)
    domain_accuracy: float = 0.0
    domain_top3_accuracy: float = 0.0

    # Overall
    num_samples: int = 0
    num_correct_routing: int = 0
    num_correct_domain: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing_accuracy": self.routing_accuracy,
            "routing_precision_per_class": self.routing_precision_per_class,
            "routing_recall_per_class": self.routing_recall_per_class,
            "routing_f1_per_class": self.routing_f1_per_class,
            "routing_confusion": self.routing_confusion,
            "domain_accuracy": self.domain_accuracy,
            "domain_top3_accuracy": self.domain_top3_accuracy,
            "num_samples": self.num_samples,
        }


# --- Sample Generation ---

# Intent-to-routing mapping: most intents route to tool_execution (Cat B),
# graph-related intents route to graph_extraction (Cat C),
# simple informational intents are self-handled by Cat A.
_GRAPH_INTENTS = {
    "workflow_visualization", "process_mapping", "graph_extraction",
    "flow_diagram_request", "dependency_mapping",
}

_SELF_HANDLE_INTENTS = {
    "greeting", "clarification", "closing", "out_of_scope",
    "repeat_rephrase", "language_switch",
}

# User message templates per routing target
_TEMPLATES: dict[str, list[str]] = {
    "tool_execution": [
        "I need to {action} for {entity}",
        "Can you help me {action}?",
        "Please {action} on my account",
        "I want to {action}",
        "I'd like to request {action}",
    ],
    "graph_extraction": [
        "Can you show me the workflow diagram for {domain}?",
        "I need a visual map of the {domain} process",
        "Generate a flowchart for the {action} workflow",
        "What does the {domain} process look like as a graph?",
        "Map out the steps for {action}",
    ],
    "self_handle": [
        "Hello, I need some help",
        "Can you repeat that?",
        "I didn't understand, can you rephrase?",
        "Thank you, that's all I needed",
        "Actually, never mind",
    ],
}


def _classify_intent_to_routing(intent: str) -> str:
    """Map an intent string to its routing target."""
    if intent in _GRAPH_INTENTS:
        return "graph_extraction"
    if intent in _SELF_HANDLE_INTENTS:
        return "self_handle"
    return "tool_execution"


def generate_intent_samples(
    num_samples: int = 500,
    seed: int = 42,
    routing_distribution: dict[str, float] | None = None,
) -> list[IntentSample]:
    """Generate intent classification test samples from the domain registry.

    Args:
        num_samples: Total number of samples to generate.
        seed: Random seed.
        routing_distribution: Target distribution for routing classes.
            Defaults to {"tool_execution": 0.60, "graph_extraction": 0.20,
            "self_handle": 0.20}.

    Returns:
        List of IntentSample with text, domain, intent, and routing_target.
    """
    from llm_workflow_agents.data.domain_registry import (
        ALL_DOMAIN_NAMES,
        CROSS_CUTTING_INTENTS,
        DOMAIN_REGISTRY,
    )

    if routing_distribution is None:
        routing_distribution = {
            "tool_execution": 0.60,
            "graph_extraction": 0.20,
            "self_handle": 0.20,
        }

    rng = random.Random(seed)
    samples: list[IntentSample] = []

    for _ in range(num_samples):
        # Select routing target based on distribution
        target = rng.choices(
            list(routing_distribution.keys()),
            weights=list(routing_distribution.values()),
            k=1,
        )[0]

        if target == "tool_execution":
            domain_key = rng.choice(ALL_DOMAIN_NAMES)
            domain_spec = DOMAIN_REGISTRY[domain_key]
            intent = rng.choice(list(domain_spec.intents))
            action = intent.replace("_", " ")
            entity = rng.choice(list(domain_spec.entity_slots)) if domain_spec.entity_slots else "my request"
            template = rng.choice(_TEMPLATES["tool_execution"])
            text = template.format(action=action, entity=entity)

        elif target == "graph_extraction":
            domain_key = rng.choice(ALL_DOMAIN_NAMES)
            domain_spec = DOMAIN_REGISTRY[domain_key]
            intent = rng.choice(list(_GRAPH_INTENTS))
            action = rng.choice(list(domain_spec.intents)).replace("_", " ")
            template = rng.choice(_TEMPLATES["graph_extraction"])
            text = template.format(domain=domain_spec.name, action=action)

        else:  # self_handle
            domain_key = "general"
            intent = rng.choice(list(_SELF_HANDLE_INTENTS))
            text = rng.choice(_TEMPLATES["self_handle"])

        samples.append(IntentSample(
            text=text,
            domain=domain_key,
            intent=intent,
            routing_target=target,
        ))

    return samples


# --- Evaluation ---


def _compute_prf1(
    predictions: list[str],
    ground_truth: list[str],
    classes: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float], dict[str, dict[str, int]]]:
    """Compute per-class precision, recall, F1 and confusion matrix."""
    confusion: dict[str, dict[str, int]] = {c: {c2: 0 for c2 in classes} for c in classes}
    for pred, gt in zip(predictions, ground_truth):
        if gt in confusion and pred in confusion[gt]:
            confusion[gt][pred] += 1

    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}

    for cls in classes:
        tp = confusion[cls][cls]
        fp = sum(confusion[other][cls] for other in classes if other != cls)
        fn = sum(confusion[cls][other] for other in classes if other != cls)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision[cls] = p
        recall[cls] = r
        f1[cls] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return precision, recall, f1, confusion


def evaluate_intent_classification(
    predictions: list[dict[str, str]],
    ground_truth: list[IntentSample],
) -> IntentClassificationMetrics:
    """Evaluate orchestrator intent classification accuracy.

    Args:
        predictions: List of dicts with keys "routing_target" and optionally
            "domain" — the model's classification output per sample.
        ground_truth: List of IntentSample with correct labels.

    Returns:
        IntentClassificationMetrics with routing and domain accuracy.
    """
    if not predictions or not ground_truth:
        return IntentClassificationMetrics()

    n = min(len(predictions), len(ground_truth))

    # Routing evaluation
    pred_routing = [p.get("routing_target", "unknown") for p in predictions[:n]]
    gt_routing = [g.routing_target for g in ground_truth[:n]]

    correct_routing = sum(1 for p, g in zip(pred_routing, gt_routing) if p == g)
    routing_accuracy = correct_routing / n

    classes = list(ROUTING_TARGETS)
    precision, recall, f1, confusion = _compute_prf1(pred_routing, gt_routing, classes)

    # Domain evaluation
    pred_domains = [p.get("domain", "unknown") for p in predictions[:n]]
    gt_domains = [g.domain for g in ground_truth[:n]]

    correct_domain = sum(1 for p, g in zip(pred_domains, gt_domains) if p == g)
    domain_accuracy = correct_domain / n

    # Top-3 domain accuracy (if predictions include ranked list)
    top3_correct = 0
    for pred, gt_sample in zip(predictions[:n], ground_truth[:n]):
        top3 = pred.get("domain_top3", [pred.get("domain", "")])
        if gt_sample.domain in top3:
            top3_correct += 1
    domain_top3_accuracy = top3_correct / n

    metrics = IntentClassificationMetrics(
        routing_accuracy=routing_accuracy,
        routing_precision_per_class=precision,
        routing_recall_per_class=recall,
        routing_f1_per_class=f1,
        routing_confusion=confusion,
        domain_accuracy=domain_accuracy,
        domain_top3_accuracy=domain_top3_accuracy,
        num_samples=n,
        num_correct_routing=correct_routing,
        num_correct_domain=correct_domain,
    )

    logger.info(
        "intent_classification_eval_complete",
        routing_accuracy=routing_accuracy,
        domain_accuracy=domain_accuracy,
        num_samples=n,
    )

    return metrics


def run_intent_benchmark(
    model_name: str,
    num_samples: int = 500,
    seed: int = 42,
    base_url: str = "http://localhost:8000/v1",
    few_shot: int = 0,
) -> IntentClassificationMetrics:
    """Run full intent classification benchmark against a vLLM-served model.

    Generates test samples, prompts the model to classify each one,
    parses the response, and computes metrics.

    Args:
        model_name: Model name for the vLLM server.
        num_samples: Number of test samples.
        seed: Random seed.
        base_url: vLLM server URL.
        few_shot: Number of few-shot exemplars (0 for zero-shot).

    Returns:
        IntentClassificationMetrics with routing and domain accuracy.
    """
    from openai import OpenAI

    samples = generate_intent_samples(num_samples=num_samples, seed=seed)
    client = OpenAI(base_url=base_url, api_key="unused")

    # Build few-shot exemplars if needed
    exemplars = ""
    if few_shot > 0:
        rng = random.Random(seed + 1)  # Different seed for exemplars
        exemplar_samples = generate_intent_samples(num_samples=few_shot * 3, seed=seed + 1)
        rng.shuffle(exemplar_samples)
        for ex in exemplar_samples[:few_shot]:
            exemplars += (
                f'\nUser: "{ex.text}"\n'
                f'Classification: {{"routing_target": "{ex.routing_target}", "domain": "{ex.domain}"}}\n'
            )

    from llm_workflow_agents.data.domain_registry import ALL_DOMAIN_NAMES

    system_prompt = (
        "You are a routing classifier. Given a user message, classify it into:\n"
        "1. routing_target: one of 'tool_execution', 'graph_extraction', 'self_handle'\n"
        f"2. domain: one of {json.dumps(ALL_DOMAIN_NAMES + ['general'])}\n\n"
        "Respond with ONLY a JSON object: {\"routing_target\": \"...\", \"domain\": \"...\"}\n"
        f"{exemplars}"
    )

    predictions: list[dict[str, str]] = []
    for sample in samples:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": sample.text},
                ],
                max_tokens=100,
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            pred = json.loads(content)
            predictions.append(pred)
        except (json.JSONDecodeError, Exception):
            predictions.append({"routing_target": "unknown", "domain": "unknown"})

    return evaluate_intent_classification(predictions, samples)
