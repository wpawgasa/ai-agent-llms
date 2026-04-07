"""Combined workflow quality benchmark for Experiment A.

Composes state machine, tool-calling, and chain propagation metrics
into a single weighted workflow quality score.

CLI usage (invoked by scripts/run_exp_a.sh):
    python -m llm_workflow_agents.eval.agent_benchmark \\
        --model  Qwen/Qwen3-32B \\
        --output results/exp_a/Qwen_Qwen3-32B_auto.json \\
        --data   data/output/benchmark/task_a/ \\
        --endpoint http://localhost:8000

Data directory is expected to contain JSONL files produced by
scripts/generate_benchmark_data.sh (one file per complexity level).
Each sample must have the schema written by generate_workflows.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from llm_workflow_agents.eval.state_accuracy import StateMachineMetrics
from llm_workflow_agents.eval.tool_call_f1 import ToolCallMetrics
from llm_workflow_agents.eval.tool_chain_propagation import ChainPropagationMetrics

logger = structlog.get_logger(__name__)


@dataclass
class WorkflowQualityMetrics:
    """Combined workflow quality metrics."""

    full_workflow_success: float = 0.0  # Target: >=55%
    weighted_workflow_score: float = 0.0  # Target: >=0.75
    latency_per_turn_median_ms: float = 0.0  # Target: <=2000 (L1-L3), <=5000 (L4-L5)

    state_metrics: StateMachineMetrics = field(default_factory=StateMachineMetrics)
    tool_metrics: ToolCallMetrics = field(default_factory=ToolCallMetrics)
    chain_metrics: ChainPropagationMetrics = field(default_factory=ChainPropagationMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_workflow_success": self.full_workflow_success,
            "weighted_workflow_score": self.weighted_workflow_score,
            "latency_per_turn_median_ms": self.latency_per_turn_median_ms,
            "state_metrics": self.state_metrics.to_dict(),
            "tool_metrics": self.tool_metrics.to_dict(),
            "chain_metrics": self.chain_metrics.to_dict(),
        }


def compute_weighted_score(
    state: StateMachineMetrics,
    tool: ToolCallMetrics,
    completion: float,
) -> float:
    """Compute weighted workflow quality score.

    Formula: 0.4 * StateTransAcc + 0.4 * ToolCallF1 + 0.2 * TaskCompletion

    Args:
        state: State machine metrics.
        tool: Tool-calling metrics.
        completion: Task completion rate (0.0-1.0).

    Returns:
        Weighted score between 0.0 and 1.0.
    """
    return (
        0.4 * state.state_transition_accuracy
        + 0.4 * tool.tool_call_f1
        + 0.2 * completion
    )


def compute_full_workflow_success(
    state: StateMachineMetrics,
    tool: ToolCallMetrics,
    chain: ChainPropagationMetrics,
) -> float:
    """Compute full workflow success rate.

    A workflow is fully successful if:
      - Task was completed (reached terminal state)
      - All tool calls were correct (F1 >= 0.8)
      - Chain propagation was correct (accuracy >= 0.7)

    Returns approximate rate based on component metrics.
    """
    # Estimate: multiply independent success probabilities
    completion_factor = state.task_completion_rate
    tool_factor = min(tool.tool_call_f1 / 0.8, 1.0) if tool.tool_call_f1 > 0 else 0.0
    chain_factor = (
        min(chain.chain_propagation_accuracy / 0.7, 1.0)
        if chain.total_chains > 0
        else 1.0  # No chains present = not a failure
    )

    return completion_factor * tool_factor * chain_factor


def compute_latency_median(latencies_ms: list[float]) -> float:
    """Compute median latency from a list of per-turn latencies."""
    if not latencies_ms:
        return 0.0
    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)
    if n % 2 == 0:
        return (sorted_lat[n // 2 - 1] + sorted_lat[n // 2]) / 2
    return sorted_lat[n // 2]


def evaluate_workflow_quality(
    state_metrics: StateMachineMetrics,
    tool_metrics: ToolCallMetrics,
    chain_metrics: ChainPropagationMetrics,
    latencies_ms: list[float] | None = None,
) -> WorkflowQualityMetrics:
    """Compute combined workflow quality metrics.

    Args:
        state_metrics: State machine adherence results.
        tool_metrics: Tool-calling accuracy results.
        chain_metrics: Tool chain propagation results.
        latencies_ms: Optional per-turn latency measurements.

    Returns:
        WorkflowQualityMetrics with combined scores.
    """
    weighted = compute_weighted_score(
        state_metrics,
        tool_metrics,
        state_metrics.task_completion_rate,
    )

    full_success = compute_full_workflow_success(
        state_metrics,
        tool_metrics,
        chain_metrics,
    )

    median_latency = compute_latency_median(latencies_ms or [])

    metrics = WorkflowQualityMetrics(
        full_workflow_success=full_success,
        weighted_workflow_score=weighted,
        latency_per_turn_median_ms=median_latency,
        state_metrics=state_metrics,
        tool_metrics=tool_metrics,
        chain_metrics=chain_metrics,
    )

    logger.info(
        "workflow_quality_eval_complete",
        weighted_score=weighted,
        full_success=full_success,
        median_latency_ms=median_latency,
    )

    return metrics


# ---------------------------------------------------------------------------
# CLI entrypoint — invoked by scripts/run_exp_a.sh
# ---------------------------------------------------------------------------

def _load_samples(data_dir: "Path") -> list[dict[str, Any]]:
    """Load all JSONL samples from a benchmark data directory."""
    import json
    from pathlib import Path

    samples: list[dict[str, Any]] = []
    for path in sorted(Path(data_dir).glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
    return samples


def _call_vllm(
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> tuple[str, float]:
    """Call the vLLM OpenAI-compatible chat completions endpoint.

    Returns:
        (content, latency_ms)
    """
    import json
    import time
    import urllib.request

    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    latency_ms = (time.monotonic() - t0) * 1000.0

    content: str = body["choices"][0]["message"]["content"] or ""
    return content, latency_ms


def _replay_conversation(
    endpoint: str,
    model: str,
    sample: dict[str, Any],
    temperature: float = 0.0,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Replay a conversation, substituting model completions at assistant turns.

    Ground-truth tool responses are kept as-is so the conversation stays on
    track regardless of whether the model's tool call was correct — this
    isolates state-transition and tool-call quality from cascading failures.

    Returns:
        (predicted_messages, latencies_ms_per_assistant_turn)
    """
    predicted: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    context: list[dict[str, Any]] = []  # sliding context sent to the model

    for msg in sample.get("messages", []):
        role = msg["role"]

        if role == "system":
            context.append({"role": "system", "content": msg["content"]})
            predicted.append(msg)

        elif role == "user":
            context.append({"role": "user", "content": msg["content"]})
            predicted.append(msg)

        elif role == "assistant":
            content, latency = _call_vllm(endpoint, model, context, temperature)
            latencies_ms.append(latency)
            logger.debug(
                "model_response",
                turn=len(predicted),
                latency_ms=round(latency, 1),
                content=content[:2000],  # truncate to avoid flooding logs
            )
            pred_msg: dict[str, Any] = {"role": "assistant", "content": content}
            context.append(pred_msg)
            predicted.append(pred_msg)

        elif role == "tool":
            # Use ground-truth tool response to avoid cascading failures
            context.append({"role": "tool", "content": msg["content"]})
            predicted.append(msg)

    return predicted, latencies_ms


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    from llm_workflow_agents.eval.state_accuracy import (
        ConversationGroundTruth,
        ConversationPrediction,
        evaluate_state_machine,
    )
    from llm_workflow_agents.eval.tool_call_f1 import (
        TurnGroundTruth,
        TurnPrediction,
        evaluate_tool_calls,
    )
    from llm_workflow_agents.eval.tool_chain_propagation import evaluate_chain_propagation

    parser = argparse.ArgumentParser(description="Experiment A: workflow quality benchmark")
    parser.add_argument("--model",    required=True,  help="Model name (must match vLLM --model)")
    parser.add_argument("--output",   required=True,  help="Path to write results JSON")
    parser.add_argument(
        "--data",
        default="data/output/benchmark/task_a",
        help="Directory containing benchmark JSONL files (default: data/output/benchmark/task_a)",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="vLLM server base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit evaluation to first N samples per level (0 = no limit, useful for smoke tests)",
    )
    parser.add_argument(
        "--stochastic-trials",
        type=int,
        default=5,
        help="Number of temperature=0.7 trials for pass^k consistency (default: 5)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO). Use DEBUG to see raw model responses.",
    )
    args = parser.parse_args()

    import logging

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, args.log_level)
        ),
    )

    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}", file=sys.stderr)
        print(
            "Run ./scripts/generate_benchmark_data.sh first to generate benchmark data.",
            file=sys.stderr,
        )
        sys.exit(1)

    samples = _load_samples(data_dir)
    if not samples:
        print(f"ERROR: no JSONL samples found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    logger.info(
        "benchmark_start",
        model=args.model,
        endpoint=args.endpoint,
        data_dir=str(data_dir),
        num_samples=len(samples),
        stochastic_trials=args.stochastic_trials,
    )

    # --- Run deterministic evaluation pass (temperature=0.0) ---
    state_predictions: list[ConversationPrediction] = []
    state_ground_truths: list[ConversationGroundTruth] = []
    tool_predictions: list[TurnPrediction] = []
    tool_ground_truths: list[TurnGroundTruth] = []
    chain_predictions: list[dict[str, Any]] = []
    chain_ground_truths: list[dict[str, Any]] = []
    all_latencies_ms: list[float] = []

    for idx, sample in enumerate(samples):
        conv_id = sample.get("conversation_id", f"sample_{idx}")
        tool_schemas = sample.get("tool_schemas", [])
        gt_truth = sample.get("ground_truth", {})
        terminal_states = [gt_truth.get("terminal_state", "")] if gt_truth.get("terminal_state") else []

        logger.info("evaluating_sample", idx=idx + 1, total=len(samples), conversation_id=conv_id)

        pred_messages, latencies = _replay_conversation(
            args.endpoint, args.model, sample, temperature=0.0
        )
        all_latencies_ms.extend(latencies)

        # State machine inputs
        state_predictions.append(ConversationPrediction(
            conversation_id=conv_id,
            messages=pred_messages,
        ))
        state_ground_truths.append(ConversationGroundTruth(
            conversation_id=conv_id,
            messages=sample.get("messages", []),
            terminal_states=terminal_states,
        ))

        # Tool-call inputs — one TurnPrediction/GroundTruth per assistant turn
        for turn_idx, (pred_msg, gt_msg) in enumerate(
            zip(pred_messages, sample.get("messages", []))
        ):
            if gt_msg.get("role") != "assistant":
                continue
            tool_predictions.append(TurnPrediction(
                turn_id=turn_idx,
                content=pred_msg.get("content", ""),
            ))
            gt_tool_calls = (gt_msg.get("annotations") or {}).get("tool_calls") or []
            tool_ground_truths.append(TurnGroundTruth(
                turn_id=turn_idx,
                tool_calls=gt_tool_calls,
            ))

        # Chain propagation inputs
        chain_predictions.append({"messages": pred_messages})
        chain_ground_truths.append({"messages": sample.get("messages", [])})

    # --- Stochastic trials for pass^k ---
    stochastic_map: dict[str, list[list[dict[str, Any]]]] = {
        s.conversation_id: [] for s in state_predictions
    }
    for trial_num in range(args.stochastic_trials):
        logger.info("stochastic_trial", trial=trial_num + 1, total=args.stochastic_trials)
        for idx, sample in enumerate(samples):
            conv_id = sample.get("conversation_id", f"sample_{idx}")
            trial_messages, _ = _replay_conversation(
                args.endpoint, args.model, sample, temperature=0.7
            )
            stochastic_map[conv_id].append(trial_messages)

    for pred in state_predictions:
        pred.stochastic_trials = stochastic_map.get(pred.conversation_id, [])

    # --- Compute metrics ---
    state_metrics = evaluate_state_machine(state_predictions, state_ground_truths)
    tool_metrics = evaluate_tool_calls(tool_predictions, tool_ground_truths)
    chain_metrics = evaluate_chain_propagation(chain_predictions, chain_ground_truths)
    quality = evaluate_workflow_quality(state_metrics, tool_metrics, chain_metrics, all_latencies_ms)

    # --- Write results ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "model": args.model,
        "data_dir": str(data_dir),
        "num_samples": len(samples),
        "stochastic_trials": args.stochastic_trials,
        "metrics": quality.to_dict(),
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("benchmark_complete", output=str(output_path), **quality.to_dict())
    print(f"\nResults written to {output_path}")
    print(f"  weighted_workflow_score : {quality.weighted_workflow_score:.3f}  (target >=0.75)")
    print(f"  full_workflow_success   : {quality.full_workflow_success:.3f}  (target >=0.55)")
    print(f"  state_transition_acc    : {quality.state_metrics.state_transition_accuracy:.3f}  (target >=0.85)")
    print(f"  tool_call_f1            : {quality.tool_metrics.tool_call_f1:.3f}  (target >=0.85)")
