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

from llm_workflow_agents.eval.state_accuracy import StateMachineMetrics, parse_state_transitions
from llm_workflow_agents.eval.tool_call_f1 import ToolCallMetrics
from llm_workflow_agents.eval.tool_chain_propagation import ChainPropagationMetrics
from llm_workflow_agents.data.system_prompt import build_enriched_system_prompt as _build_system_prompt

logger = structlog.get_logger(__name__)


@dataclass
class WorkflowQualityMetrics:
    """Combined workflow quality metrics."""

    full_workflow_success: float = 0.0  # Target: >=55%
    weighted_workflow_score: float = 0.0  # Target: >=0.75
    latency_per_turn_median_ms: float = 0.0  # Target: <=2000 (L1-L3), <=5000 (L4-L5)
    latency_per_turn_avg_ms: float = 0.0
    ttft_avg_ms: float = 0.0

    state_metrics: StateMachineMetrics = field(default_factory=StateMachineMetrics)
    tool_metrics: ToolCallMetrics = field(default_factory=ToolCallMetrics)
    tool_metrics_conversation: ToolCallMetrics = field(default_factory=ToolCallMetrics)
    chain_metrics: ChainPropagationMetrics = field(default_factory=ChainPropagationMetrics)

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_workflow_success": self.full_workflow_success,
            "weighted_workflow_score": self.weighted_workflow_score,
            "latency_per_turn_median_ms": self.latency_per_turn_median_ms,
            "latency_per_turn_avg_ms": self.latency_per_turn_avg_ms,
            "ttft_avg_ms": self.ttft_avg_ms,
            "state_metrics": self.state_metrics.to_dict(),
            "tool_metrics": self.tool_metrics.to_dict(),
            "tool_metrics_conversation": self.tool_metrics_conversation.to_dict(),
            "chain_metrics": self.chain_metrics.to_dict(),
        }


def compute_weighted_score(
    state: StateMachineMetrics,
    tool: ToolCallMetrics,
    completion: float,
) -> float:
    """Compute weighted workflow quality score.

    Uses the *better* of per-turn and conversation-level state accuracy
    (``state_sequence_accuracy``) so that models which traverse the
    correct states in fewer turns are not unfairly penalised.

    Formula: 0.4 * max(state_turn, state_seq) + 0.4 * ToolCallF1 + 0.2 * TaskCompletion

    Args:
        state: State machine metrics.
        tool: Tool-calling metrics.
        completion: Task completion rate (0.0-1.0).

    Returns:
        Weighted score between 0.0 and 1.0.
    """
    best_state_acc = max(
        state.state_transition_accuracy,
        state.state_sequence_accuracy,
    )
    return (
        0.4 * best_state_acc
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


def compute_average(values: list[float]) -> float:
    """Compute arithmetic mean from a list of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def evaluate_workflow_quality(
    state_metrics: StateMachineMetrics,
    tool_metrics: ToolCallMetrics,
    chain_metrics: ChainPropagationMetrics,
    latencies_ms: list[float] | None = None,
    ttfts_ms: list[float] | None = None,
    tool_metrics_turn: ToolCallMetrics | None = None,
    tool_metrics_conversation: ToolCallMetrics | None = None,
) -> WorkflowQualityMetrics:
    """Compute combined workflow quality metrics.

    Args:
        state_metrics: State machine adherence results.
        tool_metrics: Tool-calling metrics used for composite scoring
            (typically the better of per-turn and conversation-level).
        chain_metrics: Tool chain propagation results.
        latencies_ms: Optional per-turn latency measurements.
        ttfts_ms: Optional per-turn TTFT measurements.
        tool_metrics_turn: Per-turn tool metrics (for reporting).
        tool_metrics_conversation: Conversation-level tool metrics (for reporting).

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

    latencies = latencies_ms or []
    ttfts = ttfts_ms or []
    median_latency = compute_latency_median(latencies)
    avg_latency = compute_average(latencies)
    avg_ttft = compute_average(ttfts)

    metrics = WorkflowQualityMetrics(
        full_workflow_success=full_success,
        weighted_workflow_score=weighted,
        latency_per_turn_median_ms=median_latency,
        latency_per_turn_avg_ms=avg_latency,
        ttft_avg_ms=avg_ttft,
        state_metrics=state_metrics,
        tool_metrics=tool_metrics,
        tool_metrics_conversation=tool_metrics_conversation or tool_metrics,
        chain_metrics=chain_metrics,
    )

    logger.info(
        "workflow_quality_eval_complete",
        weighted_score=weighted,
        full_success=full_success,
        median_latency_ms=median_latency,
        avg_latency_ms=avg_latency,
        avg_ttft_ms=avg_ttft,
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
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool = False,
) -> tuple[str, list[dict[str, Any]], float, float]:
    """Call the vLLM OpenAI-compatible chat completions endpoint.

    When *tools* are provided they are included in the request so that
    vLLM can use its ``--tool-call-parser`` to emit structured tool
    calls.

    Uses streaming to measure TTFT (Time To First Token).

    Returns:
        (content, raw_tool_calls, latency_ms, ttft_ms)

    *content* has any structured tool calls serialised as
    ``<tool_call>{JSON}</tool_call>`` tags appended so that
    ``tool_call_f1.parse_tool_calls`` can extract them.

    *raw_tool_calls* is the list straight from the API response so the
    caller can build a well-formed assistant message for the context
    (with ``tool_calls`` field and matching ``tool_call_id``).
    """
    import json
    import time
    import urllib.request

    request_body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        # Qwen3-family thinking toggle. Ignored by tokenizer chat templates
        # that don't reference enable_thinking (Gemma, Mistral, etc.), so it's
        # safe to always send. Default False for fair latency comparison
        # against non-thinking models in Phase 1.
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    if tools:
        request_body["tools"] = tools
    payload = json.dumps(request_body).encode()

    req = urllib.request.Request(
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.monotonic()
    ttft_ms = 0.0
    first_token_received = False
    content_parts: list[str] = []
    raw_tool_calls: list[dict[str, Any]] = []
    # Track tool call deltas keyed by index
    tool_call_accum: dict[int, dict[str, Any]] = {}

    # Generous default because hybrid MoE models on enforce_eager (Qwen3.6
    # with TurboQuant, Gemma-4) may JIT kernels during early inference,
    # stretching any single socket read past 120 s. Override via env var if
    # needed for even slower cold starts.
    import os

    read_timeout_s = float(os.environ.get("VLLM_HTTP_READ_TIMEOUT_S", "600"))
    with urllib.request.urlopen(req, timeout=read_timeout_s) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[len("data:"):].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})

            # Measure TTFT on first content or tool_call delta
            if not first_token_received and (delta.get("content") or delta.get("tool_calls")):
                ttft_ms = (time.monotonic() - t0) * 1000.0
                first_token_received = True

            if delta.get("content"):
                content_parts.append(delta["content"])

            # Accumulate streamed tool call deltas
            for tc_delta in delta.get("tool_calls", []):
                tc_idx = tc_delta.get("index", 0)
                if tc_idx not in tool_call_accum:
                    tool_call_accum[tc_idx] = {
                        "id": tc_delta.get("id", ""),
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                entry = tool_call_accum[tc_idx]
                if tc_delta.get("id"):
                    entry["id"] = tc_delta["id"]
                fn_delta = tc_delta.get("function", {})
                if fn_delta.get("name"):
                    entry["function"]["name"] += fn_delta["name"]
                if fn_delta.get("arguments"):
                    entry["function"]["arguments"] += fn_delta["arguments"]

    latency_ms = (time.monotonic() - t0) * 1000.0

    # Build raw_tool_calls from accumulated deltas
    for idx in sorted(tool_call_accum):
        raw_tool_calls.append(tool_call_accum[idx])

    content = "".join(content_parts)

    # Append any structured tool calls as <tool_call> tags so that
    # parse_tool_calls() can extract them.
    for tc in raw_tool_calls:
        fn = tc.get("function", {})
        call_obj = {"name": fn.get("name", ""), "arguments": fn.get("arguments", {})}
        # arguments may arrive as a JSON string — parse it
        if isinstance(call_obj["arguments"], str):
            try:
                call_obj["arguments"] = json.loads(call_obj["arguments"])
            except json.JSONDecodeError:
                pass
        content += f"\n<tool_call>{json.dumps(call_obj)}</tool_call>"

    return content, raw_tool_calls, latency_ms, ttft_ms


def _replay_conversation(
    endpoint: str,
    model: str,
    sample: dict[str, Any],
    temperature: float = 0.0,
    enable_thinking: bool = False,
) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    """Replay a conversation, substituting model completions at assistant turns.

    Ground-truth tool responses are kept as-is so the conversation stays on
    track regardless of whether the model's tool call was correct — this
    isolates state-transition and tool-call quality from cascading failures.

    Returns:
        (predicted_messages, latencies_ms_per_assistant_turn, ttfts_ms_per_assistant_turn)
    """
    tools = sample.get("tool_schemas") or []
    terminal_states = set(sample.get("workflow_graph", {}).get("terminal", []))
    predicted: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    ttfts_ms: list[float] = []
    context: list[dict[str, Any]] = []  # sliding context sent to the model
    pending_tool_call_ids: list[str] = []  # ids from the latest assistant tool_calls

    for msg in sample.get("messages", []):
        role = msg["role"]

        if role == "system":
            enriched = _build_system_prompt(sample, msg["content"])
            context.append({"role": "system", "content": enriched})
            predicted.append(msg)  # keep original in predictions for eval

        elif role == "user":
            context.append({"role": "user", "content": msg["content"]})
            predicted.append(msg)

        elif role == "assistant":
            content, raw_tool_calls, latency, ttft = _call_vllm(
                endpoint, model, context, temperature, tools=tools,
                enable_thinking=enable_thinking,
            )
            latencies_ms.append(latency)
            ttfts_ms.append(ttft)
            logger.debug(
                "model_response",
                turn=len(predicted),
                latency_ms=round(latency, 1),
                ttft_ms=round(ttft, 1),
                content=content[:2000],  # truncate to avoid flooding logs
            )

            # Build a well-formed assistant message for the context so
            # that subsequent tool-role messages have matching tool_call_ids.
            # Use content *without* appended <tool_call> tags for the API context
            text_content = content.split("\n<tool_call>")[0] if raw_tool_calls else content
            ctx_msg: dict[str, Any] = {"role": "assistant", "content": text_content}
            if raw_tool_calls:
                ctx_msg["tool_calls"] = raw_tool_calls
                # Store the ids so the next tool message(s) can reference them
                pending_tool_call_ids.clear()
                for tc in raw_tool_calls:
                    tc_id = tc.get("id", "")
                    if tc_id:
                        pending_tool_call_ids.append(tc_id)
            context.append(ctx_msg)

            # For eval, store the full content with <tool_call> tags
            pred_msg: dict[str, Any] = {"role": "assistant", "content": content}
            predicted.append(pred_msg)

            # Stop if the model reached a terminal state
            if terminal_states:
                transitions = parse_state_transitions([pred_msg])
                if transitions and transitions[-1][1] in terminal_states:
                    logger.info(
                        "terminal_state_reached",
                        state=transitions[-1][1],
                        turn=len(latencies_ms),
                    )
                    break

        elif role == "tool":
            # Use ground-truth tool response to avoid cascading failures.
            # Assign tool_call_id from the model's preceding tool call so
            # the OpenAI-format conversation stays well-formed.
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "content": msg["content"],
            }
            if pending_tool_call_ids:
                tool_msg["tool_call_id"] = pending_tool_call_ids.pop(0)
            elif msg.get("tool_call_id"):
                tool_msg["tool_call_id"] = msg["tool_call_id"]
            context.append(tool_msg)
            predicted.append(msg)

    return predicted, latencies_ms, ttfts_ms


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()

    from llm_workflow_agents.eval.state_accuracy import (
        ConversationGroundTruth,
        ConversationPrediction,
        evaluate_state_machine,
    )
    from llm_workflow_agents.eval.tool_call_f1 import (
        TurnGroundTruth,
        TurnPrediction,
        evaluate_tool_calls,
        evaluate_tool_calls_conversation,
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
        "--kv-cache-dtype",
        default="auto",
        help="KV cache quantization dtype (default: auto). Recorded in result JSON.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level (default: INFO). Use DEBUG to see raw model responses.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help=(
            "Enable Qwen3-family reasoning/thinking mode during the benchmark. "
            "Default OFF for fair latency comparison against non-thinking models. "
            "Sends chat_template_kwargs={enable_thinking: True} on every request; "
            "ignored by tokenizer templates that don't reference the flag."
        ),
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
    # Per-conversation tool call lists for conversation-level eval
    conv_tool_preds: list[list[TurnPrediction]] = []
    conv_tool_gts: list[list[TurnGroundTruth]] = []
    chain_predictions: list[dict[str, Any]] = []
    chain_ground_truths: list[dict[str, Any]] = []
    all_latencies_ms: list[float] = []
    all_ttfts_ms: list[float] = []

    for idx, sample in enumerate(samples):
        conv_id = sample.get("conversation_id", f"sample_{idx}")
        tool_schemas = sample.get("tool_schemas", [])
        gt_truth = sample.get("ground_truth", {})
        terminal_states = [gt_truth.get("terminal_state", "")] if gt_truth.get("terminal_state") else []

        logger.info("evaluating_sample", idx=idx + 1, total=len(samples), conversation_id=conv_id)

        pred_messages, latencies, ttfts = _replay_conversation(
            args.endpoint, args.model, sample, temperature=0.0,
            enable_thinking=args.enable_thinking,
        )
        all_latencies_ms.extend(latencies)
        all_ttfts_ms.extend(ttfts)

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
        this_conv_preds: list[TurnPrediction] = []
        this_conv_gts: list[TurnGroundTruth] = []
        for turn_idx, (pred_msg, gt_msg) in enumerate(
            zip(pred_messages, sample.get("messages", []))
        ):
            if gt_msg.get("role") != "assistant":
                continue
            tp = TurnPrediction(
                turn_id=turn_idx,
                content=pred_msg.get("content", ""),
            )
            gt_tool_calls = (gt_msg.get("annotations") or {}).get("tool_calls") or []
            tg = TurnGroundTruth(
                turn_id=turn_idx,
                tool_calls=gt_tool_calls,
            )
            tool_predictions.append(tp)
            tool_ground_truths.append(tg)
            this_conv_preds.append(tp)
            this_conv_gts.append(tg)
        conv_tool_preds.append(this_conv_preds)
        conv_tool_gts.append(this_conv_gts)

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
            trial_messages, _, _ = _replay_conversation(
                args.endpoint, args.model, sample, temperature=0.7,
                enable_thinking=args.enable_thinking,
            )
            stochastic_map[conv_id].append(trial_messages)

    for pred in state_predictions:
        pred.stochastic_trials = stochastic_map.get(pred.conversation_id, [])

    # --- Compute metrics ---
    state_metrics = evaluate_state_machine(state_predictions, state_ground_truths)
    tool_metrics_turn = evaluate_tool_calls(tool_predictions, tool_ground_truths)
    tool_metrics_conv = evaluate_tool_calls_conversation(conv_tool_preds, conv_tool_gts)
    chain_metrics = evaluate_chain_propagation(chain_predictions, chain_ground_truths)

    # Use the better of per-turn and conversation-level tool metrics for the
    # composite score.  Conversation-level is more lenient — a correct tool
    # call at a different turn still gets credit — which is fairer for
    # pre-trained models that haven't been fine-tuned on the exact workflow.
    tool_metrics_best = (
        tool_metrics_conv if tool_metrics_conv.tool_call_f1 >= tool_metrics_turn.tool_call_f1
        else tool_metrics_turn
    )
    quality = evaluate_workflow_quality(
        state_metrics, tool_metrics_best, chain_metrics, all_latencies_ms,
        ttfts_ms=all_ttfts_ms,
        tool_metrics_turn=tool_metrics_turn,
        tool_metrics_conversation=tool_metrics_conv,
    )

    # --- Write results ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "model": args.model,
        "kv_cache_dtype": args.kv_cache_dtype,
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
    print(f"  state_trans_acc (turn)   : {quality.state_metrics.state_transition_accuracy:.3f}  (target >=0.85)")
    print(f"  state_seq_acc (conv)    : {quality.state_metrics.state_sequence_accuracy:.3f}")
    print(f"  tool_call_f1 (turn)     : {quality.tool_metrics.tool_call_f1:.3f}  (target >=0.85)")
    print(f"  tool_call_f1 (conv)     : {quality.tool_metrics_conversation.tool_call_f1:.3f}")
    print(f"  latency_per_turn_avg_ms : {quality.latency_per_turn_avg_ms:.1f}")
    print(f"  ttft_avg_ms             : {quality.ttft_avg_ms:.1f}")
