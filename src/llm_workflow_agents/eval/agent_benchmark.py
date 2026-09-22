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

``--data`` is repeatable, and naming both strata is what makes the modality
blend fire — the text stratum and the voice stratum are SIBLING directories
and the loader globs one level deep:

    python -m llm_workflow_agents.eval.agent_benchmark ... \\
        --data data/output/benchmark/task_a \\
        --data data/output/benchmark/task_a_voice

With one ``--data`` the run behaves exactly as it did before the flag became
repeatable, and ``quality_summary["quality"]`` equals the text score by float
identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from llm_workflow_agents.eval.state_accuracy import (
    STATE_ANNOTATION_PATTERN,
    ConversationGroundTruth,
    ConversationPrediction,
    StateMachineMetrics,
    evaluate_state_machine,
    parse_state_transitions,
)
from llm_workflow_agents.eval.tool_call_f1 import (
    ToolCallMetrics,
    TurnGroundTruth,
    TurnPrediction,
    evaluate_tool_calls,
    evaluate_tool_calls_conversation,
    parse_tool_calls,
)
from llm_workflow_agents.eval.tool_chain_propagation import ChainPropagationMetrics
from llm_workflow_agents.eval.composite_score import DEFAULT_VOICE_WEIGHT, blend_modality_scores
from llm_workflow_agents.eval.chunk_diagnostics import chunk_diagnostics_by_language
from llm_workflow_agents.eval.segment_scoring import SegmentStats, reply_texts, segment_scoring_view
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


def _fmt_stratum(value: float | None) -> str:
    """Render a per-stratum score for the console summary, or 'null' when empty."""
    return "null" if value is None else f"{value:.4f}"


def compute_modality_quality_summary(
    samples: list[dict[str, Any]],
    state_predictions: list[ConversationPrediction],
    state_ground_truths: list[ConversationGroundTruth],
    conv_tool_preds: list[list[TurnPrediction]],
    conv_tool_gts: list[list[TurnGroundTruth]],
    voice_weight: float = DEFAULT_VOICE_WEIGHT,
) -> dict[str, Any]:
    """Score the text and voice strata separately, then blend into one number.

    Partitions conversations by ``sample.get("modality") or "text"`` — a
    conversation with no ``modality`` field (every sample predating the voice
    feature) is grouped into the text stratum, never dropped.

    A conversation whose ``modality`` is present but is neither ``"text"``
    nor ``"voice"`` (a typo, a stale label, a modality this scorer does not
    know) is ALSO counted into the text stratum, and never dropped. Two
    choices were available: drop it, or fail. Dropping is what this function
    used to do, and it is the worse of the two — ``n_text + n_voice`` stopped
    summing to ``len(samples)`` with nothing in the output saying so, which
    turns the ranking into a measurement of a subset that still reads like a
    number. Failing is also wrong here: this function runs AFTER every
    generation in the run has been paid for, so raising would discard a
    completed benchmark over a label. So the row is folded into the
    established default stratum and the fact is made impossible to miss —
    a ``benchmark_unknown_modality`` warning, plus ``n_unknown_modality``
    and ``unknown_modalities`` in the returned dict and in the console
    summary. ``n_text + n_voice == len(samples)`` always holds. Each stratum's
    quality is computed once over its own aggregated metrics, using this
    module's own :func:`compute_weighted_score` — NOT
    ``composite_score.compute_weighted_workflow_score``, a different formula
    with a different signature. The two per-stratum scores are then blended
    with :func:`llm_workflow_agents.eval.composite_score.blend_modality_scores`,
    a weighted mean of the two stratum means, never a mean over pooled rows
    (a pooled mean's effective weight drifts with row counts every time the
    corpus is regenerated).

    With no voice conversations present, ``quality_voice`` is ``None`` and
    ``blend_modality_scores`` returns ``quality_text`` by float identity —
    this makes the change inert on every benchmark run predating the voice
    corpus.

    ``attach_chunk_diagnostics`` merges chunk-format diagnostics into this
    same summary dict as a separate step; this function owns only the
    modality partition and the blend, so the dict stays open to that
    addition without restructuring.

    Args:
        samples: The raw benchmark samples, in the same order used to build
            the prediction/ground-truth lists below.
        state_predictions: Per-conversation state predictions, one per sample.
        state_ground_truths: Per-conversation state ground truth, one per sample.
        conv_tool_preds: Per-conversation lists of turn-level tool predictions,
            one list per sample (as built for conversation-level tool eval).
        conv_tool_gts: Per-conversation lists of turn-level tool ground truth,
            one list per sample.
        voice_weight: Share of quality carried by the voice stratum.

    Returns:
        Dict with ``quality_text``, ``quality_voice`` (either ``None`` when
        that stratum is empty), ``quality`` (the blend), ``n_text``,
        ``n_voice``, ``n_unknown_modality``, ``unknown_modalities`` and
        ``voice_weight``. ``n_text`` includes any unknown-modality rows.
    """
    partitions: dict[str, list[int]] = {}
    for idx, sample in enumerate(samples):
        modality = sample.get("modality") or "text"
        partitions.setdefault(modality, []).append(idx)

    def _stratum_quality(indices: list[int]) -> float | None:
        if not indices:
            return None
        sub_state_preds = [state_predictions[i] for i in indices]
        sub_state_gts = [state_ground_truths[i] for i in indices]
        sub_conv_tool_preds = [conv_tool_preds[i] for i in indices]
        sub_conv_tool_gts = [conv_tool_gts[i] for i in indices]
        sub_turn_preds = [tp for i in indices for tp in conv_tool_preds[i]]
        sub_turn_gts = [tg for i in indices for tg in conv_tool_gts[i]]

        stratum_state_metrics = evaluate_state_machine(sub_state_preds, sub_state_gts)
        stratum_tool_turn = evaluate_tool_calls(sub_turn_preds, sub_turn_gts)
        stratum_tool_conv = evaluate_tool_calls_conversation(sub_conv_tool_preds, sub_conv_tool_gts)
        stratum_tool_best = (
            stratum_tool_conv
            if stratum_tool_conv.tool_call_f1 >= stratum_tool_turn.tool_call_f1
            else stratum_tool_turn
        )
        return compute_weighted_score(
            stratum_state_metrics,
            stratum_tool_best,
            stratum_state_metrics.task_completion_rate,
        )

    unknown = sorted(k for k in partitions if k not in ("text", "voice"))
    unknown_indices = [i for k in unknown for i in partitions[k]]
    text_indices = sorted(partitions.get("text", []) + unknown_indices)
    voice_indices = partitions.get("voice", [])
    if unknown:
        # Loud, and in the output — never dropped. A dropped row makes the
        # ranking a measurement of a subset that still looks like a whole
        # number (risk R18c's shape), and n_text + n_voice would stop summing
        # to len(samples) with nothing saying so.
        logger.warning(
            "benchmark_unknown_modality",
            modalities=unknown,
            n_unknown=len(unknown_indices),
            counted_as="text",
        )
    quality_text = _stratum_quality(text_indices)
    quality_voice = _stratum_quality(voice_indices)
    quality = blend_modality_scores(quality_text, quality_voice, voice_weight)

    return {
        "quality_text": quality_text,
        "quality_voice": quality_voice,
        "quality": quality,
        "n_text": len(text_indices),
        "n_voice": len(voice_indices),
        "n_unknown_modality": len(unknown_indices),
        "unknown_modalities": unknown,
        "voice_weight": voice_weight,
    }


def _voice_stratum_completions_by_language(
    samples: list[dict[str, Any]],
    conv_tool_preds: list[list[TurnPrediction]],
) -> dict[str, list[str]]:
    """Group VOICE-modality per-turn generated text by each sample's OWN language.

    Chunk diagnostics (``<S>...</S>`` markers) only make sense on voice
    output — a text conversation holds no chunk markers at all (see
    ``02-data-generation.md``, Voice Modality; ``data/voice_convention.py``
    checks the converse for text rows). Uses the same modality partition rule
    as :func:`compute_modality_quality_summary`: ``sample.get("modality") or
    "text"``.

    The voice stratum is mixed by design — the benchmark draws English and
    Thai at even odds per sample, independent of everything else (see
    ``docs/superpowers/specs/2026-08-21-voice-benchmark-and-prompt-switch-design.md``
    section 4) — so completions are grouped by each sample's own ``language``
    (default ``"en"``) rather than collapsed to one language for the whole
    stratum. Pass the result to :func:`chunk_diagnostics_by_language`, never
    to :func:`chunk_diagnostics` directly.
    """
    out: dict[str, list[str]] = {}
    for sample, turns in zip(samples, conv_tool_preds):
        if (sample.get("modality") or "text") != "voice":
            continue
        language = sample.get("language") or "en"
        out.setdefault(language, []).extend(tp.content for tp in turns)
    return out


def attach_chunk_diagnostics(
    quality_summary: dict[str, Any],
    voice_completions_by_language: dict[str, list[str]],
) -> dict[str, Any]:
    """Attach reference-free chunk diagnostics to a quality summary dict.

    Guardrail only (see ``chunk_diagnostics.py``'s module docstring for why):
    this merges a new ``chunk_diagnostics`` key into a *copy* of
    ``quality_summary`` without touching any existing key — in particular
    ``quality``, the number that ranks Phase 1 candidates. Chunk formatting
    is cheap for fine-tuning to install, so it must never move that ranking.

    ``voice_completions_by_language`` (from
    :func:`_voice_stratum_completions_by_language`) is scored via
    :func:`chunk_diagnostics_by_language`, which scores each language under
    its own boundary-quality convention and pools the underlying
    measurements before computing any percentile — see that function's
    docstring for why a majority-vote single language, or an average of
    per-language percentiles, would both be wrong.
    """
    return {
        **quality_summary,
        "chunk_diagnostics": chunk_diagnostics_by_language(voice_completions_by_language),
    }


# ---------------------------------------------------------------------------
# CLI entrypoint — invoked by scripts/run_exp_a.sh
# ---------------------------------------------------------------------------

#: Where the text stratum lives. The voice stratum is the SIBLING directory
#: ``data/output/benchmark/task_a_voice``; pass both with two ``--data`` flags
#: to produce a run that blends the two modality strata.
DEFAULT_DATA_DIR = "data/output/benchmark/task_a"


def resolve_data_paths(values: "list[str] | None") -> "list[Path]":
    """Resolve repeated ``--data`` values into a sorted list of paths.

    ``--data`` is repeatable so one run can name both strata: the text
    stratum and the voice stratum are sibling directories and
    :func:`_load_samples` globs ``*.jsonl`` non-recursively, so a single
    directory can never hold both. Without this the modality blend has no
    reachable invocation at all — ``blend_modality_scores`` could only ever
    take its "no voice stratum" identity branch, which is the flattering
    kind of inertness risk R16 records.

    ``argparse``'s ``action="append"`` APPENDS to a non-``None`` default
    rather than replacing it, so a default written into the flag would leak
    into every explicit invocation (the trap already hit in
    ``scripts/clean_task_a_sft.py``). The flag therefore defaults to
    ``None`` and the fallback lives here.

    The result is sorted and de-duplicated so flag ORDER cannot change a
    result: sample order feeds ``--max-samples`` and every aggregate below,
    so two runs naming the same strata in different orders must score
    identically.
    """
    from pathlib import Path

    raw = list(values or [])
    if not raw:
        raw = [DEFAULT_DATA_DIR]
    return sorted({Path(v) for v in raw}, key=lambda p: str(p))


def _load_samples(
    data_paths: "Path | list[Path]", max_samples_per_path: int = 0
) -> list[dict[str, Any]]:
    """Load benchmark samples from directories (all *.jsonl) and/or files.

    Accepts one path or several. Files within a directory are read in sorted
    filename order, and the paths themselves are read in the order given
    (:func:`resolve_data_paths` sorts them first).

    ``max_samples_per_path`` caps each PATH's contribution rather than the
    concatenated list. With one ``--data`` that is exactly the old whole-run
    cap; with two strata it stops the cap from truncating one stratum to
    zero while the console still reports ``voice_weight=0.3`` — a run that
    reads as "blend applied" while the blend is inert.
    """
    import json
    from pathlib import Path

    if isinstance(data_paths, (str, Path)):
        data_paths = [Path(data_paths)]

    samples: list[dict[str, Any]] = []
    for data_path in data_paths:
        p = Path(data_path)
        paths = [p] if p.is_file() else sorted(p.glob("*.jsonl"))
        from_this_path: list[dict[str, Any]] = []
        for path in paths:
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        from_this_path.append(json.loads(line))
        if max_samples_per_path > 0:
            from_this_path = from_this_path[:max_samples_per_path]
        samples.extend(from_this_path)
    return samples


def dataset_fingerprints(data_paths: "list[Path]") -> dict[str, str]:
    """SHA-256 of exactly the files ``_load_samples`` reads, per ``--data`` path.

    A directory's hash covers each ``*.jsonl`` directly inside it, by name and
    content, in sorted order — so a result can be tied to the benchmark version
    it scored, which a directory name alone cannot do.
    """
    import hashlib
    from pathlib import Path

    out: dict[str, str] = {}
    for data_path in data_paths:
        p = Path(data_path)
        files = [p] if p.is_file() else sorted(p.glob("*.jsonl"))
        digest = hashlib.sha256()
        for f in files:
            digest.update(f.name.encode() + b"\0")
            digest.update(hashlib.sha256(f.read_bytes()).digest())
        out[str(p)] = digest.hexdigest()
    return out


def _downgrade_tool_turns_to_text(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert past structured tool turns to plain text.

    Gemini-3 (via BiFrost) requires a server-generated ``thought_signature`` on
    every re-sent ``functionCall`` part. BiFrost's OpenAI translation strips
    that field, so any conversation containing a past ``assistant.tool_calls``
    fails the next request with HTTP 400. We sidestep the validation by
    rewriting past tool turns:

      - assistant + tool_calls -> assistant with ``<tool_call>{...}</tool_call>``
        tags appended in ``content``.
      - role=tool -> role=user with ``[Tool result] ...`` prefix.

    The ``tools=[...]`` schema in the request is unchanged, so the model can
    still emit *new* structured tool calls; only past ones are textualised.
    """
    import json as _json
    out: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant" and msg.get("tool_calls"):
            tags: list[str] = []
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except _json.JSONDecodeError:
                        pass
                tags.append(_json.dumps({"name": fn.get("name", ""), "arguments": args}))
            content = msg.get("content") or ""
            tag_block = "\n".join(f"<tool_call>{t}</tool_call>" for t in tags)
            new_content = (content + ("\n" if content else "") + tag_block).strip()
            out.append({"role": "assistant", "content": new_content})
        elif role == "tool":
            tcid = msg.get("tool_call_id", "")
            prefix = f"[Tool result {tcid}]" if tcid else "[Tool result]"
            out.append({"role": "user", "content": f"{prefix}: {msg.get('content', '')}"})
        else:
            out.append(msg)
    return out


#: Chars of a leaked/missing prefix kept in a log line — enough to see the
#: shape without flooding logs on a run where it recurs many times.
_LEAK_PREFIX_LOG_CHARS = 200


def _strip_leaked_reasoning_prefix(content: str) -> str:
    """Detect and strip a leaked-reasoning prefix before the mandatory annotation.

    FORMAT_RULES rule 1 requires every assistant turn's visible text to begin
    with a ``[STATE: X → Y]`` annotation; ``NO_REASONING_LEAK_RULE`` (added
    2026-09-07, ``system_prompt.py``) additionally forbids anything else
    appearing before it. A frontier-model benchmark run observed one turn whose
    visible content opened with internal self-talk ("So the tool was executed
    even with speech! Then why didn't `change_plan` execute? Wait!...") instead
    of the annotation — a decoding-side artifact that did not reproduce
    byte-for-byte on a second call at the same temperature, so a system-prompt
    instruction alone cannot be relied on to prevent it (see that rule's
    docstring). This is the harness-side guardrail:

    - If a valid ``[STATE: ...]`` annotation is found LATER in the content,
      strip everything before it, so the leaked prefix doesn't pollute the
      conversation context forwarded to subsequent turns (which would risk the
      model imitating its own error) and log the event for auditability.
    - If no annotation is found anywhere, the content is left untouched —
      state/tool scoring already treats a turn with no parseable annotation as
      a missing transition, so there is nothing to recover — but the event is
      still logged so a run can be diagnosed from its log alone, the way this
      exact issue first required a manual replay to find.
    - Empty or whitespace-only content (e.g. from an HTTP-error turn) is
      already logged separately (``http_error_during_call``) and is left alone.
    """
    match = STATE_ANNOTATION_PATTERN.search(content)
    if match is None:
        if content.strip():
            logger.warning(
                "turn_missing_state_annotation",
                content_preview=content[:_LEAK_PREFIX_LOG_CHARS],
            )
        return content
    prefix = content[: match.start()]
    if prefix.strip():
        logger.warning(
            "leaked_reasoning_prefix_stripped",
            prefix_preview=prefix[:_LEAK_PREFIX_LOG_CHARS],
        )
        return content[match.start():]
    return content


def _call_vllm(
    endpoint: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool = False,
    engine: str = "vllm",
) -> tuple[str, list[dict[str, Any]], float, float]:
    """Call the OpenAI-compatible chat completions endpoint (vLLM, SGLang, TRT-LLM, or BiFrost).

    When *tools* are provided they are included in the request so that
    the server can emit structured tool calls.

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
    import urllib.error
    import urllib.request

    if engine == "bifrost":
        messages = _downgrade_tool_turns_to_text(messages)

    request_body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if engine in {"vllm", "sglang"}:
        # Qwen3-family thinking toggle. Ignored by tokenizer chat templates
        # that don't reference enable_thinking (Gemma, Mistral, etc.), so it's
        # safe to always send for vLLM and SGLang (≥0.5 accepts the field).
        # TRT-LLM rejects unknown body fields, so it stays in the else branch.
        # Default False for fair latency comparison against non-thinking models.
        request_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
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
    try:
        resp_cm = urllib.request.urlopen(req, timeout=read_timeout_s)
    except urllib.error.HTTPError as e:
        # 400-class errors usually mean a malformed conversation slipped through
        # (e.g. mismatched tool_call_ids, schema-incompatible arguments). Log
        # the response body once and return an empty turn so the benchmark
        # continues instead of aborting all 200 samples.
        try:
            body = e.read().decode("utf-8", errors="replace")[:1000]
        except Exception:
            body = "<no body>"
        logger.warning(
            "http_error_during_call",
            status=e.code,
            body=body,
            messages_len=len(messages),
            last_role=messages[-1].get("role") if messages else None,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0
        return "", [], latency_ms, 0.0
    with resp_cm as resp:
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

    content = _strip_leaked_reasoning_prefix("".join(content_parts))

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


def _tool_calls_from_text(content: str) -> tuple[str, list[dict[str, Any]]]:
    """Lift ``<tool_call>{...}</tool_call>`` blocks out of *content*.

    Returns the content with every block removed and the calls as OpenAI
    structured ``tool_calls`` with fresh ids. Content with no parseable call is
    returned unchanged with an empty list.
    """
    import json as _json
    import re as _re
    import uuid as _uuid

    parsed = parse_tool_calls(content)
    if not parsed:
        return content, []
    calls = [
        {
            "id": f"call_{_uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": p.get("name", ""),
                "arguments": _json.dumps(p.get("arguments", {})),
            },
        }
        for p in parsed
    ]
    text = _re.sub(r"\s*<tool_call>.*?</tool_call>\s*", " ", content, flags=_re.DOTALL).strip()
    return text, calls


def _assistant_context_messages(
    text: str, tool_calls: list[dict[str, Any]], split_tool_call_content: bool
) -> list[dict[str, Any]]:
    """Native-format context entries for one assistant turn.

    With *split_tool_call_content*, a turn holding both text and tool calls is
    sent as two assistant messages, text first. The Gemma-4 chat template
    renders a merged message's text AFTER the tool responses and then closes the
    model turn, so a request ending in that tool result carries no generation
    cue: the model sees its own pre-call text posing as its answer to the result
    and replies with nothing. Split, the template renders text, call and result
    in the order they happened and leaves the turn open. ChatML-family
    templates render the merged form correctly and would show the split form as
    two separate assistant turns, hence the flag.
    """
    if tool_calls and split_tool_call_content and text.strip():
        return [
            {"role": "assistant", "content": text},
            {"role": "assistant", "content": "", "tool_calls": tool_calls},
        ]
    msg: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return [msg]


def _gt_turn_has_call(msg: dict[str, Any]) -> bool:
    return bool(
        (msg.get("annotations") or {}).get("tool_calls")
        or msg.get("tool_calls")
        or parse_tool_calls(msg.get("content") or "")
    )


def _replay_conversation(
    endpoint: str,
    model: str,
    sample: dict[str, Any],
    temperature: float = 0.0,
    enable_thinking: bool = False,
    engine: str = "vllm",
    tool_turn_format: str = "native",
    split_tool_call_content: bool = False,
    stats: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    """Replay a conversation, asking the model once per SEGMENT of assistant turns.

    A segment is a run of ground-truth assistant turns with no user message or
    tool result between them (``eval/segment_scoring.py``). The corpus often
    splits one piece of work across two such turns — speech, then the tool
    call — and a model may equally do both in one reply. So the model is asked
    once at the start of each segment, and when the segment owes a tool call
    that the reply did not make, asked once more: the context then ends on the
    model's own reply, which vLLM and the Gemma-4 template accept as the cue
    for a fresh turn (verified 2026-09-22 on 40 real pairs, both formats).
    Frontier APIs refuse a request ending on the model's own turn, so on
    BiFrost the second ask is skipped and counted (``second_ask_unavailable``).
    A second reply is kept only if it makes a tool call. Asked again with
    nothing new to react to, a model that had no call to make sometimes writes
    the customer's next line; such a reply goes into neither the context nor
    the prediction (``second_ask_no_call_discarded``).

    Ground truth is never copied into a prediction. An opening turn (an
    outbound conversation, before any user message) is not asked for: the
    served prompt does not say why the agent is calling, so the model can only
    invent a purpose, and its invented opener would then sit in the context
    beside ground-truth customer replies to a different one. The ground-truth
    opener goes into the CONTEXT and the segment is marked ``"unscored": True``
    so scoring drops it — on every engine, so local and frontier runs match.

    Ground-truth tool responses stand in for real tool execution, but only for
    calls that were actually made: a result reaches the context only when the
    latest reply made a tool call, one result per call. A result for a call the
    model never made would not exist in a real system (R26). The segment that
    answers a withheld result cannot be asked for — nothing new has arrived —
    so it is scored as a miss (empty prediction).

    Predictions stay aligned with the ground truth message by message: a
    segment's replies are joined into its first slot (and listed under
    ``"replies"``), and its other slots are left empty.

    ``stats``, when given, is incremented in place with counts of segments,
    second asks, unscored segments and misses.

    Returns:
        (predicted_messages, latencies_ms_per_request, ttfts_ms_per_request)
    """
    tools = sample.get("tool_schemas") or []
    terminal_states = set(sample.get("workflow_graph", {}).get("terminal", []))
    messages = sample.get("messages", [])
    predicted: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    ttfts_ms: list[float] = []
    context: list[dict[str, Any]] = []  # sliding context sent to the model
    # Ids of the calls made by the latest assistant reply that no result has
    # answered yet. Reset at every reply, so a call from an earlier reply
    # never licenses a later result.
    pending_tool_call_ids: list[str] = []
    # True when a ground-truth tool result was withheld since the last reply
    # (the call it answers was never made).
    result_withheld = False
    counts = stats if stats is not None else {}

    def _count(key: str) -> None:
        counts[key] = counts.get(key, 0) + 1

    def _add_assistant_turn(content: str, text: str, calls: list[dict[str, Any]]) -> None:
        if tool_turn_format == "text":
            # Keep the turn exactly as written. Training sees tool calls as
            # text in place, so rebuilding them from structured calls would
            # make the context drift from it.
            context.append({"role": "assistant", "content": content})
        else:
            context.extend(_assistant_context_messages(text, calls, split_tool_call_content))
        pending_tool_call_ids[:] = [tc.get("id") or f"call_{i}" for i, tc in enumerate(calls)]

    def _ask() -> tuple[str, str, list[dict[str, Any]]]:
        if tool_turn_format == "text":
            # Same format the model trained in: tool calls as text, tool
            # results as prefixed user turns, and no native tool declarations
            # (the schemas are in the system prompt). data/tool_turns.py.
            from llm_workflow_agents.data.tool_turns import to_text_tool_turns

            request_messages, request_tools = to_text_tool_turns(context), None
        else:
            request_messages, request_tools = context, tools
        content, raw_tool_calls, latency, ttft = _call_vllm(
            endpoint, model, request_messages, temperature, tools=request_tools,
            enable_thinking=enable_thinking, engine=engine,
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
        # Synthesize structured tool_calls from inline <tool_call> text when
        # the API returned none in the structured field; without them the next
        # tool message has no tool_call_id and frontier providers reject it.
        if raw_tool_calls:
            text_content = content.split("\n<tool_call>")[0]  # calls appended as tags
        else:
            text_content, raw_tool_calls = _tool_calls_from_text(content)
        return content, text_content, raw_tool_calls

    def _copy_into_context(msg: dict[str, Any]) -> None:
        # The corpus writes tool calls as <tool_call> text. Native templates
        # render a result only under structured tool_calls, so lift them out.
        gt_content = msg.get("content") or ""
        if msg.get("tool_calls"):
            gt_text, gt_calls = gt_content, list(msg["tool_calls"])
        else:
            gt_text, gt_calls = _tool_calls_from_text(gt_content)
        _add_assistant_turn(gt_content, gt_text, gt_calls)

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg["role"]

        if role == "system":
            enriched = _build_system_prompt(sample, msg["content"])
            context.append({"role": "system", "content": enriched})
            predicted.append(msg)  # keep original in predictions for eval
            i += 1

        elif role == "user":
            context.append({"role": "user", "content": msg["content"]})
            predicted.append(msg)
            i += 1

        elif role == "tool":
            # Ground-truth tool results stand in for real execution, but only
            # for a call the latest reply actually made. Predictions keep every
            # ground-truth message so alignment is unchanged.
            predicted.append(msg)
            i += 1
            if not pending_tool_call_ids:
                logger.info("tool_result_withheld_no_call", turn=len(predicted) - 1)
                result_withheld = True
                continue
            context.append({
                "role": "tool",
                "content": msg["content"],
                "tool_call_id": pending_tool_call_ids.pop(0),
            })

        elif role == "assistant":
            end = i
            while end < len(messages) and messages[end]["role"] == "assistant":
                end += 1
            segment = messages[i:end]
            _count("segments")
            last_role = context[-1].get("role") if context else None

            if last_role not in ("user", "tool") and result_withheld:
                # This segment answers a tool result the model never got,
                # because it never made the call. Nothing new has arrived to
                # react to, and copying the ground-truth answer would score the
                # model on work it could not have done — score it as a miss.
                logger.info("missed_segment_after_withheld_tool_result", turn=i, turns=len(segment))
                _count("missed_after_withheld_result")
                result_withheld = False
                pending_tool_call_ids.clear()
                predicted.extend({"role": "assistant", "content": ""} for _ in segment)
                i = end
                continue

            if last_role not in ("user", "tool"):
                # An opening turn: no user message yet (outbound conversations
                # open with the agent). See the docstring for why it is not
                # asked for.
                logger.info(
                    "unscored_opening_segment",
                    last_context_role=last_role, turn=i, turns=len(segment),
                )
                _count("unscored_opening")
                for gt_msg in segment:
                    _copy_into_context(gt_msg)
                result_withheld = False
                predicted.extend({**gt_msg, "unscored": True} for gt_msg in segment)
                i = end
                continue

            content, text_content, calls = _ask()
            _add_assistant_turn(content, text_content, calls)
            replies = [content]
            result_withheld = False

            if not calls and any(_gt_turn_has_call(m) for m in segment):
                # The segment owes a tool call and the reply made none: the
                # model may be announcing the call first. Ask once more.
                if engine == "bifrost":
                    logger.info("second_ask_unavailable", engine=engine, turn=i)
                    _count("second_ask_unavailable")
                else:
                    _count("second_asks")
                    content, text_content, calls = _ask()
                    if calls:
                        _count("second_ask_made_call")
                        _add_assistant_turn(content, text_content, calls)
                        replies.append(content)
                    else:
                        logger.info("second_ask_no_call_discarded", turn=i, content=content[:300])
                        _count("second_ask_no_call_discarded")

            pred_msg: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(r for r in replies if r),
                "replies": replies,
            }
            predicted.append(pred_msg)
            predicted.extend({"role": "assistant", "content": ""} for _ in segment[1:])

            # Do NOT stop on a terminal state: walking every ground-truth turn
            # keeps later ground-truth tool calls in the scoring alignment.
            if terminal_states:
                transitions = parse_state_transitions([pred_msg])
                if transitions and transitions[-1][1] in terminal_states:
                    logger.info(
                        "terminal_state_reached_continuing",
                        state=transitions[-1][1],
                        turn=len(latencies_ms),
                    )
            i = end

        else:
            predicted.append(msg)
            i += 1

    return predicted, latencies_ms, ttfts_ms


def build_state_machine_inputs(
    samples: list[dict[str, Any]],
    predicted_messages: list[list[dict[str, Any]]],
    ground_truth_messages: list[list[dict[str, Any]]] | None = None,
) -> tuple[list[ConversationPrediction], list[ConversationGroundTruth]]:
    """Pair each sample's replayed messages with its own ground truth.

    ``evaluate_state_machine`` matches predictions to ground truth by
    ``conversation_id``, but ids are not unique across a run: the text and
    voice strata both number their conversations ``L1_001``, ``L1_002``, ...
    Keyed by id, every text prediction was scored against the voice
    conversation sharing its id, understating whole-run state accuracy.
    Keys here combine the row position with the id, so each row can only
    ever meet its own ground truth.

    ``ground_truth_messages`` replaces each sample's own messages as the ground
    truth — the benchmark passes its segment view (``eval/segment_scoring.py``).
    """
    if len(samples) != len(predicted_messages):
        raise ValueError(
            f"{len(samples)} samples but {len(predicted_messages)} predicted conversations"
        )
    predictions: list[ConversationPrediction] = []
    ground_truths: list[ConversationGroundTruth] = []
    if ground_truth_messages is not None and len(ground_truth_messages) != len(samples):
        raise ValueError(
            f"{len(samples)} samples but {len(ground_truth_messages)} ground-truth conversations"
        )
    for idx, (sample, messages) in enumerate(zip(samples, predicted_messages)):
        key = f"{idx}:{sample.get('conversation_id', f'sample_{idx}')}"
        gt_truth = sample.get("ground_truth", {})
        terminal_states = [gt_truth["terminal_state"]] if gt_truth.get("terminal_state") else []
        predictions.append(ConversationPrediction(conversation_id=key, messages=messages))
        ground_truths.append(ConversationGroundTruth(
            conversation_id=key,
            messages=(
                ground_truth_messages[idx] if ground_truth_messages is not None
                else sample.get("messages", [])
            ),
            terminal_states=terminal_states,
        ))
    return predictions, ground_truths


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    from dotenv import load_dotenv

    load_dotenv()

    # ConversationGroundTruth, ConversationPrediction, evaluate_state_machine,
    # TurnGroundTruth, TurnPrediction, evaluate_tool_calls and
    # evaluate_tool_calls_conversation are imported at module level above
    # (needed there by compute_modality_quality_summary).
    from llm_workflow_agents.eval.tool_chain_propagation import evaluate_chain_propagation

    parser = argparse.ArgumentParser(description="Experiment A: workflow quality benchmark")
    parser.add_argument("--model",    required=True,  help="Model name (must match the endpoint's model name)")
    parser.add_argument("--output",   required=True,  help="Path to write results JSON")
    parser.add_argument(
        "--data",
        action="append",
        default=None,   # NOT the default path: action="append" appends to a
                        # non-None default instead of replacing it, so the
                        # default would leak into every explicit invocation.
                        # resolve_data_paths applies the fallback.
        help=(
            "Directory of benchmark JSONL files, or one JSONL file. Repeatable: "
            f"pass it twice to score both modality strata in one run (default: {DEFAULT_DATA_DIR}), "
            f"e.g. --data {DEFAULT_DATA_DIR} --data {DEFAULT_DATA_DIR}_voice. "
            "Paths are sorted before loading, so flag order cannot change a result."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="Server base URL (vLLM default: http://localhost:8000; BiFrost gateway: http://localhost:23040)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help=(
            "Limit evaluation to the first N samples from EACH --data path "
            "(0 = no limit, useful for smoke tests). Per path, not per run, so "
            "a two-stratum smoke run cannot truncate one stratum to zero while "
            "still reporting a voice weight."
        ),
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
    parser.add_argument(
        "--engine",
        default="vllm",
        choices=["vllm", "bifrost", "sglang", "tensorrt_llm"],
        help=(
            "Backend engine: 'vllm' (default) or 'sglang' or 'tensorrt_llm' "
            "for a local OpenAI-compatible server, 'bifrost' for the BiFrost "
            "LLM gateway. 'bifrost' omits vLLM/SGLang-specific body fields "
            "(chat_template_kwargs) that frontier provider APIs reject. "
            "'sglang' enables the Qwen3 thinking toggle like vLLM. "
            "'tensorrt_llm' skips the thinking toggle (TRT-LLM rejects "
            "unknown body fields)."
        ),
    )
    parser.add_argument(
        "--tool-turn-format",
        choices=["native", "text"],
        default="native",
        help=(
            "native (default): past tool calls and results go to the server as "
            "structured tool_calls / tool messages, with tools=. text: tool calls "
            "as <tool_call> text and results as '[Tool result]: ' user turns, no "
            "tools= — the format models trained on the Task A corpus since "
            "2026-09-17 see in training. Compare models only within one format."
        ),
    )
    parser.add_argument(
        "--split-tool-call-content",
        action="store_true",
        help=(
            "Native format only: send a reply holding both text and tool calls "
            "as two assistant messages, text first. Required for Gemma-4 "
            "templates, which render a merged message's text after the tool "
            "response and leave no generation cue (empty replies). Leave off "
            "for ChatML-family templates."
        ),
    )
    parser.add_argument(
        "--voice-weight",
        type=float,
        default=DEFAULT_VOICE_WEIGHT,
        help=(
            "Share of quality carried by the voice stratum (default: "
            f"{DEFAULT_VOICE_WEIGHT}). Ignored when the run holds no voice "
            "conversations, in which case quality equals the text score exactly."
        ),
    )
    args = parser.parse_args()

    import logging

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, args.log_level)
        ),
    )

    data_paths = resolve_data_paths(args.data)
    for _p in data_paths:
        if not _p.exists():
            print(f"ERROR: data path not found: {_p}", file=sys.stderr)
            print(
                "Run ./scripts/generate_benchmark_data.sh first to generate benchmark data.",
                file=sys.stderr,
            )
            sys.exit(1)

    samples = _load_samples(data_paths, max_samples_per_path=args.max_samples)
    if not samples:
        print(
            "ERROR: no JSONL samples found at "
            + ", ".join(str(_p) for _p in data_paths),
            file=sys.stderr,
        )
        sys.exit(1)

    # A single path keeps the old string in the log and the result JSON; a
    # multi-path run records every stratum it read.
    data_dir_repr = (
        str(data_paths[0]) if len(data_paths) == 1
        else ", ".join(str(_p) for _p in data_paths)
    )

    import re as _re_start
    _start_level_match = (
        _re_start.match(r"(l[1-5])_", data_paths[0].name)
        if len(data_paths) == 1 and data_paths[0].is_file() else None
    )
    _start_level_tag = _start_level_match.group(1).upper() if _start_level_match else "mixed"

    logger.info(
        "benchmark_start",
        model=args.model,
        engine=args.engine,
        endpoint=args.endpoint,
        data_dir=data_dir_repr,
        complexity_level=_start_level_tag,
        num_samples=len(samples),
        stochastic_trials=args.stochastic_trials,
    )

    # --- Run deterministic evaluation pass (temperature=0.0) ---
    all_pred_messages: list[list[dict[str, Any]]] = []
    tool_predictions: list[TurnPrediction] = []
    tool_ground_truths: list[TurnGroundTruth] = []
    # Per-conversation tool call lists for conversation-level eval
    conv_tool_preds: list[list[TurnPrediction]] = []
    conv_tool_gts: list[list[TurnGroundTruth]] = []
    chain_predictions: list[dict[str, Any]] = []
    chain_ground_truths: list[dict[str, Any]] = []
    all_latencies_ms: list[float] = []
    all_ttfts_ms: list[float] = []
    # Every metric reads one ground-truth turn beside one prediction slot, so
    # both sides are scored through the segment view (eval/segment_scoring.py):
    # one turn per run of consecutive ground-truth assistant turns.
    all_gt_views: list[list[dict[str, Any]]] = []
    # One entry per request, for guardrails that measure a single reply's shape.
    conv_reply_preds: list[list[TurnPrediction]] = []
    segment_stats = SegmentStats()
    replay_counts: dict[str, int] = {}

    for idx, sample in enumerate(samples):
        conv_id = sample.get("conversation_id", f"sample_{idx}")
        tool_schemas = sample.get("tool_schemas", [])

        logger.info("evaluating_sample", idx=idx + 1, total=len(samples), conversation_id=conv_id)

        pred_messages, latencies, ttfts = _replay_conversation(
            args.endpoint, args.model, sample, temperature=0.0,
            enable_thinking=args.enable_thinking, engine=args.engine,
            tool_turn_format=args.tool_turn_format,
            split_tool_call_content=args.split_tool_call_content,
            stats=replay_counts,
        )
        all_latencies_ms.extend(latencies)
        all_ttfts_ms.extend(ttfts)
        gt_view, pred_view = segment_scoring_view(
            sample.get("messages", []), pred_messages, segment_stats,
        )
        all_pred_messages.append(pred_view)
        all_gt_views.append(gt_view)
        conv_reply_preds.append([
            TurnPrediction(turn_id=k, content=text)
            for k, text in enumerate(reply_texts(pred_messages))
        ])

        # Tool-call inputs — one TurnPrediction/GroundTruth per assistant turn
        this_conv_preds: list[TurnPrediction] = []
        this_conv_gts: list[TurnGroundTruth] = []
        for turn_idx, (pred_msg, gt_msg) in enumerate(zip(pred_view, gt_view)):
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
        chain_predictions.append({"messages": pred_view})
        chain_ground_truths.append({"messages": gt_view})

    # Keyed by row position, never by conversation_id: the text and voice
    # strata reuse ids (see build_state_machine_inputs).
    state_predictions, state_ground_truths = build_state_machine_inputs(
        samples, all_pred_messages, all_gt_views,
    )

    # --- Stochastic trials for pass^k ---
    for trial_num in range(args.stochastic_trials):
        logger.info("stochastic_trial", trial=trial_num + 1, total=args.stochastic_trials)
        for idx, sample in enumerate(samples):
            trial_messages, _, _ = _replay_conversation(
                args.endpoint, args.model, sample, temperature=0.7,
                enable_thinking=args.enable_thinking, engine=args.engine,
                tool_turn_format=args.tool_turn_format,
                split_tool_call_content=args.split_tool_call_content,
            )
            _, trial_view = segment_scoring_view(sample.get("messages", []), trial_messages)
            state_predictions[idx].stochastic_trials.append(trial_view)

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

    # Score the text and voice strata separately and blend (voice weighted
    # DEFAULT_VOICE_WEIGHT by default). With no voice conversations present
    # this is inert: quality_summary["quality"] == quality_summary["quality_text"]
    # by float identity (blend_modality_scores' no-voice-stratum branch).
    quality_summary = compute_modality_quality_summary(
        samples, state_predictions, state_ground_truths,
        conv_tool_preds, conv_tool_gts,
        voice_weight=args.voice_weight,
    )

    # Attach reference-free chunk-format diagnostics over the VOICE stratum's
    # completions only (voice markers don't appear in text conversations).
    # Guardrails only — attach_chunk_diagnostics never touches
    # quality_summary["quality"], so this cannot move the Phase 1 ranking.
    quality_summary = attach_chunk_diagnostics(
        quality_summary,
        _voice_stratum_completions_by_language(samples, conv_reply_preds),
    )

    # --- Write results ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive complexity level from data path when --data is a single
    # `l[1-5]_*.jsonl` file (the per-level case). For directory inputs
    # spanning all levels, "mixed" is recorded.
    import re as _re
    level_match = (
        _re.match(r"(l[1-5])_", data_paths[0].name)
        if len(data_paths) == 1 and data_paths[0].is_file() else None
    )
    level_tag = level_match.group(1).upper() if level_match else "mixed"

    result = {
        "model": args.model,
        "engine": args.engine,
        "endpoint": args.endpoint,
        "kv_cache_dtype": args.kv_cache_dtype,
        "data_dir": data_dir_repr,
        "data_paths": [str(_p) for _p in data_paths],
        "complexity_level": level_tag,
        "num_samples": len(samples),
        "stochastic_trials": args.stochastic_trials,
        "tool_turn_format": args.tool_turn_format,
        # BiFrost rewrites every request's history to text
        # (_downgrade_tool_turns_to_text), whatever --tool-turn-format says.
        "tool_turn_format_effective": "text" if args.engine == "bifrost" else args.tool_turn_format,
        "split_tool_call_content": args.split_tool_call_content,
        "data_sha256": dataset_fingerprints(data_paths),
        # 2026-09-22: one ask per segment of consecutive ground-truth assistant
        # turns, a second ask when an owed tool call is missing, scored per
        # segment; ground truth is never copied into a prediction.
        "scoring": "segment",
        "segment_stats": {**segment_stats.to_dict(), **replay_counts},
        # 2026-09-17: tool results are shown only for calls actually made.
        "tool_result_gating": "calls_made_only",
        "metrics": quality.to_dict(),
        # quality_summary now also carries a "chunk_diagnostics" key
        # (guardrails, computed over the voice stratum only; see
        # attach_chunk_diagnostics) alongside the modality blend.
        "quality_summary": quality_summary,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(
        "benchmark_complete",
        output=str(output_path),
        **quality.to_dict(),
        **quality_summary,
    )
    print(f"\nResults written to {output_path}")
    print(f"  weighted_workflow_score : {quality.weighted_workflow_score:.3f}  (target >=0.75)")
    print(f"  full_workflow_success   : {quality.full_workflow_success:.3f}  (target >=0.55)")
    print(f"  state_trans_acc (turn)   : {quality.state_metrics.state_transition_accuracy:.3f}  (target >=0.85)")
    print(f"  state_seq_acc (conv)    : {quality.state_metrics.state_sequence_accuracy:.3f}")
    print(f"  tool_call_f1 (turn)     : {quality.tool_metrics.tool_call_f1:.3f}  (target >=0.85)")
    print(f"  tool_call_f1 (conv)     : {quality.tool_metrics_conversation.tool_call_f1:.3f}")
    print(f"  latency_per_turn_avg_ms : {quality.latency_per_turn_avg_ms:.1f}")
    print(f"  ttft_avg_ms             : {quality.ttft_avg_ms:.1f}")
    print(f"  segments (scoring unit) : {result['segment_stats']}")
    print(
        "  quality (blended)       : "
        f"{quality_summary['quality']:.4f}"
        f"  [text={_fmt_stratum(quality_summary['quality_text'])} "
        f"(n={quality_summary['n_text']}), "
        f"voice={_fmt_stratum(quality_summary['quality_voice'])} "
        f"(n={quality_summary['n_voice']}), "
        f"voice_weight={quality_summary['voice_weight']}]"
    )
    if quality_summary["n_unknown_modality"]:
        print(
            f"  WARNING: {quality_summary['n_unknown_modality']} conversation(s) "
            f"carry an unknown modality {quality_summary['unknown_modalities']} "
            f"and were counted into the TEXT stratum"
        )
    _diag = quality_summary["chunk_diagnostics"]
    print(
        "  chunk diagnostics (guardrail, voice only, not in composite): "
        f"n_turns_with_chunks={_diag['n_turns_with_chunks']}, "
        f"first_chunk_p50/p90={_diag['first_chunk_p50']:.0f}/{_diag['first_chunk_p90']:.0f}, "
        f"chunk_len_p50/p90={_diag['chunk_len_p50']:.0f}/{_diag['chunk_len_p90']:.0f}, "
        f"chunks_per_turn_p50={_diag['chunks_per_turn_p50']:.1f}, "
        f"boundary_quality={_diag['boundary_quality']:.3f} "
        f"(pooled across languages={_diag['languages']})"
    )
    # Per language, because a pooled boundary_quality cannot say WHICH
    # language drags it down, and the voice stratum is half Thai by design.
    for _lang, _sub in sorted(_diag["per_language"].items()):
        print(
            f"    {_lang}: n_turns_with_chunks={_sub['n_turns_with_chunks']}, "
            f"first_chunk_p50/p90={_sub['first_chunk_p50']:.0f}/{_sub['first_chunk_p90']:.0f}, "
            f"chunk_len_p50/p90={_sub['chunk_len_p50']:.0f}/{_sub['chunk_len_p90']:.0f}, "
            f"chunks_per_turn_p50={_sub['chunks_per_turn_p50']:.1f}, "
            f"boundary_quality={_sub['boundary_quality']:.3f}"
        )
