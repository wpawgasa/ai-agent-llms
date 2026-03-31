"""Multi-agent orchestrator for E2E workflow execution.

Deploys an orchestrator model (15-30B) alongside specialist subagents
(2-5B via vLLM LoRA multi-adapter) to route conversation turns,
record per-turn latency, tool calls, and state transitions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class WorkflowResult:
    """Result of a full workflow execution."""

    turns: list[dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    state_transitions: list[dict[str, str]] = field(default_factory=list)
    success: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "total_latency_ms": self.total_latency_ms,
            "tool_calls": self.tool_calls,
            "state_transitions": self.state_transitions,
            "success": self.success,
        }


class MultiAgentOrchestrator:
    """Orchestrates workflow execution across a 15-30B orchestrator and 2-5B specialists.

    The orchestrator model handles routing and high-level reasoning. Specialist
    models (loaded as LoRA adapters in vLLM) handle domain-specific tool calls.

    Args:
        orchestrator_config: Config dict for the orchestrator model. Expected keys:
            ``model_name`` (str), ``base_url`` (str, default "http://localhost:8000/v1").
        specialist_configs: List of config dicts for specialist models. Each dict
            should have ``model_name`` (str) and optionally ``lora_adapter`` (str).
        kv_cache_dtype: KV cache quantization dtype applied to both models.
    """

    def __init__(
        self,
        orchestrator_config: dict[str, Any],
        specialist_configs: list[dict[str, Any]],
        kv_cache_dtype: str = "turboquant",
    ) -> None:
        self.orchestrator_config = orchestrator_config
        self.specialist_configs = specialist_configs
        self.kv_cache_dtype = kv_cache_dtype
        self.base_url: str = orchestrator_config.get("base_url", "http://localhost:8000/v1")
        self.orchestrator_model: str = orchestrator_config.get("model_name", "")

        logger.info(
            "orchestrator_initialized",
            orchestrator_model=self.orchestrator_model,
            num_specialists=len(specialist_configs),
            kv_cache_dtype=kv_cache_dtype,
        )

    def _select_model(self, state: str, workflow_graph: dict[str, Any]) -> tuple[str, str | None]:
        """Select the appropriate model for the given workflow state.

        Returns:
            (model_name, lora_adapter_name) — lora_adapter is None for orchestrator.
        """
        nodes = workflow_graph.get("nodes", [])
        for node in nodes:
            node_id = node.get("id", node.get("name", ""))
            if node_id == state:
                specialist_tag = node.get("specialist")
                if specialist_tag:
                    for spec in self.specialist_configs:
                        if spec.get("tag") == specialist_tag or spec.get("model_name") == specialist_tag:
                            return spec["model_name"], spec.get("lora_adapter")
                    logger.warning(
                        "specialist_not_found",
                        state=state,
                        specialist_tag=specialist_tag,
                        fallback=self.orchestrator_model,
                    )
        return self.orchestrator_model, None

    def _extract_state_transitions(self, content: str) -> list[dict[str, str]]:
        """Extract [STATE: X -> Y] annotations from model output."""
        import re

        transitions = []
        pattern = re.compile(r"\[STATE:\s*(\w+)\s*->\s*(\w+)\]")
        for match in pattern.finditer(content or ""):
            transitions.append({"from_state": match.group(1), "to_state": match.group(2)})
        return transitions

    def _extract_tool_calls(self, message: Any) -> list[dict[str, Any]]:
        """Extract tool call info from an API response message."""
        calls = []
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return calls
        for tc in message.tool_calls:
            calls.append({
                "id": getattr(tc, "id", ""),
                "name": tc.function.name if hasattr(tc, "function") else "",
                "arguments": tc.function.arguments if hasattr(tc, "function") else "{}",
            })
        return calls

    async def run_workflow(
        self,
        conversation: list[dict[str, Any]],
        workflow_graph: dict[str, Any],
    ) -> WorkflowResult:
        """Execute a workflow turn-by-turn, routing between orchestrator and specialists.

        Args:
            conversation: List of message dicts (role/content) for the workflow.
            workflow_graph: WorkflowGraph dict with nodes, edges, initial_state, terminal_states.

        Returns:
            WorkflowResult capturing all turns, latencies, tool calls, and state transitions.
        """
        import openai

        client = openai.AsyncOpenAI(base_url=self.base_url, api_key="unused")

        current_state = workflow_graph.get("initial_state", "")
        terminal_states: set[str] = set(workflow_graph.get("terminal_states", []))
        messages = list(conversation)
        turns: list[dict[str, Any]] = []
        all_tool_calls: list[dict[str, Any]] = []
        all_transitions: list[dict[str, str]] = []
        total_latency_ms = 0.0

        logger.info(
            "workflow_started",
            initial_state=current_state,
            terminal_states=list(terminal_states),
            num_messages=len(messages),
        )

        for turn_idx, message in enumerate(messages):
            if message.get("role") != "user":
                continue

            model_name, lora_adapter = self._select_model(current_state, workflow_graph)

            extra_body: dict[str, Any] = {}
            if lora_adapter:
                extra_body["lora_request"] = {"lora_name": lora_adapter}

            t0 = time.perf_counter()
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=messages[: turn_idx + 1],
                    temperature=0.0,
                    max_tokens=1024,
                    extra_body=extra_body if extra_body else None,
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                reply = response.choices[0].message if response.choices else None
                content = reply.content if reply else ""

                # Extract state transitions and tool calls
                transitions = self._extract_state_transitions(content or "")
                tool_calls = self._extract_tool_calls(reply)

                all_transitions.extend(transitions)
                all_tool_calls.extend(tool_calls)

                if transitions:
                    current_state = transitions[-1]["to_state"]

                turn_record: dict[str, Any] = {
                    "turn": turn_idx,
                    "state": current_state,
                    "model": model_name,
                    "latency_ms": elapsed_ms,
                    "content": content,
                    "tool_calls": tool_calls,
                    "transitions": transitions,
                }
                turns.append(turn_record)
                total_latency_ms += elapsed_ms

                logger.info(
                    "turn_complete",
                    turn=turn_idx,
                    state=current_state,
                    model=model_name,
                    latency_ms=round(elapsed_ms, 2),
                )

                if current_state in terminal_states:
                    break

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                is_nemotron = "nemotron" in model_name.lower()
                logger.warning(
                    "turn_failed",
                    turn=turn_idx,
                    model=model_name,
                    nemotron_fallback=is_nemotron,
                    error=str(exc),
                )
                turns.append({
                    "turn": turn_idx,
                    "state": current_state,
                    "model": model_name,
                    "latency_ms": elapsed_ms,
                    "error": str(exc),
                })
                total_latency_ms += elapsed_ms

        success = current_state in terminal_states

        result = WorkflowResult(
            turns=turns,
            total_latency_ms=total_latency_ms,
            tool_calls=all_tool_calls,
            state_transitions=all_transitions,
            success=success,
        )

        logger.info(
            "workflow_complete",
            total_turns=len(turns),
            total_latency_ms=round(total_latency_ms, 2),
            final_state=current_state,
            success=success,
        )
        return result
