"""The agent-decidable metric must not move any existing score.

Which arguments the model owns is a decision about what the benchmark
measures (R31), and adding a metric must not take that decision quietly. So:

- no corpus declares `source` today, and while that holds the agent-decidable
  tool metrics equal the headline ones exactly;
- the composite never reads the new metric.

The first of these is asserted against the REAL v4 benchmark, so it starts
failing the day someone annotates a schema — which is the point at which the
two numbers are meant to diverge and be read separately.
"""

from __future__ import annotations

import glob
import json

import pytest

from llm_workflow_agents.eval.agent_benchmark import compute_weighted_score
from llm_workflow_agents.eval.argument_provenance import agent_decidable_call, argument_source
from llm_workflow_agents.eval.state_accuracy import StateMachineMetrics
from llm_workflow_agents.eval.tool_call_f1 import ToolCallMetrics

V4 = sorted(glob.glob("data/output/benchmark/task_a_v4/*.jsonl")) + sorted(
    glob.glob("data/output/benchmark/task_a_voice_v3/*.jsonl")
)


def test_the_composite_does_not_read_the_new_metric() -> None:
    state = StateMachineMetrics(state_transition_accuracy=0.8, state_sequence_accuracy=0.9)
    tool = ToolCallMetrics(tool_call_f1=0.7)
    before = compute_weighted_score(state, tool, completion=0.6)
    # A different agent-decidable value cannot enter: the function takes one
    # ToolCallMetrics and reads tool_call_f1 from it.
    assert before == 0.4 * 0.9 + 0.4 * 0.7 + 0.2 * 0.6


@pytest.mark.skipif(not V4, reason="v4 benchmark not materialized (dvc pull)")
def test_no_v4_schema_declares_a_source_yet_so_the_metric_is_identical() -> None:
    declared = 0
    changed = 0
    for path in V4:
        for line in open(path):
            if not line.strip():
                continue
            sample = json.loads(line)
            schemas = sample.get("tool_schemas") or []
            for schema in schemas:
                function = schema.get("function", schema)
                properties = (function.get("parameters") or {}).get("properties") or {}
                for argument in properties:
                    if argument_source(schemas, function.get("name", ""), argument) is not None:
                        declared += 1
            for message in sample.get("messages", []):
                for call in (message.get("annotations") or {}).get("tool_calls") or []:
                    if agent_decidable_call(schemas, call) != {
                        **call, "arguments": call.get("arguments") or {}
                    }:
                        changed += 1
    assert declared == 0, f"{declared} arguments now declare a source — read the two tool metrics separately"
    assert changed == 0
