"""Shared test helpers for Task C playbook-pair tests.

CompliantTeacher parses the gold graph out of a render prompt and emits a
deterministic, machine-parseable playbook that satisfies the verification
contract. compliant_back_extract is its inverse (playbook -> id-keyed eval
graph), used to make the back-extraction gate pass in integration tests.
"""

from __future__ import annotations

import json
import re
from typing import Any

from llm_workflow_agents.data._playbook_verify import assign_state_ids, graph_to_eval_shape

_JSON_BLOCK_RE = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
_PROCEED_RE = re.compile(r"Proceed to (\w+) priority (\d+)")
_BRACKET_RE = re.compile(r"\[(\w+)\]")
_HEAD_NAME_RE = re.compile(r"\[?(\w+)\]?")


def _parse_render_prompt(user_prompt: str) -> tuple[dict[str, Any], list[str]]:
    """Extract the gold graph and any distractor texts from a render user prompt."""
    match = _JSON_BLOCK_RE.search(user_prompt)
    payload = json.loads(match.group(1))
    graph = payload["graph"]
    distractors: list[str] = []
    if "Insert the following paragraphs verbatim" in user_prompt:
        tail = user_prompt.split("Insert the following paragraphs verbatim", 1)[1]
        for line in tail.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                distractors.append(stripped[2:])
    return graph, distractors


def _emit_playbook(graph: dict[str, Any], distractors: list[str], dropped: set[str]) -> str:
    """Emit one `### STATE` block per (non-dropped) state, initial first."""
    details = {sd["name"]: sd for sd in graph["state_details"]}
    outgoing: dict[str, list[dict[str, Any]]] = {}
    for t in graph["transitions"]:
        outgoing.setdefault(t["from"], []).append(t)

    order = [graph["initial"]] + [s for s in graph["states"] if s != graph["initial"]]
    blocks: list[str] = []
    for name in order:
        if name in dropped:
            continue
        sd = details.get(name, {"tools": []})
        tools = list(sd.get("tools", []))
        header = f"Handle the {name} step."
        if tools:
            header += f" Tools: {', '.join(tools)}."
        lines = [f"### {name}", header]
        for t in outgoing.get(name, []):
            if t["to"] in dropped:
                continue
            lines.append(f"Proceed to {t['to']} priority {t.get('priority', 0)}.")
        blocks.append("\n".join(lines))
    blocks.extend(distractors)
    return "\n\n".join(blocks)


class CompliantTeacher:
    """A teacher stub that returns a verification-passing playbook for any gold graph."""

    def __init__(self, drop_anchor_on_first_call: str | None = None,
                 always_drop_anchor: str | None = None) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self._drop_first = drop_anchor_on_first_call
        self._always_drop = always_drop_anchor

    def __call__(self, model: str, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        self.prompts.append(user_prompt)
        graph, distractors = _parse_render_prompt(user_prompt)
        dropped: set[str] = set()
        if self._always_drop:
            dropped.add(self._always_drop)
        if self._drop_first and self.calls == 1:
            dropped.add(self._drop_first)
        return json.dumps({"playbook": _emit_playbook(graph, distractors, dropped)})


def compliant_back_extract(playbook: str) -> str:
    """Reconstruct the id-keyed eval graph from a playbook (echo gold).

    Handles both the CompliantTeacher format (`### NAME` + `Proceed to X priority N`)
    and the programmatic state_script format (`### [NAME]` + `[TARGET]` bullets).
    """
    blocks = re.split(r"^### ", playbook, flags=re.MULTILINE)[1:]
    names: list[str] = []
    details: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    for block in blocks:
        lines = block.splitlines()
        name = _HEAD_NAME_RE.match(lines[0].strip()).group(1)
        body = "\n".join(lines[1:])
        names.append(name)
        tools_match = re.search(r"Tools: ([^.\n]+)", body)
        tools = [t.strip() for t in tools_match.group(1).split(",")] if tools_match else []
        details.append({"name": name, "tools": tools, "entry_actions": [], "instruction": ""})
        targets = {tgt for tgt, _prio in _PROCEED_RE.findall(body)} | set(_BRACKET_RE.findall(body))
        for tgt in targets:
            transitions.append({"from": name, "to": tgt, "condition": "", "priority": 0})
    with_out = {t["from"] for t in transitions}
    graph = {
        "states": names,
        "state_details": details,
        "transitions": transitions,
        "initial": names[0],
        "terminal": [n for n in names if n not in with_out] or [names[-1]],
    }
    state_ids = assign_state_ids(graph, playbook)
    return json.dumps(graph_to_eval_shape(graph, state_ids))


# A small valid id-keyed graph the invention teacher returns for any brief.
_INVENTED_GRAPH = {
    "nodes": [
        {"id": "S1", "name": "INTAKE", "tools": ["log_case"], "entry_actions": []},
        {"id": "S2", "name": "PROCESS", "tools": [], "entry_actions": []},
        {"id": "S3", "name": "CLOSE", "tools": [], "entry_actions": []},
    ],
    "edges": [
        {"from_state": "S1", "to_state": "S2", "condition": "logged", "priority": 0},
        {"from_state": "S2", "to_state": "S3", "condition": "processed", "priority": 0},
    ],
    "initial_state": "S1",
    "terminal_states": ["S3"],
}


def _patch_all_teachers(monkeypatch, teacher: CompliantTeacher, echo_gold_on_verify: bool = False) -> None:
    """Patch call_teacher_model in render, verify (back-extraction), and invention modules."""
    import llm_workflow_agents.data._graph_invention as gi
    import llm_workflow_agents.data._playbook_render as pr
    import llm_workflow_agents.data._playbook_verify as pv

    monkeypatch.setattr(pr, "call_teacher_model", teacher)
    monkeypatch.setattr(gi, "call_teacher_model", lambda model, system_prompt, user_prompt: json.dumps(_INVENTED_GRAPH))
    if echo_gold_on_verify:
        monkeypatch.setattr(pv, "call_teacher_model",
                            lambda model, system_prompt, playbook: compliant_back_extract(playbook))
