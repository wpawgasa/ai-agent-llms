"""Rewrite a domain graph so that no state offers more than one tool.

A state offering several tools leaves the model to guess which to call first,
and its instruction ("collect the satisfaction or NPS rating") cannot say how
many calls finish it -- so a model that follows the prompt exactly is scored
as missing a call (CLAUDE.md R28). Each multi-tool state in the registry
declares a ``tool_mode``, and this module splits it:

``sequence`` -- every tool is called, in the listed order. The state keeps its
name and its first tool; each later tool gets a new state, chained by
``tool_success``. Success exits leave from the last step only; error and
non-tool exits are available from every step.

``choice`` -- the customer's request decides the tool. The state keeps its
name as a text-only **router** whose instruction names every option, with one
``route`` edge to a new state per tool. One route is the spine edge (or none,
if the router keeps a required exit of its own), as ``validate_domain``
requires; subgraph selection keeps every route and the walk picks among them
at random. Tool exits (success and error) leave
from every branch; non-tool exits stay on the router.

Every original state name survives, so incoming edges, outbound reasons and
anything else that names a state stay valid. New states are named after their
tool (``SEND_PROPOSAL``), prefixed with the original state when that name is
already taken. A domain with no multi-tool state is returned unchanged -- the
same object -- so turning the rewrite on cannot alter such a domain.
"""

from __future__ import annotations

import dataclasses

from llm_workflow_agents.data.domain_registry import DomainSpec, Edge, StateNode

_TOOL_TRIGGERS = frozenset({"tool_success", "tool_error"})


def multi_tool_states(domain: DomainSpec) -> list[str]:
    """Names of the states in ``domain`` that offer more than one tool."""
    return [s.name for s in domain.states if len(s.tools) > 1]


def _descriptions(domain: DomainSpec) -> dict[str, str]:
    found: dict[str, str] = {}
    for tool in domain.tools:
        function = tool.get("function", tool)
        found[function.get("name", "")] = function.get("description", "")
    return found


def _new_name(tool: str, owner: str, taken: set[str]) -> str:
    name = tool.upper()
    if name == owner:
        # billing's PROCESS_PAYMENT router offers the process_payment tool.
        name = f"RUN_{name}"
    if name in taken:
        name = f"{owner}_{tool.upper()}"
    suffix = 2
    base = name
    while name in taken:
        name, suffix = f"{base}_{suffix}", suffix + 1
    taken.add(name)
    return name


def _describe(tool: str, descriptions: dict[str, str]) -> str:
    text = descriptions.get(tool) or tool.replace("_", " ")
    return text.rstrip(".")


def split_multi_tool_states(domain: DomainSpec) -> DomainSpec:
    """Return ``domain`` with every multi-tool state split into one-tool states."""
    targets = multi_tool_states(domain)
    if not targets:
        return domain

    descriptions = _descriptions(domain)
    taken = {s.name for s in domain.states}
    states: list[StateNode] = []
    edges: list[Edge] = [e for e in domain.edges if e.src not in targets]

    for state in domain.states:
        if state.name not in targets:
            states.append(state)
            continue
        outgoing = [e for e in domain.edges if e.src == state.name]
        tool_exits = [e for e in outgoing if e.trigger in _TOOL_TRIGGERS]
        other_exits = [e for e in outgoing if e.trigger not in _TOOL_TRIGGERS]
        k = len(state.tools)

        if state.tool_mode == "sequence":
            order = state.sequence_order or state.tools
            if sorted(order) != sorted(state.tools):
                raise ValueError(f"{domain.name}.{state.name}: sequence_order {order} does not match tools {state.tools}")
            names = [state.name] + [_new_name(t, state.name, taken) for t in order[1:]]
            for i, (name, tool) in enumerate(zip(names, order), start=1):
                states.append(StateNode(
                    name=name,
                    instruction=(
                        f"{_describe(tool, descriptions)} ({tool}). "
                        f"This is step {i} of {k} of: {state.instruction}"
                    ),
                    tools=(tool,),
                    kind=state.kind,
                ))
            for src, dst, tool in zip(names, names[1:], order):
                edges.append(Edge(src=src, dst=dst, label=f"{tool} done", trigger="tool_success"))
            for name in names:
                edges.extend(dataclasses.replace(e, src=name) for e in other_exits)
                edges.extend(dataclasses.replace(e, src=name) for e in tool_exits if e.trigger == "tool_error")
            edges.extend(dataclasses.replace(e, src=names[-1]) for e in tool_exits if e.trigger == "tool_success")

        elif state.tool_mode == "choice":
            branches = [(_new_name(t, state.name, taken), t) for t in state.tools]
            options = "; ".join(f"{_describe(t, descriptions)} -> [{name}]" for name, t in branches)
            states.append(StateNode(
                name=state.name,
                instruction=(
                    f"{state.instruction} Decide from the customer's request which one "
                    f"applies, then move to that state: {options}."
                ),
                tools=(),
                kind=state.kind,
            ))
            # validate_domain wants exactly one spine (non-optional) successor.
            # The router keeps a required exit of its own if it had one;
            # otherwise the first route is the spine. The walk picks among a
            # router's routes uniformly either way.
            router_has_spine = any(not e.optional for e in other_exits)
            for index, (name, tool) in enumerate(branches):
                states.append(StateNode(
                    name=name,
                    instruction=f"{_describe(tool, descriptions)} ({tool}).",
                    tools=(tool,),
                    kind="working",
                ))
                edges.append(Edge(
                    src=state.name, dst=name,
                    label=f"customer needs: {_describe(tool, descriptions).lower()}",
                    trigger="intent_match", route=True,
                    optional=router_has_spine or index > 0,
                ))
                edges.extend(dataclasses.replace(e, src=name) for e in tool_exits)
            edges.extend(other_exits)

        else:
            raise ValueError(
                f"{domain.name}: state '{state.name}' offers {k} tools but declares "
                f"tool_mode={state.tool_mode!r}; set 'sequence' or 'choice'"
            )

    return dataclasses.replace(domain, states=tuple(states), edges=tuple(edges))
