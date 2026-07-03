---
name: model-routing
description: Route tasks to the right Claude model tier — Fable 5 for high-leverage thinking work (spec drafting, research, planning, architecture decisions), Opus 4.8 or Sonnet 5 for implementation based on difficulty. Use this skill whenever the user mentions Fable, model selection, delegating or handing over tasks between models, "which model should do this", drafting a spec/plan before implementation, or any multi-phase workflow that separates planning from coding. Also use when the user asks to escalate a hard problem to a stronger model or to save cost by downgrading routine work. IMPORTANT — this skill requires explicit user confirmation before any Fable usage.
---

# Model Routing: Fable → Opus/Sonnet Handover

Use the most capable model where judgment quality compounds (specs, research, plans), and cheaper/faster models where the work is well-specified. A great spec executed by Sonnet beats a mediocre spec executed by Fable.

## Model tiers

| Model | Model string | Use for | Relative cost |
|---|---|---|---|
| Fable 5 | `claude-fable-5` | Spec drafting, deep research, planning, architecture trade-offs, ambiguous/novel problems | Highest — **always ask user first** |
| Opus 4.8 | `claude-opus-4-8` | Hard implementation: multi-file refactors, concurrency, performance-critical code, subtle debugging | High |
| Sonnet 5 | `claude-sonnet-5` | Routine implementation: CRUD, UI from a spec, tests, scripts, config, docs | Moderate |

## Routing decision

Classify the task **before** doing any work:

**Route to Fable (thinking work) when the task is:**
- Drafting a technical spec, PRD, or design doc from vague requirements
- Research requiring synthesis across many sources or deep domain reasoning
- System/architecture planning with non-obvious trade-offs (e.g., serving-engine selection, capacity modeling)
- Decomposing an ambiguous goal into an implementation plan
- Reviewing/critiquing a plan where a missed flaw is expensive

**Route to Opus (hard implementation) when ≥2 of these hold:**
- Touches >5 files or crosses module/service boundaries
- Involves concurrency, distributed state, or performance constraints
- Requires debugging with non-obvious root cause
- Correctness failures are costly (data loss, prod outage, security)
- The spec leaves meaningful design decisions to the implementer

**Route to Sonnet (routine implementation) when:**
- The spec/plan is complete and unambiguous
- The task is pattern-following: boilerplate, tests, UI from wireframes, glue code, migrations from a template
- Failures are cheap to detect and retry (lint/tests catch them)

Default when uncertain between Opus and Sonnet: start with Sonnet, escalate to Opus on the second failed attempt at the same problem.

## Fable confirmation gate (mandatory)

Never invoke Fable silently. Before any Fable call:

1. State **what** the Fable task is (one sentence).
2. State **why** a lower tier is insufficient.
3. Ask: "Use Fable for this? (highest-cost tier)"
4. Proceed only on explicit yes. If declined, do the task with Opus and note the downgrade.

This applies per work item, not per session — a new spec or research question needs a fresh confirmation. Batching is fine: "Use Fable for these 3 planning items?" counts as one confirmation for all three.

## Execution mechanics

**In Claude Code:**
- Delegate to a specific model via subagent: `Task(..., model="claude-sonnet-5")` (or opus/fable string)
- Or headless: `claude -p "<task>" --model claude-opus-4-8`
- Or tell the user to run `/model` to switch the interactive session

**Via API:** use the model strings from the table in the `model` field.

**If model switching isn't available in the current environment:** do the routing analysis anyway, tell the user which model should handle each phase, and produce the handover document so they can paste it into a session with the right model.

## Handover protocol (Fable → implementer)

Fable's output is only as useful as its handover. Every Fable planning/spec task must end with a handover document containing:

```markdown
# Handover: <task name>
**Target model:** claude-opus-4-8 | claude-sonnet-5 (with one-line justification)

## Objective
What done looks like, in one paragraph. Include acceptance criteria.

## Context the implementer lacks
Repo layout, key files, domain constraints, decisions already made and WHY.
Assume zero shared memory — the implementer sees only this document.

## Plan
Ordered steps. For Sonnet targets: fully specified, no open design questions.
For Opus targets: mark decisions intentionally left open with [IMPLEMENTER DECIDES].

## Constraints & non-goals
What must not change, out-of-scope items, performance/compat requirements.

## Verification
Exact commands/tests to run to confirm success.
```

Rules:
- **Sonnet handovers must contain zero open design questions.** If a question remains, either resolve it in the plan or upgrade the target to Opus.
- Include concrete file paths, function names, and interface signatures — not "update the relevant module."
- If the plan exceeds ~1 day of implementation work, split into multiple handover docs, each independently verifiable.

## Multi-phase workflow example

User: "Build a rate limiter for our voicebot API."

1. Classify: ambiguous requirements + architecture trade-offs → Fable task.
2. Ask: "This needs algorithm selection (token bucket vs sliding window) and capacity modeling — use Fable for the design spec? (highest-cost tier)"
3. On yes → Fable produces spec + handover doc. Distributed-state coordination piece → target Opus. Config, metrics endpoint, and tests → target Sonnet (separate handover).
4. Delegate each handover to its target model; run verification commands from each doc before accepting.

## Anti-patterns

- Using Fable for work a complete spec already covers — waste.
- Handing Sonnet a plan with "figure out the best approach for X" — that's an Opus task or an incomplete plan.
- Escalating to Fable after one Sonnet failure — try Opus first; Fable is for thinking, not retry-firepower.
- Skipping the confirmation because the user approved Fable earlier in the session for a different item.
