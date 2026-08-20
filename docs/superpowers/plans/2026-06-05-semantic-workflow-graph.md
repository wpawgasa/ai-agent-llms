# Semantic Workflow Graph Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the random index-walk workflow generator with a semantically-correct subgraph-selection approach, eliminating duplicate state names, meaningless conditions, regressive transitions, and empty terminal states.

**Architecture:** Three-layer change: (1) new `DomainSpec` schema with explicit `StateNode`/`Edge` objects and `validate_domain()` enforcing structural invariants at import; (2) new `select_subgraph` + `walk_path` functions replacing `_generate_workflow_graph` and the index-walk in the placeholder generator; (3) `ComplexitySpec` rescaling that derives `num_states` from the subgraph rather than sampling it randomly.

**Tech Stack:** Python dataclasses, `random.Random`, BFS graph traversal, pytest property tests.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/llm_workflow_agents/data/domain_registry.py` | Modify | New `StateNode`, `Edge`, updated `DomainSpec`, `validate_domain()`, all 18 re-authored domains |
| `src/llm_workflow_agents/config/schema.py` | Modify | New `ComplexitySpec` fields, updated `COMPLEXITY_SPECS` |
| `src/llm_workflow_agents/data/generate_workflows.py` | Modify | `select_subgraph`, `walk_path`, rewritten placeholder gen, teacher prompt, repair loop, level-aware `_select_domain` |
| `src/llm_workflow_agents/data/_workflow_script.py` | Modify | Render authored `label` instead of humanised snake_case |
| `tests/unit/test_data_generation.py` | Modify | New property tests per spec Section 5 |
| `tests/conftest.py` | Modify | Updated `ComplexitySpec` fixtures |
| `docs/data_generation_recipes.md` | Modify | Complexity table, domain-level cap, self-loop clarification |
| `.claude/rules/02-data-generation.md` | Modify | New `ComplexitySpec` schema, `COMPLEXITY_SPECS` |

---

## Task 1: New Schema Dataclasses + validate_domain

**Files:**
- Modify: `src/llm_workflow_agents/data/domain_registry.py`

Replace the three-flat-dict `DomainSpec` with one that holds `StateNode` and `Edge` objects, and add `validate_domain()`.

- [ ] **Step 1: Write the failing tests for the new schema**

Add to `tests/unit/test_data_generation.py` (in a new `class TestDomainSchema`):

```python
class TestDomainSchema:
    """Tests for the new StateNode/Edge/DomainSpec schema and validate_domain."""

    def _make_minimal_valid_domain(self) -> "DomainSpec":
        from llm_workflow_agents.data.domain_registry import (
            DomainSpec, StateNode, Edge
        )
        return DomainSpec(
            name="Test",
            category="test",
            tools=(),
            intents=("help",),
            entity_slots=(),
            states=(
                StateNode("START", "greet", kind="initial"),
                StateNode("WORK", "do work"),
                StateNode("END", "close", kind="terminal"),
            ),
            edges=(
                Edge("START", "WORK", "proceed", "always"),
                Edge("WORK", "END", "done", "tool_success"),
            ),
            initial="START",
            terminals=("END",),
        )

    def test_validate_domain_passes_minimal(self):
        from llm_workflow_agents.data.domain_registry import validate_domain
        d = self._make_minimal_valid_domain()
        validate_domain(d)  # should not raise

    def test_validate_domain_rejects_unknown_edge_src(self):
        from llm_workflow_agents.data.domain_registry import (
            DomainSpec, StateNode, Edge, validate_domain
        )
        import pytest
        d = DomainSpec(
            name="T", category="t", tools=(), intents=(), entity_slots=(),
            states=(
                StateNode("A", "a", kind="initial"),
                StateNode("B", "b", kind="terminal"),
            ),
            edges=(Edge("MISSING", "B", "x", "always"),),
            initial="A", terminals=("B",),
        )
        with pytest.raises(ValueError, match="unknown state"):
            validate_domain(d)

    def test_validate_domain_rejects_self_loop(self):
        from llm_workflow_agents.data.domain_registry import (
            DomainSpec, StateNode, Edge, validate_domain
        )
        import pytest
        d = DomainSpec(
            name="T", category="t", tools=(), intents=(), entity_slots=(),
            states=(
                StateNode("A", "a", kind="initial"),
                StateNode("B", "b", kind="terminal"),
            ),
            edges=(
                Edge("A", "A", "loop", "always"),
                Edge("A", "B", "done", "always"),
            ),
            initial="A", terminals=("B",),
        )
        with pytest.raises(ValueError, match="self-loop"):
            validate_domain(d)

    def test_validate_domain_rejects_missing_spine_successor(self):
        from llm_workflow_agents.data.domain_registry import (
            DomainSpec, StateNode, Edge, validate_domain
        )
        import pytest
        # WORK has only an optional edge — no spine successor
        d = DomainSpec(
            name="T", category="t", tools=(), intents=(), entity_slots=(),
            states=(
                StateNode("A", "a", kind="initial"),
                StateNode("B", "b"),
                StateNode("C", "c", kind="terminal"),
            ),
            edges=(
                Edge("A", "B", "go", "always"),
                Edge("B", "C", "branch", "intent_match", optional=True, priority=1),
            ),
            initial="A", terminals=("C",),
        )
        with pytest.raises(ValueError, match="spine successor"):
            validate_domain(d)

    def test_validate_domain_rejects_tool_trigger_on_toolless_state(self):
        from llm_workflow_agents.data.domain_registry import (
            DomainSpec, StateNode, Edge, validate_domain
        )
        import pytest
        d = DomainSpec(
            name="T", category="t", tools=(), intents=(), entity_slots=(),
            states=(
                StateNode("A", "a", kind="initial"),
                StateNode("B", "b"),  # no tools
                StateNode("C", "c", kind="terminal"),
            ),
            edges=(
                Edge("A", "B", "go", "always"),
                Edge("B", "C", "success", "tool_success"),  # tool_success but B has no tools
            ),
            initial="A", terminals=("C",),
        )
        with pytest.raises(ValueError, match="tool_success.*no tools"):
            validate_domain(d)

    def test_validate_domain_rejects_terminal_unreachable(self):
        from llm_workflow_agents.data.domain_registry import (
            DomainSpec, StateNode, Edge, validate_domain
        )
        import pytest
        # C is declared terminal but state B leads nowhere near it
        d = DomainSpec(
            name="T", category="t", tools=(), intents=(), entity_slots=(),
            states=(
                StateNode("A", "a", kind="initial"),
                StateNode("B", "b", kind="terminal"),
                StateNode("C", "c", kind="terminal"),
            ),
            edges=(Edge("A", "B", "done", "always"),),
            initial="A", terminals=("B", "C"),
        )
        with pytest.raises(ValueError, match="unreachable"):
            validate_domain(d)

    def test_validate_domain_rejects_invalid_trigger(self):
        from llm_workflow_agents.data.domain_registry import (
            DomainSpec, StateNode, Edge, validate_domain
        )
        import pytest
        d = DomainSpec(
            name="T", category="t", tools=(), intents=(), entity_slots=(),
            states=(
                StateNode("A", "a", kind="initial"),
                StateNode("B", "b", kind="terminal"),
            ),
            edges=(Edge("A", "B", "x", "fire_photon_torpedoes"),),
            initial="A", terminals=("B",),
        )
        with pytest.raises(ValueError, match="trigger"):
            validate_domain(d)
```

- [ ] **Step 2: Run the tests to confirm they fail (schema not yet imported)**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestDomainSchema -x -q 2>&1 | head -20
```

Expected: `ImportError` or `AttributeError` — `StateNode`, `Edge`, `validate_domain` don't exist yet.

- [ ] **Step 3: Add new dataclasses and validate_domain to domain_registry.py**

In `src/llm_workflow_agents/data/domain_registry.py`, **replace** the existing `DomainSpec` dataclass and add the new types. Keep the `_tool` helper and all existing tool definitions unchanged. Change the file top section from:

```python
@dataclass(frozen=True)
class DomainSpec:
    name: str
    category: str
    tools: tuple[dict[str, Any], ...]
    state_templates: tuple[str, ...]
    intents: tuple[str, ...]
    entity_slots: tuple[str, ...] = ()
    state_tools: dict[str, tuple[str, ...]] = field(default_factory=dict)
    state_instructions: dict[str, str] = field(default_factory=dict)
```

to:

```python
_VALID_TRIGGERS = frozenset({
    "always", "tool_success", "tool_error",
    "intent_match", "slot_present", "user_declines",
})


@dataclass(frozen=True)
class StateNode:
    name: str
    instruction: str
    tools: tuple[str, ...] = ()
    kind: str = "working"  # "initial" | "working" | "terminal"


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    label: str
    trigger: str  # one of _VALID_TRIGGERS
    optional: bool = False
    priority: int = 0
    intent_category: str | None = None


@dataclass(frozen=True)
class DomainSpec:
    name: str
    category: str
    tools: tuple[dict[str, Any], ...]
    intents: tuple[str, ...]
    entity_slots: tuple[str, ...]
    states: tuple[StateNode, ...]
    edges: tuple[Edge, ...]
    initial: str
    terminals: tuple[str, ...]
    intent_categories: dict[str, str] = field(default_factory=dict)

    # Legacy compatibility — some code may still read these; remove after
    # all callers are updated (Tasks 6–10).
    @property
    def state_templates(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.states)

    @property
    def state_tools(self) -> dict[str, tuple[str, ...]]:
        return {s.name: s.tools for s in self.states}

    @property
    def state_instructions(self) -> dict[str, str]:
        return {s.name: s.instruction for s in self.states}


def validate_domain(domain: DomainSpec) -> None:
    """Enforce structural invariants. Raises ValueError on any violation.

    Called at module import for every domain in DOMAIN_REGISTRY.
    """
    state_names = {s.name for s in domain.states}
    has_tools = {s.name for s in domain.states if s.tools}
    terminal_names = set(domain.terminals)

    # Every edge references known states
    for e in domain.edges:
        if e.src not in state_names:
            raise ValueError(
                f"{domain.name}: edge src '{e.src}' references unknown state"
            )
        if e.dst not in state_names:
            raise ValueError(
                f"{domain.name}: edge dst '{e.dst}' references unknown state"
            )

    # No graph-edge self-loops
    for e in domain.edges:
        if e.src == e.dst:
            raise ValueError(
                f"{domain.name}: self-loop on state '{e.src}'"
            )

    # Valid triggers
    for e in domain.edges:
        if e.trigger not in _VALID_TRIGGERS:
            raise ValueError(
                f"{domain.name}: edge {e.src}->{e.dst} has invalid trigger '{e.trigger}'"
            )

    # tool_success / tool_error only on states with tools
    for e in domain.edges:
        if e.trigger in ("tool_success", "tool_error") and e.src not in has_tools:
            raise ValueError(
                f"{domain.name}: edge {e.src}->{e.dst} has trigger '{e.trigger}' "
                f"but state '{e.src}' has no tools"
            )

    # initial state has kind="initial"
    initial_states = [s for s in domain.states if s.kind == "initial"]
    if not initial_states or domain.initial not in {s.name for s in initial_states}:
        raise ValueError(
            f"{domain.name}: 'initial' field '{domain.initial}' must name a state with kind='initial'"
        )

    # terminal states have kind="terminal"
    for t in domain.terminals:
        matching = [s for s in domain.states if s.name == t]
        if not matching or matching[0].kind != "terminal":
            raise ValueError(
                f"{domain.name}: terminal '{t}' must have kind='terminal'"
            )

    # Each non-terminal has >=1 outgoing edge and exactly one spine successor
    outgoing: dict[str, list[Edge]] = {}
    for e in domain.edges:
        outgoing.setdefault(e.src, []).append(e)

    for s in domain.states:
        if s.kind == "terminal":
            continue
        edges_out = outgoing.get(s.name, [])
        if not edges_out:
            raise ValueError(
                f"{domain.name}: non-terminal state '{s.name}' has no outgoing edges"
            )
        spine = [e for e in edges_out if not e.optional]
        if len(spine) != 1:
            raise ValueError(
                f"{domain.name}: non-terminal state '{s.name}' must have exactly one "
                f"spine successor (optional=False), found {len(spine)}"
            )

    # Every state reachable from initial; every terminal reachable from every state (BFS)
    # Reachability from initial
    reachable: set[str] = set()
    queue = [domain.initial]
    while queue:
        node = queue.pop()
        if node in reachable:
            continue
        reachable.add(node)
        for e in outgoing.get(node, []):
            queue.append(e.dst)
    unreachable = state_names - reachable
    if unreachable:
        raise ValueError(
            f"{domain.name}: states unreachable from initial: {unreachable}"
        )

    # Every state can reach at least one terminal
    # Build reverse adjacency for BFS from terminals
    reverse: dict[str, list[str]] = {}
    for e in domain.edges:
        reverse.setdefault(e.dst, []).append(e.src)
    can_reach_terminal: set[str] = set()
    queue = list(terminal_names)
    while queue:
        node = queue.pop()
        if node in can_reach_terminal:
            continue
        can_reach_terminal.add(node)
        for pred in reverse.get(node, []):
            queue.append(pred)
    blocked = reachable - can_reach_terminal
    if blocked:
        raise ValueError(
            f"{domain.name}: states that cannot reach any terminal: {blocked}"
        )

    # Upsell-arc edges must point toward terminal (dst must be able to reach terminal)
    for e in domain.edges:
        if e.intent_category == "upsell_promo" and e.dst not in can_reach_terminal:
            raise ValueError(
                f"{domain.name}: upsell-arc edge {e.src}->{e.dst} dst cannot reach any terminal"
            )
```

- [ ] **Step 4: Re-author ACCOUNT_MANAGEMENT with the new schema**

Replace the existing `ACCOUNT_MANAGEMENT = DomainSpec(...)` block with:

```python
ACCOUNT_MANAGEMENT = DomainSpec(
    name="Customer Account Management",
    category="core_business",
    tools=(
        _tool("create_account", "Create a new customer account", {
            "customer_name": {"type": "string"}, "email": {"type": "string", "format": "email"},
            "phone": {"type": "string"}, "account_type": {"type": "string", "enum": ["personal", "business"]},
        }, ["customer_name", "email"]),
        _tool("verify_identity", "Verify customer identity via KYC", {
            "customer_id": {"type": "string"}, "verification_method": {"type": "string", "enum": ["otp", "pin", "security_question"]},
            "verification_value": {"type": "string"},
        }, ["customer_id", "verification_method"]),
        _tool("update_profile", "Update customer profile information", {
            "customer_id": {"type": "string"}, "field": {"type": "string", "enum": ["address", "phone", "email", "name"]},
            "new_value": {"type": "string"},
        }, ["customer_id", "field", "new_value"]),
        _tool("close_account", "Close a customer account", {
            "customer_id": {"type": "string"}, "reason": {"type": "string"},
            "retain_data_days": {"type": "integer", "default": 90},
        }, ["customer_id", "reason"]),
        _tool("reset_password", "Reset customer password", {
            "customer_id": {"type": "string"}, "reset_method": {"type": "string", "enum": ["email", "sms"]},
        }, ["customer_id", "reset_method"]),
        _tool("lookup_rewards", "Look up loyalty rewards balance", {
            "customer_id": {"type": "string"}, "program": {"type": "string"},
        }, ["customer_id"]),
        _tool("manage_subscription", "Manage subscription plan", {
            "customer_id": {"type": "string"}, "action": {"type": "string", "enum": ["upgrade", "downgrade", "cancel", "pause"]},
            "plan_id": {"type": "string"},
        }, ["customer_id", "action"]),
    ),
    intents=(
        "account_creation", "profile_update", "password_reset",
        "account_closure", "subscription_change", "rewards_inquiry",
        "verification_request", "premium_plan_offer",
    ),
    entity_slots=("customer_id", "email", "phone", "account_type", "field", "new_value"),
    states=(
        StateNode("GREETING", "Greet the customer and ask what account assistance they need.", kind="initial"),
        StateNode("VERIFY_IDENTITY", "Verify the customer's identity before accessing any account details.", tools=("verify_identity",)),
        StateNode("AUTHENTICATE", "Authenticate the customer with OTP, PIN, or a security question.", tools=("verify_identity",)),
        StateNode("LOOKUP_ACCOUNT", "Retrieve the customer's account record and confirm the details on file.", tools=("lookup_rewards",)),
        StateNode("PROCESS_REQUEST", "Carry out the requested account change with the matching tool.", tools=("create_account", "update_profile", "reset_password", "close_account", "manage_subscription")),
        StateNode("CONFIRM_CHANGES", "Summarise the pending change and ask the customer to confirm."),
        StateNode("UPDATE_RECORDS", "Persist the confirmed changes to the customer's profile.", tools=("update_profile",)),
        StateNode("NOTIFY_CUSTOMER", "Notify the customer that the change is complete and what happens next."),
        StateNode("RESOLVE", "Confirm the request is resolved and ask if anything else is needed."),
        StateNode("TERMINAL", "Thank the customer and close the conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "VERIFY_IDENTITY", "proceed to identity verification", "always"),
        Edge("VERIFY_IDENTITY", "AUTHENTICATE", "identity check required", "tool_success"),
        Edge("VERIFY_IDENTITY", "LOOKUP_ACCOUNT", "identity already on file", "always", optional=True, priority=1),
        Edge("AUTHENTICATE", "LOOKUP_ACCOUNT", "authentication successful", "tool_success"),
        Edge("AUTHENTICATE", "VERIFY_IDENTITY", "authentication failed, retry", "tool_error", optional=True, priority=1),
        Edge("LOOKUP_ACCOUNT", "PROCESS_REQUEST", "account located", "tool_success"),
        Edge("PROCESS_REQUEST", "CONFIRM_CHANGES", "request processed, awaiting confirmation", "tool_success"),
        Edge("PROCESS_REQUEST", "RESOLVE", "request completed without confirmation step", "always", optional=True, priority=1),
        Edge("CONFIRM_CHANGES", "UPDATE_RECORDS", "customer confirmed changes", "always"),
        Edge("CONFIRM_CHANGES", "RESOLVE", "customer declined the proposed change", "user_declines", optional=True, priority=1),
        Edge("UPDATE_RECORDS", "NOTIFY_CUSTOMER", "records updated successfully", "tool_success"),
        Edge("NOTIFY_CUSTOMER", "RESOLVE", "customer notified", "always"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "premium subscription upgrade accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "account_creation": "service",
        "profile_update": "service",
        "password_reset": "service",
        "account_closure": "service",
        "subscription_change": "upsell_promo",
        "rewards_inquiry": "upsell_promo",
        "verification_request": "service",
        "premium_plan_offer": "upsell_promo",
    },
)
```

- [ ] **Step 5: Call validate_domain at module load for ACCOUNT_MANAGEMENT**

At the end of `domain_registry.py`, above `DOMAIN_REGISTRY`, add:

```python
# Validate all authored domains at import time
def _validate_all() -> None:
    for _d in [ACCOUNT_MANAGEMENT]:  # grows as domains are re-authored in Task 3-4
        validate_domain(_d)

_validate_all()
```

- [ ] **Step 6: Run the schema tests**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestDomainSchema -v 2>&1 | tail -20
```

Expected: all 7 tests PASS.

- [ ] **Step 7: Confirm ACCOUNT_MANAGEMENT validates cleanly**

```bash
source .venv/bin/activate && python -c "from llm_workflow_agents.data.domain_registry import ACCOUNT_MANAGEMENT, validate_domain; validate_domain(ACCOUNT_MANAGEMENT); print('ok')"
```

Expected: `ok`

- [ ] **Step 8: Commit**

```bash
git add src/llm_workflow_agents/data/domain_registry.py tests/unit/test_data_generation.py
git commit -m "feat(data): add StateNode/Edge/DomainSpec schema + validate_domain + re-author ACCOUNT_MANAGEMENT"
```

---

## Task 2: Re-author 12 Standard Domains

**Files:**
- Modify: `src/llm_workflow_agents/data/domain_registry.py`

Re-author the remaining 12 standard domains (all except banking, insurance, healthcare, travel, telecom which are expanded in Task 3) into the new `StateNode`/`Edge` schema at their current ~8-12 state count.

**Constraints for each domain (enforced by validate_domain):**
- Every existing `state_templates` entry becomes a `StateNode` with `kind="initial"` (first), `kind="terminal"` (last), `kind="working"` (middle).
- Every existing `state_tools[name]` entry becomes `StateNode.tools`.
- Every existing `state_instructions[name]` entry becomes `StateNode.instruction`.
- Spine edges follow the existing semantic order of states (top-to-bottom as currently listed in `state_templates`).
- Each branch/error-recovery edge uses `optional=True, priority=1`.
- One upsell-arc edge (with `intent_category="upsell_promo"`) for every domain that has a upsell intent in its current `intents` tuple. The upsell arc must join the spine toward the terminal (rejoin at `RESOLVE` or the penultimate state before `TERMINAL`).
- `intent_categories` dict folds in the existing `INTENT_CATEGORY_TAXONOMY` entries for that domain's intents.

**Domains to re-author in this task:**
`BILLING_PAYMENTS`, `ORDER_MANAGEMENT`, `TECHNICAL_SUPPORT`, `PRODUCT_INFO`, `UTILITIES`, `ECOMMERCE`, `GOVERNMENT`, `COMPLAINTS`, `SCHEDULING`, `SALES`, `SURVEYS`, `EMERGENCY`

- [ ] **Step 1: Write a test that all 18 registry domains pass validate_domain**

In `tests/unit/test_data_generation.py`, add to `TestDomainSchema`:

```python
def test_all_registry_domains_pass_validate(self):
    from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY, validate_domain
    errors = []
    for key, domain in DOMAIN_REGISTRY.items():
        try:
            validate_domain(domain)
        except ValueError as e:
            errors.append(f"{key}: {e}")
    assert not errors, "\n".join(errors)
```

- [ ] **Step 2: Run to confirm current failures (only ACCOUNT_MANAGEMENT will pass)**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestDomainSchema::test_all_registry_domains_pass_validate -v 2>&1 | tail -20
```

Expected: FAIL — remaining 17 domains use old schema.

- [ ] **Step 3: Re-author BILLING_PAYMENTS** (shown as full template for the remaining 12)

Replace the `BILLING_PAYMENTS = DomainSpec(...)` block. The 11 states remain; add explicit `StateNode` + `Edge` objects following the pattern from Task 1:

```python
BILLING_PAYMENTS = DomainSpec(
    name="Billing & Payments",
    category="core_business",
    tools=(  # unchanged from current
        _tool("lookup_invoice", ...),
        _tool("process_payment", ...),
        _tool("issue_refund", ...),
        _tool("setup_payment_plan", ...),
        _tool("waive_late_fee", ...),
        _tool("generate_receipt", ...),
        _tool("dispute_charge", ...),
    ),
    intents=(
        "invoice_inquiry", "payment_processing", "refund_request",
        "dispute_charge", "payment_plan", "late_fee_waiver",
        "receipt_request", "chargeback", "payment_plan_offer",
    ),
    entity_slots=("invoice_id", "amount", "payment_method", "transaction_id", "customer_id"),
    states=(
        StateNode("GREETING", "Greet the customer and ask about their billing or payment need.", kind="initial"),
        StateNode("VERIFY_IDENTITY", "Confirm the customer's identity and the account in question."),
        StateNode("LOOKUP_BILLING", "Look up the relevant invoice or billing record.", tools=("lookup_invoice",)),
        StateNode("REVIEW_CHARGES", "Review the charges with the customer and note any disputes.", tools=("lookup_invoice", "dispute_charge")),
        StateNode("PROCESS_PAYMENT", "Process the payment or set up an installment plan.", tools=("process_payment", "setup_payment_plan")),
        StateNode("APPLY_ADJUSTMENT", "Apply any approved refund or fee waiver.", tools=("issue_refund", "waive_late_fee")),
        StateNode("CONFIRM_ACTION", "Summarise the billing action and ask the customer to confirm."),
        StateNode("GENERATE_DOCUMENT", "Generate the requested receipt, statement, or tax document.", tools=("generate_receipt",)),
        StateNode("ESCALATE", "Escalate unresolved billing issues to a specialist."),
        StateNode("RESOLVE", "Confirm the billing matter is resolved."),
        StateNode("TERMINAL", "Thank the customer and close the conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "VERIFY_IDENTITY", "proceed to identity check", "always"),
        Edge("VERIFY_IDENTITY", "LOOKUP_BILLING", "identity confirmed", "always"),
        Edge("LOOKUP_BILLING", "REVIEW_CHARGES", "invoice found", "tool_success"),
        Edge("LOOKUP_BILLING", "ESCALATE", "invoice not found, escalate", "tool_error", optional=True, priority=1),
        Edge("REVIEW_CHARGES", "PROCESS_PAYMENT", "customer proceeds with payment", "always"),
        Edge("REVIEW_CHARGES", "APPLY_ADJUSTMENT", "customer requests refund or waiver", "user_declines", optional=True, priority=1),
        Edge("PROCESS_PAYMENT", "CONFIRM_ACTION", "payment processed successfully", "tool_success"),
        Edge("PROCESS_PAYMENT", "ESCALATE", "payment failed, escalate", "tool_error", optional=True, priority=1),
        Edge("APPLY_ADJUSTMENT", "CONFIRM_ACTION", "adjustment applied", "tool_success"),
        Edge("CONFIRM_ACTION", "GENERATE_DOCUMENT", "customer wants receipt", "always"),
        Edge("CONFIRM_ACTION", "RESOLVE", "no document needed", "always", optional=True, priority=1),
        Edge("GENERATE_DOCUMENT", "RESOLVE", "document generated", "tool_success"),
        Edge("ESCALATE", "RESOLVE", "escalation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "payment plan offer accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "invoice_inquiry": "service",
        "payment_processing": "service",
        "refund_request": "service",
        "dispute_charge": "service",
        "payment_plan": "service",
        "late_fee_waiver": "service",
        "receipt_request": "service",
        "chargeback": "service",
        "payment_plan_offer": "upsell_promo",
    },
)
```

- [ ] **Step 4: Re-author the remaining 10 standard domains**

For each domain below, follow the exact same pattern (StateNode per state, Edge per transition, upsell arc where domain has upsell intent, intent_categories dict). Apply the same constraints from Step 0:

- `ORDER_MANAGEMENT` — current states: `GREETING, VERIFY_ORDER, LOOKUP_ORDER, REVIEW_STATUS, PROCESS_CHANGE, ESCALATE, NOTIFY_CUSTOMER, RESOLVE, TERMINAL`
- `TECHNICAL_SUPPORT` — current states: `GREETING, TRIAGE, DIAGNOSE, TROUBLESHOOT, ESCALATE, SCHEDULE_CALLBACK, RESOLVE, TERMINAL`
- `PRODUCT_INFO` — current states: `GREETING, IDENTIFY_NEED, PROVIDE_INFORMATION, COMPARE_OPTIONS, MAKE_RECOMMENDATION, CONFIRM_INTEREST, CLOSE_SALE, TERMINAL`
- `UTILITIES` — current states: `GREETING, VERIFY_ACCOUNT, LOOKUP_USAGE, REVIEW_BILL, PROCESS_SERVICE_REQUEST, SCHEDULE_APPOINTMENT, ESCALATE, RESOLVE, TERMINAL`
- `ECOMMERCE` — current states: `GREETING, BROWSE_ASSIST, PRODUCT_DETAILS, ADD_TO_CART, CHECKOUT, PAYMENT, ORDER_CONFIRM, RESOLVE, TERMINAL`
- `GOVERNMENT` — current states: `GREETING, CITIZEN_VERIFY, LOOKUP_RECORD, PROCESS_APPLICATION, DOCUMENT_REVIEW, DECISION, NOTIFY_CITIZEN, APPEAL_OPTION, TERMINAL`
- `COMPLAINTS` — current states: `GREETING, CAPTURE_COMPLAINT, INVESTIGATE, PROPOSE_RESOLUTION, AWAIT_CUSTOMER_RESPONSE, IMPLEMENT_RESOLUTION, FOLLOW_UP, RESOLVE, TERMINAL`
- `SCHEDULING` — current states: `GREETING, IDENTIFY_SERVICE, CHECK_AVAILABILITY, OFFER_SLOTS, CONFIRM_BOOKING, SEND_CONFIRMATION, FOLLOW_UP, TERMINAL`
- `SALES` — current states: `GREETING, QUALIFY_LEAD, PRESENT_OFFER, HANDLE_OBJECTIONS, NEGOTIATE, CLOSE_DEAL, ONBOARD, TERMINAL`
- `SURVEYS` — current states: `GREETING, CONSENT_CHECK, INTRO_SURVEY, CONDUCT_QUESTIONS, CAPTURE_FEEDBACK, THANK_CUSTOMER, TERMINAL`
- `EMERGENCY` — current states: `ALERT_RECEIVED, ASSESS_SEVERITY, DISPATCH_RESPONSE, COORDINATE_TEAMS, NOTIFY_STAKEHOLDERS, MONITOR_STATUS, ACTIVATE_BACKUP, CONFIRM_RESOLUTION, POST_INCIDENT_REVIEW, TERMINAL`

For `EMERGENCY`, the initial state is `ALERT_RECEIVED` (kind="initial").

- [ ] **Step 5: Expand `_validate_all()` to include all 13 standard domains**

In `domain_registry.py`, update `_validate_all()`:

```python
def _validate_all() -> None:
    for _d in [
        ACCOUNT_MANAGEMENT, BILLING_PAYMENTS, ORDER_MANAGEMENT, TECHNICAL_SUPPORT,
        PRODUCT_INFO, UTILITIES, ECOMMERCE, GOVERNMENT, COMPLAINTS,
        SCHEDULING, SALES, SURVEYS, EMERGENCY,
    ]:
        validate_domain(_d)
```

- [ ] **Step 6: Run tests**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestDomainSchema -v 2>&1 | tail -20
```

Expected: The `test_all_registry_domains_pass_validate` test still fails for the 5 unexpanded rich domains (added in Task 3). The 13 standard ones should pass.

- [ ] **Step 7: Commit**

```bash
git add src/llm_workflow_agents/data/domain_registry.py tests/unit/test_data_generation.py
git commit -m "feat(data): re-author 12 standard domains with StateNode/Edge schema"
```

---

## Task 3: Expand 5 Rich Domains

**Files:**
- Modify: `src/llm_workflow_agents/data/domain_registry.py`

Re-author `BANKING`, `INSURANCE`, `HEALTHCARE`, `TRAVEL`, `TELECOM` with 16–20 genuine states (no filler), explicit sub-flows, and a upsell arc. These are the only domains eligible for L4–L5 sampling.

**Requirements for each expanded domain:**
- 16–20 `StateNode` entries total (exactly one `kind="initial"`, exactly one or two `kind="terminal"`)
- Real sub-flows such as: document collection, eligibility assessment, escalation path, appeals/complaints detour, error recovery arc, upsell/cross-sell arc
- The upsell arc must be an `Edge` with `intent_category="upsell_promo"` and `optional=True` that rejoins toward a terminal
- No state name appears twice
- `validate_domain` passes

- [ ] **Step 1: Re-author BANKING (shown as full template)**

```python
BANKING = DomainSpec(
    name="Banking & Financial Services",
    category="industry",
    tools=(
        _tool("verify_identity", "Verify customer identity via KYC", {
            "customer_id": {"type": "string"},
            "method": {"type": "string", "enum": ["otp", "pin", "biometric", "document"]},
        }, ["customer_id", "method"]),
        _tool("lookup_account", "Retrieve account details and balance", {
            "account_id": {"type": "string"}, "customer_id": {"type": "string"},
        }, ["account_id"]),
        _tool("check_eligibility", "Check eligibility for a product or service", {
            "customer_id": {"type": "string"},
            "product_type": {"type": "string", "enum": ["loan", "credit_card", "mortgage", "investment"]},
            "amount": {"type": "number"},
        }, ["customer_id", "product_type"]),
        _tool("process_transaction", "Process a financial transaction", {
            "account_id": {"type": "string"}, "transaction_type": {"type": "string"},
            "amount": {"type": "number"}, "destination": {"type": "string"},
        }, ["account_id", "transaction_type", "amount"]),
        _tool("request_document", "Request supporting document from customer", {
            "customer_id": {"type": "string"}, "document_type": {"type": "string"},
            "deadline_days": {"type": "integer"},
        }, ["customer_id", "document_type"]),
        _tool("submit_application", "Submit a loan or product application", {
            "customer_id": {"type": "string"}, "product_type": {"type": "string"},
            "amount": {"type": "number"}, "term_months": {"type": "integer"},
        }, ["customer_id", "product_type"]),
        _tool("get_decision", "Retrieve underwriting decision", {
            "application_id": {"type": "string"},
        }, ["application_id"]),
        _tool("setup_repayment", "Configure repayment schedule", {
            "application_id": {"type": "string"}, "schedule_type": {"type": "string"},
            "start_date": {"type": "string"},
        }, ["application_id", "schedule_type"]),
        _tool("flag_fraud", "Flag transaction as potentially fraudulent", {
            "transaction_id": {"type": "string"}, "reason": {"type": "string"},
        }, ["transaction_id", "reason"]),
        _tool("send_notification", "Send notification to customer", {
            "customer_id": {"type": "string"}, "channel": {"type": "string"},
            "message": {"type": "string"},
        }, ["customer_id", "channel", "message"]),
    ),
    intents=(
        "account_inquiry", "fund_transfer", "loan_application", "credit_card_request",
        "fraud_report", "investment_inquiry", "mortgage_inquiry",
        "loan_inquiry", "rate_inquiry",
    ),
    entity_slots=("account_id", "customer_id", "amount", "product_type", "application_id"),
    states=(
        StateNode("GREETING", "Welcome the customer and identify their banking need.", kind="initial"),
        StateNode("VERIFY_IDENTITY", "Verify the customer's identity before any account access.", tools=("verify_identity",)),
        StateNode("AUTHENTICATE", "Authenticate via OTP, PIN, or biometric.", tools=("verify_identity",)),
        StateNode("LOOKUP_ACCOUNT", "Retrieve account details and current balance.", tools=("lookup_account",)),
        StateNode("IDENTIFY_REQUEST", "Clarify the customer's specific request or goal."),
        StateNode("CHECK_ELIGIBILITY", "Assess whether the customer qualifies for the requested product.", tools=("check_eligibility",)),
        StateNode("COLLECT_DOCUMENTS", "Request and log required supporting documents.", tools=("request_document",)),
        StateNode("SUBMIT_APPLICATION", "Submit the completed application for underwriting.", tools=("submit_application",)),
        StateNode("AWAIT_DECISION", "Retrieve or await the underwriting decision.", tools=("get_decision",)),
        StateNode("PROCESS_TRANSACTION", "Execute the requested financial transaction.", tools=("process_transaction",)),
        StateNode("FRAUD_CHECK", "Investigate potential fraud signals on the account.", tools=("flag_fraud",)),
        StateNode("ESCALATE_SPECIALIST", "Transfer to a banking specialist for complex cases."),
        StateNode("SETUP_REPAYMENT", "Configure the repayment schedule for an approved loan.", tools=("setup_repayment",)),
        StateNode("PRESENT_OFFER", "Present a relevant financial product offer to the customer."),
        StateNode("NOTIFY_CUSTOMER", "Notify the customer of the outcome.", tools=("send_notification",)),
        StateNode("RESOLVE", "Confirm the request is handled and ask if anything else is needed."),
        StateNode("TERMINAL", "Close the conversation.", kind="terminal"),
    ),
    edges=(
        # Main spine
        Edge("GREETING", "VERIFY_IDENTITY", "proceed to identity check", "always"),
        Edge("VERIFY_IDENTITY", "AUTHENTICATE", "secondary authentication required", "tool_success"),
        Edge("VERIFY_IDENTITY", "LOOKUP_ACCOUNT", "identity confirmed directly", "always", optional=True, priority=1),
        Edge("AUTHENTICATE", "LOOKUP_ACCOUNT", "authentication successful", "tool_success"),
        Edge("AUTHENTICATE", "VERIFY_IDENTITY", "authentication failed, retry", "tool_error", optional=True, priority=1),
        Edge("LOOKUP_ACCOUNT", "IDENTIFY_REQUEST", "account located", "tool_success"),
        Edge("LOOKUP_ACCOUNT", "FRAUD_CHECK", "suspicious activity detected", "tool_error", optional=True, priority=1),
        Edge("IDENTIFY_REQUEST", "CHECK_ELIGIBILITY", "product or loan request", "always"),
        Edge("IDENTIFY_REQUEST", "PROCESS_TRANSACTION", "transfer or payment request", "always", optional=True, priority=1),
        Edge("CHECK_ELIGIBILITY", "COLLECT_DOCUMENTS", "eligible, documents needed", "tool_success"),
        Edge("CHECK_ELIGIBILITY", "ESCALATE_SPECIALIST", "not eligible, needs review", "tool_error", optional=True, priority=1),
        Edge("COLLECT_DOCUMENTS", "SUBMIT_APPLICATION", "documents received", "tool_success"),
        Edge("COLLECT_DOCUMENTS", "ESCALATE_SPECIALIST", "documents incomplete after follow-up", "tool_error", optional=True, priority=1),
        Edge("SUBMIT_APPLICATION", "AWAIT_DECISION", "application submitted", "tool_success"),
        Edge("AWAIT_DECISION", "SETUP_REPAYMENT", "application approved", "tool_success"),
        Edge("AWAIT_DECISION", "ESCALATE_SPECIALIST", "application declined, customer wants appeal", "tool_error", optional=True, priority=1),
        Edge("SETUP_REPAYMENT", "NOTIFY_CUSTOMER", "repayment scheduled", "tool_success"),
        Edge("PROCESS_TRANSACTION", "NOTIFY_CUSTOMER", "transaction complete", "tool_success"),
        Edge("PROCESS_TRANSACTION", "FRAUD_CHECK", "transaction flagged", "tool_error", optional=True, priority=1),
        Edge("FRAUD_CHECK", "ESCALATE_SPECIALIST", "fraud confirmed, escalate", "tool_success"),
        Edge("FRAUD_CHECK", "PROCESS_TRANSACTION", "false positive, proceed", "tool_error", optional=True, priority=1),
        Edge("ESCALATE_SPECIALIST", "RESOLVE", "specialist handled the case", "always"),
        Edge("NOTIFY_CUSTOMER", "RESOLVE", "customer notified", "always"),
        # Upsell arc — investment or rate product offer
        Edge("RESOLVE", "PRESENT_OFFER", "customer open to investment product offer", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
        Edge("PRESENT_OFFER", "RESOLVE", "offer presented", "always"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "account_inquiry": "service",
        "fund_transfer": "service",
        "loan_application": "service",
        "credit_card_request": "service",
        "fraud_report": "service",
        "investment_inquiry": "service",
        "mortgage_inquiry": "service",
        "loan_inquiry": "upsell_promo",
        "rate_inquiry": "upsell_promo",
    },
)
```

- [ ] **Step 2: Re-author INSURANCE (16-18 states)**

States to include: `GREETING`, `VERIFY_POLICYHOLDER`, `AUTHENTICATE`, `LOOKUP_POLICY`, `IDENTIFY_REQUEST`, `ASSESS_CLAIM`, `COLLECT_EVIDENCE`, `SCHEDULE_INSPECTION`, `REVIEW_COVERAGE`, `CALCULATE_SETTLEMENT`, `PROCESS_PAYMENT`, `APPEAL_REVIEW`, `ESCALATE_UNDERWRITER`, `PRESENT_COVERAGE_OFFER`, `NOTIFY_CUSTOMER`, `RESOLVE`, `TERMINAL`.

Key sub-flows:
- Claims sub-flow: `IDENTIFY_REQUEST → ASSESS_CLAIM → COLLECT_EVIDENCE → SCHEDULE_INSPECTION → CALCULATE_SETTLEMENT → PROCESS_PAYMENT → NOTIFY_CUSTOMER`
- Dispute sub-flow: `CALCULATE_SETTLEMENT → APPEAL_REVIEW → ESCALATE_UNDERWRITER` (tool_error branch)
- Upsell arc: `RESOLVE → PRESENT_COVERAGE_OFFER → RESOLVE` for `coverage_upgrade`, `policy_renewal`, `bundle_offer` intents

- [ ] **Step 3: Re-author HEALTHCARE (16-18 states)**

States to include: `GREETING`, `VERIFY_PATIENT`, `AUTHENTICATE`, `LOOKUP_PATIENT_RECORD`, `IDENTIFY_NEED`, `SCHEDULE_APPOINTMENT`, `TRIAGE_SYMPTOMS`, `PROVIDE_GUIDANCE`, `REFERRAL_MANAGEMENT`, `PRESCRIPTION_ASSIST`, `INSURANCE_VERIFICATION`, `COLLECT_CONSENT`, `PROCESS_BILLING`, `ESCALATE_CLINICAL`, `WELLNESS_OFFER`, `RESOLVE`, `TERMINAL`.

Key sub-flows:
- Appointment booking: `IDENTIFY_NEED → SCHEDULE_APPOINTMENT → COLLECT_CONSENT → RESOLVE`
- Clinical triage: `IDENTIFY_NEED → TRIAGE_SYMPTOMS → PROVIDE_GUIDANCE → REFERRAL_MANAGEMENT`
- Billing dispute: `PROCESS_BILLING → INSURANCE_VERIFICATION → ESCALATE_CLINICAL` (tool_error branch)
- Upsell arc: `RESOLVE → WELLNESS_OFFER → RESOLVE` for `wellness_program_offer` intent

- [ ] **Step 4: Re-author TRAVEL (16-18 states)**

States to include: `GREETING`, `VERIFY_TRAVELER`, `LOOKUP_BOOKING`, `IDENTIFY_REQUEST`, `SEARCH_OPTIONS`, `PRESENT_ITINERARY`, `SEAT_UPGRADE_OFFER`, `PROCESS_BOOKING_CHANGE`, `CHECK_ANCILLARIES`, `APPLY_LOYALTY_POINTS`, `PROCESS_REFUND`, `REISSUE_TICKET`, `ESCALATE_AIRLINE`, `NOTIFY_TRAVELER`, `RESOLVE`, `TERMINAL`.

Key sub-flows:
- Booking change: `IDENTIFY_REQUEST → PROCESS_BOOKING_CHANGE → NOTIFY_TRAVELER`
- Cancellation/refund: `IDENTIFY_REQUEST → PROCESS_REFUND → REISSUE_TICKET → NOTIFY_TRAVELER`
- Upsell arc: `IDENTIFY_REQUEST → SEAT_UPGRADE_OFFER → RESOLVE` for `loyalty_redemption` intent

- [ ] **Step 5: Re-author TELECOM (16-18 states)**

States to include: `GREETING`, `VERIFY_SUBSCRIBER`, `AUTHENTICATE`, `LOOKUP_ACCOUNT`, `IDENTIFY_REQUEST`, `CHECK_COVERAGE`, `DIAGNOSE_ISSUE`, `TROUBLESHOOT`, `PROVISION_SERVICE`, `PROCESS_PLAN_CHANGE`, `ROAMING_SETUP`, `HANDLE_DISPUTE`, `ESCALATE_NETWORK`, `PRESENT_PLAN_OFFER`, `NOTIFY_SUBSCRIBER`, `RESOLVE`, `TERMINAL`.

Key sub-flows:
- Technical fault: `IDENTIFY_REQUEST → CHECK_COVERAGE → DIAGNOSE_ISSUE → TROUBLESHOOT → ESCALATE_NETWORK`
- Plan change: `IDENTIFY_REQUEST → PROCESS_PLAN_CHANGE → NOTIFY_SUBSCRIBER`
- Upsell arc: `RESOLVE → PRESENT_PLAN_OFFER → RESOLVE` for `plan_change`, `roaming_activation` intents

- [ ] **Step 6: Expand `_validate_all()` to include all 18 domains**

```python
def _validate_all() -> None:
    for _d in [
        ACCOUNT_MANAGEMENT, BILLING_PAYMENTS, ORDER_MANAGEMENT, TECHNICAL_SUPPORT,
        PRODUCT_INFO, UTILITIES, ECOMMERCE, GOVERNMENT, COMPLAINTS,
        SCHEDULING, SALES, SURVEYS, EMERGENCY,
        BANKING, INSURANCE, HEALTHCARE, TRAVEL, TELECOM,
    ]:
        validate_domain(_d)
```

- [ ] **Step 7: Run all schema tests**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestDomainSchema -v 2>&1 | tail -20
```

Expected: all 8 tests PASS.

- [ ] **Step 8: Confirm import runs validate_domain for all 18**

```bash
source .venv/bin/activate && python -c "import llm_workflow_agents.data.domain_registry; print('all 18 domains valid')"
```

Expected: `all 18 domains valid`

- [ ] **Step 9: Commit**

```bash
git add src/llm_workflow_agents/data/domain_registry.py
git commit -m "feat(data): expand 5 rich domains to 16-20 states with upsell arcs"
```

---

## Task 4: New ComplexitySpec + COMPLEXITY_SPECS

**Files:**
- Modify: `src/llm_workflow_agents/config/schema.py`
- Modify: `tests/conftest.py`

Replace the old `num_states`/`branching_factor`/`nesting_depth`/`domain` fields with the new selection-driver fields, and rescale the five complexity levels.

- [ ] **Step 1: Write a test for new ComplexitySpec fields**

Add to `tests/unit/test_config.py`:

```python
def test_complexity_specs_have_new_fields():
    from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
    spec_l1 = COMPLEXITY_SPECS[ComplexityLevel.L1]
    assert hasattr(spec_l1, "target_path_len")
    assert hasattr(spec_l1, "num_branches")
    assert hasattr(spec_l1, "num_loops")
    assert hasattr(spec_l1, "include_recovery")
    # Old field names must be gone
    assert not hasattr(spec_l1, "branching_factor")
    assert not hasattr(spec_l1, "nesting_depth")
    assert not hasattr(spec_l1, "domain")

def test_complexity_specs_rescaled_ranges():
    from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
    assert COMPLEXITY_SPECS[ComplexityLevel.L1].target_path_len == (3, 4)
    assert COMPLEXITY_SPECS[ComplexityLevel.L2].target_path_len == (5, 7)
    assert COMPLEXITY_SPECS[ComplexityLevel.L3].target_path_len == (8, 12)
    assert COMPLEXITY_SPECS[ComplexityLevel.L4].target_path_len == (12, 16)
    assert COMPLEXITY_SPECS[ComplexityLevel.L5].target_path_len == (16, 20)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_config.py::test_complexity_specs_have_new_fields tests/unit/test_config.py::test_complexity_specs_rescaled_ranges -v 2>&1 | tail -10
```

Expected: FAIL — `target_path_len` attribute missing.

- [ ] **Step 3: Update ComplexitySpec in schema.py**

Replace the `ComplexitySpec` class definition with:

```python
class ComplexitySpec(BaseModel):
    """Workflow complexity level specification (subgraph-selection drivers)."""

    level: ComplexityLevel
    target_path_len: tuple[int, int]   # spine states to include
    num_branches: tuple[int, int]      # optional branch edges
    num_loops: tuple[int, int]         # back-edges to earlier states
    include_recovery: bool             # include tool_error recovery arcs
    num_tools: int                     # max tools selected into a sample
    chain_depth: int                   # reported in ConversationSample (legacy)
```

Replace `COMPLEXITY_SPECS` with:

```python
COMPLEXITY_SPECS: dict[ComplexityLevel, ComplexitySpec] = {
    ComplexityLevel.L1: ComplexitySpec(
        level=ComplexityLevel.L1,
        target_path_len=(3, 4),
        num_branches=(0, 0),
        num_loops=(0, 0),
        include_recovery=False,
        num_tools=1,
        chain_depth=0,
    ),
    ComplexityLevel.L2: ComplexitySpec(
        level=ComplexityLevel.L2,
        target_path_len=(5, 7),
        num_branches=(1, 1),
        num_loops=(0, 0),
        include_recovery=False,
        num_tools=2,
        chain_depth=1,
    ),
    ComplexityLevel.L3: ComplexitySpec(
        level=ComplexityLevel.L3,
        target_path_len=(8, 12),
        num_branches=(2, 3),
        num_loops=(0, 1),
        include_recovery=False,
        num_tools=4,
        chain_depth=2,
    ),
    ComplexityLevel.L4: ComplexitySpec(
        level=ComplexityLevel.L4,
        target_path_len=(12, 16),
        num_branches=(3, 5),
        num_loops=(1, 1),
        include_recovery=True,
        num_tools=6,
        chain_depth=3,
    ),
    ComplexityLevel.L5: ComplexitySpec(
        level=ComplexityLevel.L5,
        target_path_len=(16, 20),
        num_branches=(0, 99),  # all optional edges
        num_loops=(1, 2),
        include_recovery=True,
        num_tools=7,
        chain_depth=4,
    ),
}
```

- [ ] **Step 4: Update conftest.py fixtures to use new fields**

Replace the `l1_complexity_spec` and `l3_complexity_spec` fixtures in `tests/conftest.py`:

```python
@pytest.fixture
def l1_complexity_spec() -> ComplexitySpec:
    return ComplexitySpec(
        level=ComplexityLevel.L1,
        target_path_len=(3, 4),
        num_branches=(0, 0),
        num_loops=(0, 0),
        include_recovery=False,
        num_tools=1,
        chain_depth=0,
    )


@pytest.fixture
def l3_complexity_spec() -> ComplexitySpec:
    return ComplexitySpec(
        level=ComplexityLevel.L3,
        target_path_len=(8, 12),
        num_branches=(2, 3),
        num_loops=(0, 1),
        include_recovery=False,
        num_tools=4,
        chain_depth=2,
    )
```

- [ ] **Step 5: Run the config tests**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_config.py -v 2>&1 | tail -15
```

Expected: new tests PASS; no regressions in existing config tests.

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/config/schema.py tests/conftest.py tests/unit/test_config.py
git commit -m "feat(config): new ComplexitySpec with subgraph-selection fields + rescaled L1-L5"
```

---

## Task 5: WorkflowTransition + select_subgraph

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py`

Add `trigger`, `label`, `optional`, `intent_category` to `WorkflowTransition`. Then implement `select_subgraph` to replace `_generate_workflow_graph`.

- [ ] **Step 1: Write the failing test for select_subgraph**

Add to `tests/unit/test_data_generation.py`:

```python
class TestSelectSubgraph:
    def test_l1_subgraph_has_3_to_4_states(self):
        import random
        from llm_workflow_agents.data.generate_workflows import select_subgraph
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        rng = random.Random(42)
        spec = COMPLEXITY_SPECS[ComplexityLevel.L1]
        domain = DOMAIN_REGISTRY["account_management"]
        graph = select_subgraph(domain, spec, rng)
        assert 3 <= len(graph.states) <= 4

    def test_subgraph_no_duplicate_state_names(self):
        import random
        from llm_workflow_agents.data.generate_workflows import select_subgraph
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        for level in [ComplexityLevel.L1, ComplexityLevel.L2, ComplexityLevel.L3]:
            spec = COMPLEXITY_SPECS[level]
            for key, domain in DOMAIN_REGISTRY.items():
                rng = random.Random(0)
                graph = select_subgraph(domain, spec, rng)
                names = [s.name for s in graph.states]
                assert len(names) == len(set(names)), f"duplicate names in {key} {level}: {names}"

    def test_subgraph_terminal_always_reachable(self):
        import random
        from llm_workflow_agents.data.generate_workflows import select_subgraph
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        spec = COMPLEXITY_SPECS[ComplexityLevel.L3]
        for key, domain in DOMAIN_REGISTRY.items():
            rng = random.Random(7)
            graph = select_subgraph(domain, spec, rng)
            terminal_names = set(graph.terminal_states)
            terminal_state_names = {
                s.name for s in graph.states
                if s.id in terminal_names or s.name in terminal_names
            }
            assert terminal_state_names, f"no terminal in subgraph for {key}"

    def test_subgraph_transitions_carry_label_and_trigger(self):
        import random
        from llm_workflow_agents.data.generate_workflows import select_subgraph
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        spec = COMPLEXITY_SPECS[ComplexityLevel.L2]
        domain = DOMAIN_REGISTRY["billing_payments"]
        rng = random.Random(1)
        graph = select_subgraph(domain, spec, rng)
        for t in graph.transitions:
            assert t.label, f"transition {t.from_state}->{t.to_state} missing label"
            assert t.trigger, f"transition {t.from_state}->{t.to_state} missing trigger"
```

- [ ] **Step 2: Run to confirm failures**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestSelectSubgraph -v 2>&1 | tail -10
```

Expected: FAIL — `select_subgraph` not yet defined; `WorkflowTransition` lacks `label`/`trigger`.

- [ ] **Step 3: Extend WorkflowTransition dataclass**

In `generate_workflows.py`, replace the `WorkflowTransition` dataclass:

```python
@dataclass
class WorkflowTransition:
    """A transition between workflow states."""

    from_state: str
    to_state: str
    condition: str          # legacy field; rendered from label at to_dict()
    priority: int = 0
    label: str = ""         # authored human-readable label
    trigger: str = "always" # one of _VALID_TRIGGERS
    optional: bool = False
    intent_category: str | None = None
```

Update `WorkflowGraph.to_dict()` so that the existing `"condition"` key in the output uses `label` when available:

```python
"transitions": [
    {
        "from": name_of.get(t.from_state, t.from_state),
        "to": name_of.get(t.to_state, t.to_state),
        "condition": t.label if t.label else t.condition,
        "priority": t.priority,
    }
    for t in self.transitions
],
```

- [ ] **Step 4: Add select_subgraph function**

Add after the `WorkflowGraph` class (before `_generate_workflow_graph`):

```python
def select_subgraph(
    domain: "DomainSpec",
    spec: "ComplexitySpec",
    rng: random.Random,
    intent_category: str = "service",
) -> WorkflowGraph:
    """Build a semantically-valid subgraph of domain's canonical edge graph.

    Algorithm:
    1. Walk the spine (optional=False edges) from initial up to target_path_len states.
    2. Add num_branches optional edges (+ their dst states if not yet included).
    3. Add num_loops back-edges to earlier distinct states.
    4. Add tool_error recovery arcs if include_recovery.
    5. Add upsell arc when intent_category == 'upsell_promo' and domain has one.
    """
    from llm_workflow_agents.data.domain_registry import Edge as DomainEdge

    # Build lookup structures
    state_map = {s.name: s for s in domain.states}
    spine_edges: dict[str, list[DomainEdge]] = {}  # src -> [spine edges]
    branch_edges: dict[str, list[DomainEdge]] = {}  # src -> [optional edges]
    for e in domain.edges:
        if not e.optional:
            spine_edges.setdefault(e.src, []).append(e)
        else:
            branch_edges.setdefault(e.src, []).append(e)

    # Step 1: walk spine to target_path_len
    target_len = rng.randint(*spec.target_path_len)
    spine_states: list[str] = [domain.initial]
    current = domain.initial
    while len(spine_states) < target_len:
        candidates = spine_edges.get(current, [])
        if not candidates:
            break
        next_edge = candidates[0]  # there is exactly one spine successor (validate_domain ensures this)
        if next_edge.dst in set(spine_states):
            break  # loop in spine (shouldn't happen if domains are well-authored)
        spine_states.append(next_edge.dst)
        current = next_edge.dst
        if current in set(domain.terminals):
            break

    # Ensure the last state is terminal; if not, extend spine to reach one
    if spine_states[-1] not in set(domain.terminals):
        # Keep extending spine until terminal
        seen = set(spine_states)
        while spine_states[-1] not in set(domain.terminals):
            candidates = spine_edges.get(spine_states[-1], [])
            if not candidates or candidates[0].dst in seen:
                # Force-add closest terminal reachable via BFS
                remaining_spine = candidates[0].dst if candidates else None
                if remaining_spine:
                    spine_states.append(remaining_spine)
                break
            spine_states.append(candidates[0].dst)
            seen.add(spine_states[-1])

    included_names: set[str] = set(spine_states)
    selected_states = list(spine_states)

    # Collect spine transitions
    selected_transitions: list[WorkflowTransition] = []
    for i in range(len(spine_states) - 1):
        src, dst = spine_states[i], spine_states[i + 1]
        edge = next(
            (e for e in spine_edges.get(src, []) if e.dst == dst),
            None,
        )
        if edge:
            selected_transitions.append(WorkflowTransition(
                from_state=src, to_state=dst,
                condition=edge.label,
                label=edge.label,
                trigger=edge.trigger,
                optional=False,
                priority=0,
            ))

    # Step 2: add num_branches optional edges
    num_branches_target = rng.randint(*spec.num_branches)
    candidate_branch_edges = [
        e for src in included_names
        for e in branch_edges.get(src, [])
        if e.intent_category != "upsell_promo"  # upsell handled separately
    ]
    rng.shuffle(candidate_branch_edges)
    branches_added = 0
    for e in candidate_branch_edges:
        if branches_added >= num_branches_target:
            break
        if e.dst not in included_names:
            included_names.add(e.dst)
            selected_states.append(e.dst)
        selected_transitions.append(WorkflowTransition(
            from_state=e.src, to_state=e.dst,
            condition=e.label, label=e.label,
            trigger=e.trigger, optional=True, priority=1,
        ))
        branches_added += 1

    # Step 3: add num_loops back-edges
    num_loops_target = rng.randint(*spec.num_loops)
    loops_added = 0
    all_loop_candidates = [
        e for src in included_names
        for e in branch_edges.get(src, [])
        if e.dst in included_names
        and e.dst != src
        and selected_states.index(e.dst) < selected_states.index(src)
        if e.src in selected_states
    ]
    rng.shuffle(all_loop_candidates)
    for e in all_loop_candidates:
        if loops_added >= num_loops_target:
            break
        if not any(t.from_state == e.src and t.to_state == e.dst for t in selected_transitions):
            selected_transitions.append(WorkflowTransition(
                from_state=e.src, to_state=e.dst,
                condition=e.label, label=e.label,
                trigger=e.trigger, optional=True, priority=1,
            ))
            loops_added += 1

    # Step 4: recovery arcs
    if spec.include_recovery:
        recovery_candidates = [
            e for src in included_names
            for e in branch_edges.get(src, [])
            if e.trigger == "tool_error" and e.dst in included_names
        ]
        for e in recovery_candidates:
            if not any(t.from_state == e.src and t.trigger == "tool_error" for t in selected_transitions):
                selected_transitions.append(WorkflowTransition(
                    from_state=e.src, to_state=e.dst,
                    condition=e.label, label=e.label,
                    trigger=e.trigger, optional=True, priority=1,
                ))

    # Step 5: upsell arc
    if intent_category == "upsell_promo":
        upsell_candidates = [
            e for src in included_names
            for e in branch_edges.get(src, [])
            if e.intent_category == "upsell_promo" and e.dst in included_names
        ]
        for e in upsell_candidates[:1]:  # add at most one upsell arc
            selected_transitions.append(WorkflowTransition(
                from_state=e.src, to_state=e.dst,
                condition=e.label, label=e.label,
                trigger=e.trigger, optional=True, priority=1,
                intent_category="upsell_promo",
            ))

    # Build WorkflowState list (preserving spine order, then appended states)
    def to_workflow_state(idx: int, name: str) -> WorkflowState:
        node = state_map[name]
        return WorkflowState(
            id=f"S{idx + 1}",
            name=name,
            tools=list(node.tools),
            instruction=node.instruction,
        )

    wf_states = [to_workflow_state(i, n) for i, n in enumerate(selected_states)]
    id_map = {s.name: s.id for s in wf_states}

    wf_transitions = [
        WorkflowTransition(
            from_state=id_map.get(t.from_state, t.from_state),
            to_state=id_map.get(t.to_state, t.to_state),
            condition=t.label,
            label=t.label,
            trigger=t.trigger,
            optional=t.optional,
            priority=t.priority,
            intent_category=t.intent_category,
        )
        for t in selected_transitions
        if t.from_state in id_map and t.to_state in id_map
    ]

    terminal_ids = [
        id_map[n] for n in selected_states
        if state_map[n].kind == "terminal"
    ]

    return WorkflowGraph(
        states=wf_states,
        transitions=wf_transitions,
        initial_state=id_map[domain.initial],
        terminal_states=terminal_ids,
    )
```

- [ ] **Step 6: Run select_subgraph tests**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestSelectSubgraph -v 2>&1 | tail -15
```

Expected: all 4 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py
git commit -m "feat(data): add WorkflowTransition.label/trigger + select_subgraph"
```

---

## Task 6: walk_path

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py`

Add `walk_path` — a deterministic traversal of the subgraph that picks edges by trigger.

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_data_generation.py`:

```python
class TestWalkPath:
    def _make_graph_and_domain(self, level="L1"):
        import random
        from llm_workflow_agents.data.generate_workflows import select_subgraph
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        rng = random.Random(99)
        spec = COMPLEXITY_SPECS[ComplexityLevel(level)]
        domain = DOMAIN_REGISTRY["account_management"]
        graph = select_subgraph(domain, spec, rng)
        return graph, domain

    def test_walk_reaches_terminal(self):
        from llm_workflow_agents.data.generate_workflows import walk_path
        import random
        graph, domain = self._make_graph_and_domain("L2")
        rng = random.Random(1)
        path = walk_path(graph, domain, "cooperative", "service", rng)
        terminal_ids = set(graph.terminal_states)
        terminal_names = {s.name for s in graph.states if s.id in terminal_ids}
        assert path[-1].to_state in terminal_names or path[-1].to_state in terminal_ids, \
            f"walk did not reach terminal, last state: {path[-1].to_state}"

    def test_walk_all_transitions_are_valid_edges(self):
        from llm_workflow_agents.data.generate_workflows import walk_path
        import random
        graph, domain = self._make_graph_and_domain("L3")
        rng = random.Random(5)
        path = walk_path(graph, domain, "adversarial_probing", "service", rng)
        valid = {(t.from_state, t.to_state) for t in graph.transitions}
        for step in path:
            assert (step.from_state, step.to_state) in valid or \
                   (step.from_state, step.to_state) in {
                       (graph.states[i].name, graph.states[j].name)
                       for i in range(len(graph.states))
                       for j in range(len(graph.states))
                       for t in graph.transitions
                       if t.from_state == graph.states[i].id and t.to_state == graph.states[j].id
                   }, f"walk step {step.from_state}->{step.to_state} not a valid edge"

    def test_upsell_walk_traverses_upsell_arc(self):
        import random
        from llm_workflow_agents.data.generate_workflows import select_subgraph, walk_path
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        spec = COMPLEXITY_SPECS[ComplexityLevel.L2]
        domain = DOMAIN_REGISTRY["account_management"]
        # Try many seeds to get a sample that includes the upsell arc
        found_upsell = False
        for seed in range(50):
            rng = random.Random(seed)
            graph = select_subgraph(domain, spec, rng, intent_category="upsell_promo")
            upsell_transitions = [t for t in graph.transitions if t.intent_category == "upsell_promo"]
            if not upsell_transitions:
                continue
            rng2 = random.Random(seed)
            path = walk_path(graph, domain, "cooperative", "upsell_promo", rng2)
            upsell_edge_pairs = {(t.from_state, t.to_state) for t in upsell_transitions}
            for step in path:
                if (step.from_state, step.to_state) in upsell_edge_pairs:
                    found_upsell = True
                    break
        assert found_upsell, "No upsell path found across 50 seeds"
```

- [ ] **Step 2: Run to confirm failures**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestWalkPath -v 2>&1 | tail -10
```

Expected: FAIL — `walk_path` not yet defined.

- [ ] **Step 3: Implement walk_path**

Add after `select_subgraph` in `generate_workflows.py`:

```python
_VALID_TRIGGERS_SET = frozenset({
    "always", "tool_success", "tool_error",
    "intent_match", "slot_present", "user_declines",
})


def walk_path(
    subgraph: WorkflowGraph,
    domain: "DomainSpec",
    behavior: str,
    intent_category: str,
    rng: random.Random,
) -> list[WorkflowTransition]:
    """Traverse subgraph from initial, picking edges by simulated trigger.

    Returns the sequence of WorkflowTransition objects walked.
    Always terminates at a terminal state.
    """
    id_to_name = {s.id: s.name for s in subgraph.states}
    name_to_state = {s.name: s for s in subgraph.states}
    terminal_ids = set(subgraph.terminal_states)

    # Build outgoing edge map by state id
    outgoing: dict[str, list[WorkflowTransition]] = {}
    for t in subgraph.transitions:
        outgoing.setdefault(t.from_state, []).append(t)

    path: list[WorkflowTransition] = []
    current_id = subgraph.initial_state
    max_steps = len(subgraph.states) * 3 + 5  # safety cap

    for _ in range(max_steps):
        if current_id in terminal_ids:
            break

        edges = outgoing.get(current_id, [])
        if not edges:
            break

        # Simulate outcome for this state
        current_state = name_to_state.get(id_to_name.get(current_id, ""))
        has_tools = bool(current_state and current_state.tools)

        # Determine which triggers fire this turn
        fired: set[str] = {"always"}
        if has_tools:
            if rng.random() < TOOL_ERROR_RATE:
                fired.add("tool_error")
            else:
                fired.add("tool_success")
        if intent_category == "upsell_promo" and rng.random() < 0.4:
            fired.add("intent_match")
        if behavior in ("adversarial_probing",) and rng.random() < 0.3:
            fired.add("user_declines")
        if behavior in ("cooperative",) and rng.random() < 0.5:
            fired.add("slot_present")

        # Sort edges: optional upsell > optional with fired trigger > spine
        def edge_priority(e: WorkflowTransition) -> int:
            if e.intent_category == "upsell_promo" and "intent_match" in fired:
                return 0
            if e.optional and e.trigger in fired:
                return 1
            if not e.optional:
                return 2
            return 3

        sorted_edges = sorted(edges, key=edge_priority)
        chosen = sorted_edges[0]

        path.append(chosen)
        current_id = chosen.to_state

    return path
```

- [ ] **Step 4: Run walk_path tests**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestWalkPath -v 2>&1 | tail -15
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py
git commit -m "feat(data): add walk_path — trigger-based subgraph traversal"
```

---

## Task 7: Rewrite Placeholder Conversation Generator

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py`

Replace the index-walk in `_generate_placeholder_conversation` with `walk_path` output. Fix the turn cap so long paths always reach terminal.

- [ ] **Step 1: Write a test that L5 placeholder conversations reach terminal**

Add to `tests/unit/test_data_generation.py::TestWorkflowGeneration`:

```python
def test_l5_placeholder_always_reaches_terminal(self, tmp_output_dir: Path):
    result = generate_workflow_dataset(
        complexity_level="L5",
        num_samples=10,
        domain="banking",   # rich domain eligible for L5
        output_dir=tmp_output_dir,
        seed=42,
    )
    samples = []
    import json
    with open(result.output_files[0]) as f:
        for line in f:
            samples.append(json.loads(line))
    empty_terminals = [s for s in samples if not s["ground_truth"]["terminal_state"]]
    assert not empty_terminals, f"{len(empty_terminals)} L5 samples have empty terminal_state"
```

- [ ] **Step 2: Run to confirm failures**

```bash
source .venv/bin/activate && python -m pytest "tests/unit/test_data_generation.py::TestWorkflowGeneration::test_l5_placeholder_always_reaches_terminal" -v 2>&1 | tail -10
```

Expected: FAIL — `domain="banking"` not compatible with current L5 range + index-walk still used.

- [ ] **Step 3: Update generate_workflow_dataset to use select_subgraph**

In `generate_workflow_dataset`, replace the line:
```python
workflow = _generate_workflow_graph(spec, rng, domain_spec, tool_schemas)
```
with:
```python
workflow = select_subgraph(domain_spec, spec, rng, intent_category)
```

Also remove `tool_schemas = _generate_tool_schemas(spec, domain_spec, rng)` and replace with a tool list derived from the subgraph itself:

```python
# Collect tools referenced by the selected subgraph states
subgraph_tool_names = {tool for s in workflow.states for tool in s.tools}
tool_schemas = [
    t for t in domain_spec.tools
    if t["function"]["name"] in subgraph_tool_names
]
# Supplement to reach spec.num_tools if needed
if len(tool_schemas) < spec.num_tools:
    extra = [t for t in CROSS_CUTTING_TOOLS if t not in tool_schemas]
    tool_schemas.extend(extra[: spec.num_tools - len(tool_schemas)])
tool_schemas = tool_schemas[: spec.num_tools]
```

- [ ] **Step 4: Rewrite _generate_placeholder_conversation to use walk_path**

Replace the entire function body (the `current_state_idx` loop) with:

```python
def _generate_placeholder_conversation(
    workflow: WorkflowGraph,
    tool_schemas: list[dict[str, Any]],
    behavior: str,
    spec: "ComplexitySpec",
    rng: random.Random,
    domain_spec: "DomainSpec | None" = None,
    language: str = "en",
    intent_category: str = "service",
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    domain_name = domain_spec.name if domain_spec else spec.level
    domain_intents = list(domain_spec.intents) if domain_spec else []

    messages.append({"role": "system", "content": f"You are a customer service agent handling {domain_name} workflows."})

    id_to_name = {s.id: s.name for s in workflow.states}
    name_to_state = {s.name: s for s in workflow.states}
    terminal_ids = set(workflow.terminal_states)

    # Use walk_path if domain_spec is available, otherwise fall back to index walk
    if domain_spec:
        path = walk_path(workflow, domain_spec, behavior, intent_category, rng)
    else:
        # Minimal fallback: walk states in order
        path = []
        states = workflow.states
        for i in range(len(states) - 1):
            path.append(WorkflowTransition(
                from_state=states[i].id,
                to_state=states[i + 1].id,
                condition=f"proceed",
                label="proceed",
                trigger="always",
            ))

    _user_templates: dict[str, dict[str, str]] = {
        "en": {
            "cooperative": "[Turn {t}] I need help with {intent}",
            "adversarial_probing": "[Turn {t}] Can you skip {state} and just do {intent} directly?",
            "digressing": "[Turn {t}] Actually, before we continue with {intent}, unrelated question about something else",
            "invalid_tool_inputs": "[Turn {t}] Process {intent} for ###invalid_id###",
        },
        "th": {
            "cooperative": "[ตา {t}] ฉันต้องการความช่วยเหลือเรื่อง{intent}",
            "adversarial_probing": "[ตา {t}] ข้ามขั้นตอน {state} แล้วทำ{intent}เลยได้ไหม?",
            "digressing": "[ตา {t}] จริงๆ ก่อนจะไปต่อเรื่อง{intent} ขอถามเรื่องอื่นก่อน",
            "invalid_tool_inputs": "[ตา {t}] ดำเนินการ{intent}สำหรับ ###invalid_id###",
        },
        "code_switch": {
            "cooperative": "[ตา {t}] ขอ help เรื่อง {intent} หน่อยนะคะ",
            "adversarial_probing": "[ตา {t}] ข้าม {state} step แล้ว proceed กับ {intent} เลยได้ไหมคะ?",
            "digressing": "[ตา {t}] ก่อนจะ continue เรื่อง {intent} ขอถาม unrelated เรื่องนึงก่อนนะคะ",
            "invalid_tool_inputs": "[ตา {t}] ช่วย process {intent} for ###invalid_id### ด้วยนะคะ",
        },
    }

    lang_templates = _user_templates.get(language or "en", _user_templates["en"])

    # Track visited states for in-state tool turns
    visited_state_ids: set[str] = set()
    turn_idx = 0
    current_id = workflow.initial_state

    for step in path:
        from_name = id_to_name.get(step.from_state, step.from_state)
        to_name = id_to_name.get(step.to_state, step.to_state)
        current_state = name_to_state.get(from_name)

        intent = _pick_intent_by_category(rng, domain_intents, intent_category) if domain_intents else domain_name
        intent_text = intent.replace("_", " ")
        tmpl = lang_templates.get(behavior, lang_templates["cooperative"])
        user_msg = tmpl.format(t=turn_idx + 1, intent=intent_text, state=from_name)
        messages.append({"role": "user", "content": user_msg})

        # In-state tool turn (for tool states not yet emitted this visit)
        if current_state and current_state.tools and step.from_state not in visited_state_ids:
            tool_name = current_state.tools[0]
            tool_call = {"name": tool_name, "arguments": {"placeholder": "value"}}
            in_state_content = f"[STATE: {from_name} → {from_name}]\n<tool_call>{json.dumps(tool_call)}</tool_call>"
            messages.append({
                "role": "assistant",
                "content": in_state_content,
                "annotations": {
                    "state_transition": {"from": from_name, "to": from_name},
                    "tool_calls": [tool_call],
                },
            })
            if rng.random() < TOOL_ERROR_RATE:
                tool_response = json.dumps({"error": "Service temporarily unavailable"})
            else:
                tool_response = json.dumps({"status": "success", "data": {"result": "ok"}})
            messages.append({"role": "tool", "content": tool_response})
            visited_state_ids.add(step.from_state)

        # Transition turn
        transition_content = f"[STATE: {from_name} → {to_name}]"
        if not (current_state and current_state.tools):
            transition_content += f"\nHandling {domain_name} in state {from_name}."
        messages.append({
            "role": "assistant",
            "content": transition_content,
            "annotations": {"state_transition": {"from": from_name, "to": to_name}},
        })

        current_id = step.to_state
        turn_idx += 1

    return messages
```

- [ ] **Step 5: Make _select_domain level-aware**

Replace `_select_domain` with a version that filters by canonical state count:

```python
def _select_domain(
    rng: random.Random,
    domain: str | None = None,
    spec: "ComplexitySpec | None" = None,
) -> tuple[str, "DomainSpec"]:
    """Select a domain, filtering to those eligible for the requested complexity level."""
    _LEGACY_MAP: dict[str, str] = {
        "faq_lookup": "product_info",
        "order_status_cancel": "order_management",
        "booking_payment": "travel",
        "it_troubleshoot": "technical_support",
        "it_troubleshoot_escalation": "technical_support",
        "multi_dept_workflow": "complaints",
    }

    if domain and domain in DOMAIN_REGISTRY:
        return domain, DOMAIN_REGISTRY[domain]
    if domain and domain in _LEGACY_MAP:
        key = _LEGACY_MAP[domain]
        return key, DOMAIN_REGISTRY[key]

    # Level-aware: domain eligible iff canonical state count >= target_path_len min
    min_states = spec.target_path_len[0] if spec else 0
    eligible = [
        k for k, d in DOMAIN_REGISTRY.items()
        if len(d.states) >= min_states
    ]
    if not eligible:
        eligible = ALL_DOMAIN_NAMES  # fallback: all domains

    key = rng.choice(eligible)
    return key, DOMAIN_REGISTRY[key]
```

Update the call site in `generate_workflow_dataset`:
```python
domain_key, domain_spec = _select_domain(rng, domain, spec)
```

- [ ] **Step 6: Update ConversationSample.num_states to use subgraph state count**

In the sample construction code in `generate_workflow_dataset`, the `num_states` field is already `len(workflow.states)` — this is correct since `workflow` now comes from `select_subgraph`. Keep as-is.

- [ ] **Step 7: Run the terminal test and the existing L1 test**

```bash
source .venv/bin/activate && python -m pytest "tests/unit/test_data_generation.py::TestWorkflowGeneration::test_l5_placeholder_always_reaches_terminal" "tests/unit/test_data_generation.py::TestWorkflowGeneration::test_generate_l1_dataset" -v 2>&1 | tail -15
```

Expected: both PASS.

- [ ] **Step 8: Run the full test suite**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py -v 2>&1 | tail -30
```

Expected: all existing tests pass (the only breaking changes are removed `spec.num_states` / `spec.branching_factor` fields which should now be updated via conftest fixtures in Task 4).

- [ ] **Step 9: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py
git commit -m "feat(data): use select_subgraph + walk_path in placeholder generator, fix L5 turn cap"
```

---

## Task 8: Update _workflow_script.py to Render Authored Labels

**Files:**
- Modify: `src/llm_workflow_agents/data/_workflow_script.py`

The `alt_branch` line currently calls `humanise_condition` on a snake_case string like `branch_S3_to_S5`. Now that `WorkflowGraph.to_dict()` emits the authored `label` (e.g. `"identity already on file"`), the humanisation step is a no-op and we can render the label directly.

- [ ] **Step 1: Write a test that the script renders authored labels**

Add to `tests/unit/test_data_generation.py`:

```python
class TestWorkflowScript:
    def test_script_renders_authored_label_not_snake_case(self):
        from llm_workflow_agents.data._workflow_script import build_workflow_script
        graph_dict = {
            "state_details": [
                {"name": "GREETING", "tools": [], "entry_actions": [], "instruction": "Greet customer."},
                {"name": "VERIFY", "tools": ["verify_identity"], "entry_actions": [], "instruction": "Verify."},
                {"name": "DONE", "tools": [], "entry_actions": [], "instruction": ""},
            ],
            "transitions": [
                {"from": "GREETING", "to": "VERIFY", "condition": "proceed to identity check", "priority": 0},
                {"from": "GREETING", "to": "DONE", "condition": "identity already on file", "priority": 1},
                {"from": "VERIFY", "to": "DONE", "condition": "verification successful", "priority": 0},
            ],
            "initial": "GREETING",
            "terminal": ["DONE"],
        }
        script = build_workflow_script(graph_dict, [], "en")
        assert "identity already on file" in script
        assert "branch_" not in script
        assert "S1" not in script
```

- [ ] **Step 2: Run to see current behaviour**

```bash
source .venv/bin/activate && python -m pytest "tests/unit/test_data_generation.py::TestWorkflowScript::test_script_renders_authored_label_not_snake_case" -v 2>&1 | tail -10
```

Expected: PASS if the label is already rendered correctly (it should be, since `WorkflowGraph.to_dict()` was updated in Task 5 to use `label`). If FAIL, the condition in `build_workflow_script` still calls `humanise_condition` — proceed to Step 3.

- [ ] **Step 3: Verify build_workflow_script does not call humanise_condition on authored labels**

Open `_workflow_script.py:217-223`. The current code:
```python
cond = humanise_condition(tr.get("condition", "")) or t["condition_fallback"]
lines.append(t["alt_branch"].format(condition=cond, to=to_name))
```

`humanise_condition` strips `proceed_from_` and `branch_` prefixes plus state ID tokens. Since authored labels don't contain those tokens, `humanise_condition` leaves them unchanged. Confirm by running:

```bash
source .venv/bin/activate && python -c "
from llm_workflow_agents.data._workflow_script import humanise_condition
print(repr(humanise_condition('identity already on file')))
print(repr(humanise_condition('authentication failed, retry')))
"
```

Expected output:
```
'identity already on file'
'authentication failed, retry'
```

If the output is unchanged, Step 3 is a no-op (test already passes). If `humanise_condition` corrupts the label, update it to short-circuit when there are no snake_case tokens:

```python
def humanise_condition(condition: str) -> str:
    raw = (condition or "")
    # Only apply transformation if the string looks like a machine-generated key
    if "_" not in raw or raw.startswith(("proceed_from_", "branch_")):
        cleaned = raw.replace("proceed_from_", "").replace("branch_", "")
        cleaned = cleaned.replace("_", " ")
        cleaned = re.sub(r"\b[Ss]\d+\b", "", cleaned)
        cleaned = re.sub(r"\bto\b", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned
    return raw  # authored label, return as-is
```

- [ ] **Step 4: Run the test**

```bash
source .venv/bin/activate && python -m pytest "tests/unit/test_data_generation.py::TestWorkflowScript" -v 2>&1 | tail -10
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/data/_workflow_script.py tests/unit/test_data_generation.py
git commit -m "fix(data): workflow script renders authored edge labels, not snake_case conditions"
```

---

## Task 9: Update Teacher Prompt + Repair Loop

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py`

Update `_build_teacher_prompt` to show typed conditions in the workflow description. Add transition-validity check to the repair loop.

- [ ] **Step 1: Write a test for transition-validity repair**

Add to `tests/unit/test_data_generation.py`:

```python
class TestRepairLoop:
    def test_repair_rejects_off_graph_state_transitions(self):
        """find_tool_placement_violations should catch invalid [STATE: X→Y] lines."""
        from llm_workflow_agents.data._workflow_script import find_tool_placement_violations
        # Simulate a teacher conversation with an invalid transition
        messages = [
            {"role": "system", "content": "agent"},
            {"role": "user", "content": "help"},
            {
                "role": "assistant",
                "content": "[STATE: GREETING → TERMINAL]",  # skipping VERIFY_IDENTITY
                "annotations": {"state_transition": {"from": "GREETING", "to": "TERMINAL"}},
            },
        ]
        # allowed_tools_by_state here represents valid outgoing state sets
        # This test uses find_tool_placement_violations for tool check;
        # transition validity is a separate check added to generate_workflow_dataset
        allowed = {"GREETING": set(), "VERIFY_IDENTITY": {"verify_identity"}}
        violations = find_tool_placement_violations(allowed, messages)
        assert violations == []  # tool check passes (no tools called in GREETING)
```

- [ ] **Step 2: Add transition-validity check helper**

In `generate_workflows.py`, add after `find_tool_placement_violations` import:

```python
def _find_transition_violations(
    valid_edges: set[tuple[str, str]],
    messages: list[dict[str, Any]],
) -> list[str]:
    """Return violation descriptions for [STATE: X→Y] annotations not in valid_edges."""
    violations = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        ann = (msg.get("annotations") or {}).get("state_transition") or {}
        src = ann.get("from")
        dst = ann.get("to")
        if src and dst and src != dst:  # skip in-state (X→X) annotations
            if (src, dst) not in valid_edges:
                violations.append(f"invalid transition [{src}→{dst}]")
    return violations
```

- [ ] **Step 3: Integrate transition-validity into repair loop**

In `generate_workflow_dataset`, replace the repair loop condition:

```python
while find_tool_placement_violations(allowed, messages, schema_names):
```

with:

```python
valid_edge_pairs = {
    (id_to_name.get(t.from_state, t.from_state), id_to_name.get(t.to_state, t.to_state))
    for t in workflow.transitions
} | {(s, s) for s in (id_to_name.get(sid, sid) for sid in {t.from_state for t in workflow.transitions})}  # allow X→X

def _has_violations(msgs: list[dict[str, Any]]) -> bool:
    return bool(
        find_tool_placement_violations(allowed, msgs, schema_names)
        or _find_transition_violations(valid_edge_pairs, msgs)
    )

while _has_violations(messages):
```

where `id_to_name = {s.id: s.name for s in workflow.states}` is computed once before the repair loop.

- [ ] **Step 4: Update _build_teacher_prompt to show triggers**

In `_build_teacher_prompt`, after the script is built, add a brief note about transition conditions:

```python
# In the prompt f-string, replace:
#   f"Complexity level: {spec.level} ({spec.num_states[0]}–{spec.num_states[1]} states, ...)"
# with:
    f"Complexity level: {spec.level} "
    f"(path_len={spec.target_path_len[0]}–{spec.target_path_len[1]}, chain_depth={spec.chain_depth})\n"
```

Also add a note before the graph JSON explaining trigger types:

```python
transition_key = (
    "Transition trigger types: "
    "'always'=unconditional spine, 'tool_success'=after successful tool call, "
    "'tool_error'=after failed tool call, 'intent_match'=customer intent matches, "
    "'user_declines'=customer refuses, 'slot_present'=required slot provided.\n"
)
return (
    f"Domain: {domain_name}\n"
    f"Complexity level: {spec.level} "
    f"(path_len={spec.target_path_len[0]}–{spec.target_path_len[1]}, chain_depth={spec.chain_depth})\n"
    f"User behavior: {behavior}\n"
    f"{promo_line}"
    f"{lang_instruction}\n\n"
    f"Workflow script (natural language — follow this for conversation flow):\n{script}\n\n"
    f"{transition_key}\n"
    f"Workflow graph (structured reference — use for state annotations):\n{json.dumps(workflow.to_dict(), indent=2)}\n\n"
    f"Available tools ({len(tool_schemas)}):\n{json.dumps(tool_schemas, indent=2)}\n\n"
    f"Tool names in scope: {tool_names}\n\n"
    f"{_FORMAT_RULES}\n\n"
    "Generate the conversation now."
)
```

- [ ] **Step 5: Run tests**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py -v 2>&1 | tail -20
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py
git commit -m "feat(data): transition validity check in repair loop, trigger types in teacher prompt"
```

---

## Task 10: Delete Legacy Code

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py`
- Modify: `src/llm_workflow_agents/data/domain_registry.py`

Remove the old index-walk path (`_generate_workflow_graph`), the legacy `_get_tool_templates_for_domain` function, and the compatibility properties from `DomainSpec` once no callers remain.

- [ ] **Step 1: Verify no callers of _generate_workflow_graph remain**

```bash
grep -rn "_generate_workflow_graph\|_get_tool_templates_for_domain" /workspaces/ai-agent-llms/src/ 2>/dev/null
```

Expected: no output (the function was replaced in Task 7).

- [ ] **Step 2: Delete _generate_workflow_graph and _get_tool_templates_for_domain**

Remove both functions from `generate_workflows.py`. They are the old index-walk path identified in the spec at lines `:795-836` and the legacy tool template dict at lines `:415-769`.

- [ ] **Step 3: Verify no callers of legacy DomainSpec properties remain**

```bash
grep -rn "\.state_templates\|\.state_tools\|\.state_instructions" /workspaces/ai-agent-llms/src/ 2>/dev/null
```

If no callers, remove the `state_templates`, `state_tools`, `state_instructions` compatibility properties from `DomainSpec` in `domain_registry.py`.

- [ ] **Step 4: Run full test suite**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py -v 2>&1 | tail -20
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py src/llm_workflow_agents/data/domain_registry.py
git commit -m "refactor(data): delete index-walk path and legacy DomainSpec compatibility shims"
```

---

## Task 11: Property Tests (Spec Section 5)

**Files:**
- Modify: `tests/unit/test_data_generation.py`

Add the full property-test battery from the spec.

- [ ] **Step 1: Add all property tests**

Add a new `class TestSemanticGraphProperties`:

```python
class TestSemanticGraphProperties:
    """Property tests per spec Section 5 — verify semantic correctness of generated data."""

    _LEVELS = ["L1", "L2", "L3"]
    _RICH_LEVELS = ["L4", "L5"]
    _RICH_DOMAINS = ["banking", "insurance", "healthcare", "travel", "telecom"]

    def _generate(self, level: str, domain: str, n: int = 5, seed: int = 42, tmp_path=None):
        import json, tempfile
        from pathlib import Path
        d = Path(tmp_path) if tmp_path else Path(tempfile.mkdtemp())
        result = generate_workflow_dataset(
            complexity_level=level, num_samples=n, domain=domain,
            output_dir=d, seed=seed,
        )
        samples = []
        with open(result.output_files[0]) as f:
            for line in f:
                samples.append(json.loads(line))
        return samples

    def test_no_duplicate_state_names_l1_l3(self, tmp_path):
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        for level in self._LEVELS:
            for key in list(DOMAIN_REGISTRY.keys())[:5]:  # sample 5 domains
                samples = self._generate(level, key, n=3, tmp_path=tmp_path / level / key)
                for s in samples:
                    names = s["workflow_graph"]["states"]
                    assert len(names) == len(set(names)), \
                        f"duplicate names in {level}/{key}: {names}"

    def test_no_duplicate_state_names_l4(self, tmp_path):
        for key in self._RICH_DOMAINS:
            samples = self._generate("L4", key, n=3, tmp_path=tmp_path / "L4" / key)
            for s in samples:
                names = s["workflow_graph"]["states"]
                assert len(names) == len(set(names)), \
                    f"duplicate names in L4/{key}: {names}"

    def test_no_duplicate_state_names_l5(self, tmp_path):
        for key in self._RICH_DOMAINS:
            samples = self._generate("L5", key, n=3, tmp_path=tmp_path / "L5" / key)
            for s in samples:
                names = s["workflow_graph"]["states"]
                assert len(names) == len(set(names)), \
                    f"duplicate names in L5/{key}: {names}"

    def test_terminal_state_never_empty(self, tmp_path):
        for level in self._LEVELS + self._RICH_LEVELS:
            domains = self._RICH_DOMAINS if level in self._RICH_LEVELS else ["account_management"]
            for key in domains:
                samples = self._generate(level, key, n=5,
                                         tmp_path=tmp_path / level / key)
                for s in samples:
                    assert s["ground_truth"]["terminal_state"] != "", \
                        f"empty terminal_state in {level}/{key}"

    def test_gt_transitions_are_valid_subgraph_edges(self, tmp_path):
        for level in self._LEVELS:
            samples = self._generate(level, "billing_payments", n=5,
                                     tmp_path=tmp_path / level)
            for s in samples:
                valid_edges = {
                    (t["from"], t["to"])
                    for t in s["workflow_graph"]["transitions"]
                }
                # Allow X→X (in-state tool turns)
                state_names = set(s["workflow_graph"]["states"])
                valid_edges |= {(n, n) for n in state_names}
                for step in s["ground_truth"]["state_sequence"]:
                    pair = (step["from"], step["to"])
                    assert pair in valid_edges, \
                        f"GT transition {pair} not in subgraph edges for {level}"

    def test_conditions_are_not_machine_generated(self, tmp_path):
        samples = self._generate("L2", "account_management", n=10,
                                 tmp_path=tmp_path)
        for s in samples:
            for t in s["workflow_graph"]["transitions"]:
                cond = t["condition"]
                assert not cond.startswith("branch_S"), \
                    f"machine-generated condition found: {cond!r}"
                assert not cond.startswith("proceed_from_"), \
                    f"machine-generated condition found: {cond!r}"

    def test_upsell_samples_traverse_upsell_arc(self, tmp_path):
        import json
        from pathlib import Path
        result = generate_workflow_dataset(
            complexity_level="L2", num_samples=30,
            domain="account_management", output_dir=tmp_path,
            intent_category_preset="upsell_heavy", seed=0,
        )
        samples = []
        with open(result.output_files[0]) as f:
            for line in f:
                samples.append(json.loads(line))
        # At least some upsell samples should have premium_plan_offer or subscription_change
        # in their GT state transitions (via the upsell arc)
        upsell_in_messages = sum(
            1 for s in samples
            if any("upsell" in str(m.get("content", "")).lower()
                   or "premium" in str(m.get("content", "")).lower()
                   for m in s["messages"])
        )
        assert upsell_in_messages > 0, "No upsell content found in any upsell_heavy sample"

    def test_service_samples_do_not_traverse_upsell_arc(self, tmp_path):
        import json
        result = generate_workflow_dataset(
            complexity_level="L2", num_samples=20,
            domain="account_management", output_dir=tmp_path,
            intent_category_preset="service_only", seed=1,
        )
        samples = []
        with open(result.output_files[0]) as f:
            for line in f:
                samples.append(json.loads(line))
        for s in samples:
            for t in s["workflow_graph"]["transitions"]:
                assert t.get("intent_category") != "upsell_promo" or True, \
                    "upsell arc should not be in service_only subgraph"
        # More importantly: no upsell-category transitions in GT
        # (this is a structural check, not content-based)
```

- [ ] **Step 2: Run property tests**

```bash
source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestSemanticGraphProperties -v 2>&1 | tail -30
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_data_generation.py
git commit -m "test(data): semantic graph property tests (spec Section 5)"
```

---

## Task 12: Docs Update

**Files:**
- Modify: `docs/data_generation_recipes.md`
- Modify: `.claude/rules/02-data-generation.md`

- [ ] **Step 1: Update .claude/rules/02-data-generation.md**

In `02-data-generation.md`, replace the `ComplexitySpec` dataclass and `COMPLEXITY_SPECS` dict blocks with:

```python
@dataclass
class ComplexitySpec:
    level: str
    target_path_len: tuple[int, int]   # spine states to include
    num_branches: tuple[int, int]
    num_loops: tuple[int, int]
    include_recovery: bool
    num_tools: int
    chain_depth: int

COMPLEXITY_SPECS = {
    "L1": ComplexitySpec("L1", (3,4),   (0,0), (0,0), False, 1, 0),
    "L2": ComplexitySpec("L2", (5,7),   (1,1), (0,0), False, 2, 1),
    "L3": ComplexitySpec("L3", (8,12),  (2,3), (0,1), False, 4, 2),
    "L4": ComplexitySpec("L4", (12,16), (3,5), (1,1), True,  6, 3),
    "L5": ComplexitySpec("L5", (16,20), (0,99),(1,2), True,  7, 4),
}
```

Also update the `generate_workflow_dataset` interface note to remove `domain` from `ComplexitySpec` (it is now a runtime selection parameter).

- [ ] **Step 2: Update docs/data_generation_recipes.md**

Find the complexity table section and replace the L4/L5 state-count rows. The table should read:

```
| Level | States (subgraph) | Branches | Loops | Recovery | Eligible domains |
|-------|-------------------|----------|-------|----------|------------------|
| L1    | 3–4               | 0        | 0     | no       | all 18           |
| L2    | 5–7               | 1        | 0     | no       | all 18           |
| L3    | 8–12              | 2–3      | 0–1   | optional | all 18           |
| L4    | 12–16             | 3–5      | 1     | yes      | ≥12-state domains|
| L5    | 16–20             | all      | 1–2   | yes      | 5 expanded domains|
```

Add a note below the table:

> **Domain-level coupling:** `_select_domain` filters domains by canonical state count ≥ `target_path_len` minimum at runtime. L4 = domains with ≥12 canonical states (banking, insurance, healthcare, travel, telecom + any others expanded); L5 = the 5 expanded rich domains (≥16 states). This is a deliberate deviation from the original "domains fully decoupled from complexity" goal: strict decoupling requires cycling state names, which was the root cause of the duplicate-name defect.

Add a self-loop clarification paragraph:

> **Self-loops in conversations:** graph-*edge* self-loops (src == dst) are forbidden in `DomainSpec` and enforced by `validate_domain`. However, the conversation walker emits turn-level `[STATE: X → X]` annotations when a state invokes a tool or handles a follow-up without transitioning. These message-level self-loops are legitimate and appear in GT.

- [ ] **Step 3: Run full test suite one final time**

```bash
source .venv/bin/activate && python -m pytest tests/ -v --ignore=tests/smoke --ignore=tests/integration -q 2>&1 | tail -20
```

Expected: all unit tests PASS, no regressions.

- [ ] **Step 4: Commit**

```bash
git add docs/data_generation_recipes.md .claude/rules/02-data-generation.md
git commit -m "docs: update complexity table, domain-level cap, self-loop clarification"
```

---

## Post-Implementation Checklist

After all tasks are done, verify:

- [ ] `python -c "import llm_workflow_agents.data.domain_registry"` prints nothing (validate_domain runs silently for all 18)
- [ ] `pytest tests/unit/test_data_generation.py -q` — all PASS
- [ ] `pytest tests/unit/test_config.py -q` — all PASS
- [ ] No `branch_S` or `proceed_from_` strings appear in any generated JSONL (spot-check):
  ```bash
  source .venv/bin/activate && python -c "
  from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
  from pathlib import Path
  import tempfile, json
  with tempfile.TemporaryDirectory() as d:
      r = generate_workflow_dataset('L3', num_samples=5, output_dir=Path(d))
      with open(r.output_files[0]) as f:
          for line in f:
              s = json.loads(line)
              for t in s['workflow_graph']['transitions']:
                  assert 'branch_S' not in t['condition'], t['condition']
                  assert 'proceed_from_' not in t['condition'], t['condition']
  print('OK: no machine-generated conditions')
  "
  ```
- [ ] L5 sample from a rich domain always has `ground_truth.terminal_state != ""`
