# Outbound (Support-Initiated) Conversations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add support-initiated ("outbound") Task A conversations — where the support agent opens the call with a purpose (sales promo, prescription follow-up, renewal reminder) instead of the customer stating an intent — and make them flow correctly through validation and training.

**Architecture:** Three layers. (1) Data model: a new `OutboundReason` dataclass and an `outbound_reasons` field on `DomainSpec`, authored for a curated subset of domains. (2) Generation: a new `initiation_preset` knob mixes inbound/outbound per sample; outbound samples reuse the domain's **existing** canonical graph but emit an assistant **opener** at the initial state and carry `conversation_initiator="agent"` + `outbound_reason`. (3) Downstream: the validator and the GRPO data loader are widened so the opener (an assistant turn preceded by `system`, not `user`) is a valid training row; SFT masking already handles it.

**Tech Stack:** Python 3, dataclasses, `random.Random`, pytest. Use `uv` for env (`source .venv/bin/activate && …`).

---

## Target message shape

Inbound (unchanged): `[system, user, assistant(+tool), user, assistant, …]`

Outbound (new): `[system, assistant(outreach+purpose), user(responds), assistant(+tool), …]`

`messages[0]` is still `system` everywhere — the only structural change is that **`messages[1]` is `assistant`** for outbound. `state_sequence` still starts at the initial state.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/llm_workflow_agents/data/domain_registry.py` | Modify | `OutboundReason` dataclass, `outbound_reasons` field, curated reasons on 7 domains |
| `src/llm_workflow_agents/data/generate_workflows.py` | Modify | `INITIATION_PRESETS`, `_select_initiator`, opener + outbound user templates, teacher-prompt outbound block, `ConversationSample` fields, `generate_workflow_dataset` wiring, `_select_domain` outbound filter |
| `src/llm_workflow_agents/data/data_validator.py` | Modify | initiator-aware `messages[1]` role check |
| `src/llm_workflow_agents/training/grpo.py` | Modify | accept `system`-preceded assistant opener as a training row |
| `tests/unit/test_data_generation.py` | Modify | outbound generation + schema tests |
| `tests/unit/test_grpo_outbound.py` | Create | GRPO loader keeps the opener row |
| `tests/unit/test_sft_outbound.py` | Create | SFT masking keeps the opener unmasked |
| `tests/unit/test_chat_templates.py` | Modify | outbound conversion across families |
| `.claude/rules/02-data-generation.md` | Modify | document the new knob, schema, output fields |

---

## Task 1: `OutboundReason` dataclass + `outbound_reasons` field

**Files:**
- Modify: `src/llm_workflow_agents/data/domain_registry.py` (add dataclass near `Edge` ~line 47; add field to `DomainSpec` ~lines 48-73)
- Test: `tests/unit/test_data_generation.py`

- [ ] **Step 1: Write the failing test**

Add a new class to `tests/unit/test_data_generation.py`:

```python
class TestOutboundSchema:
    """Tests for the OutboundReason schema on DomainSpec."""

    def test_outbound_reason_dataclass_defaults(self):
        from llm_workflow_agents.data.domain_registry import OutboundReason
        r = OutboundReason(key="promo", description="offer a promotion")
        assert r.key == "promo"
        assert r.description == "offer a promotion"
        assert r.intent_category == "service"

    def test_domainspec_has_outbound_reasons_default_empty(self):
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        # A domain not in the curated subset has no outbound reasons.
        assert DOMAIN_REGISTRY["government"].outbound_reasons == ()

    def test_outbound_reason_categories_are_valid(self):
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        for key, dom in DOMAIN_REGISTRY.items():
            for r in dom.outbound_reasons:
                assert r.intent_category in ("service", "upsell_promo"), \
                    f"{key}/{r.key} has bad category {r.intent_category}"
```

- [ ] **Step 2: Run to confirm it fails**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestOutboundSchema -q`
Expected: FAIL — `ImportError: cannot import name 'OutboundReason'`.

- [ ] **Step 3: Add the dataclass and field**

In `src/llm_workflow_agents/data/domain_registry.py`, immediately after the `Edge` dataclass (after line 45's `intent_category: str | None = None` block, before `DomainSpec`), add:

```python
@dataclass(frozen=True)
class OutboundReason:
    """A support-side reason for initiating an outbound conversation.

    ``description`` is woven into the agent's opening turn and the teacher
    prompt (e.g. "follow up on the patient's prescription"). ``intent_category``
    selects which subgraph arcs the conversation pulls — "upsell_promo" for
    sales/cross-sell outreach, "service" for reminders and follow-ups.
    """

    key: str
    description: str
    intent_category: str = "service"  # "service" | "upsell_promo"
```

Then add one field to `DomainSpec` (after `intent_categories` at line 73):

```python
    outbound_reasons: tuple[OutboundReason, ...] = ()
```

- [ ] **Step 4: Run to confirm it passes**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestOutboundSchema -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/data/domain_registry.py tests/unit/test_data_generation.py
git commit -m "feat(data): add OutboundReason schema + outbound_reasons field on DomainSpec"
```

---

## Task 2: Author curated `outbound_reasons` on 7 domains

**Files:**
- Modify: `src/llm_workflow_agents/data/domain_registry.py` (the 7 named `DomainSpec` blocks)
- Test: `tests/unit/test_data_generation.py`

- [ ] **Step 1: Write the failing test**

Add to `class TestOutboundSchema`:

```python
    def test_curated_domains_have_outbound_reasons(self):
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        expected = {
            "sales", "banking", "insurance", "healthcare",
            "telecom", "travel", "scheduling",
        }
        for key in expected:
            reasons = DOMAIN_REGISTRY[key].outbound_reasons
            assert reasons, f"{key} should have outbound_reasons"
            keys = {r.key for r in reasons}
            assert len(keys) == len(reasons), f"{key} has duplicate reason keys"

    def test_healthcare_has_prescription_followup(self):
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        keys = {r.key for r in DOMAIN_REGISTRY["healthcare"].outbound_reasons}
        assert "prescription_followup" in keys
```

- [ ] **Step 2: Run to confirm it fails**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestOutboundSchema::test_curated_domains_have_outbound_reasons -q`
Expected: FAIL — `outbound_reasons` is empty for these domains.

- [ ] **Step 3: Add `outbound_reasons=(...)` to each of the 7 domain blocks**

In `src/llm_workflow_agents/data/domain_registry.py`, add an `outbound_reasons=(...)` argument to each named `DomainSpec(...)` (place it after `intent_categories={...}` in each block). Use exactly these:

`SALES`:
```python
    outbound_reasons=(
        OutboundReason("promotion_offer", "offer you a limited-time promotion on your plan", "upsell_promo"),
        OutboundReason("cross_sell", "tell you about a product that complements your account", "upsell_promo"),
    ),
```

`BANKING`:
```python
    outbound_reasons=(
        OutboundReason("loan_offer", "let you know about a pre-approved loan offer", "upsell_promo"),
        OutboundReason("rate_review", "review a better savings rate you now qualify for", "upsell_promo"),
        OutboundReason("card_activation_reminder", "remind you to activate the card we recently issued", "service"),
    ),
```

`INSURANCE`:
```python
    outbound_reasons=(
        OutboundReason("renewal_reminder", "remind you that your policy is up for renewal soon", "service"),
        OutboundReason("coverage_upgrade", "offer an upgrade that broadens your current coverage", "upsell_promo"),
    ),
```

`HEALTHCARE`:
```python
    outbound_reasons=(
        OutboundReason("prescription_followup", "follow up on your current prescription and refills", "service"),
        OutboundReason("appointment_reminder", "remind you about your upcoming appointment", "service"),
        OutboundReason("wellness_program_offer", "invite you to enrol in our wellness programme", "upsell_promo"),
    ),
```

`TELECOM`:
```python
    outbound_reasons=(
        OutboundReason("plan_upgrade_offer", "offer you an upgraded mobile plan at a better rate", "upsell_promo"),
        OutboundReason("roaming_activation_reminder", "remind you to activate roaming before your trip", "service"),
    ),
```

`TRAVEL`:
```python
    outbound_reasons=(
        OutboundReason("loyalty_upgrade_offer", "offer a loyalty upgrade for your upcoming trip", "upsell_promo"),
        OutboundReason("trip_reminder", "remind you about details of your upcoming booking", "service"),
    ),
```

`SCHEDULING`:
```python
    outbound_reasons=(
        OutboundReason("appointment_reminder", "remind you about your scheduled appointment", "service"),
        OutboundReason("reschedule_followup", "follow up about rescheduling your missed appointment", "service"),
    ),
```

- [ ] **Step 4: Run to confirm import still valid and tests pass**

Run: `source .venv/bin/activate && python -c "import llm_workflow_agents.data.domain_registry; print('ok')" && python -m pytest tests/unit/test_data_generation.py::TestOutboundSchema -q`
Expected: `ok` then PASS (all `TestOutboundSchema` tests).

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/data/domain_registry.py tests/unit/test_data_generation.py
git commit -m "feat(data): author curated outbound_reasons on 7 domains"
```

---

## Task 3: `INITIATION_PRESETS` + `_select_initiator`

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py` (presets near line 96; helper near `_select_intent_category` line 672)
- Test: `tests/unit/test_data_generation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_data_generation.py`:

```python
class TestInitiatorSelection:
    def test_initiation_presets_shape(self):
        from llm_workflow_agents.data.generate_workflows import INITIATION_PRESETS
        assert INITIATION_PRESETS["default"] == {"user": 1.00, "agent": 0.00}
        assert set(INITIATION_PRESETS["balanced"]) == {"user", "agent"}

    def test_select_initiator_default_always_user(self):
        import random
        from llm_workflow_agents.data.generate_workflows import (
            _select_initiator, INITIATION_PRESETS,
        )
        rng = random.Random(0)
        picks = {_select_initiator(rng, INITIATION_PRESETS["default"]) for _ in range(50)}
        assert picks == {"user"}

    def test_select_initiator_outbound_heavy_yields_agents(self):
        import random
        from llm_workflow_agents.data.generate_workflows import (
            _select_initiator, INITIATION_PRESETS,
        )
        rng = random.Random(0)
        picks = [_select_initiator(rng, INITIATION_PRESETS["outbound_heavy"]) for _ in range(200)]
        assert picks.count("agent") > 0
```

- [ ] **Step 2: Run to confirm it fails**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestInitiatorSelection -q`
Expected: FAIL — `cannot import name 'INITIATION_PRESETS'`.

- [ ] **Step 3: Add the presets**

In `src/llm_workflow_agents/data/generate_workflows.py`, directly after `INTENT_CATEGORY_PRESETS` (ends line 100), add:

```python
INITIATION_PRESETS: dict[str, dict[str, float]] = {
    "default":         {"user": 1.00, "agent": 0.00},  # 100% inbound (back-compat)
    "balanced":        {"user": 0.70, "agent": 0.30},
    "outbound_heavy":  {"user": 0.40, "agent": 0.60},
}
```

- [ ] **Step 4: Add the selector**

After `_select_intent_category` (ends line 679), add:

```python
def _select_initiator(
    rng: random.Random,
    distribution: dict[str, float],
) -> str:
    """Select who opens the conversation ('user' inbound | 'agent' outbound)."""
    cats = list(distribution.keys())
    weights = list(distribution.values())
    return rng.choices(cats, weights=weights, k=1)[0]
```

- [ ] **Step 5: Run to confirm it passes**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestInitiatorSelection -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py tests/unit/test_data_generation.py
git commit -m "feat(data): add INITIATION_PRESETS and _select_initiator"
```

---

## Task 4: `ConversationSample` carries initiator + reason

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py` (dataclass lines 625-658)
- Test: `tests/unit/test_data_generation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_data_generation.py`:

```python
class TestConversationSampleOutboundFields:
    def test_to_dict_includes_initiator_fields(self):
        from llm_workflow_agents.data.generate_workflows import ConversationSample
        s = ConversationSample(
            conversation_id="L1_001", complexity_level="L1", domain="sales",
            num_states=3, num_tools=1, chain_depth=0,
            workflow_graph={}, workflow_script="", tool_schemas=[],
            messages=[], user_behavior="cooperative",
            conversation_initiator="agent", outbound_reason="promotion_offer",
        )
        d = s.to_dict()
        assert d["conversation_initiator"] == "agent"
        assert d["outbound_reason"] == "promotion_offer"

    def test_defaults_are_inbound(self):
        from llm_workflow_agents.data.generate_workflows import ConversationSample
        s = ConversationSample(
            conversation_id="L1_001", complexity_level="L1", domain="sales",
            num_states=3, num_tools=1, chain_depth=0,
            workflow_graph={}, workflow_script="", tool_schemas=[],
            messages=[], user_behavior="cooperative",
        )
        d = s.to_dict()
        assert d["conversation_initiator"] == "user"
        assert d["outbound_reason"] is None
```

- [ ] **Step 2: Run to confirm it fails**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestConversationSampleOutboundFields -q`
Expected: FAIL — `TypeError: unexpected keyword argument 'conversation_initiator'`.

- [ ] **Step 3: Add the fields**

In `ConversationSample` (line 625), add two fields after `ground_truth` (line 641):

```python
    conversation_initiator: str = "user"
    outbound_reason: str | None = None
```

And in `to_dict` (after the `"ground_truth": self.ground_truth,` line 657), add:

```python
            "conversation_initiator": self.conversation_initiator,
            "outbound_reason": self.outbound_reason,
```

- [ ] **Step 4: Run to confirm it passes**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestConversationSampleOutboundFields -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py tests/unit/test_data_generation.py
git commit -m "feat(data): ConversationSample carries conversation_initiator + outbound_reason"
```

---

## Task 5: `_select_domain` outbound filter

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py` (`_select_domain` lines 693-724)
- Test: `tests/unit/test_data_generation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_data_generation.py`:

```python
class TestSelectDomainOutbound:
    def test_outbound_only_picks_outbound_capable_domain(self):
        import random
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        spec = COMPLEXITY_SPECS[ComplexityLevel.L1]
        for s in range(30):
            rng = random.Random(s)
            key, dom = _select_domain(rng, None, spec, outbound_only=True)
            assert dom.outbound_reasons, f"{key} has no outbound_reasons"

    def test_pinned_domain_ignores_outbound_filter(self):
        import random
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        spec = COMPLEXITY_SPECS[ComplexityLevel.L1]
        rng = random.Random(0)
        key, dom = _select_domain(rng, "government", spec, outbound_only=True)
        assert key == "government"  # explicit pin always honored
```

- [ ] **Step 2: Run to confirm it fails**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestSelectDomainOutbound -q`
Expected: FAIL — `_select_domain() got an unexpected keyword argument 'outbound_only'`.

- [ ] **Step 3: Add the `outbound_only` parameter**

In `_select_domain` (line 693), add the parameter:

```python
def _select_domain(
    rng: random.Random,
    domain: str | None = None,
    spec: "ComplexitySpec | None" = None,
    outbound_only: bool = False,
) -> tuple[str, DomainSpec]:
```

The pinned-domain early returns (lines 708-712) are unchanged — an explicit pin is always honored. Then change the eligibility comprehension (lines 716-719) to also require outbound reasons when requested:

```python
    min_states = spec.target_path_len[0] if spec else 0
    eligible = [
        k for k, d in DOMAIN_REGISTRY.items()
        if len(d.states) >= min_states
        and (not outbound_only or d.outbound_reasons)
    ]
```

- [ ] **Step 4: Run to confirm it passes**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestSelectDomainOutbound -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py tests/unit/test_data_generation.py
git commit -m "feat(data): _select_domain outbound-capable filter"
```

---

## Task 6: Placeholder generator opener + outbound user templates

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py` (`_generate_placeholder_conversation` lines 728-839; import line for `OutboundReason`)
- Test: `tests/unit/test_data_generation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_data_generation.py`:

```python
class TestPlaceholderOutbound:
    def _build(self, seed=0):
        import random
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY, OutboundReason
        rng = random.Random(seed)
        spec = COMPLEXITY_SPECS[ComplexityLevel.L2]
        dom = DOMAIN_REGISTRY["healthcare"]
        wf = gw.select_subgraph(dom, spec, rng, "service")
        tools = [t for t in dom.tools]
        reason = OutboundReason("prescription_followup", "follow up on your prescription", "service")
        msgs = gw._generate_placeholder_conversation(
            wf, tools, "cooperative", spec, rng, dom, "en", "service",
            initiator="agent", outbound_reason=reason,
        )
        return msgs

    def test_outbound_opener_is_assistant(self):
        msgs = self._build()
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "assistant"

    def test_outbound_opener_has_state_annotation_and_reason(self):
        msgs = self._build()
        opener = msgs[1]["content"]
        assert "[STATE:" in opener
        assert "prescription" in opener.lower()

    def test_inbound_still_user_first(self):
        import random
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
        rng = random.Random(0)
        spec = COMPLEXITY_SPECS[ComplexityLevel.L2]
        dom = DOMAIN_REGISTRY["healthcare"]
        wf = gw.select_subgraph(dom, spec, rng, "service")
        msgs = gw._generate_placeholder_conversation(
            wf, [t for t in dom.tools], "cooperative", spec, rng, dom, "en", "service",
        )
        assert msgs[1]["role"] == "user"
```

- [ ] **Step 2: Run to confirm it fails**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestPlaceholderOutbound -q`
Expected: FAIL — `_generate_placeholder_conversation() got an unexpected keyword argument 'initiator'`.

- [ ] **Step 3: Extend the signature and add the opener + outbound templates**

In `src/llm_workflow_agents/data/generate_workflows.py`, ensure `OutboundReason` is importable. Find the existing import of `DomainSpec` from `.domain_registry` (top of file) and add `OutboundReason` to it.

Change the signature of `_generate_placeholder_conversation` (line 728) to add two params at the end:

```python
def _generate_placeholder_conversation(
    workflow: WorkflowGraph,
    tool_schemas: list[dict[str, Any]],
    behavior: str,
    spec: ComplexitySpec,
    rng: random.Random,
    domain_spec: DomainSpec | None = None,
    language: str = "en",
    intent_category: str = "service",
    initiator: str = "user",
    outbound_reason: "OutboundReason | None" = None,
) -> list[dict[str, Any]]:
```

After the system message is appended (line 748) and after `id_to_name`/`name_to_state` are built (lines 750-751), insert the opener emission (place it right before the `if domain_spec:` walk_path block at line 754-755):

```python
    # Outbound: the AGENT opens at the initial state stating the purpose,
    # before the customer says anything. Inbound is unchanged (user-first).
    if initiator == "agent" and outbound_reason is not None:
        initial_name = id_to_name.get(workflow.initial_state, workflow.initial_state)
        opener = (
            f"[STATE: {initial_name} → {initial_name}]\n"
            f"Hello, this is {domain_name} support reaching out. "
            f"I'm calling to {outbound_reason.description}."
        )
        messages.append({
            "role": "assistant",
            "content": opener,
            "annotations": {"state_transition": {"from": initial_name, "to": initial_name}},
        })
```

Add an outbound user-response template table directly after the existing `_user_templates` dict (after line 789, before `lang_templates = ...`):

```python
    _outbound_user_templates: dict[str, dict[str, str]] = {
        "en": {
            "cooperative": "[Turn {t}] Oh, hi — yes, I have a moment. What about {intent}?",
            "adversarial_probing": "[Turn {t}] How did you get my number? Can we skip {state} and get to the point?",
            "digressing": "[Turn {t}] Before that — unrelated, but I had another question first.",
            "invalid_tool_inputs": "[Turn {t}] Sure, my reference is ###invalid_id### if you need it.",
        },
        "th": {
            "cooperative": "[ตา {t}] อ้อ สวัสดีค่ะ ว่างพอดี เรื่อง{intent}ใช่ไหมคะ?",
            "adversarial_probing": "[ตา {t}] ได้เบอร์ฉันมาจากไหน? ข้าม {state} แล้วเข้าเรื่องเลยได้ไหม?",
            "digressing": "[ตา {t}] เดี๋ยวก่อนนะคะ ขอถามเรื่องอื่นก่อนได้ไหม",
            "invalid_tool_inputs": "[ตา {t}] ได้ค่ะ รหัสอ้างอิงของฉันคือ ###invalid_id### นะคะ",
        },
        "code_switch": {
            "cooperative": "[ตา {t}] อ้อ hi ค่ะ ว่างพอดี เรื่อง {intent} ใช่ไหมคะ?",
            "adversarial_probing": "[ตา {t}] ได้ number ฉันมาจากไหนคะ? ข้าม {state} แล้ว get to the point เลยได้ไหม?",
            "digressing": "[ตา {t}] เดี๋ยวก่อนค่ะ ขอถาม unrelated question ก่อนนะคะ",
            "invalid_tool_inputs": "[ตา {t}] ได้ค่ะ my reference คือ ###invalid_id### นะคะ",
        },
    }
```

Finally, inside the `for step in path:` loop, replace the template pick at line 803:

```python
        tmpl = lang_templates.get(behavior, lang_templates["cooperative"])
```

with an outbound-aware pick (the customer's *first* reply reacts to the outreach):

```python
        if initiator == "agent" and turn_idx == 0:
            resp = _outbound_user_templates.get(language or "en", _outbound_user_templates["en"])
            tmpl = resp.get(behavior, resp["cooperative"])
        else:
            tmpl = lang_templates.get(behavior, lang_templates["cooperative"])
```

- [ ] **Step 4: Run to confirm it passes**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestPlaceholderOutbound -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py tests/unit/test_data_generation.py
git commit -m "feat(data): placeholder generator emits outbound opener + response templates"
```

---

## Task 7: Teacher prompt outbound block

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py` (`_build_teacher_prompt` lines 893-933; `_generate_teacher_conversation` signature lines 1032+ and its fallback line 1062)
- Test: `tests/unit/test_data_generation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_data_generation.py`:

```python
class TestTeacherPromptOutbound:
    def _prompt(self, initiator, reason=None):
        import random
        from llm_workflow_agents.config.schema import COMPLEXITY_SPECS, ComplexityLevel
        from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY, OutboundReason
        rng = random.Random(0)
        spec = COMPLEXITY_SPECS[ComplexityLevel.L2]
        dom = DOMAIN_REGISTRY["sales"]
        wf = gw.select_subgraph(dom, spec, rng, "service")
        tools = [t for t in dom.tools]
        r = reason or OutboundReason("promotion_offer", "offer a promotion", "upsell_promo")
        return gw._build_teacher_prompt(
            wf, tools, "cooperative", spec, dom, "en", "service",
            initiator=initiator, outbound_reason=(r if initiator == "agent" else None),
        )

    def test_outbound_prompt_mentions_agent_initiation(self):
        p = self._prompt("agent")
        assert "OUTBOUND" in p
        assert "offer a promotion" in p

    def test_inbound_prompt_has_no_outbound_block(self):
        p = self._prompt("user")
        assert "OUTBOUND" not in p
```

- [ ] **Step 2: Run to confirm it fails**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestTeacherPromptOutbound -q`
Expected: FAIL — `_build_teacher_prompt() got an unexpected keyword argument 'initiator'`.

- [ ] **Step 3: Add params and the outbound block**

Change `_build_teacher_prompt` signature (line 893) to add two params at the end:

```python
def _build_teacher_prompt(
    workflow: WorkflowGraph,
    tool_schemas: list[dict[str, Any]],
    behavior: str,
    spec: ComplexitySpec,
    domain_spec: DomainSpec | None,
    language: str = "en",
    intent_category: str = "service",
    initiator: str = "user",
    outbound_reason: "OutboundReason | None" = None,
) -> str:
```

After the `promo_line = (...)` assignment (ends line 912), add:

```python
    outbound_line = ""
    if initiator == "agent" and outbound_reason is not None:
        outbound_line = (
            "Conversation initiation: OUTBOUND. The support AGENT initiates this contact. "
            "The FIRST message after the system message MUST be the assistant introducing "
            f"themselves and stating the reason for reaching out — {outbound_reason.description}. "
            "The customer responds only after that. The workflow must still reach a terminal state.\n"
        )
```

Insert `outbound_line` into the returned string — change the `f"User behavior: {behavior}\n"` line (923) region so the block reads:

```python
        f"User behavior: {behavior}\n"
        f"{outbound_line}"
        f"{promo_line}"
```

- [ ] **Step 4: Thread params through `_generate_teacher_conversation`**

`_generate_teacher_conversation` (def ~line 1032) calls `_build_teacher_prompt` and has a placeholder fallback at line 1062. Add `initiator: str = "user"` and `outbound_reason: "OutboundReason | None" = None` to its signature, pass them into the `_build_teacher_prompt(...)` call, and pass them into the fallback call at line 1062:

```python
        return _generate_placeholder_conversation(
            workflow, tool_schemas, behavior, spec, rng, domain_spec, language,
            intent_category, initiator, outbound_reason,
        )
```

- [ ] **Step 5: Run to confirm it passes**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestTeacherPromptOutbound -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py tests/unit/test_data_generation.py
git commit -m "feat(data): teacher prompt outbound-initiation instruction block"
```

---

## Task 8: Wire `initiation_preset` through `generate_workflow_dataset`

**Files:**
- Modify: `src/llm_workflow_agents/data/generate_workflows.py` (signature line 1067; validation ~1131; loop 1172-1311; stats ~1318)
- Test: `tests/unit/test_data_generation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_data_generation.py`:

```python
class TestGenerateOutboundDataset:
    def test_outbound_heavy_emits_agent_samples(self, tmp_output_dir: Path) -> None:
        meta = generate_workflow_dataset(
            complexity_level="L2", num_samples=30,
            initiation_preset="outbound_heavy",
            output_dir=tmp_output_dir, seed=7,
        )
        agent_samples = []
        with open(meta.output_files[0]) as f:
            for line in f:
                s = json.loads(line)
                if s["conversation_initiator"] == "agent":
                    agent_samples.append(s)
        assert agent_samples, "expected at least one outbound sample"
        for s in agent_samples:
            assert s["messages"][1]["role"] == "assistant"   # opener
            assert s["outbound_reason"]                        # reason recorded
            # outbound only chosen for outbound-capable domains
            from llm_workflow_agents.data.domain_registry import DOMAIN_REGISTRY
            assert DOMAIN_REGISTRY[s["domain"]].outbound_reasons

    def test_default_preset_is_all_inbound(self, tmp_output_dir: Path) -> None:
        meta = generate_workflow_dataset(
            complexity_level="L1", num_samples=10,
            output_dir=tmp_output_dir, seed=1,
        )
        with open(meta.output_files[0]) as f:
            for line in f:
                s = json.loads(line)
                assert s["conversation_initiator"] == "user"
                assert s["messages"][1]["role"] == "user"

    def test_unknown_initiation_preset_raises(self, tmp_output_dir: Path) -> None:
        with pytest.raises(ValueError, match="initiation_preset"):
            generate_workflow_dataset(
                complexity_level="L1", num_samples=1,
                initiation_preset="bogus", output_dir=tmp_output_dir, seed=1,
            )
```

- [ ] **Step 2: Run to confirm it fails**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestGenerateOutboundDataset -q`
Expected: FAIL — `unexpected keyword argument 'initiation_preset'`.

- [ ] **Step 3: Add the parameter and its validation**

Add `initiation_preset: str = "default"` to the signature (after `intent_category_preset` line 1077). After the `intent_category_preset` validation block (ends line 1135), add:

```python
    if initiation_preset not in INITIATION_PRESETS:
        raise ValueError(
            f"Unknown initiation_preset {initiation_preset!r}. "
            f"Valid options: {list(INITIATION_PRESETS)}"
        )
    active_initiation_dist = INITIATION_PRESETS[initiation_preset]
```

- [ ] **Step 4: Add counters and per-sample selection**

After `intent_category_counts = {...}` (line 1165), add:

```python
    initiator_counts: dict[str, int] = {"user": 0, "agent": 0}
    outbound_reason_counts: dict[str, int] = {}
    outbound_fallback_inbound = 0
```

Replace the domain + intent-category selection block (lines 1180-1188) with initiator-aware selection:

```python
        # Decide who initiates this conversation.
        initiator = _select_initiator(rng, active_initiation_dist)
        outbound_reason = None

        if initiator == "agent":
            domain_key, domain_spec = _select_domain(rng, domain, spec, outbound_only=True)
            if domain_spec.outbound_reasons:
                outbound_reason = rng.choice(list(domain_spec.outbound_reasons))
                intent_category = outbound_reason.intent_category
            else:
                # Pinned/eligible domain has no outbound reasons → fall back to inbound.
                initiator = "user"
                outbound_fallback_inbound += 1
                intent_category = _select_intent_category(rng, active_intent_dist)
        else:
            domain_key, domain_spec = _select_domain(rng, domain, spec)
            intent_category = _select_intent_category(rng, active_intent_dist)

        domain_counts[domain_key] = domain_counts.get(domain_key, 0) + 1
        intent_category_counts[intent_category] = intent_category_counts.get(intent_category, 0) + 1
        initiator_counts[initiator] += 1
        if outbound_reason is not None:
            outbound_reason_counts[outbound_reason.key] = \
                outbound_reason_counts.get(outbound_reason.key, 0) + 1

        workflow = select_subgraph(domain_spec, spec, rng, intent_category)
```

- [ ] **Step 5: Thread initiator/reason into generation calls**

Update the `_placeholder()` closure (lines 1201-1205) to pass the new args:

```python
        def _placeholder() -> list[dict[str, Any]]:
            return _generate_placeholder_conversation(
                workflow, tool_schemas, behavior, spec, rng, domain_spec, sample_language,
                intent_category, initiator, outbound_reason,
            )
```

Update both `_generate_teacher_conversation(...)` calls (lines 1208-1211 and 1242-1245) to append `initiator, outbound_reason` as the final two positional args, e.g.:

```python
            messages = _generate_teacher_conversation(
                workflow, tool_schemas, behavior, spec, rng, domain_spec, teacher_model,
                sample_language, intent_category, initiator, outbound_reason,
            )
```

- [ ] **Step 6: Record fields on the sample + stats**

In the `ConversationSample(...)` construction (lines 1296-1310), add two kwargs after `ground_truth=...`:

```python
            conversation_initiator=initiator,
            outbound_reason=(outbound_reason.key if outbound_reason else None),
```

In the `stats = {...}` dict (starts line 1318), add three keys:

```python
        "initiator_distribution": initiator_counts,
        "outbound_reason_distribution": outbound_reason_counts,
        "outbound_fallback_inbound": outbound_fallback_inbound,
```

- [ ] **Step 7: Run to confirm it passes**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestGenerateOutboundDataset -q`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/llm_workflow_agents/data/generate_workflows.py tests/unit/test_data_generation.py
git commit -m "feat(data): initiation_preset wiring in generate_workflow_dataset"
```

---

## Task 9: Validator initiator-aware `messages[1]` check

**Files:**
- Modify: `src/llm_workflow_agents/data/data_validator.py` (`_validate_workflow_sample` lines 66-71)
- Test: `tests/unit/test_data_generation.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_data_generation.py`:

```python
class TestValidatorOutbound:
    def _base(self, msgs, initiator):
        return {
            "conversation_id": "L1_001", "complexity_level": "L1", "domain": "sales",
            "workflow_graph": {"states": ["GREETING", "TERMINAL"],
                               "initial": "GREETING", "terminal": ["TERMINAL"]},
            "tool_schemas": [], "messages": msgs, "user_behavior": "cooperative",
            "ground_truth": {}, "conversation_initiator": initiator,
        }

    def test_outbound_with_assistant_second_passes(self):
        from llm_workflow_agents.data.data_validator import _validate_workflow_sample
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "[STATE: GREETING → GREETING]\nHi, calling about X."},
            {"role": "user", "content": "ok"},
        ]
        errs = _validate_workflow_sample(self._base(msgs, "agent"), 0)
        assert not any("second message" in e for e in errs)

    def test_outbound_with_user_second_fails(self):
        from llm_workflow_agents.data.data_validator import _validate_workflow_sample
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hello?"},
        ]
        errs = _validate_workflow_sample(self._base(msgs, "agent"), 0)
        assert any("second message" in e for e in errs)
```

- [ ] **Step 2: Run to confirm it fails**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestValidatorOutbound -q`
Expected: FAIL — the user-second outbound sample currently produces no "second message" error.

- [ ] **Step 3: Add the check**

In `src/llm_workflow_agents/data/data_validator.py`, after the existing first-message block (lines 67-71), add:

```python
    # Second-message role depends on who initiated the conversation.
    initiator = sample.get("conversation_initiator", "user")
    if len(messages) > 1:
        second_role = messages[1].get("role")
        if initiator == "agent":
            if second_role != "assistant":
                errors.append(
                    f"Sample {idx}: outbound (agent-initiated) conversation must have "
                    f"an assistant second message, got '{second_role}'"
                )
        elif second_role != "user":
            errors.append(
                f"Sample {idx}: inbound conversation must have a user second message, "
                f"got '{second_role}'"
            )
```

- [ ] **Step 4: Run to confirm it passes**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py::TestValidatorOutbound -q`
Expected: PASS.

- [ ] **Step 5: Run the full data-generation suite for regressions**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_data_generation.py -q`
Expected: PASS (no regressions in existing inbound tests, including `test_messages_start_with_system`).

- [ ] **Step 6: Commit**

```bash
git add src/llm_workflow_agents/data/data_validator.py tests/unit/test_data_generation.py
git commit -m "feat(data): validator enforces initiator-aware second-message role"
```

---

## Task 10: GRPO loader keeps the outbound opener as a training row

**Files:**
- Modify: `src/llm_workflow_agents/training/grpo.py` (filter lines 248-251)
- Test: `tests/unit/test_grpo_outbound.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_grpo_outbound.py`:

```python
"""GRPO loader must keep the outbound opener (assistant preceded by system)."""

from __future__ import annotations

import json
from pathlib import Path

from llm_workflow_agents.training.grpo import _load_grpo_jsonl


def _write(tmp_path: Path) -> Path:
    conv = {
        "messages": [
            {"role": "system", "content": "You are a sales agent."},
            {
                "role": "assistant",
                "content": "[STATE: GREETING → GREETING]\nHi, calling to offer a promotion.",
                "annotations": {"state_transition": {"from": "GREETING", "to": "GREETING"}},
            },
            {"role": "user", "content": "Oh, sure."},
            {
                "role": "assistant",
                "content": "[STATE: GREETING → QUALIFY_LEAD]\nGreat, let me check your account.",
                "annotations": {"state_transition": {"from": "GREETING", "to": "QUALIFY_LEAD"}},
            },
        ],
        "conversation_initiator": "agent",
        "ground_truth": {"terminal_state": "", "terminal_reached": False},
    }
    p = tmp_path / "train.jsonl"
    p.write_text(json.dumps(conv) + "\n")
    return tmp_path


def test_opener_becomes_a_row_with_system_only_prompt(tmp_path):
    _write(tmp_path)
    ds = _load_grpo_jsonl(tmp_path, split="train")
    prompts = [r["prompt"] for r in ds]
    # The opener row's prompt is exactly the system message.
    assert any(len(p) == 1 and p[0]["role"] == "system" for p in prompts), \
        f"opener row missing; prompts={prompts}"
    # Two assistant turns → two rows.
    assert len(ds) == 2
```

- [ ] **Step 2: Run to confirm it fails**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_grpo_outbound.py -q`
Expected: FAIL — only 1 row; the opener (preceded by `system`) is dropped, so no system-only prompt exists.

- [ ] **Step 3: Widen the filter**

In `src/llm_workflow_agents/training/grpo.py`, change the `valid_pairs` comprehension (lines 248-251) from:

```python
            valid_pairs = [
                i for i in asst_indices
                if i > 0 and raw_msgs[i - 1].get("role") == "user"
            ]
```

to:

```python
            valid_pairs = [
                i for i in asst_indices
                if i > 0 and raw_msgs[i - 1].get("role") in ("user", "system")
            ]
```

This admits the outbound opener (assistant preceded by `system`) while still skipping
mid-state tool-preceded assistant turns. Inbound data is unaffected — inbound has no
system-preceded assistant turn.

- [ ] **Step 4: Run to confirm it passes**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_grpo_outbound.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/llm_workflow_agents/training/grpo.py tests/unit/test_grpo_outbound.py
git commit -m "fix(grpo): keep outbound opener (system-preceded assistant) as a training row"
```

---

## Task 11: SFT masking keeps the opener unmasked (lock behavior)

**Files:**
- Test: `tests/unit/test_sft_outbound.py` (create) — no production change expected
- Modify (only if Step 2 reveals a gap): `src/llm_workflow_agents/training/sft.py`

- [ ] **Step 1: Write the test**

Create `tests/unit/test_sft_outbound.py`:

```python
"""SFT response-only masking must keep the outbound opener (assistant) unmasked."""

from __future__ import annotations

from llm_workflow_agents.training.sft import render_response_only_sample


class _StubTok:
    """Deterministic, prefix-extending chat template: 1 token per char."""

    def apply_chat_template(self, msgs, tokenize=True, add_generation_prompt=False):
        s = "".join(f"{m['role']}|{m['content']}\n" for m in msgs)
        return [ord(c) % 256 for c in s]


def test_outbound_opener_tokens_are_unmasked():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "[STATE: G -> G] Hi, calling about X."},
        {"role": "user", "content": "ok"},
        {"role": "assistant", "content": "[STATE: G -> V] Let me check."},
    ]
    out = render_response_only_sample(messages, _StubTok(), max_seq_length=10_000)
    labels = out["labels"]
    assert len(labels) == len(out["input_ids"])
    # At least some tokens are unmasked (the two assistant turns).
    assert any(l != -100 for l in labels)
    # System tokens (the prefix) are masked.
    assert labels[0] == -100


def test_system_then_assistant_only_keeps_assistant():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "opener"},
    ]
    out = render_response_only_sample(messages, _StubTok(), max_seq_length=10_000)
    # The very last token belongs to the assistant opener → unmasked.
    assert out["labels"][-1] != -100
```

- [ ] **Step 2: Run the test**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_sft_outbound.py -q`
Expected: PASS with **no production change** — `render_response_only_sample` is already
role-agnostic (keeps `assistant`, masks the rest). If it fails, fix `sft.py` so an
assistant turn at index 1 is preserved, then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_sft_outbound.py
git commit -m "test(sft): lock outbound-opener masking behavior"
```

> **Real-tokenizer caveat (manual, not a unit test):** Some HF chat templates require a
> user-first turn and may reject `[system, assistant, …]` in `apply_chat_template`
> (notably Gemma → `model`, Mistral). This must be verified against each Phase-1 winner's
> tokenizer (see Verification step 3). If a family rejects the ordering, exclude outbound
> rows from that family's SFT conversion and document it in `.claude/rules/03-training.md`.

---

## Task 12: Chat-template conversion for outbound + docs

**Files:**
- Modify: `tests/unit/test_chat_templates.py`
- Modify: `.claude/rules/02-data-generation.md`

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/test_chat_templates.py`. The public entry point is
`convert_to_model_format(input_jsonl: Path, model_family: ModelFamilyName, output_path: Path)
-> ConversionStats`; it reads/writes JSONL files:

```python
def test_outbound_conversation_converts_for_gemma(tmp_path):
    import json
    from pathlib import Path
    from llm_workflow_agents.data.chat_template_converter import convert_to_model_format
    src = tmp_path / "outbound.jsonl"
    src.write_text(json.dumps({
        "messages": [
            {"role": "system", "content": "You are support."},
            {"role": "assistant", "content": "[STATE: G -> G] Hi, calling about your refill."},
            {"role": "user", "content": "Oh, hi."},
            {"role": "assistant", "content": "[STATE: G -> V] Let me verify you."},
        ],
        "conversation_initiator": "agent",
    }) + "\n")
    out_path = tmp_path / "gemma.jsonl"
    convert_to_model_format(src, "gemma", out_path)
    rows = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    roles = [m["role"] for m in rows[0]["messages"]]
    # Gemma renames assistant -> model; ordering is preserved, system stays first.
    assert roles[0] == "system"
    assert roles[1] == "model"
```

- [ ] **Step 2: Run to confirm it fails or passes**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_chat_templates.py -q`
Expected: PASS — the converters in `chat_template_converter.py` are role-agnostic, so this
locks the behavior. If it fails on signature mismatch, fix the test call (not production).

- [ ] **Step 3: Update the data-generation rule doc**

In `.claude/rules/02-data-generation.md`, under Task A, add documentation for:
- The new `initiation_preset: str = "default"` parameter on `generate_workflow_dataset`,
  with the `INITIATION_PRESETS` table (`default` 100/0, `balanced` 70/30, `outbound_heavy` 40/60).
- The `OutboundReason(key, description, intent_category)` schema and the `outbound_reasons`
  field on `DomainSpec`, listing the 7 curated domains.
- The new per-sample output fields `conversation_initiator` ("user" | "agent") and
  `outbound_reason` (key | null), and the outbound message shape
  `[system, assistant(opener), user, assistant, …]`.

- [ ] **Step 4: Run both test files**

Run: `source .venv/bin/activate && python -m pytest tests/unit/test_chat_templates.py tests/unit/test_data_generation.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_chat_templates.py .claude/rules/02-data-generation.md
git commit -m "test(data): outbound chat-template conversion + document initiation_preset"
```

---

## Final Verification

- [ ] **Full unit suite green**

Run: `source .venv/bin/activate && python -m pytest tests/unit -q`
Expected: PASS.

- [ ] **End-to-end smoke (real generation, eyeball outbound)**

```bash
source .venv/bin/activate && python -c "
from llm_workflow_agents.data.generate_workflows import generate_workflow_dataset
from llm_workflow_agents.data.data_validator import validate_dataset
from pathlib import Path
import json
m = generate_workflow_dataset('L2', num_samples=20, initiation_preset='outbound_heavy',
                              output_dir=Path('data/output/_outbound_smoke'), seed=42)
print('stats:', m.stats.get('initiator_distribution'), m.stats.get('outbound_reason_distribution'))
rows = [json.loads(l) for l in m.output_files[0].read_text().splitlines()]
ob = [r for r in rows if r['conversation_initiator']=='agent'][0]
print('opener:', ob['messages'][1]['role'], '|', ob['messages'][1]['content'][:80])
print('validate:', validate_dataset(m.output_files[0], dataset_type='workflow'))
"
```
Expected: outbound samples present; `messages[1].role == "assistant"`; opener states the
reason; validation reports no errors.

- [ ] **Per-family `apply_chat_template` check (SFT risk)**

For each Phase-1 winner tokenizer, render an outbound conversation through
`render_response_only_sample` with the real tokenizer and confirm no exception. For any
family that rejects system→assistant, apply the documented fallback (Task 11 caveat).

- [ ] **GRPO loader sanity**

Confirm `skipped_tool_preceded_turns` in the `grpo_data_loaded` log does **not** count the
outbound opener (it now becomes a row).

---

## Self-Review Notes (for the implementer)

- **Type consistency:** `OutboundReason` is defined in Task 1 and imported into
  `generate_workflows.py` in Task 6; its fields (`key`, `description`, `intent_category`)
  are used unchanged in Tasks 2, 6, 7, 8.
- **Back-compat:** every new parameter defaults to the inbound behavior
  (`initiation_preset="default"` → 100% user; `conversation_initiator="user"`;
  `outbound_reasons=()`), so existing datasets, tests, and the GRPO/SFT loaders are
  unaffected unless outbound is explicitly requested.
- **No graph changes:** outbound reuses each domain's canonical graph; the opener is an
  in-state `[STATE: initial → initial]` turn, so `validate_domain` and `select_subgraph`
  are untouched.
