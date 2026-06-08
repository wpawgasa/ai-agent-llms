"""Domain registry for workflow conversation generation.

18 call center domains with tool schemas, state templates, intent examples,
and entity slots. Domains are decoupled from complexity levels — any domain
can appear at any L1–L5 complexity, with the complexity controlling structural
parameters (states, branching, tools, depth) and the domain controlling
content (tool schemas, state names, intents).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# New semantic graph schema (StateNode, Edge, updated DomainSpec)
# ---------------------------------------------------------------------------

_VALID_TRIGGERS = frozenset({
    "always", "tool_success", "tool_error",
    "intent_match", "slot_present", "user_declines",
})


@dataclass(frozen=True)
class StateNode:
    """A single state in the canonical domain workflow graph."""

    name: str
    instruction: str
    tools: tuple[str, ...] = ()
    kind: str = "working"  # "initial" | "working" | "terminal"


@dataclass(frozen=True)
class Edge:
    """A directed transition in the canonical domain workflow graph."""

    src: str
    dst: str
    label: str          # human-readable authored label
    trigger: str        # one of _VALID_TRIGGERS
    optional: bool = False
    priority: int = 0
    intent_category: str | None = None  # set to "upsell_promo" on upsell-arc edges


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


@dataclass(frozen=True)
class DomainSpec:
    """Specification for a single call center domain.

    Supports both legacy flat-dict schema (state_templates / state_tools /
    state_instructions) and the new semantic-graph schema (states / edges).
    During migration, legacy fields default to empty; new fields default to
    empty tuples/strings. ``validate_domain`` only runs invariant checks on
    domains that have ``states`` populated (i.e., migrated to the new schema).
    """

    name: str
    category: str
    tools: tuple[dict[str, Any], ...]
    intents: tuple[str, ...]
    entity_slots: tuple[str, ...] = ()
    # Legacy flat-dict fields (unmigrated domains; kept for backward compat)
    state_templates: tuple[str, ...] = ()
    state_tools: dict[str, tuple[str, ...]] = field(default_factory=dict)
    state_instructions: dict[str, str] = field(default_factory=dict)
    # New semantic graph fields
    states: tuple[StateNode, ...] = ()
    edges: tuple[Edge, ...] = ()
    initial: str = ""
    terminals: tuple[str, ...] = ()
    intent_categories: dict[str, str] = field(default_factory=dict)
    outbound_reasons: tuple[OutboundReason, ...] = ()


def validate_domain(domain: DomainSpec) -> None:
    """Enforce structural invariants on a new-schema domain.

    Skips silently for unmigrated domains (empty ``states``).
    Raises ValueError on any violation.
    Called at module import for every migrated domain.
    """
    if not domain.states:
        return  # unmigrated domain — skip

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
                f"but state '{e.src}' has no tools — tool_success/tool_error "
                f"requires the source state to have at least one tool"
            )

    # initial state must have kind="initial"
    initial_states = {s.name for s in domain.states if s.kind == "initial"}
    if not domain.initial or domain.initial not in initial_states:
        raise ValueError(
            f"{domain.name}: 'initial' field '{domain.initial}' must name a state "
            f"with kind='initial'"
        )

    # terminal states must have kind="terminal"
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

    # Reachability from initial (BFS)
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

    # Every state must be able to reach at least one terminal (reverse BFS)
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

    # Upsell-arc edges must rejoin toward a terminal
    for e in domain.edges:
        if e.intent_category == "upsell_promo" and e.dst not in can_reach_terminal:
            raise ValueError(
                f"{domain.name}: upsell-arc edge {e.src}->{e.dst} dst cannot reach "
                f"any terminal"
            )


def _tool(name: str, desc: str, params: dict[str, Any], required: list[str]) -> dict[str, Any]:
    """Build an OpenAI-style function tool schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": required,
            },
        },
    }


# ---------------------------------------------------------------------------
# Core Business Domains
# ---------------------------------------------------------------------------

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

BILLING_PAYMENTS = DomainSpec(
    name="Billing & Payments",
    category="core_business",
    tools=(
        _tool("lookup_invoice", "Look up invoice details", {
            "invoice_id": {"type": "string"}, "customer_id": {"type": "string"},
        }, ["invoice_id"]),
        _tool("process_payment", "Process a payment", {
            "invoice_id": {"type": "string"}, "amount": {"type": "number"},
            "payment_method": {"type": "string", "enum": ["credit_card", "debit_card", "bank_transfer", "digital_wallet"]},
        }, ["invoice_id", "amount", "payment_method"]),
        _tool("issue_refund", "Issue a refund to customer", {
            "transaction_id": {"type": "string"}, "amount": {"type": "number"},
            "reason": {"type": "string"}, "refund_method": {"type": "string", "enum": ["original_method", "credit", "check"]},
        }, ["transaction_id", "amount", "reason"]),
        _tool("setup_payment_plan", "Set up installment payment plan", {
            "customer_id": {"type": "string"}, "total_amount": {"type": "number"},
            "num_installments": {"type": "integer"}, "start_date": {"type": "string", "format": "date"},
        }, ["customer_id", "total_amount", "num_installments"]),
        _tool("waive_late_fee", "Waive a late payment fee", {
            "invoice_id": {"type": "string"}, "fee_amount": {"type": "number"},
            "waiver_reason": {"type": "string"},
        }, ["invoice_id", "waiver_reason"]),
        _tool("generate_receipt", "Generate payment receipt or tax document", {
            "transaction_id": {"type": "string"}, "document_type": {"type": "string", "enum": ["receipt", "tax_invoice", "statement"]},
        }, ["transaction_id", "document_type"]),
        _tool("dispute_charge", "File a billing dispute", {
            "invoice_id": {"type": "string"}, "disputed_amount": {"type": "number"},
            "dispute_reason": {"type": "string"},
        }, ["invoice_id", "disputed_amount", "dispute_reason"]),
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
        Edge("GREETING", "VERIFY_IDENTITY", "proceed to identity verification", "always"),
        Edge("VERIFY_IDENTITY", "LOOKUP_BILLING", "identity confirmed", "always"),
        Edge("LOOKUP_BILLING", "REVIEW_CHARGES", "invoice retrieved", "tool_success"),
        Edge("REVIEW_CHARGES", "PROCESS_PAYMENT", "charges reviewed, proceed to payment", "tool_success"),
        Edge("REVIEW_CHARGES", "APPLY_ADJUSTMENT", "dispute or adjustment requested", "intent_match", optional=True, priority=1),
        Edge("APPLY_ADJUSTMENT", "CONFIRM_ACTION", "adjustment applied", "tool_success"),
        Edge("PROCESS_PAYMENT", "CONFIRM_ACTION", "payment processed successfully", "tool_success"),
        Edge("PROCESS_PAYMENT", "ESCALATE", "payment failed, escalating", "tool_error", optional=True, priority=1),
        Edge("CONFIRM_ACTION", "RESOLVE", "action confirmed by customer", "always"),
        Edge("CONFIRM_ACTION", "GENERATE_DOCUMENT", "receipt or tax document requested", "slot_present", optional=True, priority=1),
        Edge("GENERATE_DOCUMENT", "RESOLVE", "document generated", "tool_success"),
        Edge("ESCALATE", "RESOLVE", "issue escalated to specialist", "always"),
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

ORDER_MANAGEMENT = DomainSpec(
    name="Order Management",
    category="core_business",
    tools=(
        _tool("lookup_order", "Look up order details by order ID", {
            "order_id": {"type": "string"},
        }, ["order_id"]),
        _tool("track_delivery", "Track delivery status of an order", {
            "order_id": {"type": "string"}, "tracking_number": {"type": "string"},
        }, ["order_id"]),
        _tool("cancel_order", "Cancel an active order", {
            "order_id": {"type": "string"}, "reason": {"type": "string"},
        }, ["order_id", "reason"]),
        _tool("modify_order", "Modify an existing order", {
            "order_id": {"type": "string"}, "modifications": {"type": "object"},
        }, ["order_id", "modifications"]),
        _tool("initiate_return", "Initiate a return or exchange", {
            "order_id": {"type": "string"}, "items": {"type": "array", "items": {"type": "string"}},
            "return_type": {"type": "string", "enum": ["return", "exchange"]},
            "reason": {"type": "string"},
        }, ["order_id", "items", "return_type"]),
        _tool("report_damaged_item", "Report a damaged or missing item", {
            "order_id": {"type": "string"}, "item_id": {"type": "string"},
            "issue_type": {"type": "string", "enum": ["damaged", "missing", "wrong_item"]},
            "description": {"type": "string"},
        }, ["order_id", "item_id", "issue_type"]),
        _tool("reschedule_delivery", "Reschedule delivery date/time", {
            "order_id": {"type": "string"}, "new_date": {"type": "string", "format": "date"},
            "time_slot": {"type": "string"},
        }, ["order_id", "new_date"]),
    ),
    intents=(
        "order_tracking", "order_cancellation", "order_modification",
        "return_request", "exchange_request", "damaged_item_report",
        "delivery_reschedule", "warranty_claim", "accessory_upsell",
    ),
    entity_slots=("order_id", "tracking_number", "item_id", "return_type"),
    states=(
        StateNode("GREETING", "Greet the customer and ask which order they need help with.", kind="initial"),
        StateNode("VERIFY_IDENTITY", "Confirm the customer's identity and the order owner."),
        StateNode("LOOKUP_ORDER", "Look up the order by its ID.", tools=("lookup_order",)),
        StateNode("CHECK_STATUS", "Check the order's current status or delivery tracking.", tools=("track_delivery",)),
        StateNode("PROCESS_MODIFICATION", "Modify or cancel the order as requested.", tools=("modify_order", "cancel_order")),
        StateNode("INITIATE_RETURN", "Start a return, exchange, or damaged-item report.", tools=("initiate_return", "report_damaged_item")),
        StateNode("ARRANGE_DELIVERY", "Reschedule or re-track delivery as needed.", tools=("reschedule_delivery", "track_delivery")),
        StateNode("CONFIRM_ACTION", "Summarise the order action and ask the customer to confirm."),
        StateNode("ESCALATE", "Escalate unresolved order issues to a specialist."),
        StateNode("RESOLVE", "Confirm the order matter is resolved."),
        StateNode("TERMINAL", "Thank the customer and close the conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "VERIFY_IDENTITY", "proceed to identity verification", "always"),
        Edge("VERIFY_IDENTITY", "LOOKUP_ORDER", "identity confirmed", "always"),
        Edge("LOOKUP_ORDER", "CHECK_STATUS", "order retrieved", "tool_success"),
        Edge("CHECK_STATUS", "PROCESS_MODIFICATION", "status checked, proceeding with modification", "tool_success"),
        Edge("CHECK_STATUS", "INITIATE_RETURN", "return or damaged-item request", "intent_match", optional=True, priority=1),
        Edge("CHECK_STATUS", "ARRANGE_DELIVERY", "delivery reschedule requested", "intent_match", optional=True, priority=2),
        Edge("PROCESS_MODIFICATION", "CONFIRM_ACTION", "modification applied", "tool_success"),
        Edge("PROCESS_MODIFICATION", "ESCALATE", "modification failed, escalating", "tool_error", optional=True, priority=1),
        Edge("INITIATE_RETURN", "CONFIRM_ACTION", "return or exchange initiated", "tool_success"),
        Edge("ARRANGE_DELIVERY", "CONFIRM_ACTION", "delivery rescheduled", "tool_success"),
        Edge("CONFIRM_ACTION", "RESOLVE", "action confirmed by customer", "always"),
        Edge("ESCALATE", "RESOLVE", "issue escalated to specialist", "always"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "accessory upsell accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "order_tracking": "service",
        "order_cancellation": "service",
        "order_modification": "service",
        "return_request": "service",
        "exchange_request": "service",
        "damaged_item_report": "service",
        "delivery_reschedule": "service",
        "warranty_claim": "service",
        "accessory_upsell": "upsell_promo",
    },
)

TECHNICAL_SUPPORT = DomainSpec(
    name="Technical Support",
    category="core_business",
    tools=(
        _tool("check_system_status", "Check status of a system or service", {
            "system_name": {"type": "string"},
            "check_type": {"type": "string", "enum": ["health", "connectivity", "performance"]},
        }, ["system_name"]),
        _tool("run_diagnostic", "Run a diagnostic test", {
            "system_name": {"type": "string"}, "diagnostic_type": {"type": "string"},
            "verbose": {"type": "boolean", "default": False},
        }, ["system_name", "diagnostic_type"]),
        _tool("apply_fix", "Apply a known fix to an issue", {
            "system_name": {"type": "string"}, "fix_id": {"type": "string"},
            "force": {"type": "boolean", "default": False},
        }, ["system_name", "fix_id"]),
        _tool("check_compatibility", "Check device or software compatibility", {
            "device_model": {"type": "string"}, "target_version": {"type": "string"},
        }, ["device_model", "target_version"]),
        _tool("create_bug_report", "Create a bug report", {
            "title": {"type": "string"}, "description": {"type": "string"},
            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "steps_to_reproduce": {"type": "string"},
        }, ["title", "severity"]),
        _tool("restart_service", "Restart a system service", {
            "service_name": {"type": "string"}, "graceful": {"type": "boolean", "default": True},
        }, ["service_name"]),
        _tool("escalate_to_engineer", "Escalate to engineering team", {
            "ticket_id": {"type": "string"}, "priority": {"type": "string", "enum": ["normal", "urgent", "critical"]},
            "notes": {"type": "string"},
        }, ["ticket_id", "priority"]),
    ),
    intents=(
        "setup_help", "troubleshoot", "bug_report", "compatibility_check",
        "update_guidance", "remote_assistance", "escalation",
        "extended_warranty_offer",
    ),
    entity_slots=("system_name", "device_model", "fix_id", "severity", "ticket_id"),
    states=(
        StateNode("GREETING", "Greet the customer and ask what technical problem they're facing.", kind="initial"),
        StateNode("IDENTIFY_ISSUE", "Identify the affected system and check its status or compatibility.", tools=("check_system_status", "check_compatibility")),
        StateNode("COLLECT_DIAGNOSTICS", "Collect diagnostic information about the issue.", tools=("check_system_status",)),
        StateNode("RUN_TESTS", "Run the appropriate diagnostic test.", tools=("run_diagnostic",)),
        StateNode("ATTEMPT_FIX", "Apply a known fix or restart the affected service.", tools=("apply_fix", "restart_service")),
        StateNode("VERIFY_RESOLUTION", "Verify the issue is resolved after the fix.", tools=("check_system_status",)),
        StateNode("ESCALATE_ENGINEERING", "Escalate to engineering and file a bug report if unresolved.", tools=("escalate_to_engineer", "create_bug_report")),
        StateNode("REMOTE_ASSIST", "Guide the customer through remote assistance steps.", tools=("restart_service",)),
        StateNode("CONFIRM_FIX", "Confirm with the customer that the fix worked."),
        StateNode("RESOLVE", "Confirm the technical issue is resolved."),
        StateNode("TERMINAL", "Thank the customer and close the conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "IDENTIFY_ISSUE", "proceed to issue identification", "always"),
        Edge("IDENTIFY_ISSUE", "COLLECT_DIAGNOSTICS", "issue identified", "tool_success"),
        Edge("IDENTIFY_ISSUE", "REMOTE_ASSIST", "remote session requested", "intent_match", optional=True, priority=1),
        Edge("COLLECT_DIAGNOSTICS", "RUN_TESTS", "diagnostics collected", "tool_success"),
        Edge("RUN_TESTS", "ATTEMPT_FIX", "tests complete, applying fix", "tool_success"),
        Edge("ATTEMPT_FIX", "VERIFY_RESOLUTION", "fix applied, verifying", "tool_success"),
        Edge("ATTEMPT_FIX", "ESCALATE_ENGINEERING", "fix not available, escalating", "tool_error", optional=True, priority=1),
        Edge("VERIFY_RESOLUTION", "CONFIRM_FIX", "issue resolved", "tool_success"),
        Edge("VERIFY_RESOLUTION", "ATTEMPT_FIX", "issue persists, retrying fix", "tool_error", optional=True, priority=1),
        Edge("ESCALATE_ENGINEERING", "RESOLVE", "case escalated to engineering", "tool_success"),
        Edge("REMOTE_ASSIST", "ATTEMPT_FIX", "remote session complete", "tool_success"),
        Edge("CONFIRM_FIX", "RESOLVE", "customer confirmed fix", "always"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "extended warranty offer accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "setup_help": "service",
        "troubleshoot": "service",
        "bug_report": "service",
        "compatibility_check": "service",
        "update_guidance": "service",
        "remote_assistance": "service",
        "escalation": "service",
        "extended_warranty_offer": "upsell_promo",
    },
)

PRODUCT_INFO = DomainSpec(
    name="Product & Service Information",
    category="core_business",
    tools=(
        _tool("search_products", "Search product catalog", {
            "query": {"type": "string"}, "category": {"type": "string"},
            "price_range": {"type": "object", "properties": {"min": {"type": "number"}, "max": {"type": "number"}}},
        }, ["query"]),
        _tool("get_product_details", "Get detailed product specifications", {
            "product_id": {"type": "string"},
        }, ["product_id"]),
        _tool("check_availability", "Check product availability", {
            "product_id": {"type": "string"}, "location": {"type": "string"},
        }, ["product_id"]),
        _tool("compare_products", "Compare two or more products", {
            "product_ids": {"type": "array", "items": {"type": "string"}},
        }, ["product_ids"]),
        _tool("get_pricing", "Get current pricing and promotions", {
            "product_id": {"type": "string"}, "customer_tier": {"type": "string", "enum": ["standard", "premium", "enterprise"]},
        }, ["product_id"]),
        _tool("recommend_upgrade", "Recommend product upgrade path", {
            "current_product_id": {"type": "string"}, "usage_profile": {"type": "string"},
        }, ["current_product_id"]),
    ),
    intents=(
        "product_inquiry", "pricing_inquiry", "availability_check",
        "feature_comparison", "upgrade_recommendation", "promotion_inquiry",
    ),
    entity_slots=("product_id", "category", "price_range", "customer_tier"),
    states=(
        StateNode("GREETING", "Greet the customer and ask what product information they need.", kind="initial"),
        StateNode("UNDERSTAND_NEEDS", "Clarify the customer's requirements and use case."),
        StateNode("SEARCH_CATALOG", "Search the product catalog for matching items.", tools=("search_products",)),
        StateNode("PRESENT_OPTIONS", "Present the matching products with their key details.", tools=("get_product_details",)),
        StateNode("COMPARE_FEATURES", "Compare the shortlisted products' features.", tools=("compare_products",)),
        StateNode("CHECK_AVAILABILITY", "Check availability for the chosen product.", tools=("check_availability",)),
        StateNode("QUOTE_PRICING", "Provide current pricing and any promotions.", tools=("get_pricing",)),
        StateNode("RECOMMEND_UPGRADE", "Recommend a suitable upgrade path if relevant.", tools=("recommend_upgrade",)),
        StateNode("RESOLVE", "Confirm the customer has the information they need."),
        StateNode("TERMINAL", "Thank the customer and close the conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "UNDERSTAND_NEEDS", "proceed to needs clarification", "always"),
        Edge("UNDERSTAND_NEEDS", "SEARCH_CATALOG", "needs clarified, searching catalog", "always"),
        Edge("SEARCH_CATALOG", "PRESENT_OPTIONS", "products found", "tool_success"),
        Edge("PRESENT_OPTIONS", "CHECK_AVAILABILITY", "option selected", "tool_success"),
        Edge("PRESENT_OPTIONS", "COMPARE_FEATURES", "multiple options to compare", "slot_present", optional=True, priority=1),
        Edge("COMPARE_FEATURES", "CHECK_AVAILABILITY", "comparison complete", "tool_success"),
        Edge("CHECK_AVAILABILITY", "RESOLVE", "availability confirmed", "tool_success"),
        Edge("CHECK_AVAILABILITY", "QUOTE_PRICING", "pricing details requested", "intent_match", optional=True, priority=1),
        Edge("CHECK_AVAILABILITY", "RECOMMEND_UPGRADE", "upgrade path requested", "intent_match", optional=True, priority=2),
        Edge("QUOTE_PRICING", "RESOLVE", "pricing provided", "tool_success"),
        Edge("RECOMMEND_UPGRADE", "RESOLVE", "upgrade recommended", "tool_success"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "upgrade or promotion offer accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "product_inquiry": "service",
        "pricing_inquiry": "upsell_promo",
        "availability_check": "service",
        "feature_comparison": "service",
        "upgrade_recommendation": "upsell_promo",
        "promotion_inquiry": "upsell_promo",
    },
)

# ---------------------------------------------------------------------------
# Industry-Specific Domains
# ---------------------------------------------------------------------------

HEALTHCARE = DomainSpec(
    name="Healthcare & Insurance",
    category="industry",
    tools=(
        _tool("schedule_appointment", "Schedule a medical appointment", {
            "patient_id": {"type": "string"}, "provider_id": {"type": "string"},
            "appointment_type": {"type": "string", "enum": ["consultation", "follow_up", "procedure", "lab_work"]},
            "preferred_date": {"type": "string", "format": "date"},
        }, ["patient_id", "appointment_type"]),
        _tool("request_prescription_refill", "Request a prescription refill", {
            "patient_id": {"type": "string"}, "prescription_id": {"type": "string"},
            "pharmacy_id": {"type": "string"},
        }, ["patient_id", "prescription_id"]),
        _tool("check_claim_status", "Check insurance claim status", {
            "claim_id": {"type": "string"}, "patient_id": {"type": "string"},
        }, ["claim_id"]),
        _tool("verify_coverage", "Verify insurance benefits and coverage", {
            "patient_id": {"type": "string"}, "procedure_code": {"type": "string"},
            "provider_id": {"type": "string"},
        }, ["patient_id", "procedure_code"]),
        _tool("request_referral", "Request a specialist referral", {
            "patient_id": {"type": "string"}, "specialty": {"type": "string"},
            "reason": {"type": "string"},
        }, ["patient_id", "specialty"]),
        _tool("submit_prior_auth", "Submit prior authorization request", {
            "patient_id": {"type": "string"}, "procedure_code": {"type": "string"},
            "supporting_docs": {"type": "array", "items": {"type": "string"}},
        }, ["patient_id", "procedure_code"]),
    ),
    intents=(
        "appointment_scheduling", "prescription_refill", "claim_status",
        "coverage_verification", "referral_request", "prior_authorization",
        "wellness_program_offer",
    ),
    entity_slots=("patient_id", "claim_id", "prescription_id", "procedure_code", "provider_id"),
    states=(
        StateNode("GREETING", "Greet the patient and ask how you can help with their healthcare needs.", kind="initial"),
        StateNode("VERIFY_PATIENT", "Verify the patient's identity and confirm their account before accessing records."),
        StateNode("CHECK_ELIGIBILITY", "Verify the patient's insurance coverage and benefits for the requested service.", tools=("verify_coverage",)),
        StateNode("REVIEW_RECORDS", "Review the patient's medical records or current claim status.", tools=("check_claim_status",)),
        StateNode("SCHEDULE_SERVICE", "Schedule the requested medical appointment.", tools=("schedule_appointment",)),
        StateNode("PROCESS_REQUEST", "Process the prescription refill or specialist referral request.", tools=("request_prescription_refill", "request_referral")),
        StateNode("REFERRAL_PROCESS", "Coordinate the specialist referral, confirming provider availability and next steps.", tools=("request_referral",)),
        StateNode("SUBMIT_AUTHORIZATION", "Submit the prior-authorization request for the procedure or service.", tools=("submit_prior_auth",)),
        StateNode("REVIEW_AUTHORIZATION", "Review the outcome of a submitted prior-authorization request.", tools=("check_claim_status",)),
        StateNode("APPEAL_DENIAL", "Handle a formal appeal when a prior-authorization or claim is denied."),
        StateNode("WELLNESS_ENROLL", "Enroll the patient in a wellness or preventive care program."),
        StateNode("CONFIRM_DETAILS", "Confirm the appointment, referral, or authorization details with the patient."),
        StateNode("NOTIFY_PATIENT", "Notify the patient of outcomes, next steps, or appointment confirmations."),
        StateNode("ESCALATE_CLINICAL", "Escalate clinical questions or urgent cases to a qualified healthcare provider."),
        StateNode("RESOLVE", "Confirm the patient's request is fully resolved."),
        StateNode("TERMINAL", "Thank the patient and close the healthcare conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "VERIFY_PATIENT", "proceed to patient verification", "always"),
        Edge("VERIFY_PATIENT", "CHECK_ELIGIBILITY", "patient verified", "always"),
        Edge("CHECK_ELIGIBILITY", "REVIEW_RECORDS", "eligibility confirmed", "tool_success"),
        Edge("CHECK_ELIGIBILITY", "ESCALATE_CLINICAL", "complex coverage issue requiring clinical review", "tool_error", optional=True, priority=1),
        Edge("REVIEW_RECORDS", "SCHEDULE_SERVICE", "records reviewed, scheduling service", "tool_success"),
        Edge("REVIEW_RECORDS", "PROCESS_REQUEST", "prescription or referral request", "intent_match", optional=True, priority=1),
        Edge("REVIEW_RECORDS", "SUBMIT_AUTHORIZATION", "prior authorization required", "intent_match", optional=True, priority=2),
        Edge("SCHEDULE_SERVICE", "CONFIRM_DETAILS", "appointment scheduled", "tool_success"),
        Edge("PROCESS_REQUEST", "REFERRAL_PROCESS", "referral pathway initiated", "tool_success"),
        Edge("PROCESS_REQUEST", "CONFIRM_DETAILS", "prescription refill processed", "always", optional=True, priority=1),
        Edge("REFERRAL_PROCESS", "CONFIRM_DETAILS", "referral coordinated", "tool_success"),
        Edge("SUBMIT_AUTHORIZATION", "REVIEW_AUTHORIZATION", "authorization submitted", "tool_success"),
        Edge("REVIEW_AUTHORIZATION", "CONFIRM_DETAILS", "authorization approved", "tool_success"),
        Edge("REVIEW_AUTHORIZATION", "APPEAL_DENIAL", "authorization denied, initiating appeal", "tool_error", optional=True, priority=1),
        Edge("APPEAL_DENIAL", "ESCALATE_CLINICAL", "appeal requires clinical escalation", "always"),
        Edge("CONFIRM_DETAILS", "WELLNESS_ENROLL", "wellness program offered", "intent_match", optional=True, priority=1),
        Edge("CONFIRM_DETAILS", "NOTIFY_PATIENT", "details confirmed", "always"),
        Edge("WELLNESS_ENROLL", "NOTIFY_PATIENT", "wellness enrollment processed", "always"),
        Edge("ESCALATE_CLINICAL", "NOTIFY_PATIENT", "clinical escalation complete", "always"),
        Edge("NOTIFY_PATIENT", "RESOLVE", "patient notified", "always"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "wellness program enrollment accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "appointment_scheduling": "service",
        "prescription_refill": "service",
        "claim_status": "service",
        "coverage_verification": "service",
        "referral_request": "service",
        "prior_authorization": "service",
        "wellness_program_offer": "upsell_promo",
    },
    outbound_reasons=(
        OutboundReason("prescription_followup", "follow up on your current prescription and refills", "service"),
        OutboundReason("appointment_reminder", "remind you about your upcoming appointment", "service"),
        OutboundReason("wellness_program_offer", "invite you to enrol in our wellness programme", "upsell_promo"),
    ),
)

BANKING = DomainSpec(
    name="Banking & Financial Services",
    category="industry",
    tools=(
        _tool("check_balance", "Check account balance and recent transactions", {
            "account_id": {"type": "string"}, "include_pending": {"type": "boolean", "default": True},
        }, ["account_id"]),
        _tool("transfer_funds", "Transfer funds between accounts", {
            "from_account": {"type": "string"}, "to_account": {"type": "string"},
            "amount": {"type": "number"}, "currency": {"type": "string", "default": "USD"},
        }, ["from_account", "to_account", "amount"]),
        _tool("block_card", "Block or freeze a debit/credit card", {
            "card_id": {"type": "string"}, "reason": {"type": "string", "enum": ["lost", "stolen", "suspicious_activity", "temporary_hold"]},
        }, ["card_id", "reason"]),
        _tool("report_fraud", "Report a fraudulent transaction", {
            "transaction_id": {"type": "string"}, "description": {"type": "string"},
            "disputed_amount": {"type": "number"},
        }, ["transaction_id", "description"]),
        _tool("apply_for_loan", "Submit a loan application", {
            "customer_id": {"type": "string"}, "loan_type": {"type": "string", "enum": ["personal", "mortgage", "auto", "business"]},
            "requested_amount": {"type": "number"},
        }, ["customer_id", "loan_type", "requested_amount"]),
        _tool("activate_card", "Activate a new card", {
            "card_id": {"type": "string"}, "last_four_ssn": {"type": "string"},
        }, ["card_id", "last_four_ssn"]),
        _tool("inquiry_interest_rate", "Get current interest rates", {
            "product_type": {"type": "string", "enum": ["savings", "cd", "mortgage", "personal_loan"]},
        }, ["product_type"]),
    ),
    intents=(
        "balance_inquiry", "fund_transfer", "card_block", "fraud_report",
        "loan_inquiry", "card_activation", "rate_inquiry", "wire_request",
    ),
    entity_slots=("account_id", "card_id", "transaction_id", "amount", "loan_type"),
    states=(
        StateNode("GREETING", "Greet the customer and ask how you can help with their banking needs.", kind="initial"),
        StateNode("VERIFY_IDENTITY", "Verify the customer's identity before accessing any account information."),
        StateNode("AUTHENTICATE_2FA", "Complete two-factor authentication for sensitive banking actions."),
        StateNode("LOOKUP_ACCOUNT", "Look up the account balance and recent transaction history.", tools=("check_balance",)),
        StateNode("REVIEW_TRANSACTIONS", "Review recent transactions with the customer and note any concerns.", tools=("check_balance",)),
        StateNode("PROCESS_REQUEST", "Carry out the requested banking transaction using the appropriate tool.", tools=("transfer_funds", "apply_for_loan", "activate_card", "inquiry_interest_rate", "block_card")),
        StateNode("FRAUD_INVESTIGATION", "Investigate suspected fraud, block the card if necessary, and file a report.", tools=("report_fraud", "block_card")),
        StateNode("FRAUD_APPEAL", "Review a previously filed fraud decision and escalate if the customer disputes the outcome."),
        StateNode("APPROVAL_CHECK", "Check whether the transaction or loan request requires additional approval."),
        StateNode("COLLECT_DOCUMENTS", "Guide the customer through submitting the documents required for a loan application."),
        StateNode("SUBMIT_LOAN_APPLICATION", "Submit the completed loan application with all collected documents.", tools=("apply_for_loan",)),
        StateNode("RATE_COMPARISON", "Retrieve and compare current interest rates across savings, CD, and loan products.", tools=("inquiry_interest_rate",)),
        StateNode("CONFIRM_ACTION", "Summarise the transaction or request and ask the customer to confirm."),
        StateNode("ESCALATE_COMPLIANCE", "Escalate flagged transactions or policy-sensitive cases to the compliance team."),
        StateNode("NOTIFY_CUSTOMER", "Notify the customer of the outcome and any next steps."),
        StateNode("RESOLVE", "Confirm the banking request is fully resolved."),
        StateNode("TERMINAL", "Thank the customer and close the banking conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "VERIFY_IDENTITY", "proceed to identity verification", "always"),
        Edge("VERIFY_IDENTITY", "AUTHENTICATE_2FA", "identity check requires 2FA", "always"),
        Edge("AUTHENTICATE_2FA", "LOOKUP_ACCOUNT", "authentication successful", "always"),
        Edge("LOOKUP_ACCOUNT", "REVIEW_TRANSACTIONS", "account retrieved", "tool_success"),
        Edge("REVIEW_TRANSACTIONS", "PROCESS_REQUEST", "transactions reviewed, proceeding with request", "tool_success"),
        Edge("REVIEW_TRANSACTIONS", "FRAUD_INVESTIGATION", "suspicious activity detected", "intent_match", optional=True, priority=1),
        Edge("REVIEW_TRANSACTIONS", "RATE_COMPARISON", "interest rate comparison requested", "intent_match", optional=True, priority=2),
        Edge("PROCESS_REQUEST", "APPROVAL_CHECK", "request requires approval gate", "tool_success"),
        Edge("PROCESS_REQUEST", "COLLECT_DOCUMENTS", "loan application requires document collection", "intent_match", optional=True, priority=1),
        Edge("COLLECT_DOCUMENTS", "SUBMIT_LOAN_APPLICATION", "all documents collected", "always"),
        Edge("SUBMIT_LOAN_APPLICATION", "APPROVAL_CHECK", "loan application submitted", "tool_success"),
        Edge("FRAUD_INVESTIGATION", "FRAUD_APPEAL", "customer disputes fraud decision", "tool_success", optional=True, priority=1),
        Edge("FRAUD_INVESTIGATION", "ESCALATE_COMPLIANCE", "fraud confirmed, escalating", "tool_success"),
        Edge("FRAUD_APPEAL", "ESCALATE_COMPLIANCE", "appeal requires compliance review", "always"),
        Edge("RATE_COMPARISON", "CONFIRM_ACTION", "rate comparison complete", "tool_success"),
        Edge("APPROVAL_CHECK", "CONFIRM_ACTION", "approval check complete", "always"),
        Edge("ESCALATE_COMPLIANCE", "NOTIFY_CUSTOMER", "compliance case escalated", "always"),
        Edge("CONFIRM_ACTION", "NOTIFY_CUSTOMER", "action confirmed by customer", "always"),
        Edge("NOTIFY_CUSTOMER", "RESOLVE", "customer notified of outcome", "always"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "loan or rate product offer accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "balance_inquiry": "service",
        "fund_transfer": "service",
        "card_block": "service",
        "fraud_report": "service",
        "loan_inquiry": "upsell_promo",
        "card_activation": "service",
        "rate_inquiry": "upsell_promo",
        "wire_request": "service",
    },
    outbound_reasons=(
        OutboundReason("loan_offer", "let you know about a pre-approved loan offer", "upsell_promo"),
        OutboundReason("rate_review", "review a better savings rate you now qualify for", "upsell_promo"),
        OutboundReason("card_activation_reminder", "remind you to activate the card we recently issued", "service"),
    ),
)

TELECOM = DomainSpec(
    name="Telecommunications",
    category="industry",
    tools=(
        _tool("change_plan", "Change mobile/broadband plan", {
            "account_id": {"type": "string"}, "new_plan_id": {"type": "string"},
            "effective_date": {"type": "string", "format": "date"},
        }, ["account_id", "new_plan_id"]),
        _tool("port_number", "Port phone number from another carrier", {
            "phone_number": {"type": "string"}, "current_carrier": {"type": "string"},
            "account_number": {"type": "string"},
        }, ["phone_number", "current_carrier"]),
        _tool("report_outage", "Report a network outage", {
            "location": {"type": "string"}, "service_type": {"type": "string", "enum": ["mobile", "broadband", "tv"]},
            "description": {"type": "string"},
        }, ["location", "service_type"]),
        _tool("check_data_usage", "Check data usage and allowance", {
            "account_id": {"type": "string"}, "period": {"type": "string", "enum": ["current", "last_month"]},
        }, ["account_id"]),
        _tool("unlock_device", "Unlock a device from carrier", {
            "device_imei": {"type": "string"}, "account_id": {"type": "string"},
        }, ["device_imei", "account_id"]),
        _tool("activate_roaming", "Activate international roaming", {
            "account_id": {"type": "string"}, "destination_country": {"type": "string"},
            "roaming_plan": {"type": "string", "enum": ["basic", "premium", "unlimited"]},
        }, ["account_id", "destination_country"]),
    ),
    intents=(
        "plan_change", "number_porting", "outage_report", "data_usage",
        "device_unlock", "roaming_activation", "sim_replacement",
    ),
    entity_slots=("account_id", "phone_number", "device_imei", "plan_id"),
    states=(
        StateNode("GREETING", "Greet the customer and ask what telecom service they need help with.", kind="initial"),
        StateNode("VERIFY_ACCOUNT", "Verify the customer's account and identity before making any changes."),
        StateNode("CHECK_ELIGIBILITY", "Check whether the customer is eligible for the requested change or service."),
        StateNode("REVIEW_PLAN", "Review the current plan and data usage with the customer.", tools=("check_data_usage",)),
        StateNode("USAGE_ANALYSIS", "Analyse the customer's data usage patterns and recommend a suitable plan.", tools=("check_data_usage",)),
        StateNode("UPGRADE_OFFER", "Present a plan upgrade recommendation based on usage analysis.", tools=("change_plan",)),
        StateNode("PROCESS_CHANGE", "Apply the plan change as requested by the customer.", tools=("change_plan",)),
        StateNode("PORT_NUMBER_PROCESS", "Carry out the number porting procedure from the previous carrier.", tools=("port_number",)),
        StateNode("SIM_REPLACEMENT", "Process a SIM card replacement for the customer.", tools=("unlock_device",)),
        StateNode("TECHNICAL_CHECK", "Check for network outages or technical issues affecting the service.", tools=("report_outage",)),
        StateNode("ACTIVATE_SERVICE", "Activate roaming or unlock the device as requested.", tools=("activate_roaming", "unlock_device")),
        StateNode("CONFIRM_CHANGES", "Summarise all pending changes and ask the customer to confirm."),
        StateNode("NOTIFY_CUSTOMER", "Notify the customer of the completed changes and any important details."),
        StateNode("ESCALATE", "Escalate unresolved telecom issues to a specialist."),
        StateNode("RESOLVE", "Confirm the telecom request is fully resolved."),
        StateNode("TERMINAL", "Thank the customer and close the telecom conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "VERIFY_ACCOUNT", "proceed to account verification", "always"),
        Edge("VERIFY_ACCOUNT", "CHECK_ELIGIBILITY", "account verified", "always"),
        Edge("CHECK_ELIGIBILITY", "REVIEW_PLAN", "eligibility confirmed, reviewing plan", "always"),
        Edge("REVIEW_PLAN", "USAGE_ANALYSIS", "detailed usage analysis needed", "tool_success"),
        Edge("REVIEW_PLAN", "PROCESS_CHANGE", "plan change directly requested", "intent_match", optional=True, priority=1),
        Edge("REVIEW_PLAN", "PORT_NUMBER_PROCESS", "number porting requested", "intent_match", optional=True, priority=2),
        Edge("REVIEW_PLAN", "SIM_REPLACEMENT", "SIM replacement requested", "intent_match", optional=True, priority=3),
        Edge("REVIEW_PLAN", "TECHNICAL_CHECK", "outage or technical issue reported", "intent_match", optional=True, priority=4),
        Edge("USAGE_ANALYSIS", "UPGRADE_OFFER", "upgrade recommended based on usage", "tool_success"),
        Edge("USAGE_ANALYSIS", "PROCESS_CHANGE", "plan change confirmed without upsell", "always", optional=True, priority=1),
        Edge("UPGRADE_OFFER", "PROCESS_CHANGE", "upgrade accepted", "tool_success"),
        Edge("UPGRADE_OFFER", "CONFIRM_CHANGES", "upgrade declined, confirming original request", "always", optional=True, priority=1),
        Edge("PROCESS_CHANGE", "ACTIVATE_SERVICE", "plan changed, activating roaming or device", "tool_success", optional=True, priority=1),
        Edge("PROCESS_CHANGE", "CONFIRM_CHANGES", "plan change complete", "tool_success"),
        Edge("PORT_NUMBER_PROCESS", "CONFIRM_CHANGES", "number porting initiated", "tool_success"),
        Edge("PORT_NUMBER_PROCESS", "ESCALATE", "porting failed, escalating", "tool_error", optional=True, priority=1),
        Edge("SIM_REPLACEMENT", "CONFIRM_CHANGES", "SIM replacement processed", "tool_success"),
        Edge("TECHNICAL_CHECK", "CONFIRM_CHANGES", "technical issue reported", "tool_success"),
        Edge("TECHNICAL_CHECK", "ESCALATE", "outage confirmed, escalating", "tool_error", optional=True, priority=1),
        Edge("ACTIVATE_SERVICE", "CONFIRM_CHANGES", "service activated", "tool_success"),
        Edge("CONFIRM_CHANGES", "NOTIFY_CUSTOMER", "customer confirmed changes", "always"),
        Edge("ESCALATE", "NOTIFY_CUSTOMER", "issue escalated to specialist", "always"),
        Edge("NOTIFY_CUSTOMER", "RESOLVE", "customer notified", "always"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "plan upgrade or roaming package accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "plan_change": "upsell_promo",
        "number_porting": "service",
        "outage_report": "service",
        "data_usage": "service",
        "device_unlock": "service",
        "roaming_activation": "upsell_promo",
        "sim_replacement": "service",
    },
    outbound_reasons=(
        OutboundReason("plan_upgrade_offer", "offer you an upgraded mobile plan at a better rate", "upsell_promo"),
        OutboundReason("roaming_activation_reminder", "remind you to activate roaming before your trip", "service"),
    ),
)

UTILITIES = DomainSpec(
    name="Utilities (Electric, Water, Gas)",
    category="industry",
    tools=(
        _tool("submit_meter_reading", "Submit a meter reading", {
            "account_id": {"type": "string"}, "reading_value": {"type": "number"},
            "meter_type": {"type": "string", "enum": ["electric", "gas", "water"]},
        }, ["account_id", "reading_value", "meter_type"]),
        _tool("dispute_bill", "Dispute a utility bill", {
            "account_id": {"type": "string"}, "bill_period": {"type": "string"},
            "dispute_reason": {"type": "string"},
        }, ["account_id", "bill_period", "dispute_reason"]),
        _tool("request_connection", "Request new service connection", {
            "address": {"type": "string"}, "service_type": {"type": "string", "enum": ["electric", "gas", "water"]},
            "move_in_date": {"type": "string", "format": "date"},
        }, ["address", "service_type", "move_in_date"]),
        _tool("report_outage", "Report a utility outage", {
            "address": {"type": "string"}, "service_type": {"type": "string"},
            "severity": {"type": "string", "enum": ["partial", "complete", "emergency"]},
        }, ["address", "service_type"]),
        _tool("analyze_usage", "Analyze energy/water usage patterns", {
            "account_id": {"type": "string"}, "period_months": {"type": "integer", "default": 12},
        }, ["account_id"]),
        _tool("enroll_green_energy", "Enroll in green energy program", {
            "account_id": {"type": "string"}, "program": {"type": "string", "enum": ["solar", "wind", "carbon_offset"]},
        }, ["account_id", "program"]),
    ),
    intents=(
        "meter_reading", "billing_dispute", "new_connection", "disconnection",
        "outage_report", "usage_analysis", "green_program_enrollment",
        "green_energy_upgrade",
    ),
    entity_slots=("account_id", "address", "meter_type", "reading_value"),
    states=(
        StateNode("GREETING", "Greet the customer and ask about their utility service.", kind="initial"),
        StateNode("VERIFY_ACCOUNT", "Verify the customer's utility account."),
        StateNode("REVIEW_BILLING", "Review the bill and usage, noting any disputes.", tools=("dispute_bill", "analyze_usage")),
        StateNode("CHECK_METER", "Record or check the customer's meter reading.", tools=("submit_meter_reading",)),
        StateNode("PROCESS_REQUEST", "Process the connection or program-enrollment request.", tools=("request_connection", "enroll_green_energy")),
        StateNode("DISPATCH_TECHNICIAN", "Report the outage and dispatch a technician if needed.", tools=("report_outage",)),
        StateNode("CONFIRM_ACTION", "Summarise the action and ask the customer to confirm."),
        StateNode("SCHEDULE_SERVICE", "Schedule the service connection or visit.", tools=("request_connection",)),
        StateNode("ESCALATE", "Escalate unresolved utility issues to a specialist."),
        StateNode("RESOLVE", "Confirm the utility request is resolved."),
        StateNode("TERMINAL", "Thank the customer and close the conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "VERIFY_ACCOUNT", "proceed to account verification", "always"),
        Edge("VERIFY_ACCOUNT", "REVIEW_BILLING", "account verified, reviewing billing", "always"),
        Edge("VERIFY_ACCOUNT", "CHECK_METER", "meter reading submission", "intent_match", optional=True, priority=1),
        Edge("REVIEW_BILLING", "PROCESS_REQUEST", "billing reviewed, processing request", "tool_success"),
        Edge("REVIEW_BILLING", "CHECK_METER", "meter discrepancy, checking reading", "intent_match", optional=True, priority=1),
        Edge("CHECK_METER", "CONFIRM_ACTION", "meter reading submitted", "tool_success"),
        Edge("PROCESS_REQUEST", "CONFIRM_ACTION", "request processed", "tool_success"),
        Edge("PROCESS_REQUEST", "SCHEDULE_SERVICE", "site visit needed", "intent_match", optional=True, priority=1),
        Edge("PROCESS_REQUEST", "DISPATCH_TECHNICIAN", "outage reported", "intent_match", optional=True, priority=2),
        Edge("SCHEDULE_SERVICE", "CONFIRM_ACTION", "service scheduled", "tool_success"),
        Edge("DISPATCH_TECHNICIAN", "CONFIRM_ACTION", "technician dispatched", "tool_success"),
        Edge("CONFIRM_ACTION", "RESOLVE", "customer confirmed", "always"),
        Edge("CONFIRM_ACTION", "ESCALATE", "issue unresolved, escalating", "user_declines", optional=True, priority=1),
        Edge("ESCALATE", "RESOLVE", "escalated to specialist", "always"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "green energy upgrade accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "meter_reading": "service",
        "billing_dispute": "service",
        "new_connection": "service",
        "disconnection": "service",
        "outage_report": "service",
        "usage_analysis": "service",
        "green_program_enrollment": "upsell_promo",
        "green_energy_upgrade": "upsell_promo",
    },
)

TRAVEL = DomainSpec(
    name="Travel & Hospitality",
    category="industry",
    tools=(
        _tool("search_flights", "Search available flights", {
            "origin": {"type": "string"}, "destination": {"type": "string"},
            "departure_date": {"type": "string", "format": "date"},
            "passengers": {"type": "integer"}, "cabin_class": {"type": "string", "enum": ["economy", "business", "first"]},
        }, ["origin", "destination", "departure_date"]),
        _tool("book_reservation", "Book flight, hotel, or car", {
            "reservation_type": {"type": "string", "enum": ["flight", "hotel", "car"]},
            "option_id": {"type": "string"}, "passenger_name": {"type": "string"},
        }, ["reservation_type", "option_id", "passenger_name"]),
        _tool("cancel_reservation", "Cancel an existing reservation", {
            "reservation_id": {"type": "string"}, "reason": {"type": "string"},
        }, ["reservation_id"]),
        _tool("modify_reservation", "Modify an existing reservation", {
            "reservation_id": {"type": "string"}, "changes": {"type": "object"},
        }, ["reservation_id", "changes"]),
        _tool("redeem_points", "Redeem loyalty points for booking", {
            "loyalty_id": {"type": "string"}, "points_to_redeem": {"type": "integer"},
            "reservation_id": {"type": "string"},
        }, ["loyalty_id", "points_to_redeem"]),
        _tool("file_travel_claim", "File a travel insurance claim", {
            "policy_id": {"type": "string"}, "claim_type": {"type": "string", "enum": ["cancellation", "delay", "medical", "baggage"]},
            "amount": {"type": "number"}, "description": {"type": "string"},
        }, ["policy_id", "claim_type"]),
        _tool("check_visa_requirements", "Check visa and documentation requirements", {
            "nationality": {"type": "string"}, "destination": {"type": "string"},
            "trip_purpose": {"type": "string", "enum": ["tourism", "business", "transit"]},
        }, ["nationality", "destination"]),
    ),
    intents=(
        "flight_search", "hotel_booking", "car_rental", "reservation_change",
        "cancellation", "loyalty_redemption", "insurance_claim",
        "visa_inquiry", "checkin_help",
    ),
    entity_slots=("reservation_id", "origin", "destination", "departure_date", "loyalty_id"),
    states=(
        StateNode("GREETING", "Greet the traveler and ask about their trip planning or booking needs.", kind="initial"),
        StateNode("VERIFY_TRAVELER", "Verify the traveler's identity and confirm any existing reservations."),
        StateNode("SEARCH_OPTIONS", "Search available flights, hotels, or car rentals.", tools=("search_flights",)),
        StateNode("CHECK_VISA_REQUIREMENTS", "Check visa and travel documentation requirements for the destination.", tools=("check_visa_requirements",)),
        StateNode("COLLECT_VISA_DOCS", "Guide the traveler through collecting and submitting visa documentation.", tools=("check_visa_requirements",)),
        StateNode("PRESENT_ITINERARY", "Present the proposed itinerary options to the traveler."),
        StateNode("PROCESS_BOOKING", "Book the selected flight, hotel, or car reservation.", tools=("book_reservation",)),
        StateNode("PAYMENT_PROCESSING", "Process payment for the booking, applying loyalty points if requested.", tools=("redeem_points",)),
        StateNode("LOYALTY_UPGRADE", "Present a loyalty tier upgrade offer or apply a points redemption.", tools=("redeem_points",)),
        StateNode("ISSUE_DOCUMENTS", "Issue travel documents and confirm all requirements are met.", tools=("check_visa_requirements",)),
        StateNode("HANDLE_CHANGES", "Modify or cancel an existing reservation as requested.", tools=("modify_reservation", "cancel_reservation")),
        StateNode("DISRUPTION_HANDLING", "Handle travel disruption such as delays, cancellations, or missed connections.", tools=("modify_reservation", "cancel_reservation")),
        StateNode("REBOOK_FLIGHTS", "Rebook affected flights due to disruption or schedule change.", tools=("book_reservation",)),
        StateNode("FILE_CLAIM", "File the travel insurance claim for the incident.", tools=("file_travel_claim",)),
        StateNode("CONFIRM_DETAILS", "Confirm the booking or change details with the traveler."),
        StateNode("RESOLVE", "Confirm the travel request is fully resolved."),
        StateNode("TERMINAL", "Thank the traveler and close the conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "VERIFY_TRAVELER", "proceed to traveler verification", "always"),
        Edge("VERIFY_TRAVELER", "SEARCH_OPTIONS", "traveler verified, searching options", "always"),
        Edge("SEARCH_OPTIONS", "CHECK_VISA_REQUIREMENTS", "visa check needed for destination", "tool_success"),
        Edge("SEARCH_OPTIONS", "HANDLE_CHANGES", "change or cancellation requested", "intent_match", optional=True, priority=1),
        Edge("SEARCH_OPTIONS", "DISRUPTION_HANDLING", "travel disruption reported", "intent_match", optional=True, priority=2),
        Edge("CHECK_VISA_REQUIREMENTS", "COLLECT_VISA_DOCS", "visa documents required", "tool_success"),
        Edge("CHECK_VISA_REQUIREMENTS", "PRESENT_ITINERARY", "no additional visa docs needed", "always", optional=True, priority=1),
        Edge("COLLECT_VISA_DOCS", "PRESENT_ITINERARY", "visa documents collected", "tool_success"),
        Edge("PRESENT_ITINERARY", "PROCESS_BOOKING", "itinerary selected", "always"),
        Edge("PROCESS_BOOKING", "PAYMENT_PROCESSING", "booking confirmed", "tool_success"),
        Edge("PAYMENT_PROCESSING", "LOYALTY_UPGRADE", "loyalty tier upgrade offered", "tool_success", optional=True, priority=1),
        Edge("PAYMENT_PROCESSING", "ISSUE_DOCUMENTS", "payment complete", "always"),
        Edge("LOYALTY_UPGRADE", "ISSUE_DOCUMENTS", "loyalty upgrade processed", "tool_success"),
        Edge("ISSUE_DOCUMENTS", "CONFIRM_DETAILS", "documents issued", "tool_success"),
        Edge("HANDLE_CHANGES", "CONFIRM_DETAILS", "reservation change applied", "tool_success"),
        Edge("HANDLE_CHANGES", "FILE_CLAIM", "cancellation insurance claim required", "intent_match", optional=True, priority=1),
        Edge("DISRUPTION_HANDLING", "REBOOK_FLIGHTS", "rebooking required", "tool_success"),
        Edge("DISRUPTION_HANDLING", "FILE_CLAIM", "disruption insurance claim required", "intent_match", optional=True, priority=1),
        Edge("REBOOK_FLIGHTS", "CONFIRM_DETAILS", "rebooking complete", "tool_success"),
        Edge("FILE_CLAIM", "CONFIRM_DETAILS", "insurance claim filed", "tool_success"),
        Edge("CONFIRM_DETAILS", "RESOLVE", "details confirmed by traveler", "always"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "loyalty upgrade or points redemption accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "flight_search": "service",
        "hotel_booking": "service",
        "car_rental": "service",
        "reservation_change": "service",
        "cancellation": "service",
        "loyalty_redemption": "upsell_promo",
        "insurance_claim": "service",
        "visa_inquiry": "service",
        "checkin_help": "service",
    },
    outbound_reasons=(
        OutboundReason("loyalty_upgrade_offer", "offer a loyalty upgrade for your upcoming trip", "upsell_promo"),
        OutboundReason("trip_reminder", "remind you about details of your upcoming booking", "service"),
    ),
)

ECOMMERCE = DomainSpec(
    name="E-Commerce & Retail",
    category="industry",
    tools=(
        _tool("search_products", "Search product catalog", {
            "query": {"type": "string"}, "category": {"type": "string"},
            "sort_by": {"type": "string", "enum": ["relevance", "price_low", "price_high", "rating"]},
        }, ["query"]),
        _tool("check_stock", "Check product stock availability", {
            "product_id": {"type": "string"}, "store_id": {"type": "string"},
        }, ["product_id"]),
        _tool("apply_coupon", "Apply a coupon or voucher code", {
            "order_id": {"type": "string"}, "coupon_code": {"type": "string"},
        }, ["order_id", "coupon_code"]),
        _tool("price_match", "Request price match with competitor", {
            "product_id": {"type": "string"}, "competitor_price": {"type": "number"},
            "competitor_url": {"type": "string"},
        }, ["product_id", "competitor_price"]),
        _tool("recommend_products", "Get product recommendations", {
            "customer_id": {"type": "string"}, "based_on": {"type": "string", "enum": ["history", "trending", "similar"]},
        }, ["customer_id"]),
        _tool("check_backorder", "Check back-order status and ETA", {
            "product_id": {"type": "string"},
        }, ["product_id"]),
    ),
    intents=(
        "product_search", "stock_check", "coupon_application",
        "price_match_request", "recommendation", "backorder_inquiry",
        "bundle_promotion",
    ),
    entity_slots=("product_id", "coupon_code", "store_id", "order_id"),
    states=(
        StateNode("GREETING", "Greet the shopper and ask what they're looking for.", kind="initial"),
        StateNode("UNDERSTAND_NEEDS", "Clarify the shopper's needs and suggest relevant products.", tools=("recommend_products",)),
        StateNode("SEARCH_CATALOG", "Search the catalog for matching products.", tools=("search_products",)),
        StateNode("CHECK_AVAILABILITY", "Check stock or back-order status for the chosen item.", tools=("check_stock", "check_backorder")),
        StateNode("APPLY_PROMOTIONS", "Apply coupons or a price match if eligible.", tools=("apply_coupon", "price_match")),
        StateNode("PROCESS_ORDER", "Place the order for the shopper."),
        StateNode("CONFIRM_PURCHASE", "Summarise the purchase and ask the shopper to confirm."),
        StateNode("ARRANGE_DELIVERY", "Arrange delivery details for the order."),
        StateNode("RESOLVE", "Confirm the shopping request is resolved."),
        StateNode("TERMINAL", "Thank the shopper and close the conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "UNDERSTAND_NEEDS", "proceed to needs clarification", "always"),
        Edge("UNDERSTAND_NEEDS", "SEARCH_CATALOG", "needs clarified, searching", "tool_success"),
        Edge("SEARCH_CATALOG", "CHECK_AVAILABILITY", "products found", "tool_success"),
        Edge("CHECK_AVAILABILITY", "PROCESS_ORDER", "item available, placing order", "tool_success"),
        Edge("CHECK_AVAILABILITY", "APPLY_PROMOTIONS", "promotion code available", "slot_present", optional=True, priority=1),
        Edge("APPLY_PROMOTIONS", "PROCESS_ORDER", "promotion applied", "tool_success"),
        Edge("PROCESS_ORDER", "CONFIRM_PURCHASE", "order placed", "always"),
        Edge("PROCESS_ORDER", "ARRANGE_DELIVERY", "custom delivery required", "intent_match", optional=True, priority=1),
        Edge("ARRANGE_DELIVERY", "CONFIRM_PURCHASE", "delivery arranged", "always"),
        Edge("CONFIRM_PURCHASE", "RESOLVE", "purchase confirmed", "always"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "bundle promotion accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "product_search": "service",
        "stock_check": "service",
        "coupon_application": "service",
        "price_match_request": "service",
        "recommendation": "upsell_promo",
        "backorder_inquiry": "service",
        "bundle_promotion": "upsell_promo",
    },
)

GOVERNMENT = DomainSpec(
    name="Government & Public Services",
    category="industry",
    tools=(
        _tool("check_benefit_eligibility", "Check eligibility for social benefits", {
            "citizen_id": {"type": "string"}, "benefit_type": {"type": "string", "enum": ["unemployment", "disability", "housing", "food_assistance"]},
        }, ["citizen_id", "benefit_type"]),
        _tool("check_application_status", "Check status of a permit or license application", {
            "application_id": {"type": "string"},
        }, ["application_id"]),
        _tool("file_complaint", "File a formal complaint or grievance", {
            "department": {"type": "string"}, "subject": {"type": "string"},
            "description": {"type": "string"}, "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        }, ["department", "subject", "description"]),
        _tool("verify_document", "Verify authenticity of an official document", {
            "document_type": {"type": "string", "enum": ["id_card", "passport", "license", "certificate"]},
            "document_number": {"type": "string"},
        }, ["document_type", "document_number"]),
        _tool("schedule_appointment", "Schedule an in-person appointment at a government office", {
            "office_id": {"type": "string"}, "service_type": {"type": "string"},
            "preferred_date": {"type": "string", "format": "date"},
        }, ["office_id", "service_type"]),
        _tool("submit_tax_inquiry", "Submit a tax-related inquiry", {
            "taxpayer_id": {"type": "string"}, "tax_year": {"type": "integer"},
            "inquiry_type": {"type": "string", "enum": ["filing_status", "refund_status", "payment", "amendment"]},
        }, ["taxpayer_id", "tax_year", "inquiry_type"]),
    ),
    intents=(
        "benefit_inquiry", "application_status", "complaint_filing",
        "document_verification", "appointment_scheduling", "tax_inquiry",
    ),
    entity_slots=("citizen_id", "application_id", "document_number", "taxpayer_id"),
    states=(
        StateNode("GREETING", "Greet the citizen and ask which public service they need.", kind="initial"),
        StateNode("VERIFY_CITIZEN", "Verify the citizen's identity."),
        StateNode("CHECK_ELIGIBILITY", "Check eligibility for the requested benefit.", tools=("check_benefit_eligibility",)),
        StateNode("REVIEW_APPLICATION", "Review the status of the citizen's application.", tools=("check_application_status",)),
        StateNode("PROCESS_REQUEST", "Process the complaint or tax inquiry.", tools=("file_complaint", "submit_tax_inquiry")),
        StateNode("VERIFY_DOCUMENTS", "Verify the authenticity of submitted documents.", tools=("verify_document",)),
        StateNode("SUBMIT_FORM", "Submit the required form on the citizen's behalf.", tools=("submit_tax_inquiry",)),
        StateNode("SCHEDULE_VISIT", "Schedule an in-person office appointment.", tools=("schedule_appointment",)),
        StateNode("ESCALATE_SUPERVISOR", "Escalate complex cases to a supervisor."),
        StateNode("RESOLVE", "Confirm the citizen's request is resolved."),
        StateNode("TERMINAL", "Thank the citizen and close the conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "VERIFY_CITIZEN", "proceed to citizen verification", "always"),
        Edge("VERIFY_CITIZEN", "CHECK_ELIGIBILITY", "identity verified", "always"),
        Edge("CHECK_ELIGIBILITY", "REVIEW_APPLICATION", "eligibility checked", "tool_success"),
        Edge("CHECK_ELIGIBILITY", "VERIFY_DOCUMENTS", "documents required", "slot_present", optional=True, priority=1),
        Edge("VERIFY_DOCUMENTS", "SUBMIT_FORM", "documents verified", "tool_success"),
        Edge("SUBMIT_FORM", "RESOLVE", "form submitted", "tool_success"),
        Edge("REVIEW_APPLICATION", "PROCESS_REQUEST", "application reviewed, processing request", "tool_success"),
        Edge("REVIEW_APPLICATION", "SCHEDULE_VISIT", "in-person visit required", "intent_match", optional=True, priority=1),
        Edge("PROCESS_REQUEST", "RESOLVE", "request processed", "tool_success"),
        Edge("PROCESS_REQUEST", "ESCALATE_SUPERVISOR", "complex case, escalating", "tool_error", optional=True, priority=1),
        Edge("SCHEDULE_VISIT", "RESOLVE", "visit scheduled", "tool_success"),
        Edge("ESCALATE_SUPERVISOR", "RESOLVE", "escalated to supervisor", "always"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "benefit_inquiry": "service",
        "application_status": "service",
        "complaint_filing": "service",
        "document_verification": "service",
        "appointment_scheduling": "service",
        "tax_inquiry": "service",
    },
)

INSURANCE = DomainSpec(
    name="Insurance",
    category="industry",
    tools=(
        _tool("file_claim", "File an insurance claim", {
            "policy_id": {"type": "string"},
            "claim_type": {"type": "string", "enum": ["life", "health", "auto", "home"]},
            "incident_date": {"type": "string", "format": "date"},
            "description": {"type": "string"},
        }, ["policy_id", "claim_type", "incident_date"]),
        _tool("check_claim_status", "Check the status of an existing claim", {
            "claim_id": {"type": "string"},
        }, ["claim_id"]),
        _tool("verify_policy", "Verify policyholder details and active coverage", {
            "policy_id": {"type": "string"}, "policyholder_id": {"type": "string"},
        }, ["policy_id", "policyholder_id"]),
        _tool("update_policy", "Update policy information", {
            "policy_id": {"type": "string"},
            "field": {"type": "string", "enum": ["beneficiary", "address", "coverage_level", "payment_method"]},
            "new_value": {"type": "string"},
        }, ["policy_id", "field", "new_value"]),
        _tool("quote_premium", "Get a premium quote for new or upgraded coverage", {
            "coverage_type": {"type": "string", "enum": ["life", "health", "auto", "home", "bundle"]},
            "coverage_amount": {"type": "number"},
            "age": {"type": "integer"},
            "risk_profile": {"type": "string", "enum": ["low", "medium", "high"]},
        }, ["coverage_type", "coverage_amount"]),
        _tool("renew_policy", "Renew an expiring policy", {
            "policy_id": {"type": "string"}, "duration_months": {"type": "integer"},
        }, ["policy_id", "duration_months"]),
        _tool("cancel_policy", "Cancel an active insurance policy", {
            "policy_id": {"type": "string"}, "reason": {"type": "string"},
            "effective_date": {"type": "string", "format": "date"},
        }, ["policy_id", "reason"]),
        _tool("request_claim_documents", "Request supporting documents for a claim", {
            "claim_id": {"type": "string"},
            "doc_type": {"type": "string", "enum": ["police_report", "medical_record", "repair_estimate", "photo"]},
        }, ["claim_id", "doc_type"]),
    ),
    intents=(
        "file_claim", "check_claim_status", "update_beneficiary",
        "policy_verification", "cancel_policy",
        "quote_request", "coverage_upgrade", "policy_renewal", "bundle_offer",
    ),
    entity_slots=("policy_id", "claim_id", "policyholder_id", "coverage_type",
                  "incident_date", "vin", "beneficiary"),
    states=(
        StateNode("GREETING", "Greet the policyholder and ask how you can help with their insurance needs.", kind="initial"),
        StateNode("VERIFY_POLICYHOLDER", "Verify the policyholder's identity and confirm active coverage.", tools=("verify_policy",)),
        StateNode("REVIEW_POLICY", "Review the policy details, including coverage levels and any pending renewals.", tools=("update_policy", "renew_policy", "cancel_policy", "quote_premium")),
        StateNode("AMEND_POLICY", "Apply a policy amendment such as beneficiary update or coverage level change.", tools=("update_policy",)),
        StateNode("CLAIM_INTAKE", "Take down the details of the new claim and file it.", tools=("file_claim",)),
        StateNode("ASSESS_COVERAGE", "Assess whether the incident is covered and check the current claim status.", tools=("check_claim_status",)),
        StateNode("REQUEST_DOCUMENTATION", "Request supporting documents required for the claim evaluation.", tools=("request_claim_documents",)),
        StateNode("EVALUATE_CLAIM", "Evaluate the submitted claim against the policy terms and documentation.", tools=("check_claim_status",)),
        StateNode("APPROVE_OR_DENY", "Determine whether to approve or deny the claim and record the decision."),
        StateNode("APPEAL_CLAIM", "Handle a formal appeal from the policyholder against a denied claim."),
        StateNode("NOTIFY_OUTCOME", "Notify the policyholder of the claim decision and any next steps."),
        StateNode("PROCESS_PAYOUT", "Process the approved claim payout to the policyholder."),
        StateNode("BUNDLE_OFFER", "Present a bundle coverage offer to the policyholder.", tools=("quote_premium",)),
        StateNode("CONFIRM_SETTLEMENT", "Confirm the settlement amount and payment method with the policyholder."),
        StateNode("RESOLVE", "Confirm the insurance request is fully resolved."),
        StateNode("TERMINAL", "Thank the policyholder and close the insurance conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "VERIFY_POLICYHOLDER", "proceed to policyholder verification", "always"),
        Edge("VERIFY_POLICYHOLDER", "REVIEW_POLICY", "policyholder verified", "tool_success"),
        Edge("REVIEW_POLICY", "CLAIM_INTAKE", "new claim to file", "tool_success"),
        Edge("REVIEW_POLICY", "AMEND_POLICY", "policy amendment requested", "intent_match", optional=True, priority=1),
        Edge("REVIEW_POLICY", "BUNDLE_OFFER", "bundle coverage offer presented", "intent_match", optional=True, priority=2),
        Edge("AMEND_POLICY", "NOTIFY_OUTCOME", "policy amendment applied", "tool_success"),
        Edge("CLAIM_INTAKE", "ASSESS_COVERAGE", "claim filed, assessing coverage", "tool_success"),
        Edge("ASSESS_COVERAGE", "REQUEST_DOCUMENTATION", "documents required for evaluation", "tool_success"),
        Edge("ASSESS_COVERAGE", "APPROVE_OR_DENY", "coverage assessed, no additional docs needed", "always", optional=True, priority=1),
        Edge("REQUEST_DOCUMENTATION", "EVALUATE_CLAIM", "documents received", "tool_success"),
        Edge("EVALUATE_CLAIM", "APPROVE_OR_DENY", "evaluation complete", "tool_success"),
        Edge("APPROVE_OR_DENY", "PROCESS_PAYOUT", "claim approved", "always"),
        Edge("APPROVE_OR_DENY", "APPEAL_CLAIM", "claim denied, appeal filed", "intent_match", optional=True, priority=1),
        Edge("APPEAL_CLAIM", "NOTIFY_OUTCOME", "appeal decision reached", "always"),
        Edge("PROCESS_PAYOUT", "CONFIRM_SETTLEMENT", "payout ready, confirming settlement", "always"),
        Edge("CONFIRM_SETTLEMENT", "NOTIFY_OUTCOME", "settlement confirmed by policyholder", "always"),
        Edge("BUNDLE_OFFER", "NOTIFY_OUTCOME", "bundle offer presented", "tool_success"),
        Edge("NOTIFY_OUTCOME", "RESOLVE", "policyholder notified", "always"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "coverage upgrade or renewal accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "file_claim": "service",
        "check_claim_status": "service",
        "update_beneficiary": "service",
        "policy_verification": "service",
        "cancel_policy": "service",
        "quote_request": "service",
        "coverage_upgrade": "upsell_promo",
        "policy_renewal": "upsell_promo",
        "bundle_offer": "upsell_promo",
    },
    outbound_reasons=(
        OutboundReason("renewal_reminder", "remind you that your policy is up for renewal soon", "service"),
        OutboundReason("coverage_upgrade", "offer an upgrade that broadens your current coverage", "upsell_promo"),
    ),
)

# ---------------------------------------------------------------------------
# Operational Domains
# ---------------------------------------------------------------------------

COMPLAINTS = DomainSpec(
    name="Complaints & Escalations",
    category="operational",
    tools=(
        _tool("register_complaint", "Register a formal complaint", {
            "customer_id": {"type": "string"}, "category": {"type": "string"},
            "description": {"type": "string"}, "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        }, ["customer_id", "category", "description"]),
        _tool("escalate_to_supervisor", "Escalate case to supervisor", {
            "case_id": {"type": "string"}, "reason": {"type": "string"},
            "urgency": {"type": "string", "enum": ["normal", "urgent", "immediate"]},
        }, ["case_id", "reason"]),
        _tool("offer_goodwill", "Offer goodwill gesture or compensation", {
            "customer_id": {"type": "string"}, "gesture_type": {"type": "string", "enum": ["discount", "credit", "free_service", "upgrade"]},
            "value": {"type": "number"},
        }, ["customer_id", "gesture_type"]),
        _tool("check_sla_status", "Check SLA compliance for a case", {
            "case_id": {"type": "string"},
        }, ["case_id"]),
        _tool("close_case", "Close a complaint case with resolution", {
            "case_id": {"type": "string"}, "resolution_summary": {"type": "string"},
            "customer_satisfied": {"type": "boolean"},
        }, ["case_id", "resolution_summary"]),
    ),
    intents=(
        "complaint_registration", "escalation_request", "service_recovery",
        "sla_inquiry", "case_followup", "case_closure", "goodwill_upgrade_offer",
    ),
    entity_slots=("case_id", "customer_id", "severity", "gesture_type"),
    states=(
        StateNode("GREETING", "Greet the customer and invite them to share their concern.", kind="initial"),
        StateNode("LISTEN_COMPLAINT", "Listen to the complaint and register it formally.", tools=("register_complaint",)),
        StateNode("ACKNOWLEDGE_ISSUE", "Acknowledge the issue and empathise with the customer."),
        StateNode("INVESTIGATE", "Investigate the case and check SLA status.", tools=("check_sla_status",)),
        StateNode("OFFER_RESOLUTION", "Offer a resolution or goodwill gesture.", tools=("offer_goodwill",)),
        StateNode("ESCALATE_SUPERVISOR", "Escalate the case to a supervisor when needed.", tools=("escalate_to_supervisor",)),
        StateNode("APPLY_COMPENSATION", "Apply the agreed compensation.", tools=("offer_goodwill",)),
        StateNode("VERIFY_SATISFACTION", "Confirm the customer is satisfied with the resolution."),
        StateNode("CLOSE_CASE", "Close the complaint case with a resolution summary.", tools=("close_case",)),
        StateNode("TERMINAL", "Thank the customer and close the conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "LISTEN_COMPLAINT", "proceed to complaint registration", "always"),
        Edge("LISTEN_COMPLAINT", "ACKNOWLEDGE_ISSUE", "complaint registered", "tool_success"),
        Edge("ACKNOWLEDGE_ISSUE", "INVESTIGATE", "issue acknowledged, investigating", "always"),
        Edge("INVESTIGATE", "OFFER_RESOLUTION", "investigation complete", "tool_success"),
        Edge("INVESTIGATE", "ESCALATE_SUPERVISOR", "SLA breach or high severity", "tool_success", optional=True, priority=1),
        Edge("OFFER_RESOLUTION", "VERIFY_SATISFACTION", "resolution offered", "tool_success"),
        Edge("OFFER_RESOLUTION", "APPLY_COMPENSATION", "compensation agreed", "intent_match", optional=True, priority=1),
        Edge("ESCALATE_SUPERVISOR", "OFFER_RESOLUTION", "supervisor decision reached", "tool_success"),
        Edge("APPLY_COMPENSATION", "VERIFY_SATISFACTION", "compensation applied", "tool_success"),
        Edge("VERIFY_SATISFACTION", "CLOSE_CASE", "customer satisfied", "always"),
        Edge("CLOSE_CASE", "TERMINAL", "case closed", "tool_success"),
        Edge("CLOSE_CASE", "TERMINAL", "goodwill upgrade accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "complaint_registration": "service",
        "escalation_request": "service",
        "service_recovery": "service",
        "sla_inquiry": "service",
        "case_followup": "service",
        "case_closure": "service",
        "goodwill_upgrade_offer": "upsell_promo",
    },
)

SCHEDULING = DomainSpec(
    name="Appointment & Scheduling",
    category="operational",
    tools=(
        _tool("book_appointment", "Book an appointment", {
            "service_type": {"type": "string"}, "preferred_date": {"type": "string", "format": "date"},
            "preferred_time": {"type": "string"}, "customer_id": {"type": "string"},
        }, ["service_type", "customer_id"]),
        _tool("reschedule_appointment", "Reschedule an existing appointment", {
            "appointment_id": {"type": "string"}, "new_date": {"type": "string", "format": "date"},
            "new_time": {"type": "string"},
        }, ["appointment_id"]),
        _tool("cancel_appointment", "Cancel an appointment", {
            "appointment_id": {"type": "string"}, "reason": {"type": "string"},
        }, ["appointment_id"]),
        _tool("check_availability", "Check available time slots", {
            "service_type": {"type": "string"}, "date": {"type": "string", "format": "date"},
            "location": {"type": "string"},
        }, ["service_type", "date"]),
        _tool("join_waitlist", "Add customer to waitlist", {
            "customer_id": {"type": "string"}, "service_type": {"type": "string"},
            "preferred_date_range": {"type": "string"},
        }, ["customer_id", "service_type"]),
        _tool("send_reminder", "Send appointment reminder", {
            "appointment_id": {"type": "string"}, "channel": {"type": "string", "enum": ["sms", "email", "push"]},
        }, ["appointment_id"]),
    ),
    intents=(
        "book_appointment", "reschedule", "cancel_appointment",
        "check_availability", "waitlist_request", "reminder_request",
        "premium_slot_offer",
    ),
    entity_slots=("appointment_id", "service_type", "date", "time", "location"),
    states=(
        StateNode("GREETING", "Greet the customer and ask what they'd like to schedule.", kind="initial"),
        StateNode("IDENTIFY_SERVICE", "Identify the service the customer wants to book."),
        StateNode("CHECK_AVAILABILITY", "Check available time slots for the service.", tools=("check_availability",)),
        StateNode("SELECT_SLOT", "Help the customer select a slot and book it.", tools=("book_appointment",)),
        StateNode("CONFIRM_BOOKING", "Confirm the booking details with the customer.", tools=("book_appointment",)),
        StateNode("SEND_CONFIRMATION", "Send a confirmation or reminder.", tools=("send_reminder",)),
        StateNode("HANDLE_RESCHEDULE", "Reschedule or cancel the appointment as requested.", tools=("reschedule_appointment", "cancel_appointment")),
        StateNode("MANAGE_WAITLIST", "Add the customer to the waitlist if no slot is open.", tools=("join_waitlist",)),
        StateNode("RESOLVE", "Confirm the scheduling request is resolved."),
        StateNode("TERMINAL", "Thank the customer and close the conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "IDENTIFY_SERVICE", "proceed to service identification", "always"),
        Edge("IDENTIFY_SERVICE", "CHECK_AVAILABILITY", "service identified", "always"),
        Edge("CHECK_AVAILABILITY", "SELECT_SLOT", "slots available, selecting", "tool_success"),
        Edge("CHECK_AVAILABILITY", "MANAGE_WAITLIST", "no slots available, joining waitlist", "tool_error", optional=True, priority=1),
        Edge("SELECT_SLOT", "CONFIRM_BOOKING", "slot booked", "tool_success"),
        Edge("CONFIRM_BOOKING", "SEND_CONFIRMATION", "booking confirmed", "tool_success"),
        Edge("CONFIRM_BOOKING", "HANDLE_RESCHEDULE", "reschedule or cancellation requested", "intent_match", optional=True, priority=1),
        Edge("SEND_CONFIRMATION", "RESOLVE", "confirmation sent", "tool_success"),
        Edge("HANDLE_RESCHEDULE", "SEND_CONFIRMATION", "reschedule applied", "tool_success"),
        Edge("MANAGE_WAITLIST", "RESOLVE", "added to waitlist", "tool_success"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "premium slot upgrade accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "book_appointment": "service",
        "reschedule": "service",
        "cancel_appointment": "service",
        "check_availability": "service",
        "waitlist_request": "service",
        "reminder_request": "service",
        "premium_slot_offer": "upsell_promo",
    },
    outbound_reasons=(
        OutboundReason("appointment_reminder", "remind you about your scheduled appointment", "service"),
        OutboundReason("reschedule_followup", "follow up about rescheduling your missed appointment", "service"),
    ),
)

SALES = DomainSpec(
    name="Sales & Lead Generation",
    category="operational",
    tools=(
        _tool("qualify_lead", "Qualify a sales lead", {
            "contact_name": {"type": "string"}, "company": {"type": "string"},
            "budget_range": {"type": "string"}, "timeline": {"type": "string"},
        }, ["contact_name"]),
        _tool("create_quote", "Create a price quote", {
            "customer_id": {"type": "string"}, "products": {"type": "array", "items": {"type": "string"}},
            "discount_pct": {"type": "number", "default": 0},
        }, ["customer_id", "products"]),
        _tool("schedule_demo", "Schedule a product demo", {
            "contact_name": {"type": "string"}, "product": {"type": "string"},
            "preferred_date": {"type": "string", "format": "date"},
        }, ["contact_name", "product"]),
        _tool("check_contract_renewal", "Check contract renewal status", {
            "contract_id": {"type": "string"},
        }, ["contract_id"]),
        _tool("process_upsell", "Process an upsell or cross-sell offer", {
            "customer_id": {"type": "string"}, "current_product": {"type": "string"},
            "recommended_product": {"type": "string"}, "offer_details": {"type": "string"},
        }, ["customer_id", "recommended_product"]),
        _tool("send_proposal", "Send a formal sales proposal", {
            "quote_id": {"type": "string"}, "recipient_email": {"type": "string"},
        }, ["quote_id", "recipient_email"]),
    ),
    intents=(
        "sales_inquiry", "demo_request", "quote_request", "contract_renewal",
        "upsell_offer", "proposal_request", "pricing_negotiation",
    ),
    entity_slots=("contact_name", "company", "contract_id", "quote_id", "product"),
    states=(
        StateNode("GREETING", "Greet the prospect and open the conversation.", kind="initial"),
        StateNode("QUALIFY_PROSPECT", "Qualify the lead's budget, timeline, and fit.", tools=("qualify_lead",)),
        StateNode("IDENTIFY_NEEDS", "Identify the prospect's needs and pain points."),
        StateNode("PRESENT_SOLUTION", "Present the solution, offering a demo if useful.", tools=("schedule_demo",)),
        StateNode("HANDLE_OBJECTIONS", "Address the prospect's objections."),
        StateNode("CREATE_PROPOSAL", "Create a quote and send a proposal.", tools=("create_quote", "send_proposal")),
        StateNode("NEGOTIATE_TERMS", "Negotiate terms, including any upsell.", tools=("process_upsell",)),
        StateNode("CLOSE_DEAL", "Close the deal and check renewal terms.", tools=("check_contract_renewal",)),
        StateNode("FOLLOW_UP", "Follow up with the prospect after the meeting.", tools=("send_proposal",)),
        StateNode("RESOLVE", "Confirm next steps are agreed."),
        StateNode("TERMINAL", "Thank the prospect and close the conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "QUALIFY_PROSPECT", "proceed to lead qualification", "always"),
        Edge("QUALIFY_PROSPECT", "IDENTIFY_NEEDS", "lead qualified", "tool_success"),
        Edge("IDENTIFY_NEEDS", "PRESENT_SOLUTION", "needs identified", "always"),
        Edge("PRESENT_SOLUTION", "HANDLE_OBJECTIONS", "solution presented", "tool_success"),
        Edge("PRESENT_SOLUTION", "NEGOTIATE_TERMS", "price negotiation needed", "intent_match", optional=True, priority=1),
        Edge("HANDLE_OBJECTIONS", "CREATE_PROPOSAL", "objections addressed", "always"),
        Edge("NEGOTIATE_TERMS", "CREATE_PROPOSAL", "terms negotiated", "tool_success"),
        Edge("CREATE_PROPOSAL", "CLOSE_DEAL", "proposal sent", "tool_success"),
        Edge("CREATE_PROPOSAL", "FOLLOW_UP", "follow-up scheduled", "intent_match", optional=True, priority=1),
        Edge("CLOSE_DEAL", "RESOLVE", "deal closed", "tool_success"),
        Edge("CLOSE_DEAL", "FOLLOW_UP", "deal needs follow-up", "intent_match", optional=True, priority=1),
        Edge("FOLLOW_UP", "RESOLVE", "follow-up complete", "tool_success"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
        Edge("RESOLVE", "TERMINAL", "upsell or contract renewal accepted", "intent_match", optional=True, priority=1, intent_category="upsell_promo"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "sales_inquiry": "service",
        "demo_request": "service",
        "quote_request": "upsell_promo",
        "contract_renewal": "upsell_promo",
        "upsell_offer": "upsell_promo",
        "proposal_request": "service",
        "pricing_negotiation": "upsell_promo",
    },
    outbound_reasons=(
        OutboundReason("promotion_offer", "offer you a limited-time promotion on your plan", "upsell_promo"),
        OutboundReason("cross_sell", "tell you about a product that complements your account", "upsell_promo"),
    ),
)

SURVEYS = DomainSpec(
    name="Surveys & Feedback",
    category="operational",
    tools=(
        _tool("collect_csat", "Collect customer satisfaction score", {
            "interaction_id": {"type": "string"}, "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "comments": {"type": "string"},
        }, ["interaction_id", "score"]),
        _tool("collect_nps", "Collect Net Promoter Score", {
            "customer_id": {"type": "string"}, "score": {"type": "integer", "minimum": 0, "maximum": 10},
            "reason": {"type": "string"},
        }, ["customer_id", "score"]),
        _tool("submit_feedback", "Submit product or service feedback", {
            "customer_id": {"type": "string"}, "feedback_type": {"type": "string", "enum": ["product", "service", "experience"]},
            "description": {"type": "string"}, "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
        }, ["customer_id", "feedback_type", "description"]),
        _tool("log_complaint_trend", "Log complaint for trend analysis", {
            "category": {"type": "string"}, "description": {"type": "string"},
            "region": {"type": "string"},
        }, ["category", "description"]),
    ),
    intents=(
        "csat_survey", "nps_survey", "product_feedback",
        "service_feedback", "complaint_trend_report",
    ),
    entity_slots=("interaction_id", "score", "feedback_type", "sentiment"),
    states=(
        StateNode("GREETING", "Greet the customer and introduce the survey.", kind="initial"),
        StateNode("EXPLAIN_SURVEY", "Explain the survey's purpose and length."),
        StateNode("COLLECT_RATING", "Collect the customer's satisfaction or NPS rating.", tools=("collect_csat", "collect_nps")),
        StateNode("COLLECT_COMMENTS", "Collect any additional comments or feedback.", tools=("submit_feedback",)),
        StateNode("THANK_CUSTOMER", "Thank the customer for their feedback."),
        StateNode("ESCALATE_LOW_SCORE", "Log low scores for trend analysis and follow-up.", tools=("log_complaint_trend",)),
        StateNode("RESOLVE", "Confirm the survey is complete."),
        StateNode("TERMINAL", "Thank the customer and close the conversation.", kind="terminal"),
    ),
    edges=(
        Edge("GREETING", "EXPLAIN_SURVEY", "proceed to survey explanation", "always"),
        Edge("EXPLAIN_SURVEY", "COLLECT_RATING", "survey explained", "always"),
        Edge("COLLECT_RATING", "COLLECT_COMMENTS", "rating collected", "tool_success"),
        Edge("COLLECT_RATING", "ESCALATE_LOW_SCORE", "critical low score detected", "intent_match", optional=True, priority=1),
        Edge("COLLECT_COMMENTS", "THANK_CUSTOMER", "comments collected", "tool_success"),
        Edge("ESCALATE_LOW_SCORE", "RESOLVE", "low score logged", "tool_success"),
        Edge("THANK_CUSTOMER", "RESOLVE", "customer thanked", "always"),
        Edge("RESOLVE", "TERMINAL", "conversation complete", "always"),
    ),
    initial="GREETING",
    terminals=("TERMINAL",),
    intent_categories={
        "csat_survey": "service",
        "nps_survey": "service",
        "product_feedback": "service",
        "service_feedback": "service",
        "complaint_trend_report": "service",
    },
)

EMERGENCY = DomainSpec(
    name="Emergency & Critical Services",
    category="operational",
    tools=(
        _tool("dispatch_emergency", "Dispatch emergency response team", {
            "location": {"type": "string"}, "emergency_type": {"type": "string", "enum": ["fire", "medical", "security", "hazmat", "infrastructure"]},
            "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
            "description": {"type": "string"},
        }, ["location", "emergency_type", "severity"]),
        _tool("report_incident", "Report a critical incident", {
            "incident_type": {"type": "string"}, "location": {"type": "string"},
            "affected_services": {"type": "array", "items": {"type": "string"}},
            "estimated_impact": {"type": "string"},
        }, ["incident_type", "location"]),
        _tool("send_mass_notification", "Send mass notification to affected parties", {
            "message": {"type": "string"}, "channels": {"type": "array", "items": {"type": "string"}},
            "priority": {"type": "string", "enum": ["info", "warning", "critical"]},
        }, ["message", "priority"]),
        _tool("check_safety_status", "Check safety status of a person or location", {
            "identifier": {"type": "string"}, "check_type": {"type": "string", "enum": ["person", "location", "facility"]},
        }, ["identifier", "check_type"]),
        _tool("activate_backup_systems", "Activate backup or failover systems", {
            "system_name": {"type": "string"}, "failover_type": {"type": "string", "enum": ["hot", "warm", "cold"]},
        }, ["system_name"]),
    ),
    intents=(
        "emergency_dispatch", "incident_report", "safety_check",
        "mass_notification", "backup_activation", "status_update",
    ),
    entity_slots=("location", "emergency_type", "severity", "incident_type"),
    states=(
        StateNode("ALERT_RECEIVED", "Receive the alert and record the incident details.", kind="initial", tools=("report_incident",)),
        StateNode("ASSESS_SEVERITY", "Assess the severity and safety status.", tools=("check_safety_status",)),
        StateNode("DISPATCH_RESPONSE", "Dispatch the appropriate emergency response team.", tools=("dispatch_emergency",)),
        StateNode("COORDINATE_TEAMS", "Coordinate the responding teams."),
        StateNode("NOTIFY_STAKEHOLDERS", "Send notifications to affected stakeholders.", tools=("send_mass_notification",)),
        StateNode("MONITOR_STATUS", "Monitor the ongoing status of the incident.", tools=("check_safety_status",)),
        StateNode("ACTIVATE_BACKUP", "Activate backup or failover systems if required.", tools=("activate_backup_systems",)),
        StateNode("CONFIRM_RESOLUTION", "Confirm the incident is resolved."),
        StateNode("POST_INCIDENT_REVIEW", "Conduct a post-incident review."),
        StateNode("TERMINAL", "Close out the incident record.", kind="terminal"),
    ),
    edges=(
        Edge("ALERT_RECEIVED", "ASSESS_SEVERITY", "incident recorded", "tool_success"),
        Edge("ASSESS_SEVERITY", "DISPATCH_RESPONSE", "severity assessed, dispatching", "tool_success"),
        Edge("DISPATCH_RESPONSE", "COORDINATE_TEAMS", "response dispatched", "tool_success"),
        Edge("DISPATCH_RESPONSE", "ACTIVATE_BACKUP", "backup systems required", "intent_match", optional=True, priority=1),
        Edge("ACTIVATE_BACKUP", "COORDINATE_TEAMS", "backup activated", "tool_success"),
        Edge("COORDINATE_TEAMS", "NOTIFY_STAKEHOLDERS", "teams coordinated", "always"),
        Edge("COORDINATE_TEAMS", "MONITOR_STATUS", "ongoing monitoring needed", "intent_match", optional=True, priority=1),
        Edge("NOTIFY_STAKEHOLDERS", "CONFIRM_RESOLUTION", "stakeholders notified", "tool_success"),
        Edge("MONITOR_STATUS", "CONFIRM_RESOLUTION", "situation stable", "tool_success"),
        Edge("CONFIRM_RESOLUTION", "POST_INCIDENT_REVIEW", "incident confirmed resolved", "always"),
        Edge("POST_INCIDENT_REVIEW", "TERMINAL", "review complete", "always"),
    ),
    initial="ALERT_RECEIVED",
    terminals=("TERMINAL",),
    intent_categories={
        "emergency_dispatch": "service",
        "incident_report": "service",
        "safety_check": "service",
        "mass_notification": "service",
        "backup_activation": "service",
        "status_update": "service",
    },
)

# ---------------------------------------------------------------------------
# Cross-Cutting Intents (available in all domains)
# ---------------------------------------------------------------------------

CROSS_CUTTING_TOOLS = (
    _tool("verify_identity", "Verify customer identity (OTP/PIN/KYC)", {
        "customer_id": {"type": "string"},
        "method": {"type": "string", "enum": ["otp", "pin", "security_question", "biometric"]},
    }, ["customer_id", "method"]),
    _tool("transfer_to_agent", "Transfer call to a live agent", {
        "department": {"type": "string"}, "reason": {"type": "string"},
        "priority": {"type": "string", "enum": ["normal", "urgent"]},
    }, ["department"]),
    _tool("create_ticket", "Create a support ticket for follow-up", {
        "customer_id": {"type": "string"}, "subject": {"type": "string"},
        "description": {"type": "string"}, "priority": {"type": "string", "enum": ["low", "medium", "high"]},
    }, ["customer_id", "subject"]),
    _tool("send_notification", "Send notification to customer", {
        "customer_id": {"type": "string"}, "channel": {"type": "string", "enum": ["email", "sms", "push"]},
        "message": {"type": "string"},
    }, ["customer_id", "channel", "message"]),
)

CROSS_CUTTING_INTENTS = (
    "greeting", "clarification", "hold_transfer", "authentication",
    "closing", "language_switch", "repeat_rephrase", "out_of_scope",
    "human_handoff",
)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DOMAIN_REGISTRY: dict[str, DomainSpec] = {
    # Core Business
    "account_management": ACCOUNT_MANAGEMENT,
    "billing_payments": BILLING_PAYMENTS,
    "order_management": ORDER_MANAGEMENT,
    "technical_support": TECHNICAL_SUPPORT,
    "product_info": PRODUCT_INFO,
    # Industry-Specific
    "healthcare": HEALTHCARE,
    "banking": BANKING,
    "telecom": TELECOM,
    "utilities": UTILITIES,
    "travel": TRAVEL,
    "ecommerce": ECOMMERCE,
    "government": GOVERNMENT,
    "insurance": INSURANCE,
    # Operational
    "complaints": COMPLAINTS,
    "scheduling": SCHEDULING,
    "sales": SALES,
    "surveys": SURVEYS,
    "emergency": EMERGENCY,
}

ALL_DOMAIN_NAMES: list[str] = list(DOMAIN_REGISTRY.keys())

# ---------------------------------------------------------------------------
# Intent-category taxonomy for promo/upsell biasing
# ---------------------------------------------------------------------------

INTENT_CATEGORY_TAXONOMY: dict[str, str] = {
    # account_management
    "subscription_change": "upsell_promo",
    "rewards_inquiry": "upsell_promo",
    "premium_plan_offer": "upsell_promo",
    # billing_payments
    "payment_plan_offer": "upsell_promo",
    # order_management
    "accessory_upsell": "upsell_promo",
    # technical_support
    "extended_warranty_offer": "upsell_promo",
    # product_info
    "promotion_inquiry": "upsell_promo",
    "upgrade_recommendation": "upsell_promo",
    "pricing_inquiry": "upsell_promo",
    # healthcare
    "wellness_program_offer": "upsell_promo",
    # banking
    "loan_inquiry": "upsell_promo",
    "rate_inquiry": "upsell_promo",
    # telecom
    "plan_change": "upsell_promo",
    "roaming_activation": "upsell_promo",
    # utilities
    "green_energy_upgrade": "upsell_promo",
    "green_program_enrollment": "upsell_promo",
    # travel
    "loyalty_redemption": "upsell_promo",
    # ecommerce
    "bundle_promotion": "upsell_promo",
    "recommendation": "upsell_promo",
    # complaints
    "goodwill_upgrade_offer": "upsell_promo",
    # scheduling
    "premium_slot_offer": "upsell_promo",
    # sales
    "upsell_offer": "upsell_promo",
    "quote_request": "upsell_promo",
    "pricing_negotiation": "upsell_promo",
    "contract_renewal": "upsell_promo",
    # insurance
    "coverage_upgrade": "upsell_promo",
    "policy_renewal": "upsell_promo",
    "bundle_offer": "upsell_promo",
}


def classify_intent(intent: str) -> str:
    """Return the intent category ('service' or 'upsell_promo') for an intent name."""
    return INTENT_CATEGORY_TAXONOMY.get(intent, "service")


# ---------------------------------------------------------------------------
# Import-time validation for migrated domains
# ---------------------------------------------------------------------------

def _validate_all() -> None:
    """Run validate_domain for all migrated domains at import time."""
    for _d in [
        ACCOUNT_MANAGEMENT,
        BILLING_PAYMENTS,
        ORDER_MANAGEMENT,
        TECHNICAL_SUPPORT,
        PRODUCT_INFO,
        UTILITIES,
        ECOMMERCE,
        GOVERNMENT,
        COMPLAINTS,
        SCHEDULING,
        SALES,
        SURVEYS,
        EMERGENCY,
        # Newly migrated rich domains
        BANKING,
        INSURANCE,
        HEALTHCARE,
        TRAVEL,
        TELECOM,
    ]:
        validate_domain(_d)


_validate_all()
