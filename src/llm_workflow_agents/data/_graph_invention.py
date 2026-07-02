"""Task C gold-graph invention and validation.

Provides out-of-registry "domain briefs" and a teacher-driven inventor that
produces novel workflow graphs for the playbook->graph dataset (see
docs/data_generation_recipes_task_c.md, §Gold Graph Sourcing). Invented graphs
are validated against the same predicates the eval harness uses, so a gold graph
can never score zero on the structural metrics.

The teacher emits the id-keyed eval schema (data/templates/graph_output_schema.json);
after validation, graphs are normalized to the name-keyed interchange shape
(WorkflowGraph.to_dict()) used everywhere else in the pipeline.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import jsonschema
import structlog

from llm_workflow_agents.data._teacher_client import call_teacher_model
from llm_workflow_agents.eval.graph_extraction_eval import (
    WorkflowGraph,
    check_mermaid_renderability,
    check_structural_validity,
)

logger = structlog.get_logger(__name__)

_SCHEMA_PATH = Path(__file__).parents[3] / "data/templates/graph_output_schema.json"
_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_SCHEMA_CACHE: dict[str, Any] | None = None


def _schema() -> dict[str, Any]:
    """Load the graph-output JSON schema lazily (avoids file I/O at import time)."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is None:
        _SCHEMA_CACHE = json.loads(_SCHEMA_PATH.read_text())
    return _SCHEMA_CACHE


@dataclass(frozen=True)
class DomainBrief:
    """A short description of a workflow domain outside the 18-domain registry."""

    slug: str
    title: str
    description: str
    suggested_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class InventedGraph:
    """A teacher-invented gold graph, stored in the name-keyed interchange shape."""

    brief_slug: str
    graph: dict[str, Any]


# 25 domains deliberately outside DOMAIN_REGISTRY, to exercise OOD generalization.
DOMAIN_BRIEFS: tuple[DomainBrief, ...] = (
    DomainBrief("hr_onboarding", "Employee Onboarding",
                "Onboard a new hire: collect documents, provision accounts, assign training, confirm start.",
                ("create_account", "assign_training", "verify_documents", "schedule_orientation")),
    DomainBrief("it_incident_response", "IT Incident Response",
                "Triage and resolve an IT incident: classify severity, diagnose, mitigate, escalate, close.",
                ("open_ticket", "run_diagnostic", "apply_mitigation", "escalate_incident", "close_ticket")),
    DomainBrief("loan_underwriting", "Loan Underwriting",
                "Underwrite a loan application: verify applicant, pull credit, assess risk, decide, notify.",
                ("verify_applicant", "pull_credit_report", "assess_risk", "record_decision")),
    DomainBrief("restaurant_reservations", "Restaurant Reservations",
                "Handle a table reservation: check availability, capture party details, confirm, remind.",
                ("check_availability", "book_table", "send_confirmation", "cancel_booking")),
    DomainBrief("warehouse_returns", "Warehouse Returns Processing",
                "Process a returned item: authorize return, inspect, restock or scrap, issue refund.",
                ("authorize_return", "inspect_item", "restock_item", "issue_refund")),
    DomainBrief("clinical_trial_screening", "Clinical Trial Screening",
                "Screen a candidate for a clinical trial: consent, eligibility, enroll or reject, schedule.",
                ("record_consent", "check_eligibility", "enroll_subject", "schedule_visit")),
    DomainBrief("freight_dispatch", "Freight Dispatch",
                "Dispatch a freight shipment: intake order, assign carrier, route, track, confirm delivery.",
                ("intake_order", "assign_carrier", "plan_route", "track_shipment")),
    DomainBrief("campus_admissions", "Campus Admissions",
                "Process an admissions application: verify transcripts, review, interview, decide, notify.",
                ("verify_transcripts", "review_application", "schedule_interview", "record_decision")),
    DomainBrief("visa_processing", "Visa Application Processing",
                "Process a visa application: verify identity, check documents, review, approve or deny.",
                ("verify_identity", "check_documents", "review_case", "issue_decision")),
    DomainBrief("mortgage_servicing", "Mortgage Servicing",
                "Service a mortgage account: authenticate, review balance, process payment or modification.",
                ("authenticate_borrower", "review_balance", "process_payment", "modify_terms")),
    DomainBrief("field_service_dispatch", "Field Service Dispatch",
                "Dispatch a field technician: log request, diagnose remotely, schedule visit, confirm fix.",
                ("log_request", "remote_diagnose", "schedule_visit", "confirm_resolution")),
    DomainBrief("grant_application", "Research Grant Application",
                "Review a grant application: verify eligibility, score, panel review, award or reject.",
                ("verify_eligibility", "score_proposal", "panel_review", "record_award")),
    DomainBrief("device_activation", "Device Activation",
                "Activate a new device: verify purchase, register serial, provision service, confirm.",
                ("verify_purchase", "register_serial", "provision_service", "confirm_activation")),
    DomainBrief("subscription_billing_dispute", "Subscription Billing Dispute",
                "Resolve a billing dispute: authenticate, review charges, adjudicate, refund or deny.",
                ("authenticate_user", "review_charges", "adjudicate_dispute", "issue_adjustment")),
    DomainBrief("pet_adoption", "Pet Adoption",
                "Process a pet adoption: screen adopter, match pet, home check, finalize, follow up.",
                ("screen_adopter", "match_pet", "schedule_home_check", "finalize_adoption")),
    DomainBrief("event_registration", "Event Registration",
                "Register an attendee: capture details, select tickets, process payment, confirm, remind.",
                ("capture_details", "select_tickets", "process_payment", "send_confirmation")),
    DomainBrief("software_license_provisioning", "Software License Provisioning",
                "Provision a software license: validate contract, allocate seats, deliver keys, confirm.",
                ("validate_contract", "allocate_seats", "deliver_keys", "confirm_provisioning")),
    DomainBrief("warranty_claim", "Warranty Claim",
                "Process a warranty claim: verify coverage, assess defect, repair or replace, close.",
                ("verify_coverage", "assess_defect", "authorize_repair", "close_claim")),
    DomainBrief("background_check", "Background Check",
                "Run a background check: obtain consent, query records, review flags, report result.",
                ("obtain_consent", "query_records", "review_flags", "issue_report")),
    DomainBrief("utility_connection", "New Utility Connection",
                "Set up a new utility connection: verify address, schedule install, activate, confirm.",
                ("verify_address", "schedule_install", "activate_meter", "confirm_service")),
    DomainBrief("apartment_leasing", "Apartment Leasing",
                "Lease an apartment: screen tenant, tour unit, application review, sign lease, hand keys.",
                ("screen_tenant", "schedule_tour", "review_application", "execute_lease")),
    DomainBrief("expense_reimbursement", "Expense Reimbursement",
                "Process an expense report: validate receipts, check policy, approve, reimburse or reject.",
                ("validate_receipts", "check_policy", "approve_report", "issue_reimbursement")),
    DomainBrief("vaccination_scheduling", "Vaccination Scheduling",
                "Schedule a vaccination: verify eligibility, book slot, screen contraindications, confirm.",
                ("verify_eligibility", "book_slot", "screen_contraindications", "send_reminder")),
    DomainBrief("supply_procurement", "Supply Procurement",
                "Process a procurement request: validate need, source vendor, approve budget, order, receive.",
                ("validate_need", "source_vendor", "approve_budget", "place_order")),
    DomainBrief("membership_renewal", "Membership Renewal",
                "Renew a membership: authenticate, review tier, process payment, apply benefits, confirm.",
                ("authenticate_member", "review_tier", "process_payment", "apply_benefits")),
)


def validate_gold_graph(graph_dict: dict[str, Any]) -> list[str]:
    """Validate an id-keyed eval-shape graph. Returns itemized violations (empty = valid)."""
    try:
        jsonschema.validate(graph_dict, _schema())
    except jsonschema.ValidationError as exc:
        return [f"schema: {exc.message}"]

    violations: list[str] = []
    names = [n["name"] for n in graph_dict["nodes"]]
    ids = {n["id"] for n in graph_dict["nodes"]}

    if len(set(names)) != len(names):
        violations.append("duplicate state name")
    for name in names:
        if not _NAME_RE.match(name):
            violations.append(f"state name not SCREAMING_SNAKE: {name}")

    outgoing: dict[str, list[int]] = {node_id: [] for node_id in ids}
    for edge in graph_dict["edges"]:
        src, dst = edge["from_state"], edge["to_state"]
        if src == dst:
            violations.append(f"self-loop edge: {src}")
        if src in outgoing:
            outgoing[src].append(edge.get("priority", 0))

    terminals = set(graph_dict["terminal_states"])
    for node_id, priorities in outgoing.items():
        if node_id not in terminals and not priorities:
            violations.append(f"non-terminal state has no outgoing edge: {node_id}")
        if len(priorities) != len(set(priorities)):
            violations.append(f"duplicate edge priorities from: {node_id}")

    eval_graph = WorkflowGraph(
        nodes=graph_dict["nodes"],
        edges=graph_dict["edges"],
        initial_state=graph_dict["initial_state"],
        terminal_states=graph_dict["terminal_states"],
    )
    if not check_structural_validity(eval_graph):
        violations.append("structural validity failed (initial/terminal reachability or orphan nodes)")
    if not check_mermaid_renderability(eval_graph):
        violations.append("mermaid renderability failed (invalid ids or dangling edge endpoints)")

    return violations


def _normalize_to_name_keyed(eval_dict: dict[str, Any]) -> dict[str, Any]:
    """Convert a validated id-keyed eval graph to the name-keyed interchange shape."""
    id_to_name = {n["id"]: n["name"] for n in eval_dict["nodes"]}
    return {
        "states": [n["name"] for n in eval_dict["nodes"]],
        "state_details": [
            {
                "name": n["name"],
                "tools": list(n.get("tools", [])),
                "entry_actions": list(n.get("entry_actions", [])),
                "instruction": n.get("instruction", ""),
            }
            for n in eval_dict["nodes"]
        ],
        "transitions": [
            {
                "from": id_to_name[e["from_state"]],
                "to": id_to_name[e["to_state"]],
                "condition": e.get("condition", ""),
                "priority": e.get("priority", 0),
            }
            for e in eval_dict["edges"]
        ],
        "initial": id_to_name[eval_dict["initial_state"]],
        "terminal": [id_to_name[t] for t in eval_dict["terminal_states"]],
    }


_INVENTION_SYSTEM = (
    "You are a workflow architect. Given a business-process brief, design a "
    "state-machine workflow graph and return it as a single JSON object matching "
    "this schema exactly:\n"
    '{"nodes": [{"id": "S<n>", "name": "<SCREAMING_SNAKE>", "tools": [...], '
    '"entry_actions": [...]}], "edges": [{"from_state": "S<n>", "to_state": "S<m>", '
    '"condition": "<nl condition>", "priority": <int>}], "initial_state": "S1", '
    '"terminal_states": ["S<k>"]}\n'
    "Rules: 4-14 states; state names UPPER_SNAKE_CASE and unique; no self-loop edges; "
    "every non-terminal state has at least one outgoing edge; when a state has multiple "
    "outgoing edges, give each a distinct integer priority (0 = default path); the "
    "initial state must be reachable to at least one terminal state; place tool names on "
    "the states that use them, drawn from the brief's suggested tools where sensible. "
    "Return ONLY the JSON object, no prose."
)


def _build_invention_prompt(brief: DomainBrief, corrections: list[str] | None) -> str:
    tools = ", ".join(brief.suggested_tools) if brief.suggested_tools else "(none suggested)"
    prompt = (
        f"## DOMAIN BRIEF\n"
        f"Title: {brief.title}\n"
        f"Description: {brief.description}\n"
        f"Suggested tools: {tools}\n\n"
        f"Design the workflow graph for this domain and return the JSON object."
    )
    if corrections:
        feedback = "\n".join(f"- {c}" for c in corrections)
        prompt += (
            "\n\n## CORRECTIONS REQUIRED\n"
            "Your previous attempt was rejected by an automated validator for the issues "
            "below. Regenerate the ENTIRE graph from scratch, fixing every one of them:\n"
            f"{feedback}"
        )
    return prompt


def invent_novel_graphs(
    briefs: Sequence[DomainBrief],
    count: int,
    teacher_model: str,
    rng: random.Random,
    max_repair_retries: int = 2,
) -> list[InventedGraph]:
    """Invent `count` novel gold graphs via the teacher, validating and repairing each.

    A graph that fails validation is retried up to `max_repair_retries` times with
    itemized corrections; still-invalid graphs (and teacher errors) are dropped.
    """
    if not briefs:
        return []

    results: list[InventedGraph] = []
    for _ in range(count):
        brief = rng.choice(list(briefs))
        corrections: list[str] | None = None
        for _attempt in range(max_repair_retries + 1):
            try:
                raw = call_teacher_model(
                    teacher_model, _INVENTION_SYSTEM, _build_invention_prompt(brief, corrections)
                )
                eval_dict = json.loads(raw)
            except Exception as exc:  # noqa: BLE001 - teacher/parse failures are non-fatal
                logger.warning("graph_invention_teacher_failure", brief=brief.slug, error=str(exc))
                break
            violations = validate_gold_graph(eval_dict)
            if not violations:
                results.append(InventedGraph(brief.slug, _normalize_to_name_keyed(eval_dict)))
                break
            corrections = violations
        # exhausted repairs or teacher error -> this slot is dropped
    return results
