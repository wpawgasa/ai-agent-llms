"""Domain registry for workflow conversation generation.

17 call center domains with tool schemas, state templates, intent examples,
and entity slots. Domains are decoupled from complexity levels — any domain
can appear at any L1–L5 complexity, with the complexity controlling structural
parameters (states, branching, tools, depth) and the domain controlling
content (tool schemas, state names, intents).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DomainSpec:
    """Specification for a single call center domain."""

    name: str
    category: str
    tools: tuple[dict[str, Any], ...]
    state_templates: tuple[str, ...]
    intents: tuple[str, ...]
    entity_slots: tuple[str, ...] = ()


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
    state_templates=(
        "GREETING", "VERIFY_IDENTITY", "AUTHENTICATE", "LOOKUP_ACCOUNT",
        "PROCESS_REQUEST", "CONFIRM_CHANGES", "UPDATE_RECORDS",
        "NOTIFY_CUSTOMER", "RESOLVE", "TERMINAL",
    ),
    intents=(
        "account_creation", "profile_update", "password_reset",
        "account_closure", "subscription_change", "rewards_inquiry",
        "verification_request",
    ),
    entity_slots=("customer_id", "email", "phone", "account_type", "field", "new_value"),
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
    state_templates=(
        "GREETING", "VERIFY_IDENTITY", "LOOKUP_BILLING", "REVIEW_CHARGES",
        "PROCESS_PAYMENT", "APPLY_ADJUSTMENT", "CONFIRM_ACTION",
        "GENERATE_DOCUMENT", "ESCALATE", "RESOLVE", "TERMINAL",
    ),
    intents=(
        "invoice_inquiry", "payment_processing", "refund_request",
        "dispute_charge", "payment_plan", "late_fee_waiver",
        "receipt_request", "chargeback",
    ),
    entity_slots=("invoice_id", "amount", "payment_method", "transaction_id", "customer_id"),
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
    state_templates=(
        "GREETING", "VERIFY_IDENTITY", "LOOKUP_ORDER", "CHECK_STATUS",
        "PROCESS_MODIFICATION", "INITIATE_RETURN", "ARRANGE_DELIVERY",
        "CONFIRM_ACTION", "ESCALATE", "RESOLVE", "TERMINAL",
    ),
    intents=(
        "order_tracking", "order_cancellation", "order_modification",
        "return_request", "exchange_request", "damaged_item_report",
        "delivery_reschedule", "warranty_claim",
    ),
    entity_slots=("order_id", "tracking_number", "item_id", "return_type"),
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
    state_templates=(
        "GREETING", "IDENTIFY_ISSUE", "COLLECT_DIAGNOSTICS", "RUN_TESTS",
        "ATTEMPT_FIX", "VERIFY_RESOLUTION", "ESCALATE_ENGINEERING",
        "REMOTE_ASSIST", "CONFIRM_FIX", "RESOLVE", "TERMINAL",
    ),
    intents=(
        "setup_help", "troubleshoot", "bug_report", "compatibility_check",
        "update_guidance", "remote_assistance", "escalation",
    ),
    entity_slots=("system_name", "device_model", "fix_id", "severity", "ticket_id"),
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
    state_templates=(
        "GREETING", "UNDERSTAND_NEEDS", "SEARCH_CATALOG", "PRESENT_OPTIONS",
        "COMPARE_FEATURES", "CHECK_AVAILABILITY", "QUOTE_PRICING",
        "RECOMMEND_UPGRADE", "RESOLVE", "TERMINAL",
    ),
    intents=(
        "product_inquiry", "pricing_inquiry", "availability_check",
        "feature_comparison", "upgrade_recommendation", "promotion_inquiry",
    ),
    entity_slots=("product_id", "category", "price_range", "customer_tier"),
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
    state_templates=(
        "GREETING", "VERIFY_PATIENT", "CHECK_ELIGIBILITY", "REVIEW_RECORDS",
        "SCHEDULE_SERVICE", "PROCESS_REQUEST", "SUBMIT_AUTHORIZATION",
        "CONFIRM_DETAILS", "ESCALATE_CLINICAL", "RESOLVE", "TERMINAL",
    ),
    intents=(
        "appointment_scheduling", "prescription_refill", "claim_status",
        "coverage_verification", "referral_request", "prior_authorization",
    ),
    entity_slots=("patient_id", "claim_id", "prescription_id", "procedure_code", "provider_id"),
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
    state_templates=(
        "GREETING", "VERIFY_IDENTITY", "AUTHENTICATE_2FA", "LOOKUP_ACCOUNT",
        "REVIEW_TRANSACTIONS", "PROCESS_REQUEST", "FRAUD_INVESTIGATION",
        "APPROVAL_CHECK", "CONFIRM_ACTION", "ESCALATE_COMPLIANCE",
        "RESOLVE", "TERMINAL",
    ),
    intents=(
        "balance_inquiry", "fund_transfer", "card_block", "fraud_report",
        "loan_inquiry", "card_activation", "rate_inquiry", "wire_request",
    ),
    entity_slots=("account_id", "card_id", "transaction_id", "amount", "loan_type"),
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
    state_templates=(
        "GREETING", "VERIFY_ACCOUNT", "CHECK_ELIGIBILITY", "REVIEW_PLAN",
        "PROCESS_CHANGE", "TECHNICAL_CHECK", "ACTIVATE_SERVICE",
        "CONFIRM_CHANGES", "ESCALATE", "RESOLVE", "TERMINAL",
    ),
    intents=(
        "plan_change", "number_porting", "outage_report", "data_usage",
        "device_unlock", "roaming_activation", "sim_replacement",
    ),
    entity_slots=("account_id", "phone_number", "device_imei", "plan_id"),
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
    state_templates=(
        "GREETING", "VERIFY_ACCOUNT", "REVIEW_BILLING", "CHECK_METER",
        "PROCESS_REQUEST", "DISPATCH_TECHNICIAN", "CONFIRM_ACTION",
        "SCHEDULE_SERVICE", "ESCALATE", "RESOLVE", "TERMINAL",
    ),
    intents=(
        "meter_reading", "billing_dispute", "new_connection", "disconnection",
        "outage_report", "usage_analysis", "green_program_enrollment",
    ),
    entity_slots=("account_id", "address", "meter_type", "reading_value"),
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
    state_templates=(
        "GREETING", "VERIFY_TRAVELER", "SEARCH_OPTIONS", "PRESENT_ITINERARY",
        "PROCESS_BOOKING", "PAYMENT_PROCESSING", "ISSUE_DOCUMENTS",
        "HANDLE_CHANGES", "FILE_CLAIM", "CONFIRM_DETAILS", "RESOLVE", "TERMINAL",
    ),
    intents=(
        "flight_search", "hotel_booking", "car_rental", "reservation_change",
        "cancellation", "loyalty_redemption", "insurance_claim",
        "visa_inquiry", "checkin_help",
    ),
    entity_slots=("reservation_id", "origin", "destination", "departure_date", "loyalty_id"),
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
    state_templates=(
        "GREETING", "UNDERSTAND_NEEDS", "SEARCH_CATALOG", "CHECK_AVAILABILITY",
        "APPLY_PROMOTIONS", "PROCESS_ORDER", "CONFIRM_PURCHASE",
        "ARRANGE_DELIVERY", "RESOLVE", "TERMINAL",
    ),
    intents=(
        "product_search", "stock_check", "coupon_application",
        "price_match_request", "recommendation", "backorder_inquiry",
    ),
    entity_slots=("product_id", "coupon_code", "store_id", "order_id"),
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
    state_templates=(
        "GREETING", "VERIFY_CITIZEN", "CHECK_ELIGIBILITY", "REVIEW_APPLICATION",
        "PROCESS_REQUEST", "VERIFY_DOCUMENTS", "SUBMIT_FORM",
        "SCHEDULE_VISIT", "ESCALATE_SUPERVISOR", "RESOLVE", "TERMINAL",
    ),
    intents=(
        "benefit_inquiry", "application_status", "complaint_filing",
        "document_verification", "appointment_scheduling", "tax_inquiry",
    ),
    entity_slots=("citizen_id", "application_id", "document_number", "taxpayer_id"),
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
    state_templates=(
        "GREETING", "LISTEN_COMPLAINT", "ACKNOWLEDGE_ISSUE", "INVESTIGATE",
        "OFFER_RESOLUTION", "ESCALATE_SUPERVISOR", "APPLY_COMPENSATION",
        "VERIFY_SATISFACTION", "CLOSE_CASE", "TERMINAL",
    ),
    intents=(
        "complaint_registration", "escalation_request", "service_recovery",
        "sla_inquiry", "case_followup", "case_closure",
    ),
    entity_slots=("case_id", "customer_id", "severity", "gesture_type"),
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
    state_templates=(
        "GREETING", "IDENTIFY_SERVICE", "CHECK_AVAILABILITY", "SELECT_SLOT",
        "CONFIRM_BOOKING", "SEND_CONFIRMATION", "HANDLE_RESCHEDULE",
        "MANAGE_WAITLIST", "RESOLVE", "TERMINAL",
    ),
    intents=(
        "book_appointment", "reschedule", "cancel_appointment",
        "check_availability", "waitlist_request", "reminder_request",
    ),
    entity_slots=("appointment_id", "service_type", "date", "time", "location"),
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
    state_templates=(
        "GREETING", "QUALIFY_PROSPECT", "IDENTIFY_NEEDS", "PRESENT_SOLUTION",
        "HANDLE_OBJECTIONS", "CREATE_PROPOSAL", "NEGOTIATE_TERMS",
        "CLOSE_DEAL", "FOLLOW_UP", "RESOLVE", "TERMINAL",
    ),
    intents=(
        "sales_inquiry", "demo_request", "quote_request", "contract_renewal",
        "upsell_offer", "proposal_request", "pricing_negotiation",
    ),
    entity_slots=("contact_name", "company", "contract_id", "quote_id", "product"),
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
    state_templates=(
        "GREETING", "EXPLAIN_SURVEY", "COLLECT_RATING", "COLLECT_COMMENTS",
        "THANK_CUSTOMER", "ESCALATE_LOW_SCORE", "RESOLVE", "TERMINAL",
    ),
    intents=(
        "csat_survey", "nps_survey", "product_feedback",
        "service_feedback", "complaint_trend_report",
    ),
    entity_slots=("interaction_id", "score", "feedback_type", "sentiment"),
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
    state_templates=(
        "ALERT_RECEIVED", "ASSESS_SEVERITY", "DISPATCH_RESPONSE",
        "COORDINATE_TEAMS", "NOTIFY_STAKEHOLDERS", "MONITOR_STATUS",
        "ACTIVATE_BACKUP", "CONFIRM_RESOLUTION", "POST_INCIDENT_REVIEW",
        "TERMINAL",
    ),
    intents=(
        "emergency_dispatch", "incident_report", "safety_check",
        "mass_notification", "backup_activation", "status_update",
    ),
    entity_slots=("location", "emergency_type", "severity", "incident_type"),
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
    # Operational
    "complaints": COMPLAINTS,
    "scheduling": SCHEDULING,
    "sales": SALES,
    "surveys": SURVEYS,
    "emergency": EMERGENCY,
}

ALL_DOMAIN_NAMES: list[str] = list(DOMAIN_REGISTRY.keys())
