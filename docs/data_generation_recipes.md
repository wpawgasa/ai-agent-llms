# Data Generation Recipes

This document describes the two data generation scripts for Task A (multi-turn workflow conversations). Both call `generate_workflow_dataset` from `src/llm_workflow_agents/data/generate_workflows.py`.

| Split | Purpose | Generator | Approx. size |
|-------|---------|-----------|--------------|
| Benchmark | Phase 1 pre-trained model ranking | `generate_benchmark_data_teacher.sh` (Gemini teachers) | **~200 conversations** (40 per level) |
| SFT | Phase 2 SFT **and** GRPO; supplies train / validation / test splits | `generate_sft_data.sh` (GPT + Gemini legs) | **~5000 conversations** (4 445 after cleanup) |

There is **no separate GRPO dataset** and **no separately generated held-out eval set**: GRPO reuses the SFT corpus, and the validation / test sets are deterministic 85 / 10 / 5 splits of the cleaned SFT data (see [Train/Val/Test Splits](#trainvaltest-splits) below).

## Shared Concepts

### Complexity Levels

| Level | States (subgraph) | Branches | Loops | Recovery | Eligible domains |
|-------|-------------------|----------|-------|----------|------------------|
| L1    | 3–4               | 0        | 0     | no       | all 18           |
| L2    | 5–7               | 1        | 0     | no       | all 18           |
| L3    | 8–12              | 2–3      | 0–1   | optional | all 18           |
| L4    | 12–16             | 3–5      | 1     | yes      | ≥12-state domains|
| L5    | 16–20             | all      | 1–2   | yes      | 5 expanded domains|

> **Domain-level coupling:** `_select_domain` filters domains by canonical state count ≥ `target_path_len` minimum at runtime. L4 = domains with ≥12 canonical states (banking, insurance, healthcare, travel, telecom + any others expanded); L5 = the 5 expanded rich domains (≥16 states). This is a deliberate deviation from the original "domains fully decoupled from complexity" goal: strict decoupling requires cycling state names, which was the root cause of the duplicate-name defect.

> **Self-loops in conversations:** graph-*edge* self-loops (src == dst) are forbidden in `DomainSpec` and enforced by `validate_domain`. However, the conversation walker emits turn-level `[STATE: X → X]` annotations when a state invokes a tool or handles a follow-up without transitioning. These message-level self-loops are legitimate and appear in GT.

### Domains

The registry (`src/llm_workflow_agents/data/domain_registry.py`) contains **18 domains** decoupled from complexity levels. When `domain=None` (the default), each sample draws a domain uniformly at random. Pass `domain="banking"` (or any key below) to pin all samples to one domain.

| Key | Name | Category |
|-----|------|----------|
| `account_management` | Customer Account Management | core_business |
| `billing_payments` | Billing & Payments | core_business |
| `order_management` | Order Management | core_business |
| `technical_support` | Technical Support | core_business |
| `product_info` | Product & Service Information | core_business |
| `healthcare` | Healthcare & Insurance | industry |
| `banking` | Banking & Financial Services | industry |
| `telecom` | Telecommunications | industry |
| `utilities` | Utilities (Electric, Water, Gas) | industry |
| `travel` | Travel & Hospitality | industry |
| `ecommerce` | E-Commerce & Retail | industry |
| `government` | Government & Public Services | industry |
| `insurance` | Insurance | industry |
| `complaints` | Complaints & Escalations | operational |
| `scheduling` | Appointment & Scheduling | operational |
| `sales` | Sales & Lead Generation | operational |
| `surveys` | Surveys & Feedback | operational |
| `emergency` | Emergency & Critical Services | operational |

#### Insurance domain

The `insurance` domain covers life, health, auto, and home insurance workflows:

- **Tools:** `file_claim`, `check_claim_status`, `verify_policy`, `update_policy`, `quote_premium`, `renew_policy`, `cancel_policy`, `request_claim_documents`
- **State flow:** `VERIFY_POLICYHOLDER → REVIEW_POLICY → CLAIM_INTAKE → ASSESS_COVERAGE → REQUEST_DOCUMENTATION → EVALUATE_CLAIM → APPROVE_OR_DENY → PROCESS_PAYOUT → RESOLVE → TERMINAL`
- **Intents (service):** `file_claim`, `check_claim_status`, `update_beneficiary`, `policy_verification`, `cancel_policy`
- **Intents (upsell/promo):** `quote_request`, `coverage_upgrade`, `policy_renewal`, `bundle_offer`

### Behavior Presets

| Preset | cooperative | adversarial_probing | digressing | invalid_tool_inputs |
|--------|-------------|---------------------|------------|---------------------|
| `default` | 60% | 15% | 10% | 15% |
| `adversarial` | 45% | 25% | 15% | 15% |
| `balanced` | 25% | 25% | 25% | 25% |

### Intent Category Presets

Controls the share of promotion/upsell-focused conversations across **all** domains. The generator selects an intent category per sample, then picks a domain intent matching that category (falling back to any intent if the domain has none tagged as `upsell_promo`).

| Preset | service | upsell_promo |
|--------|---------|--------------|
| `default` | 70% | 30% |
| `service_only` | 100% | 0% |
| `upsell_heavy` | 50% | 50% |

Intent categories are defined in `INTENT_CATEGORY_TAXONOMY` in `domain_registry.py`. The following intents are tagged `upsell_promo` (all others default to `service`):

| Domain | Upsell/promo intents |
|--------|----------------------|
| `account_management` | `subscription_change`, `rewards_inquiry`, `premium_plan_offer` |
| `billing_payments` | `payment_plan_offer` |
| `order_management` | `accessory_upsell` |
| `technical_support` | `extended_warranty_offer` |
| `product_info` | `promotion_inquiry`, `upgrade_recommendation`, `pricing_inquiry` |
| `healthcare` | `wellness_program_offer` |
| `banking` | `loan_inquiry`, `rate_inquiry` |
| `telecom` | `plan_change`, `roaming_activation` |
| `utilities` | `green_energy_upgrade`, `green_program_enrollment` |
| `travel` | `loyalty_redemption` |
| `ecommerce` | `bundle_promotion`, `recommendation` |
| `complaints` | `goodwill_upgrade_offer` |
| `scheduling` | `premium_slot_offer` |
| `sales` | `upsell_offer`, `quote_request`, `pricing_negotiation`, `contract_renewal` |
| `insurance` | `coverage_upgrade`, `policy_renewal`, `bundle_offer`, `quote_request` |

Domains `government`, `surveys`, and `emergency` have no tagged upsell intents; the selector falls back to service intents for those domains regardless of preset.

When `teacher_model` is set and a sample is selected as `upsell_promo`, the teacher prompt receives an extra instruction line:
> *"Conversation focus: naturally weave in promotion, cross-sell, or upsell opportunities relevant to this domain. The workflow must still reach a terminal state; the upsell is a secondary arc, not a hijack."*

The dataset stats dict includes `intent_category_distribution` so you can verify the effective split after generation.

### Language Options

| Value | Effect |
|-------|--------|
| `en` | All turns in English |
| `th` | All turns in Thai (annotations stay in ASCII) |
| `code_switch` | Thai-English code-switching within each conversation — Thai sentence structure with embedded English terms, mirroring real call-centre interactions |
| *(omitted)* | Each sample randomly assigned `en` or `th` (50/50) |

### System Prompt Contract

Every generated sample's `messages[0]` is a fully enriched system prompt assembled by `data/system_prompt.py::build_enriched_system_prompt`. It contains four blocks, in this order:

1. **Role line** — e.g. `"You are a customer service agent handling account_management workflows."`
2. **Workflow script** — the same natural-language `### [STATE]` format used in the teacher prompt (produced by `_graph_to_script`)
3. **Structured reference** — initial state, terminal states, tool names (from `workflow_graph` and `tool_schemas`)
4. **Format rules** — the 7-rule block (STATE annotations, tool_call wrapping, terminal-state requirement, etc.)

Enrichment is applied uniformly after message generation — for both teacher-generated and placeholder conversations — so training and benchmark eval always see the same prompt shape. The helper is **idempotent**: if `"Workflow script"` is already present in the content (e.g. legacy enriched data), the function returns it unchanged.

The benchmark harness (`eval/agent_benchmark.py`) also calls `build_enriched_system_prompt` as a safety net for old bare-prompt JSONL files. On new data this is a no-op.

### Teacher-authored rich system prompts (30%)

When `teacher_model` is set, **30% of samples** receive a teacher-authored rich natural-language system prompt in place of the bare role line. The authored prompt is structured like a production voicebot handoff document:

- A one-line **persona** sentence naming the agent's role and domain.
- A `## GOAL` section stating the call's purpose in plain language.
- One `### [state_name]` section per workflow state, using the exact state names from the graph. Each section contains suggested dialogue lines in quotes, intent-based branching bullets ("If the customer confirms… → follow the [next_state] path"), and tool-call instructions for states that invoke tools.
- Cross-references between sections using `[state_name]` notation that mirrors the graph transitions.

**Critical invariant:** the `_graph_to_script` workflow script + structured reference (initial/terminal states, tool names) + 7-rule format block are **still appended to every sample** by `build_enriched_system_prompt`, regardless of whether the sample used a rich or bare role line. Rich prompts augment the persona/flow layer; they never replace the structured reference.

On teacher failure (JSON parse error, empty `system_prompt` key), the sample silently falls back to the bare role line — no sample is lost. Dataset metadata includes `rich_prompt_count` and `rich_prompt_rate_effective` so the caller can verify the actual ratio achieved. The rate can be adjusted via the `rich_prompt_rate` kwarg to `generate_workflow_dataset` (`0.0` to disable, `1.0` to force all samples). The legacy placeholder (no-teacher) path ignores `rich_prompt_rate` entirely and always uses the bare role line — but the canonical benchmark path is now teacher-driven (Gemini), so benchmark data does include the ~30% rich-prompt share.

### Seed Allocation

Seeds are assigned per-generator to guarantee no sample overlap between benchmark and SFT data:

| Dataset | Seed |
|---------|------|
| Benchmark | 100 |
| SFT | 42 |

Train / validation / test splits are derived deterministically from the SFT cleaned set with a separate split seed (see [Train/Val/Test Splits](#trainvaltest-splits)).

### Teacher Models

| Model | API key env var | Used in |
|-------|----------------|---------|
| `gemini-3-flash` | `GEMINI_API_KEY` | Benchmark |
| `gemini-3.1-flash-lite` | `GEMINI_API_KEY` | Benchmark |
| `gpt-5.4-mini-2026-03-17` | `OPENAI_API_KEY` | SFT (en leg) |
| `gpt-5.4-nano-2026-03-17` | `OPENAI_API_KEY` | SFT (code_switch leg) |
| `gemini-3-flash-preview` | `GEMINI_API_KEY` | SFT (th leg) |
| *(none — placeholder)* | — | Smoke test / CI only (legacy path) |

**Why Gemini for the benchmark split:** Phase 1 model ranking only needs ~200 conversations, but they have to be cheap, fast to produce, and reliable JSON emitters for the state-machine and tool-call annotations. `gemini-3-flash` and `gemini-3.1-flash-lite` fit that profile better than the GPT-class teachers reserved for the higher-cost SFT rich-prompt + multilingual legs.

---

## Output Data Format

Every recipe writes JSONL where each line is one conversation sample. The schema is identical across all splits (benchmark, SFT, val, test) — they differ only in volume, behavior mix, and teacher model.

### Top-level fields

| Field | Type | Description |
|-------|------|-------------|
| `conversation_id` | str | Stable id like `"L3_002"` (level + 0-indexed within level) |
| `complexity_level` | str | `"L1"` … `"L5"` |
| `domain` | str | Domain key from `domain_registry.py` (e.g. `"healthcare"`) |
| `num_states`, `num_tools`, `chain_depth` | int | Convenience counters derived from the graph |
| `language` | str | `"en"`, `"th"`, or `"code_switch"` |
| `user_behavior` | str | `"cooperative"`, `"adversarial_probing"`, `"digressing"`, or `"invalid_tool_inputs"` (sampled per the active behavior preset) |
| `workflow_graph` | dict | State machine (see below) |
| `workflow_script` | str | Human-readable rendering of `workflow_graph` (also embedded inside `messages[0]`) |
| `tool_schemas` | list[dict] | OpenAI/Hermes-style tool definitions available to the agent |
| `messages` | list[dict] | The conversation turns — the only field training actually consumes |
| `ground_truth` | dict | Reference labels used by `eval/agent_benchmark.py` and the reward functions |

### `workflow_graph`

```json
{
  "states": ["GREETING", "VERIFY_PATIENT", "CHECK_ELIGIBILITY",
             "REVIEW_RECORDS", "SCHEDULE_SERVICE", "PROCESS_REQUEST", "TERMINAL"],
  "state_details": [
    {"name": "GREETING",        "tools": [],                                       "entry_actions": []},
    {"name": "REVIEW_RECORDS",  "tools": ["schedule_appointment","request_referral"], "entry_actions": []},
    ...
  ],
  "transitions": [
    {"from": "GREETING",         "to": "VERIFY_PATIENT",     "condition": "patient identifies self",      "priority": 0},
    {"from": "CHECK_ELIGIBILITY","to": "SCHEDULE_SERVICE",    "condition": "eligibility confirmed",        "priority": 1},
    ...
  ],
  "initial":  "GREETING",
  "terminal": ["TERMINAL"]
}
```

- `states` and `state_details` are kept in sync (state_details is just the per-state metadata).
- `transitions[].priority = 0` is the default forward edge; higher priority values mark alternative branches taken when conditions match.
- `terminal` is always a list — a workflow can legitimately have multiple terminal states (e.g. `RESOLVED` vs `ESCALATE_SUPERVISOR`).

### `tool_schemas`

OpenAI tools format (Hermes-compatible). Each entry is a single function definition:

```json
{
  "type": "function",
  "function": {
    "name": "request_referral",
    "description": "Request a specialist referral",
    "parameters": {
      "type": "object",
      "properties": {
        "patient_id": {"type": "string"},
        "specialty":  {"type": "string"},
        "reason":     {"type": "string"}
      },
      "required": ["patient_id", "specialty"]
    }
  }
}
```

### `messages`

OpenAI ChatML-style messages with two conventions layered on top:

1. **State annotations.** Every assistant turn begins with `[STATE: FROM → TO]` on its own line. The parser in `eval/reward_utils.py::extract_state_annotations` reads these.
2. **Tool calls.** Tool invocations are inlined as `<tool_call>{...JSON...}</tool_call>` blocks within the assistant content. Tool *responses* are separate `{"role": "tool", "content": "..."}` turns. (Some samples store the parsed call in `annotations.tool_calls` as well, but `<tool_call>` is the authoritative form.)

The `messages[0]` system prompt is the **enriched prompt** described in [System Prompt Contract](#system-prompt-contract): role line + workflow script + structured reference + 7-rule format guide.

```json
{"role": "user",
 "content": "Hello, I need help with my insurance."}

{"role": "assistant",
 "content": "[STATE: GREETING → VERIFY_PATIENT]\nHello! ... could you please provide your full name and patient ID for verification?"}

{"role": "user",
 "content": "My name is John Doe, ID is JD12345."}

{"role": "assistant",
 "content": "[STATE: REVIEW_RECORDS → REVIEW_RECORDS]\n<tool_call>{\"name\": \"check_claim_status\", \"arguments\": {\"claim_id\": \"CLM98765\", \"patient_id\": \"JD12345\"}}</tool_call>"}

{"role": "tool",
 "content": "{\"status\": \"processed\", \"amount\": \"$250.00\"}"}
```

A self-loop transition like `REVIEW_RECORDS → REVIEW_RECORDS` is legitimate and signals "stayed in the same state to invoke a tool / handle a follow-up turn."

### `ground_truth`

Reference labels for scoring — independent of the model's actual output. Consumed by `state_accuracy.py`, `tool_call_f1.py`, `tool_chain_propagation.py`, and the three GRPO reward functions.

```json
{
  "state_sequence": [
    {"from": "GREETING",         "to": "VERIFY_PATIENT"},
    {"from": "VERIFY_PATIENT",   "to": "CHECK_ELIGIBILITY"},
    {"from": "REVIEW_RECORDS",   "to": "REVIEW_RECORDS"},
    {"from": "REVIEW_RECORDS",   "to": "SCHEDULE_SERVICE"},
    {"from": "CONFIRM_DETAILS",  "to": "TERMINAL"}
  ],
  "tool_calls": [
    {"name": "check_claim_status", "arguments": {"claim_id": "CLM98765", "patient_id": "JD12345"}},
    {"name": "verify_coverage",    "arguments": {"patient_id": "JD12345", "procedure_code": "SURG001"}}
  ],
  "tool_chain_dependencies": [
    [
      {"name": "check_claim_status", "arguments": {"claim_id": "CLM98765", "patient_id": "JD12345"}}
    ]
  ],
  "terminal_state": "TERMINAL"
}
```

- `state_sequence` is the ordered list of transitions the agent should make. Self-loops are kept (they pair with tool-call turns).
- `tool_calls` is the flat sequence of every tool invocation the agent should make, in order. Argument matching is AST sub-tree comparison (BFCL-style), not string equality.
- `tool_chain_dependencies` is a list of *chains* (each chain is a list of tools where each tool's output feeds the next tool's arguments). The eval in `tool_chain_propagation.py` checks that the model carries values like `patient_id` correctly across the chain.
- `terminal_state` is the single state the conversation should end in. If empty (`"terminal_reached": false` flag in the cleaned SFT corpus), the conversation was truncated; reward functions rescale weights to skip the terminal component.

### Complete sample

A real benchmark conversation (`l3_mixed_gemini-3_merged.jsonl`, conv `L3_002`, domain `healthcare`, language `en`), abbreviated:

```json
{
  "conversation_id": "L3_002",
  "complexity_level": "L3",
  "domain": "healthcare",
  "num_states": 9,
  "num_tools": 4,
  "chain_depth": 1,
  "language": "en",
  "user_behavior": "adversarial_probing",
  "workflow_graph": { ... 9 states, 11 transitions, terminal=["TERMINAL"] ... },
  "workflow_script": "### [GREETING]  (initial state)\nNo tools available...\n\n### [REVIEW_RECORDS]\nAvailable tools: check_claim_status (...), verify_coverage (...)\n- On success: proceed to [SCHEDULE_SERVICE]\n...",
  "tool_schemas": [
    {"type": "function", "function": {"name": "check_claim_status", ...}},
    {"type": "function", "function": {"name": "verify_coverage", ...}},
    ...
  ],
  "messages": [
    {"role": "system",    "content": "You are a helpful healthcare assistant. ...\n\n### [GREETING]  (initial state)\n...\n\nFormat rules:\n1. Every assistant turn must start with [STATE: X → Y]\n2. Tool calls must use <tool_call>{...}</tool_call> syntax\n..."},
    {"role": "user",      "content": "Hello, I need help with my insurance."},
    {"role": "assistant", "content": "[STATE: GREETING → VERIFY_PATIENT]\nHello! ... could you please provide your full name and patient ID?"},
    {"role": "user",      "content": "My name is John Doe, ID is JD12345. Why do you need my ID? Is this secure?"},
    {"role": "assistant", "content": "[STATE: VERIFY_PATIENT → CHECK_ELIGIBILITY]\nThank you, John. ... HIPAA-compliant. Your identity is verified."},
    {"role": "user",      "content": "Fine, but make it quick. I also want to check the status of my pending claim."},
    {"role": "assistant", "content": "[STATE: REVIEW_RECORDS → REVIEW_RECORDS]\n<tool_call>{\"name\": \"check_claim_status\", \"arguments\": {\"claim_id\": \"CLM98765\", \"patient_id\": \"JD12345\"}}</tool_call>"},
    {"role": "tool",      "content": "{\"status\": \"processed\", \"amount\": \"$250.00\"}"},
    ...
    {"role": "assistant", "content": "[STATE: CONFIRM_DETAILS → TERMINAL]\nYour referral has been submitted. Is there anything else?"}
  ],
  "ground_truth": {
    "state_sequence": [
      {"from": "GREETING", "to": "VERIFY_PATIENT"},
      {"from": "VERIFY_PATIENT", "to": "CHECK_ELIGIBILITY"},
      ...
      {"from": "CONFIRM_DETAILS", "to": "TERMINAL"}
    ],
    "tool_calls": [
      {"name": "check_claim_status", "arguments": {"claim_id": "CLM98765", "patient_id": "JD12345"}},
      {"name": "verify_coverage",    "arguments": {"patient_id": "JD12345", "procedure_code": "SURG001"}}
    ],
    "tool_chain_dependencies": [
      [{"name": "check_claim_status", "arguments": {"claim_id": "CLM98765", "patient_id": "JD12345"}}]
    ],
    "terminal_state": "TERMINAL"
  }
}
```

### What training and eval actually consume

- **SFT** trains on the `messages` field directly (chat-template-formatted). `workflow_graph`, `workflow_script`, `tool_schemas`, and `ground_truth` are *not* fed into the loss — they only inform the enriched system prompt at `messages[0]` and the held-out reward evaluation.
- **GRPO** uses `messages[0]` + `messages[1]` (the user opener) as the prompt and rolls out the rest of the conversation under the policy; rewards compare generated state annotations / tool calls against `ground_truth`.
- **`eval/agent_benchmark.py`** replays the conversation turn-by-turn: at each user turn it queries the model, parses `[STATE: …]` and `<tool_call>` from the response, and scores against the corresponding `ground_truth.state_sequence` and `ground_truth.tool_calls` entries.

---

## Recipes

### Benchmark (`generate_benchmark_data_teacher.sh`)

Generates ~200 conversations for Phase 1 model ranking using Gemini teachers. Two teacher runs are produced and then merged into the per-level artifact.

```
Levels:       L1–L5
Samples/level: 40 (after merge)
Total:        ~200
Language:     mixed (50/50 en/th per sample)
Teachers:     gemini-3-flash + gemini-3.1-flash-lite
              (two runs, merged into one *_merged.jsonl per level)
Behavior:     default
Seed:         100
Output:       data/output/benchmark/task_a/l{1..5}_mixed_gemini-3_merged.jsonl
```

```bash
# Canonical teacher-driven path (requires GEMINI_API_KEY in .env)
GEMINI_API_KEY=... ./scripts/generate_benchmark_data_teacher.sh \
    --teacher gemini-3-flash

GEMINI_API_KEY=... ./scripts/generate_benchmark_data_teacher.sh \
    --teacher gemini-3.1-flash-lite

# Then merge per-level outputs into l{level}_mixed_gemini-3_merged.jsonl
# (rich-prompt diversity from both teacher families is preserved post-merge).
```

**System-prompt enrichment in the benchmark split.** Every benchmark sample's `messages[0]` passes through `build_enriched_system_prompt` at generation time, so each prompt carries the structured workflow script + tool list + 7-rule format guide. Because the teacher-driven path honours `rich_prompt_rate=0.30`, roughly 30% of benchmark prompts also have the teacher-authored natural-language description in their role line; the other ~70% use the bare role line. Both variants share the same trailing structured reference — the workflow script is **always** present in structured `### [STATE]` format.

**Legacy placeholder path (smoke test / CI only):** `./scripts/generate_benchmark_data.sh` runs `generate_workflow_dataset` with no teacher model — useful when no API key is available, but produces less diverse conversations and **no rich prompts**. Not used for the published Phase 1 results.

---

### SFT (`generate_sft_data.sh`)

Generates ~5 000 curriculum-weighted conversations for supervised fine-tuning. Each complexity level is split across three teacher / language legs to expose the model to diverse linguistic registers from the start. The same corpus, after cleanup and splitting (see below), feeds **both** Stage-1 SFT and Stage-2 GRPO — no separate GRPO dataset is generated.

```
Levels:       L1–L5
Behavior:     mostly adversarial (~87.5%) + cooperative_only (~12.5%)
Seed:         42
Output:       data/output/sft/task_a/            (raw)
              data/output/sft/task_a_cleaned/    (after scripts/clean_task_a_sft.py)
```

**On-disk counts (current revision, after `clean_task_a_sft.py`):**

| | Raw | Cleaned |
|---|---|---|
| Total samples | 4 450 | **4 445** |
| Files | many per-(level, lang, teacher, behavior_preset, timestamp) | same |

Per-leg distribution (approximate, varies by level): each level is produced in three legs — `en` / `gpt-5.4-mini-2026-03-17`, `th` / `gemini-3-flash-preview`, `code_switch` / `gpt-5.4-nano-2026-03-17` — and each leg covers both `adversarial` and `cooperative_only` behavior presets so the model sees the full distribution. Files are ~50 conversations each.

**Cleanup** (`scripts/clean_task_a_sft.py`): drops 5 truncated rows, strips 264 role-confused tool messages, flags 476 empty-terminal conversations with `terminal_reached=false`. Net: 4 450 → 4 445 conversations.

**Rationale for the mostly-adversarial behavior mix:** SFT needs to show the model how to handle user attempts to bypass workflow steps, provide invalid inputs, and go off-topic. Starting from an adversarial-heavy distribution produces a more robust SFT baseline before GRPO refines it further. The smaller `cooperative_only` slice keeps the model fluent on benign paths.

```bash
OPENAI_API_KEY=sk-... GEMINI_API_KEY=... ./scripts/generate_sft_data.sh

# Dry run
./scripts/generate_sft_data.sh --dry-run

# Cleanup (idempotent)
python scripts/clean_task_a_sft.py \
    --input-dir data/output/sft/task_a \
    --output-dir data/output/sft/task_a_cleaned
```

---

### Train/Val/Test Splits

There is **no separate held-out generator**. The cleaned SFT corpus is split deterministically into train / validation / test at an **85 / 10 / 5** ratio, and these splits are reused across:

- **SFT training** — fits on `train.jsonl`, early-stops on `validation.jsonl`.
- **GRPO training** — samples prompts from the same `train.jsonl` (rewards are recomputed online from generations, so no separate prompt set is needed).
- **Final evaluation** — Phase 1 ranking uses the benchmark split; downstream comparisons against the fine-tuned models use `test.jsonl`.

The split is produced by `scripts/split_task_a_sft.py` (DVC stage `task_a_sft_splits`, seed 42) and written to `data/output/sft/task_a_splits/`.

| Split | File | Conversations |
|-------|------|--------------|
| train | `task_a_splits/train.jsonl` | ~4 414 |
| validation | `task_a_splits/validation.jsonl` | ~519 |
| test | `task_a_splits/test.jsonl` | ~261 |
| **Total** | | **~5 194** |

> **Important:** the `test.jsonl` split must not be used during training or hyperparameter selection. Reserve it for final evaluation only.

GRPO does not consume the SFT splits directly — `scripts/filter_grpo_data.py` (DVC stage `task_a_grpo`) filters the train + validation splits to L3-L5 and writes them to `data/output/grpo/task_a/`. See the section below for rationale.

#### Note: Reusing SFT data for GRPO

Reusing the SFT corpus as the GRPO prompt set is a common and defensible recipe in modern open-source RL pipelines (TRL's own GRPO tutorials, DeepSeek-R1, Tülu 3, Unsloth's GRPO notebooks). GRPO scores **model generations**, not labels — so it doesn't need new ground truth, only prompts that (a) elicit measurable behavior under the reward function and (b) leave the policy headroom to improve. The SFT corpus already satisfies both.

Things to keep in mind so the recipe stays honest:

1. **Saturation risk.** If the SFT checkpoint already nails most prompts, GRPO has nothing to optimize. The standard refinement is to **filter** `train.jsonl` rather than generate a new corpus — e.g., keep only L3–L5 conversations (where reward shaping helps most), or apply rejection sampling and keep prompts on which the SFT model scores below a reward threshold.
2. **Distribution amplification.** RL amplifies whatever the prompt distribution covers. The SFT mix (~87.5% adversarial + ~12.5% cooperative_only) will carry through to the GRPO policy — fine for robustness, but expect adversarial lean.
3. **Held-out discipline.** Using `train.jsonl` for both SFT and GRPO is safe **only if** `validation.jsonl` drives the GRPO held-out check (reward-hacking detector, Risk R5 in the spec) and `test.jsonl` stays untouched until final evaluation.
4. **No judge-model contamination** to worry about here — rewards are verifiable (state F1, tool call F1, graph-edit distance), not LLM-as-judge.

What labs sometimes do differently — and we have not done — includes generating RL-specific harder prompts (DeepSeek-style cold-start math), and mixing in fresh prompts to avoid the policy overfitting to the SFT distribution. With only ~4 445 conversations and verifiable rewards, prompt-filtering is the higher-value refinement than a separate generator.

---

## Full Generation Order

Run the scripts in this order. Each depends on having a clean `data/output/` directory and the relevant API keys set.

```bash
# 1. Benchmark — Gemini teachers (two runs, merged per level)
GEMINI_API_KEY=... ./scripts/generate_benchmark_data_teacher.sh --teacher gemini-3-flash
GEMINI_API_KEY=... ./scripts/generate_benchmark_data_teacher.sh --teacher gemini-3.1-flash-lite
# (then merge per-level outputs to l{level}_mixed_gemini-3_merged.jsonl)

# 2. SFT training data (also used by GRPO)
OPENAI_API_KEY=sk-... GEMINI_API_KEY=... ./scripts/generate_sft_data.sh

# 3. Cleanup
python scripts/clean_task_a_sft.py \
    --input-dir data/output/sft/task_a \
    --output-dir data/output/sft/task_a_cleaned

# 4. Deterministic 85/10/5 train/val/test split (seed=42)
python scripts/split_task_a_sft.py \
    --input-dir data/output/sft/task_a_cleaned \
    --output-dir data/output/sft/task_a_splits

# 5. GRPO prompts — L3-L5 filter over the SFT splits (no API calls)
python scripts/filter_grpo_data.py \
    --input-dir data/output/sft/task_a_splits \
    --output-dir data/output/grpo/task_a
```

**Total samples generated:** ~4 645 (benchmark ~200 + SFT ~4 445 cleaned). GRPO and final evaluation reuse splits of the SFT corpus rather than generating new data.

---

## Output File Naming

Each script writes one JSONL file per (level, language, teacher, behavior preset) combination:

```
{level}_conversations_{lang}_{model}{_behavior_preset}_{timestamp}.jsonl
```

The `intent_category_preset` is recorded in dataset metadata (`stats["intent_category_distribution"]`) but not in the filename.

Examples (SFT / GRPO / eval splits):
```
l1_conversations_en_gpt-5-4-nano-2026-03-17_adversarial_20260402_143201.jsonl
l3_conversations_th_gemini-3-flash-preview_adversarial_20260402_143415.jsonl
l4_conversations_code_switch_gpt-5-4-nano-2026-03-17_adversarial_20260402_143602.jsonl
```

**Benchmark split** uses a compact merged-artifact naming after the two Gemini teacher runs are combined per level:

```
l{level}_{lang}_gemini-3_merged.jsonl
```

Example: `l1_mixed_gemini-3_merged.jsonl` … `l5_mixed_gemini-3_merged.jsonl` (the on-disk artifacts at `data/output/benchmark/task_a/`).
