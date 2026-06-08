# Data Generation Module

## Overview
`data/` handles synthetic data generation for all three task categories using teacher models (GPT-4o / Claude Sonnet 4).

## Files
- `generate_workflows.py` — Task A: multi-turn workflow conversations with state-machine annotations
- `generate_tool_call_data.py` — Task B: tool-call fine-tuning data (merge external + synthetic)
- `generate_graph_pairs.py` — Task C: (prompt, graph) pair generation
- `chat_template_converter.py` — Convert unified JSONL to model-specific formats
- `data_validator.py` — Schema validation and quality checks
- `templates/` — Prompt templates, tool schemas (L1-L5), graph output schema

## Task A: generate_workflows.py

### Interface
```python
@dataclass
class ComplexitySpec:
    level: str
    target_path_len: tuple[int, int]   # spine states to include (terminal counts)
    num_branches: tuple[int, int]      # optional off-spine branch edges
    num_loops: tuple[int, int]         # retry / escalation back-edges
    include_recovery: bool             # add tool_error recovery arcs
    num_tools: int                     # minimum tool count (never truncated above)
    chain_depth: int

COMPLEXITY_SPECS = {
    "L1": ComplexitySpec("L1", (3,4),   (0,0), (0,0), False, 1, 0),
    "L2": ComplexitySpec("L2", (5,7),   (1,1), (0,0), False, 2, 1),
    "L3": ComplexitySpec("L3", (8,12),  (2,3), (0,1), False, 4, 2),
    "L4": ComplexitySpec("L4", (12,16), (3,5), (1,1), True,  6, 3),
    "L5": ComplexitySpec("L5", (16,20), (0,99),(1,2), True,  7, 4),
}

def generate_workflow_dataset(
    complexity_level: Literal["L1","L2","L3","L4","L5"],
    num_samples: int = 200,
    domain: str | None = None,          # None → random; pin with e.g. "banking"
    intent_category_preset: str = "default",  # "default"|"service_only"|"upsell_heavy"
    initiation_preset: str = "default",  # "default"|"balanced"|"outbound_heavy"
    teacher_model: str = "gpt-4o",
    output_dir: Path = Path("data/output/task_a"),
    seed: int = 42,
) -> DatasetMetadata
```

**Domain eligibility by level:** `_select_domain` filters domains by canonical state count ≥ `target_path_len` minimum. L1–L3 draw from all 18 domains; L4 requires ≥12-state domains; L5 requires the 5 expanded rich domains (banking, insurance, healthcare, travel, telecom).

### Conversation Initiation (inbound vs outbound)
By default every conversation is **inbound** (user-initiated): after the `system` message the customer speaks first. `initiation_preset` mixes in **outbound** (support-initiated) conversations where the agent opens the call with a purpose.

| Preset | user (inbound) | agent (outbound) |
|--------|---------------|------------------|
| `default` | 100% | 0% |
| `balanced` | 70% | 30% |
| `outbound_heavy` | 40% | 60% |

Outbound samples reuse each domain's canonical graph: the agent opens at the existing initial state stating the reason (`messages[1].role == "assistant"`); `state_sequence` still starts at the initial state. Outbound is only chosen for domains carrying an `outbound_reasons` tuple — a curated subset: **sales, banking, insurance, healthcare, telecom, travel, scheduling**. Each reason is an `OutboundReason(key, description, intent_category)` (e.g. healthcare `prescription_followup`, insurance `renewal_reminder`, sales `promotion_offer`); its `intent_category` ("service" | "upsell_promo") drives subgraph arc selection. If an outbound sample lands on a domain without reasons it falls back to inbound (counted in stats `outbound_fallback_inbound`).

### User Behavior Distribution
- cooperative: 60%, adversarial_probing: 15%, digressing: 10%, invalid_tool_inputs: 15%
- Tool error rate: 20% of tool calls return error payloads

### Output Format (per sample)
Each sample includes: conversation_id, complexity_level, domain, workflow_graph, tool_schemas, messages (with `[STATE: X → Y]` and `<tool_call>` annotations), user_behavior, ground_truth (state_sequence, tool_chain_dependencies, terminal_state), `conversation_initiator` ("user" | "agent"), and `outbound_reason` (reason key | null). Outbound message shape: `[system, assistant(opener), user, assistant, …]`.

## Task B: generate_tool_call_data.py

### Interface
```python
def generate_tool_call_dataset(
    external_sources: list[str] = ["Salesforce/xlam-function-calling-60k", "ToolBench"],
    custom_synthetic_size: int = 15000,
    teacher_model: str = "gpt-4o",
    negative_ratio: float = 0.15,
    hermes_format: bool = True,
    output_dir: Path = Path("data/output/task_b"),
    seed: int = 42,
) -> DatasetSplits  # train 85% / val 10% / test 5%
```

### Negative Examples (15% total)
- Wrong tool selected: 5%, Hallucinated tool: 4%, Invalid state transition: 3%, Error recovery: 3%

## Task C: generate_graph_pairs.py

### Interface
```python
def generate_graph_pairs(
    workflow_prompts_dir: Path,       # From Task A (1000 prompts)
    gold_annotations: int = 200,
    teacher_generated: int = 800,
    augmentation_target: int = 5000,
    output_dir: Path = Path("data/output/task_c"),
) -> DatasetSplits  # 4000 train / 500 val / 500 test
```

### Graph Output Schema
WorkflowGraph: nodes (id, name, tools, entry_actions), edges (from, to, condition, priority), initial_state, terminal_states

## Chat Template Converter
Supported formats: qwen (ChatML + hermes tool), qwen35 (ChatML + qwen3_coder), gemma, mistral_instruct_v3, nemotron, glm_chatml

## Checklist
- [x] Implement ComplexitySpec dataclass and COMPLEXITY_SPECS (with nesting_depth)
- [x] Implement generate_workflows.py — output to data/output/task_a/
- [x] Create tool_schemas_L1_to_L5.json and workflow_prompt_template.txt
- [x] Implement generate_tool_call_data.py — output to data/output/task_b/
- [x] Implement generate_graph_pairs.py — output to data/output/task_c/
- [x] Create graph_output_schema.json
- [x] Implement chat_template_converter.py for all 6 model families (add qwen35 with qwen3_coder)
- [x] Implement data_validator.py with schema validation
- [x] Write test_data_generation.py tests
- [x] Write test_chat_templates.py tests
