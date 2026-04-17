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
    num_states: tuple[int, int]
    branching_factor: tuple[int, int]
    num_tools: int
    chain_depth: int
    nesting_depth: int
    num_samples: int
    domain: str

COMPLEXITY_SPECS = {
    "L1": ComplexitySpec("L1", (3,4),   (1,2), 1, 0, 0, 200, "faq_lookup"),
    "L2": ComplexitySpec("L2", (5,7),   (2,3), 2, 1, 1, 200, "order_status_cancel"),
    "L3": ComplexitySpec("L3", (8,12),  (3,5), 4, 2, 2, 200, "booking_payment"),
    "L4": ComplexitySpec("L4", (13,20), (5,8), 6, 3, 3, 200, "it_troubleshoot"),
    "L5": ComplexitySpec("L5", (21,30), (8,99),7, 4, 4, 200, "multi_dept_workflow"),
}

def generate_workflow_dataset(
    complexity_level: Literal["L1","L2","L3","L4","L5"],
    teacher_model: str = "gpt-4o",
    output_dir: Path = Path("data/output/task_a"),
    seed: int = 42,
) -> DatasetMetadata
```

### User Behavior Distribution
- cooperative: 60%, adversarial_probing: 15%, digressing: 10%, invalid_tool_inputs: 15%
- Tool error rate: 20% of tool calls return error payloads

### Output Format (per sample)
Each sample includes: conversation_id, complexity_level, domain, workflow_graph, tool_schemas, messages (with `[STATE: X → Y]` and `<tool_call>` annotations), user_behavior, ground_truth (state_sequence, tool_chain_dependencies, terminal_state).

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
