# Training Module

## Overview
`training/` implements Phase 2: two-stage fine-tuning (SFT then GRPO RL) for the 3 category winners selected in Phase 1. Uses Unsloth for 2× speed and 70% less VRAM.

## Files
- `sft.py` — Unsloth SFT entry point
- `grpo.py` — Unsloth GRPO RL entry point
- `rewards/reward_business_logic.py` — Cat A reward function
- `rewards/reward_subagent.py` — Cat B reward function
- `rewards/reward_graph_extraction.py` — Cat C reward function
- `reward_utils.py` — Shared reward computation helpers
- `lora_targets.py` — Per-architecture LoRA module registry
- `merge_adapter.py` — LoRA adapter merge and export
- `pilot_check.py` — 100-step pilot SFT on top-2 candidates (Risk R3)

## sft.py — Unsloth SFT

### Pipeline
1. Load base model via `FastLanguageModel.from_pretrained()` — QLoRA 4-bit for MoE, BF16 for dense ≤8GB
2. Apply LoRA via `FastLanguageModel.get_peft_model()` with targets from `lora_targets.py`
3. Configure SFTTrainer with ConstantLengthDataset packing + per-model chat template
4. Train for num_epochs, checkpoint every 500 steps
5. Select best checkpoint by validation loss
6. Return SFTResult with checkpoint path + training metrics

### Interface
```python
def train_sft(config_path: Path) -> SFTResult
```

### Key Unsloth Pattern
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=config.model.name,
    max_seq_length=config.training.max_seq_length,
    dtype=None,
    load_in_4bit=config.training.precision == "qlora_4bit",
)
model = FastLanguageModel.get_peft_model(
    model,
    r=config.lora.rank,
    lora_alpha=config.lora.alpha,
    lora_dropout=config.lora.dropout,
    target_modules=resolve_lora_targets(config),
    use_gradient_checkpointing="unsloth",
)
```

## grpo.py — Unsloth GRPO RL

### Pipeline
1. Load SFT checkpoint via `FastLanguageModel.from_pretrained()`
2. Configure GRPOTrainer with task-specific reward function, vLLM generation backend, FP8 RL, DAPO normalization, num_generations=4, beta=0.04 KL penalty
3. Train for 500–1000 steps
4. Monitor: reward curve, held-out eval every 50 steps (R5 hacking detection), KL divergence
5. Auto-stop if held-out metric drops while reward increases
6. Return GRPOResult with checkpoint path + reward curves

### Interface
```python
def train_grpo(config_path: Path) -> GRPOResult
```

### Key GRPO Config
- `normalization: DAPO` — removes length bias
- `generation_backend: vllm` — 11× faster RL inference
- `fp8_rl: true` — H100: 1.4× faster, 60% less VRAM during RL
- `beta: 0.04` — KL penalty to stay near SFT policy

## Reward Functions (`training/rewards/`)

### reward_business_logic.py — Cat A
```python
def reward_business_logic(prompts, completions, ground_truths) -> list[float]:
    # R1: state_transition_correctness  0.30
    # R2: tool_call_f1 (AST match)      0.30
    # R3: chain_propagation_accuracy    0.20
    # R4: format_compliance             0.10
    # R5: task_completion               0.10
    # Returns list of scalars ∈ [0.0, 1.0]
```

### reward_subagent.py — Cat B
```python
def reward_subagent(prompts, completions, ground_truths) -> list[float]:
    # tool_call_f1:         0.40
    # slot_extraction_acc:  0.30
    # state_sequence_match: 0.20
    # format_compliance:    0.10
```

### reward_graph_extraction.py — Cat C
```python
def reward_graph_extraction(prompts, completions, ground_truths) -> list[float]:
    # json_validity_bonus: 0.10  (early exit → 0.0 if invalid JSON)
    # node_f1:             0.35
    # edge_f1:             0.35
    # structural_validity: 0.10
    # 1 - normalized_GED:  0.10
```

## reward_utils.py — Shared Helpers

```python
def extract_state_annotations(text) -> list[tuple[str, str]]
def extract_tool_calls(text) -> list[dict]
def state_sequence_match(predicted, ground_truth) -> float
def tool_call_f1(predicted, ground_truth) -> float
def chain_propagation_score(tools, chains) -> float
def format_compliance_check(text) -> float
def reached_terminal(text, expected_terminal) -> bool
def node_f1(predicted_nodes, gold_nodes) -> float
def edge_f1(predicted_edges, gold_edges) -> float
def structural_validity(graph) -> float
def normalized_graph_edit_distance(predicted, gold) -> float
```

## lora_targets.py — Per-Architecture Registry

| Model | Notes |
|-------|-------|
| qwen25_3b, qwen3_32b, gemma_2b, gemma3_4b, mistral_24b | Standard q/k/v/o + gate/up/down_proj |
| qwen35_4b, qwen35_35b_a3b | + DeltaNet: in_proj_qkv/z/b/a, out_proj. Freeze mlp.gate. QLoRA 4-bit for 35B (~17.5GB) |
| glm47_flash | MLA: q_a/q_b/kv_a_proj_with_mqa/kv_b/o_proj + shared_experts. Freeze mlp.gate. ~60GB. Auto-fallback rank 64→32 (R7) |
| nemotron_30b | Mamba layers: Unsloth auto-detect. Freeze mlp.gate. vLLM compat uncertain (R6) |

## pilot_check.py — Risk R3 Mitigation
```python
def run_pilot_sft(top_2_models, task_data, pilot_steps=100) -> dict[str, PilotResult]
```
Run 100-step SFT on top-2 Phase 1 candidates. If #1 shows degradation, auto-fallback to #2.

## merge_adapter.py
```python
def merge_and_export(base_model, adapter_path, output_path, push_to_hub=False,
                     quantize_merged=None) -> None
```
Load base + LoRA adapter, merge via `model.merge_and_unload()`, optionally quantize to FP8 for deployment.

## Checklist
- [x] Implement sft.py with Unsloth FastLanguageModel pipeline
- [x] Implement grpo.py with GRPOTrainer + vLLM backend + FP8 RL
- [x] Implement reward_business_logic.py (5 components)
- [x] Implement reward_subagent.py (4 components)
- [x] Implement reward_graph_extraction.py (early-exit on invalid JSON)
- [x] Implement reward_utils.py with all shared helpers
- [x] Update lora_targets.py to include qwen35_35b_a3b and nemotron_30b entries
- [x] Implement pilot_check.py with top-2 auto-fallback logic
- [x] Update merge_adapter.py with optional quantize_merged param
- [x] Add W&B integration (reward curves, KL divergence, held-out eval)
- [x] Write test_reward_functions.py (known-answer tests, edge cases)
