"""Merge a Gemma-4 LoRA adapter into its base, and prove the merge happened.

Usage: python scripts/merge_lora_verified.py <adapter_dir> <output_dir>

The base model comes from the adapter's own adapter_config.json, and the model
class from the base config's architectures, so one script covers the Gemma-4
adapters (E4B, 12B, 26B-A4B). Merging runs on CPU in bf16.

Not training/merge_adapter.py: that loads AutoModelForCausalLM, but these
adapters were trained on the multimodal ConditionalGeneration class, whose text
layers live under model.language_model.layers. PEFT only warns on unmatched
adapter keys, so a mismatched load can merge nothing and save the base model.
This script fails unless every non-zero lora_B tensor loaded and sampled
weights changed, then requires the saved tensor names to match the base's
exactly (copying any the base has that transformers did not save, such as
E-series KV-shared k_proj / v_proj / k_norm).
"""

from __future__ import annotations

import glob
import json
import re
import shutil
import sys
from pathlib import Path

import torch
import transformers
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoProcessor

adapter_dir = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
marker = out_dir / "merge_ok.json"
EXTRA_SHARD = "model-completed-from-base.safetensors"

adapter_cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
BASE = adapter_cfg["base_model_name_or_path"]


def base_tensor_files() -> list[Path]:
    snap = Path(snapshot_download(BASE, allow_patterns=["*.safetensors", "*.json"]))
    return sorted(Path(p) for p in glob.glob(str(snap / "*.safetensors")))


def complete_from_base(out: Path) -> None:
    base_owner = {}
    for f in base_tensor_files():
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                base_owner[k] = f
    shards = sorted(Path(p).name for p in glob.glob(str(out / "*.safetensors")))
    owner = {}
    for shard in shards:
        with safe_open(str(out / shard), "pt") as h:
            for k in h.keys():
                owner[k] = shard
    missing = sorted(set(base_owner) - set(owner))
    extra = sorted(set(owner) - set(base_owner))
    if extra:
        sys.exit(f"[merge] merged checkpoint has {len(extra)} tensors the base lacks, e.g. {extra[:3]}")
    if missing:
        bad = [k for k in missing if not re.search(r"\.self_attn\.(k_proj|v_proj|k_norm)\.weight$", k)]
        if bad:
            sys.exit(f"[merge] unexpected tensors missing from merged checkpoint, e.g. {bad[:3]}")
        tensors = {}
        for k in missing:
            with safe_open(str(base_owner[k]), "pt") as h:
                tensors[k] = h.get_tensor(k).contiguous()
        save_file(tensors, str(out / EXTRA_SHARD), metadata={"format": "pt"})
        for k in missing:
            owner[k] = EXTRA_SHARD
        print(f"[merge] added {len(missing)} base tensors as {EXTRA_SHARD}")
    else:
        print("[merge] no tensors missing relative to the base")
    total_size = sum((out / s).stat().st_size for s in set(owner.values()))
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(owner.items()))}, indent=2
    ))
    if set(owner) != set(base_owner):
        sys.exit("[merge] merged checkpoint still does not match the base tensor names")
    print(f"[merge] tensor names match the base exactly: {len(owner)}")


if marker.exists():
    print(f"[merge] already merged and verified: {marker}")
    complete_from_base(out_dir)
    sys.exit(0)

cfg = AutoConfig.from_pretrained(BASE)
arch = cfg.architectures[0]
print(f"[merge] base {BASE} architecture {arch}")
model = getattr(transformers, arch).from_pretrained(BASE, dtype=torch.bfloat16, device_map="cpu")

state = model.state_dict()
probe_patterns = {
    "q_proj": r"language_model\.layers\.0\.self_attn\.q_proj\.weight$",
    "gate_proj": r"language_model\.layers\.0\.mlp\.gate_proj\.weight$",
}
probes = {}
for label, pattern in probe_patterns.items():
    names = [n for n in state if re.search(pattern, n)]
    if len(names) != 1:
        sys.exit(f"[merge] expected exactly one {label} probe weight, found {names}")
    probes[label] = (names[0], state[names[0]].detach().clone())
del state

with safe_open(str(adapter_dir / "adapter_model.safetensors"), "pt") as f:
    file_b = [k for k in f.keys() if ".lora_B." in k]
    file_b_nonzero = sum(1 for k in file_b if f.get_tensor(k).abs().max().item() > 0)
print(f"[merge] adapter file: {len(file_b)} lora_B tensors, {file_b_nonzero} non-zero")
if file_b_nonzero == 0:
    sys.exit("[merge] adapter file has no non-zero lora_B tensors; nothing was trained")

peft_model = PeftModel.from_pretrained(model, str(adapter_dir))
model_b = [(n, p) for n, p in peft_model.named_parameters() if ".lora_B." in n]
model_b_nonzero = sum(1 for _, p in model_b if p.abs().max().item() > 0)
print(f"[merge] loaded into model: {len(model_b)} lora_B tensors, {model_b_nonzero} non-zero")
if model_b_nonzero != file_b_nonzero:
    sys.exit(
        f"[merge] only {model_b_nonzero} of {file_b_nonzero} non-zero lora_B tensors loaded; "
        "adapter keys did not match this architecture"
    )

merged = peft_model.merge_and_unload()
merged_state = merged.state_dict()
deltas = {}
for label, (name, before) in probes.items():
    delta = (merged_state[name].float() - before.float()).abs().max().item()
    deltas[label] = delta
    print(f"[merge] {label} max |merged - base| = {delta:.3e} ({name})")
    if delta == 0:
        sys.exit(f"[merge] {label} is unchanged by the merge")
del merged_state

out_dir.mkdir(parents=True, exist_ok=True)
merged.save_pretrained(str(out_dir), safe_serialization=True)
AutoProcessor.from_pretrained(BASE).save_pretrained(str(out_dir))
for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    if (adapter_dir / name).exists():
        shutil.copy2(adapter_dir / name, out_dir / name)

saved_arch = json.loads((out_dir / "config.json").read_text()).get("architectures")
if saved_arch != [arch]:
    sys.exit(f"[merge] saved config architectures {saved_arch}, expected {[arch]}")

complete_from_base(out_dir)

marker.write_text(json.dumps({
    "base": BASE,
    "architecture": arch,
    "adapter": str(adapter_dir),
    "lora_B_tensors": len(file_b),
    "lora_B_nonzero_loaded": model_b_nonzero,
    "probe_max_abs_delta": deltas,
}, indent=2))
print(f"[merge] done: {out_dir}")
