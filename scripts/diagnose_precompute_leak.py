#!/usr/bin/env python3
"""Find what the DPO reference-logprob precompute retains, 64 MB at a time.

WHY: two 500-step runs were OOM-killed during `precompute_ref_log_probs`,
never during training. Sampling VmRSS against the precompute's row counter
measured a linear 64 MB per row, projecting to ~320 GB over the 5,000-row
split (docs/dpo_memory_ceiling_investigation.md section 10). Pinned host
buffers were ruled out — the slope is 65 MB/row with `dataloader_pin_memory`
off — and the offloaded-gradient-checkpointing hypothesis could not be tested,
because Unsloth leaves `gradient_checkpointing=True` on the layers even when
`from_pretrained` is told not to use it.

So stop guessing and look. This drives the precompute loop by hand and takes a
census of every live torch tensor between rows, grouped by (device, dtype,
shape). Whatever grows monotonically at ~64 MB per row is the retainer.

The census walks `gc.get_objects()`, so it sees only tensors reachable from
Python. **That is the point of the second number it prints.** If the
Python-visible total stays flat while RSS climbs, the retention is on the C++
side — a caching allocator, a pinned block, or a buffer no Python object owns —
and the search moves there instead. Either answer narrows it.

Usage (about 15 minutes, well under any memory floor):

    .venv-train/bin/python scripts/diagnose_precompute_leak.py --rows 300
"""

from __future__ import annotations

import argparse
import collections
import gc
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _rss_gb() -> float:
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1048576
    return 0.0


def _census() -> tuple[collections.Counter, int]:
    """Live torch tensors reachable from Python, keyed by (device, dtype, shape)."""
    import torch

    by_key: collections.Counter = collections.Counter()
    total = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                t = obj
            elif isinstance(obj, torch.nn.Parameter):
                t = obj.data
            else:
                continue
            nbytes = t.numel() * t.element_size()
            by_key[(str(t.device), str(t.dtype), tuple(t.shape))] += nbytes
            total += nbytes
        except Exception:  # a tensor mid-teardown must not stop the census
            continue
    return by_key, total


def _report(tag: str, before: collections.Counter, after: collections.Counter,
            rows: int) -> None:
    grown = {k: after[k] - before.get(k, 0) for k in after}
    grown = {k: v for k, v in grown.items() if v > 0}
    print(f"\n--- {tag}: top growth over {rows} rows ---")
    if not grown:
        print("   (no Python-visible tensor group grew)")
        return
    for key, delta in sorted(grown.items(), key=lambda kv: -kv[1])[:10]:
        dev, dtype, shape = key
        print(f"   +{delta/1048576:9.1f} MB  {dev:12s} {dtype:16s} {shape}")



def _run_real(args) -> int:
    """Run the real train_dpo and census from a background thread.

    The reconstruction in main() does not leak, so the difference is somewhere
    in the real call path that a rebuilt loop does not exercise. Rather than
    guess which line, run the real thing and watch it from inside the same
    process.
    """
    import threading
    import time
    from pathlib import Path as _P

    import unsloth  # noqa: F401
    import torch

    from llm_workflow_agents.training.dpo import train_dpo

    stop = threading.Event()

    def sampler() -> None:
        prev_rss = _rss_gb()
        prev_t = time.time()
        while not stop.wait(60):
            gc.collect()
            by_key, total = _census()
            rss = _rss_gb()
            dt = time.time() - prev_t
            print(
                f"\n[census] RSS {rss:7.1f} GB (+{rss-prev_rss:.2f} in {dt:.0f}s)"
                f" | python-visible {total/1073741824:6.2f} GB"
                f" | cuda {torch.cuda.memory_allocated()/1073741824:6.2f} GB",
                flush=True,
            )
            # Is the growth Unsloth's global CPU buffer pool, or free-floating
            # tensors? These two numbers separate them.
            try:
                from unsloth_zoo import gradient_checkpointing as _gc
                bufs = getattr(_gc, "CPU_BUFFERS", [])
                nbuf = len(bufs)
                bbytes = sum(b.numel() * b.element_size() for b in bufs)
            except Exception as exc:  # noqa: BLE001
                nbuf, bbytes = -1, 0
            hidden = 2816
            hs = {k: v for k, v in by_key.items()
                  if k[0] == "cpu" and k[1] == "torch.bfloat16"
                  and len(k[2]) == 1 and k[2][0] % (2 * hidden) == 0}
            print(f"   unsloth CPU_BUFFERS: {nbuf} buffers, "
                  f"{bbytes/1073741824:.2f} GB", flush=True)
            print(f"   hidden-state-shaped cpu bf16: {len(hs)} groups, "
                  f"{sum(hs.values())/1073741824:.2f} GB", flush=True)
            cpu = {k: v for k, v in by_key.items() if k[0] == "cpu"}
            top = sorted(cpu.items(), key=lambda kv: -kv[1])[:6]
            for (dev, dtype, shape), nbytes in top:
                print(f"   cpu {nbytes/1048576:9.1f} MB  {dtype:16s} {shape}",
                      flush=True)
            prev_rss, prev_t = rss, time.time()

    threading.Thread(target=sampler, daemon=True).start()
    result = train_dpo(_P(args.config))
    stop.set()
    print(f"train_dpo returned: error={result.error} steps={result.total_steps}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path,
                    default=Path("configs/training/dpo_cat_a.yaml"))
    ap.add_argument("--rows", type=int, default=300,
                    help="rows to push through the precompute loop")
    ap.add_argument("--every", type=int, default=50,
                    help="census interval, in rows")
    ap.add_argument("--mimic-trl", action="store_true",
                    help="also run gather_for_metrics and the .cpu() appends, "
                         "i.e. the two lines _precompute_ref_logps has that a "
                         "bare compute_ref_log_probs loop does not")
    ap.add_argument("--real", action="store_true",
                    help="census from a thread INSIDE the real train_dpo call, "
                         "rather than reconstructing the loop. Use when the "
                         "reconstruction does not reproduce the leak.")
    args = ap.parse_args()

    os.environ.pop("UNSLOTH_VLLM_STANDBY", None)

    if args.real:
        return _run_real(args)

    import unsloth  # noqa: F401  must precede torch/transformers
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from unsloth import FastLanguageModel

    from llm_workflow_agents.training._utils import (
        unwrap_unsloth_gemma4_kv_zero_proxy,
    )
    from llm_workflow_agents.training.dpo import (
        _dpo_trainer_kwargs,
        _filter_dpo_config_kwargs,
        _load_dpo_dataset,
        _resolve_trl_classes,
    )

    cfg = yaml.safe_load(args.config.read_text())
    dpo_cfg = cfg["dpo"]
    data_cfg = dict(cfg["data"])
    data_cfg["max_train_rows"] = args.rows

    config_cls, trainer_cls = _resolve_trl_classes(dpo_cfg.get("method", "dpo"))

    unwrap_unsloth_gemma4_kv_zero_proxy()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model"]["sft_checkpoint"],
        max_seq_length=dpo_cfg.get("max_seq_length", 6144),
        dtype=None,
        load_in_4bit=True,
    )
    from llm_workflow_agents.training.sft import _patch_gemma4_rope_stride

    _patch_gemma4_rope_stride(model)

    train_ds, eval_ds = _load_dpo_dataset(data_cfg, seed=dpo_cfg.get("seed", 42))

    # Build the trainer with the precompute OFF so __init__ does not run the
    # very loop under investigation; we drive it by hand below.
    kwargs = _dpo_trainer_kwargs(dpo_cfg, "dpo", "/tmp/leak_probe")
    kwargs["precompute_ref_log_probs"] = False
    kwargs["report_to"] = "none"
    kwargs, _dropped = _filter_dpo_config_kwargs(kwargs, "dpo")
    trainer = trainer_cls(
        model=model,
        args=config_cls(**kwargs),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    # Mirror _precompute_ref_logps' dataloader exactly.
    loader = DataLoader(
        trainer.train_dataset,
        batch_size=1,
        collate_fn=trainer.data_collator,
        num_workers=0,
        pin_memory=False,
        shuffle=False,
    )
    loader = trainer.accelerator.prepare(loader)

    gc.collect()
    base_by_key, base_total = _census()
    base_rss = _rss_gb()
    print(f"\nbaseline: RSS {base_rss:.1f} GB | python-visible tensors "
          f"{base_total/1073741824:.2f} GB | cuda_allocated "
          f"{torch.cuda.memory_allocated()/1073741824:.2f} GB")

    prev_by_key, prev_total, prev_rss, prev_n = base_by_key, base_total, base_rss, 0
    ref_chosen_logps: list = []
    ref_rejected_logps: list = []
    with torch.no_grad():
        for n, batch in enumerate(loader, start=1):
            out = trainer.compute_ref_log_probs(batch)
            if args.mimic_trl:
                # The only two lines _precompute_ref_logps has that the bare
                # loop above does not. A no-op on one process, in principle.
                chosen, rejected = trainer.accelerator.gather_for_metrics(out)
                ref_chosen_logps.append(chosen.cpu())
                ref_rejected_logps.append(rejected.cpu())
            if n % args.every:
                continue
            gc.collect()
            by_key, total = _census()
            rss = _rss_gb()
            d_rows = n - prev_n
            print(
                f"\n=== row {n} ===\n"
                f"  RSS              {rss:7.1f} GB  (+{rss-prev_rss:.2f} = "
                f"{(rss-prev_rss)*1024/d_rows:.0f} MB/row)\n"
                f"  python tensors   {total/1073741824:7.2f} GB  "
                f"(+{(total-prev_total)/1048576:.0f} MB = "
                f"{(total-prev_total)/1048576/d_rows:.0f} MB/row)\n"
                f"  cuda allocated   {torch.cuda.memory_allocated()/1073741824:7.2f} GB"
            )
            _report("python-visible", prev_by_key, by_key, d_rows)
            prev_by_key, prev_total, prev_rss, prev_n = by_key, total, rss, n
            if n >= args.rows:
                break

    rss_grew = (prev_rss - base_rss) * 1024
    py_grew = (prev_total - base_total) / 1048576
    rate = rss_grew / max(prev_n, 1)
    print(f"\n  measured rate         {rate:8.1f} MB/row over {prev_n} rows")
    if rate < 5:
        print("  -> NO LEAK reproduced in this harness (real run: 64 MB/row).")
        print("     The retention is in the caller, not in this loop.")
        return 0
    print("\n================ VERDICT ================")
    print(f"  RSS growth            {rss_grew:8.0f} MB")
    print(f"  python-visible growth {py_grew:8.0f} MB")
    if rss_grew > 500 and py_grew < rss_grew * 0.25:
        print("  -> the retention is NOT reachable from Python.")
        print("     Look at the C++ side: caching allocators, pinned blocks,")
        print("     or a buffer no Python object owns.")
    elif py_grew >= rss_grew * 0.25:
        print("  -> a Python-reachable tensor group accounts for the growth.")
        print("     The largest group listed above is the retainer.")
    else:
        print("  -> no meaningful growth reproduced in this harness;")
        print("     the leak needs TRL's own loop, not this one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
