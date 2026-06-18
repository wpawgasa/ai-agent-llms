# LMCache CPU KV-Cache Offload — Serving Knob

**Date:** 2026-06-18
**Status:** Approved (brainstorming) → ready for implementation plan
**Scope:** `deployments/models/` docker serving stack only

## Goal

Make full-precision KV-cache CPU offload an **opt-in, env-driven knob** in the
existing docker serving stack, following the established "single generic `vllm`
service, configured entirely through `.env`" pattern. When the feature flag is
blank, the launch path is byte-for-byte identical to today.

LMCache offloads full-precision KV from GPU HBM to CPU RAM and pulls it back on
prefix reuse. Its two wins for this project — cross-request prefix reuse (Task A
conversations share large system prompts + tool schemas) and freeing HBM for
more concurrent sessions — are both available once the knob is on. Measuring
those effects is left to the operator; this work only makes the knob available
and correctly wired.

## Non-goals

- No Phase 3 quantization benchmark column (this is orthogonal to TurboQuant/FP8).
- No derived Dockerfile / new image artifact — `lmcache` is installed at runtime.
- No changes to `serving/launch_vllm.sh` (the docker stack is the operator's path).

## Decisions

- **Install method:** runtime `pip install lmcache` in the container entrypoint,
  gated by the enable flag. Matches the all-via-env philosophy; no build step.
- **CPU cache default:** `VLLM_LMCACHE_CPU_SIZE=20` (GB).
- **Coverage:** both `docker-compose.yml` (H100) and `docker-compose.l4.yml` (L4).
- **Quant conflict:** hard-fail when combined with `turboquant*`/`rotorquant*`.

## Component 1 — Env knobs

Added to both compose files' `environment:` block and to `.env.example`:

| Var | Default | Maps to |
|-----|---------|---------|
| `VLLM_LMCACHE_ENABLED` | blank (off) | feature gate; non-empty turns it on |
| `VLLM_LMCACHE_CPU_SIZE` | `20` | `LMCACHE_MAX_LOCAL_CPU_SIZE` (GB) |
| `VLLM_LMCACHE_CHUNK_SIZE` | `256` | `LMCACHE_CHUNK_SIZE` (tokens) |
| `VLLM_LMCACHE_VERSION` | blank (unpinned) | pip pin for `lmcache` |
| `VLLM_LMCACHE_KV_TRANSFER_CONFIG` | `{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}` | the `--kv-transfer-config` JSON |

Making the transfer-config JSON an overridable env var is deliberate: it absorbs
version drift (the LMCache docs note vLLM 0.20.0+ may want a
`kv_connector_module_path`) without a code change.

## Component 2 — Entrypoint logic (both compose files)

A gated block inserted in the inline entrypoint before `exec`, using `$$` so
expansion happens in-container, after `args=( … )` is built:

```bash
if [ -n "$${VLLM_LMCACHE_ENABLED:-}" ]; then
  # LMCache's own KV cache I/O is incompatible with the custom quant backends.
  case "$${VLLM_KV_CACHE_DTYPE:-}" in
    turboquant*|rotorquant*)
      echo "ERROR: VLLM_LMCACHE_ENABLED + VLLM_KV_CACHE_DTYPE=$${VLLM_KV_CACHE_DTYPE} is unsupported." >&2
      exit 1 ;;
  esac
  pip install --no-cache-dir "lmcache$${VLLM_LMCACHE_VERSION:+==$${VLLM_LMCACHE_VERSION}}"
  export LMCACHE_LOCAL_CPU=true
  export LMCACHE_MAX_LOCAL_CPU_SIZE="$${VLLM_LMCACHE_CPU_SIZE:-20}"
  export LMCACHE_CHUNK_SIZE="$${VLLM_LMCACHE_CHUNK_SIZE:-256}"
  args+=(--kv-transfer-config "$${VLLM_LMCACHE_KV_TRANSFER_CONFIG:-{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}}")
fi
```

## Component 3 — Error handling & constraints

- **Install failure** → `set -eu` aborts the container non-zero; the error is in logs.
- **Quant conflict** → hard-fail (above). `fp8` KV is *not* blocked but is untested
  with LMCache → documented as a caveat, not an error.
- **Cold-start cost** → runtime `pip install` adds ~10–30s per container start; the
  existing 600s healthcheck `start_period` absorbs it. Documented in `.env.example`.

## Component 4 — Docs

- `.env.example`: a new "LMCache CPU offload" preset block (5 vars + caveats),
  matching the existing preset-comment style.
- `CLAUDE.md`: a short note that LMCache CPU offload is an alternative to (not
  stackable with) the custom KV-quant backends, mirroring the existing R-note style.

## Verification (manual)

1. `VLLM_LMCACHE_ENABLED=1 docker compose up` → logs show `pip install lmcache`
   success, the `Launching: …` line includes `--kv-transfer-config`, and LMCache
   prints its init/CPU-buffer log line.
2. `/health` returns 200; two requests sharing a long system prompt → the second
   shows lower TTFT (prefix hit from CPU).
3. Default `.env` (flag blank) → `Launching:` line identical to today (regression).

## Open item for implementation time

Confirm the exact `lmcache` version that pairs with the image's vLLM 0.20.0 and
the precise connector name / module-path. Both are env-overridable, so adjusting
them post-hoc requires no code change — low risk.
