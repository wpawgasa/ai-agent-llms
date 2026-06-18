# LMCache CPU KV-Cache Offload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in, env-driven LMCache CPU KV-cache offload knob to the `deployments/models` docker serving stack (both H100 and L4 compose files), with docs and tests.

**Architecture:** Both compose files run a single generic `vllm` service whose flags are built at runtime by an inline bash entrypoint. We add (a) new `VLLM_LMCACHE_*` env passthroughs to each service's `environment:` block, and (b) a gated block in each entrypoint that, when `VLLM_LMCACHE_ENABLED` is non-empty, `pip install`s `lmcache`, exports the `LMCACHE_*` runtime vars, and appends `--kv-transfer-config` to the vLLM args. A `VLLM_ENTRYPOINT_DRY_RUN` affordance lets the entrypoint print resolved args and exit without launching, making the logic unit-testable without docker or a GPU.

**Tech Stack:** docker compose, bash, Python/pytest (test harness extracts the entrypoint script from the compose YAML and runs it under bash), LMCache, vLLM 0.20.0.

**Reference spec:** `docs/superpowers/specs/2026-06-18-lmcache-cpu-offload-design.md`

## Global Constraints

- Default path (flag blank) MUST be byte-for-byte identical to today — the LMCache block only runs when `VLLM_LMCACHE_ENABLED` is non-empty.
- Both `deployments/models/docker-compose.yml` (H100) and `deployments/models/docker-compose.l4.yml` (L4) get identical entrypoint + env changes. Their entrypoints are currently byte-identical; keep them so.
- Hard-fail (exit 1) when `VLLM_LMCACHE_ENABLED` is set together with `VLLM_KV_CACHE_DTYPE` matching `turboquant*` or `rotorquant*`.
- Default CPU cache: `VLLM_LMCACHE_CPU_SIZE=20` (GB). Default chunk size: `256`.
- Default transfer config: `{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}`, overridable via `VLLM_LMCACHE_KV_TRANSFER_CONFIG`.
- In compose YAML, in-container shell variable refs are written `$$` so docker compose passes them through literally; keep that convention for all new entrypoint lines.
- `.env` is gitignored; only `.env.example` is edited.

---

## File Structure

- Modify: `deployments/models/docker-compose.yml` — add `VLLM_LMCACHE_*` + `VLLM_ENTRYPOINT_DRY_RUN` to `environment:`; add LMCache gated block + dry-run guard to entrypoint.
- Modify: `deployments/models/docker-compose.l4.yml` — identical changes.
- Modify: `deployments/models/.env.example` — add the new knobs to the optional-knob list + a "LMCache CPU offload" preset/usage block.
- Modify: `CLAUDE.md` — short note that LMCache CPU offload is an alternative to (not stackable with) the custom KV-quant backends.
- Create: `tests/test_lmcache_entrypoint.py` — extract-and-run dry-run tests over both compose files + a docs-parity test.

---

## Task 1: H100 compose — LMCache entrypoint block + env passthrough (TDD)

**Files:**
- Create: `tests/test_lmcache_entrypoint.py`
- Modify: `deployments/models/docker-compose.yml` (environment block + entrypoint script)

**Interfaces:**
- Produces: a compose entrypoint that, given env, either (dry-run) prints a `Launching: …` line plus `LMCACHE_*` lines and exits 0, or hard-fails on a quant conflict. Consumed by Task 2 (same shape for L4) and the parametrized tests.
- Test helpers produced for reuse in Task 2/3: `_extract_entrypoint_script(compose_path)`, `_base_env()`, `_run(compose_path, overrides)`, module constant `COMPOSE_FILES`.

- [ ] **Step 1: Write the failing test file**

Create `tests/test_lmcache_entrypoint.py`:

```python
"""Tests for the LMCache CPU-offload knob in the deployments/models compose entrypoints.

The entrypoint is an inline bash script embedded in each compose file. These tests
extract that script, apply docker compose's `$$`->`$` escaping (the only
entrypoint-relevant substitution compose performs), and run it under bash with
VLLM_ENTRYPOINT_DRY_RUN=1 so it prints resolved args and exits without launching.
"""

import subprocess
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "deployments" / "models"
COMPOSE_FILES = [
    MODELS_DIR / "docker-compose.yml",
    MODELS_DIR / "docker-compose.l4.yml",
]


def _extract_entrypoint_script(compose_path: Path) -> str:
    cfg = yaml.safe_load(compose_path.read_text())
    entrypoint = cfg["services"]["vllm"]["entrypoint"]  # ["/bin/bash", "-c", "<script>"]
    return entrypoint[2].replace("$$", "$")


def _base_env() -> dict:
    """Mirror the compose `environment:` defaults so the script runs under `set -u`."""
    return {
        "PATH": "/usr/bin:/bin",
        "VLLM_MODEL": "test/model",
        "VLLM_DTYPE": "bfloat16",
        "VLLM_TOOL_CALL_PARSER": "hermes",
        "VLLM_GPU_MEMORY_UTILIZATION": "0.90",
        "VLLM_MAX_MODEL_LEN": "8192",
        "VLLM_HF_OVERRIDES": "",
        "VLLM_KV_CACHE_DTYPE": "",
        "VLLM_ENFORCE_EAGER": "",
        "VLLM_TRUST_REMOTE_CODE": "",
        "VLLM_EXTRA_ARGS": "",
        "VLLM_USE_V2_MODEL_RUNNER": "",
        "VLLM_LMCACHE_ENABLED": "",
        "VLLM_LMCACHE_CPU_SIZE": "20",
        "VLLM_LMCACHE_CHUNK_SIZE": "256",
        "VLLM_LMCACHE_VERSION": "",
        "VLLM_LMCACHE_KV_TRANSFER_CONFIG": "",
        "VLLM_ENTRYPOINT_DRY_RUN": "1",
    }


def _run(compose_path: Path, overrides: dict) -> subprocess.CompletedProcess:
    env = _base_env()
    env.update(overrides)
    script = _extract_entrypoint_script(compose_path)
    return subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True
    )


@pytest.mark.parametrize("compose", COMPOSE_FILES, ids=lambda p: p.name)
def test_disabled_omits_kv_transfer_config(compose):
    r = _run(compose, {"VLLM_LMCACHE_ENABLED": ""})
    assert r.returncode == 0, r.stderr
    assert "Launching:" in r.stdout
    assert "--kv-transfer-config" not in r.stdout


@pytest.mark.parametrize("compose", COMPOSE_FILES, ids=lambda p: p.name)
def test_enabled_defaults(compose):
    r = _run(compose, {"VLLM_LMCACHE_ENABLED": "1"})
    assert r.returncode == 0, r.stderr
    assert "--kv-transfer-config" in r.stdout
    assert "LMCacheConnectorV1" in r.stdout
    assert '"kv_role":"kv_both"' in r.stdout
    assert "LMCACHE_MAX_LOCAL_CPU_SIZE=20" in r.stdout
    assert "LMCACHE_CHUNK_SIZE=256" in r.stdout


@pytest.mark.parametrize("compose", COMPOSE_FILES, ids=lambda p: p.name)
def test_enabled_custom_size_and_chunk(compose):
    r = _run(compose, {
        "VLLM_LMCACHE_ENABLED": "1",
        "VLLM_LMCACHE_CPU_SIZE": "40",
        "VLLM_LMCACHE_CHUNK_SIZE": "128",
    })
    assert r.returncode == 0, r.stderr
    assert "LMCACHE_MAX_LOCAL_CPU_SIZE=40" in r.stdout
    assert "LMCACHE_CHUNK_SIZE=128" in r.stdout


@pytest.mark.parametrize("compose", COMPOSE_FILES, ids=lambda p: p.name)
def test_enabled_custom_transfer_config(compose):
    r = _run(compose, {
        "VLLM_LMCACHE_ENABLED": "1",
        "VLLM_LMCACHE_KV_TRANSFER_CONFIG":
            '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}',
    })
    assert r.returncode == 0, r.stderr
    assert "LMCacheMPConnector" in r.stdout
    assert "LMCacheConnectorV1" not in r.stdout


@pytest.mark.parametrize("compose", COMPOSE_FILES, ids=lambda p: p.name)
def test_enabled_version_pin_in_dry_run(compose):
    r = _run(compose, {"VLLM_LMCACHE_ENABLED": "1", "VLLM_LMCACHE_VERSION": "0.3.1"})
    assert r.returncode == 0, r.stderr
    assert "lmcache==0.3.1" in r.stdout


@pytest.mark.parametrize("compose", COMPOSE_FILES, ids=lambda p: p.name)
@pytest.mark.parametrize("dtype", ["turboquant_3bit_nc", "rotorquant", "turboquant"])
def test_quant_conflict_hard_fails(compose, dtype):
    r = _run(compose, {"VLLM_LMCACHE_ENABLED": "1", "VLLM_KV_CACHE_DTYPE": dtype})
    assert r.returncode != 0
    assert "unsupported" in r.stderr.lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_lmcache_entrypoint.py -v`
Expected: FAIL — the `test_enabled_*` and `test_quant_conflict_*` cases fail because the current entrypoint ignores `VLLM_LMCACHE_ENABLED` (no `--kv-transfer-config`, no `LMCACHE_*` lines, no hard-fail, and no dry-run exit so the script reaches `exec` and errors). The L4 cases may also error during extraction if the dry-run guard is absent. This confirms the test exercises the new behavior.

- [ ] **Step 3: Add the LMCache env passthroughs to `docker-compose.yml`**

In `deployments/models/docker-compose.yml`, find this block in `services.vllm.environment`:

```yaml
      VLLM_USE_V2_MODEL_RUNNER: ${VLLM_USE_V2_MODEL_RUNNER:-}
    volumes:
```

Replace it with (insert the LMCache + dry-run vars before `volumes:`):

```yaml
      VLLM_USE_V2_MODEL_RUNNER: ${VLLM_USE_V2_MODEL_RUNNER:-}
      # --- LMCache CPU KV-cache offload (opt-in; see .env.example) ----------
      VLLM_LMCACHE_ENABLED: ${VLLM_LMCACHE_ENABLED:-}
      VLLM_LMCACHE_CPU_SIZE: ${VLLM_LMCACHE_CPU_SIZE:-20}
      VLLM_LMCACHE_CHUNK_SIZE: ${VLLM_LMCACHE_CHUNK_SIZE:-256}
      VLLM_LMCACHE_VERSION: ${VLLM_LMCACHE_VERSION:-}
      VLLM_LMCACHE_KV_TRANSFER_CONFIG: ${VLLM_LMCACHE_KV_TRANSFER_CONFIG:-}
      # Testing affordance: when non-empty, the entrypoint prints resolved args +
      # LMCache env and exits 0 without pip-installing or launching vLLM.
      VLLM_ENTRYPOINT_DRY_RUN: ${VLLM_ENTRYPOINT_DRY_RUN:-}
    volumes:
```

- [ ] **Step 4: Add the LMCache block + dry-run guard to the `docker-compose.yml` entrypoint**

In the same file, find the tail of the entrypoint script:

```yaml
        if [ -n "$${VLLM_EXTRA_ARGS}" ];     then args+=($${VLLM_EXTRA_ARGS}); fi
        echo "Launching: python3 -m vllm.entrypoints.openai.api_server $${args[*]}"
        exec python3 -m vllm.entrypoints.openai.api_server "$${args[@]}"
```

Replace it with:

```yaml
        if [ -n "$${VLLM_EXTRA_ARGS}" ];     then args+=($${VLLM_EXTRA_ARGS}); fi
        # LMCache CPU KV-cache offload (opt-in). Full-precision KV is offloaded
        # GPU->CPU and re-fetched on prefix reuse. Incompatible with the custom
        # turboquant*/rotorquant* backends (both hook KV cache I/O) -> hard fail.
        if [ -n "$${VLLM_LMCACHE_ENABLED:-}" ]; then
          case "$${VLLM_KV_CACHE_DTYPE:-}" in
            turboquant*|rotorquant*)
              echo "ERROR: VLLM_LMCACHE_ENABLED + VLLM_KV_CACHE_DTYPE=$${VLLM_KV_CACHE_DTYPE} is unsupported (LMCache and the custom quant backends both hook KV cache I/O)." >&2
              exit 1 ;;
          esac
          if [ -n "$${VLLM_ENTRYPOINT_DRY_RUN:-}" ]; then
            echo "DRY_RUN: would pip install lmcache$${VLLM_LMCACHE_VERSION:+==$${VLLM_LMCACHE_VERSION}}"
          else
            pip install --no-cache-dir "lmcache$${VLLM_LMCACHE_VERSION:+==$${VLLM_LMCACHE_VERSION}}"
          fi
          export LMCACHE_LOCAL_CPU=true
          export LMCACHE_MAX_LOCAL_CPU_SIZE="$${VLLM_LMCACHE_CPU_SIZE:-20}"
          export LMCACHE_CHUNK_SIZE="$${VLLM_LMCACHE_CHUNK_SIZE:-256}"
          kv_xfer="$${VLLM_LMCACHE_KV_TRANSFER_CONFIG:-}"
          if [ -z "$${kv_xfer}" ]; then
            kv_xfer='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
          fi
          args+=(--kv-transfer-config "$${kv_xfer}")
        fi
        echo "Launching: python3 -m vllm.entrypoints.openai.api_server $${args[*]}"
        if [ -n "$${VLLM_ENTRYPOINT_DRY_RUN:-}" ]; then
          echo "LMCACHE_LOCAL_CPU=$${LMCACHE_LOCAL_CPU:-}"
          echo "LMCACHE_MAX_LOCAL_CPU_SIZE=$${LMCACHE_MAX_LOCAL_CPU_SIZE:-}"
          echo "LMCACHE_CHUNK_SIZE=$${LMCACHE_CHUNK_SIZE:-}"
          exit 0
        fi
        exec python3 -m vllm.entrypoints.openai.api_server "$${args[@]}"
```

- [ ] **Step 5: Run the H100 cases to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_lmcache_entrypoint.py -v -k docker-compose.yml`
Expected: PASS for all `docker-compose.yml` parametrizations (disabled, enabled defaults, custom size/chunk, custom transfer config, version pin, all three quant-conflict dtypes).

Note: `docker-compose.l4.yml` cases still FAIL here — that's expected; Task 2 fixes them.

- [ ] **Step 6: Commit**

```bash
git add tests/test_lmcache_entrypoint.py deployments/models/docker-compose.yml
git commit -m "feat(serving): LMCache CPU offload knob in H100 compose entrypoint

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: L4 compose — mirror the LMCache changes

**Files:**
- Modify: `deployments/models/docker-compose.l4.yml` (environment block + entrypoint script)

**Interfaces:**
- Consumes: the entrypoint shape and env vars defined in Task 1; the L4 entrypoint must end up byte-identical to the H100 one (it is identical today).
- Produces: L4 compose passing the same parametrized tests from Task 1.

- [ ] **Step 1: Confirm the L4 test cases currently fail**

Run: `source .venv/bin/activate && pytest tests/test_lmcache_entrypoint.py -v -k docker-compose.l4.yml`
Expected: FAIL — the L4 entrypoint has no LMCache handling yet.

- [ ] **Step 2: Add the LMCache env passthroughs to `docker-compose.l4.yml`**

In `deployments/models/docker-compose.l4.yml`, find:

```yaml
      VLLM_USE_V2_MODEL_RUNNER: ${VLLM_USE_V2_MODEL_RUNNER:-}
    volumes:
```

Replace with:

```yaml
      VLLM_USE_V2_MODEL_RUNNER: ${VLLM_USE_V2_MODEL_RUNNER:-}
      # --- LMCache CPU KV-cache offload (opt-in; see .env.example) ----------
      VLLM_LMCACHE_ENABLED: ${VLLM_LMCACHE_ENABLED:-}
      VLLM_LMCACHE_CPU_SIZE: ${VLLM_LMCACHE_CPU_SIZE:-20}
      VLLM_LMCACHE_CHUNK_SIZE: ${VLLM_LMCACHE_CHUNK_SIZE:-256}
      VLLM_LMCACHE_VERSION: ${VLLM_LMCACHE_VERSION:-}
      VLLM_LMCACHE_KV_TRANSFER_CONFIG: ${VLLM_LMCACHE_KV_TRANSFER_CONFIG:-}
      # Testing affordance: when non-empty, the entrypoint prints resolved args +
      # LMCache env and exits 0 without pip-installing or launching vLLM.
      VLLM_ENTRYPOINT_DRY_RUN: ${VLLM_ENTRYPOINT_DRY_RUN:-}
    volumes:
```

- [ ] **Step 3: Add the LMCache block + dry-run guard to the `docker-compose.l4.yml` entrypoint**

In the same file, find:

```yaml
        if [ -n "$${VLLM_EXTRA_ARGS}" ];     then args+=($${VLLM_EXTRA_ARGS}); fi
        echo "Launching: python3 -m vllm.entrypoints.openai.api_server $${args[*]}"
        exec python3 -m vllm.entrypoints.openai.api_server "$${args[@]}"
```

Replace with (identical to the Task 1 entrypoint replacement):

```yaml
        if [ -n "$${VLLM_EXTRA_ARGS}" ];     then args+=($${VLLM_EXTRA_ARGS}); fi
        # LMCache CPU KV-cache offload (opt-in). Full-precision KV is offloaded
        # GPU->CPU and re-fetched on prefix reuse. Incompatible with the custom
        # turboquant*/rotorquant* backends (both hook KV cache I/O) -> hard fail.
        if [ -n "$${VLLM_LMCACHE_ENABLED:-}" ]; then
          case "$${VLLM_KV_CACHE_DTYPE:-}" in
            turboquant*|rotorquant*)
              echo "ERROR: VLLM_LMCACHE_ENABLED + VLLM_KV_CACHE_DTYPE=$${VLLM_KV_CACHE_DTYPE} is unsupported (LMCache and the custom quant backends both hook KV cache I/O)." >&2
              exit 1 ;;
          esac
          if [ -n "$${VLLM_ENTRYPOINT_DRY_RUN:-}" ]; then
            echo "DRY_RUN: would pip install lmcache$${VLLM_LMCACHE_VERSION:+==$${VLLM_LMCACHE_VERSION}}"
          else
            pip install --no-cache-dir "lmcache$${VLLM_LMCACHE_VERSION:+==$${VLLM_LMCACHE_VERSION}}"
          fi
          export LMCACHE_LOCAL_CPU=true
          export LMCACHE_MAX_LOCAL_CPU_SIZE="$${VLLM_LMCACHE_CPU_SIZE:-20}"
          export LMCACHE_CHUNK_SIZE="$${VLLM_LMCACHE_CHUNK_SIZE:-256}"
          kv_xfer="$${VLLM_LMCACHE_KV_TRANSFER_CONFIG:-}"
          if [ -z "$${kv_xfer}" ]; then
            kv_xfer='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
          fi
          args+=(--kv-transfer-config "$${kv_xfer}")
        fi
        echo "Launching: python3 -m vllm.entrypoints.openai.api_server $${args[*]}"
        if [ -n "$${VLLM_ENTRYPOINT_DRY_RUN:-}" ]; then
          echo "LMCACHE_LOCAL_CPU=$${LMCACHE_LOCAL_CPU:-}"
          echo "LMCACHE_MAX_LOCAL_CPU_SIZE=$${LMCACHE_MAX_LOCAL_CPU_SIZE:-}"
          echo "LMCACHE_CHUNK_SIZE=$${LMCACHE_CHUNK_SIZE:-}"
          exit 0
        fi
        exec python3 -m vllm.entrypoints.openai.api_server "$${args[@]}"
```

- [ ] **Step 4: Verify the two entrypoints stayed identical**

Run:
```bash
python3 -c "
import yaml
a=yaml.safe_load(open('deployments/models/docker-compose.yml'))['services']['vllm']['entrypoint']
b=yaml.safe_load(open('deployments/models/docker-compose.l4.yml'))['services']['vllm']['entrypoint']
assert a==b, 'entrypoints diverged'
print('entrypoints identical OK')
"
```
Expected: `entrypoints identical OK`

- [ ] **Step 5: Run the full test file**

Run: `source .venv/bin/activate && pytest tests/test_lmcache_entrypoint.py -v`
Expected: PASS — all parametrizations over both compose files.

- [ ] **Step 6: Commit**

```bash
git add deployments/models/docker-compose.l4.yml
git commit -m "feat(serving): mirror LMCache CPU offload knob into L4 compose

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Operator docs — `.env.example` knobs + preset, CLAUDE.md note, parity test

**Files:**
- Modify: `deployments/models/.env.example`
- Modify: `CLAUDE.md`
- Modify: `tests/test_lmcache_entrypoint.py` (add a docs-parity test)

**Interfaces:**
- Consumes: the `VLLM_LMCACHE_*` var names finalized in Task 1.
- Produces: a parity test asserting every `VLLM_LMCACHE_*` var is documented in `.env.example`.

- [ ] **Step 1: Write the failing docs-parity test**

Append to `tests/test_lmcache_entrypoint.py`:

```python
LMCACHE_VARS = [
    "VLLM_LMCACHE_ENABLED",
    "VLLM_LMCACHE_CPU_SIZE",
    "VLLM_LMCACHE_CHUNK_SIZE",
    "VLLM_LMCACHE_VERSION",
    "VLLM_LMCACHE_KV_TRANSFER_CONFIG",
]


@pytest.mark.parametrize("var", LMCACHE_VARS)
def test_env_example_documents_lmcache_var(var):
    text = (MODELS_DIR / ".env.example").read_text()
    assert var in text, f"{var} not documented in .env.example"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_lmcache_entrypoint.py -v -k env_example`
Expected: FAIL — `.env.example` does not yet mention the `VLLM_LMCACHE_*` vars.

- [ ] **Step 3: Add the knobs to the optional-knob list in `.env.example`**

In `deployments/models/.env.example`, find:

```bash
# Extra raw vLLM flags appended verbatim, e.g. --tensor-parallel-size 2. Leave blank for none.
VLLM_EXTRA_ARGS=
```

Insert immediately after it:

```bash

# --- LMCache CPU KV-cache offload (opt-in) -----------------------------------
# Offloads full-precision KV cache from GPU HBM to CPU RAM and re-fetches it on
# prefix reuse (e.g. shared system prompts + tool schemas across Task A turns).
# Set VLLM_LMCACHE_ENABLED to any non-empty value to turn it on; leave blank for
# the default behavior (no offload). When enabled, the entrypoint pip-installs
# lmcache at container start (adds ~10-30s; absorbed by the 600s healthcheck
# start_period) and passes --kv-transfer-config to vLLM.
# NOTE: mutually exclusive with VLLM_KV_CACHE_DTYPE=turboquant*/rotorquant* (the
# entrypoint hard-fails). fp8 KV is allowed but untested with LMCache.
VLLM_LMCACHE_ENABLED=
# GB of CPU RAM for the offloaded KV cache (-> LMCACHE_MAX_LOCAL_CPU_SIZE).
VLLM_LMCACHE_CPU_SIZE=20
# Tokens per cache chunk (-> LMCACHE_CHUNK_SIZE). 256 is the production default.
VLLM_LMCACHE_CHUNK_SIZE=256
# Optional pip version pin for lmcache (e.g. 0.3.1). Leave blank for unpinned.
# Confirm the version matching the image's vLLM (0.20.0) before pinning.
VLLM_LMCACHE_VERSION=
# Override the --kv-transfer-config JSON. Leave blank for the in-process default
# {"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}. Set this if the
# lmcache<->vLLM pairing needs a kv_connector_module_path or a different connector.
VLLM_LMCACHE_KV_TRANSFER_CONFIG=
```

- [ ] **Step 4: Run the parity test to verify it passes**

Run: `source .venv/bin/activate && pytest tests/test_lmcache_entrypoint.py -v -k env_example`
Expected: PASS for all five `VLLM_LMCACHE_*` vars.

- [ ] **Step 5: Add the CLAUDE.md note**

In `CLAUDE.md`, find the end of the "Pending Work: Custom KV Cache Backends" section — the line that begins `Phase 3 benchmark matrix:` and ends with `pending the microbenchmark result.` Immediately after that paragraph, add:

```markdown

### KV Cache CPU Offload (LMCache) — serving knob, not a benchmark target

The docker serving stack (`deployments/models/docker-compose.yml` + `.l4.yml`) exposes an opt-in LMCache CPU KV-cache offload knob via `VLLM_LMCACHE_ENABLED` (+ `VLLM_LMCACHE_CPU_SIZE`/`_CHUNK_SIZE`/`_VERSION`/`_KV_TRANSFER_CONFIG`; see `.env.example`). When enabled, the entrypoint pip-installs `lmcache` at container start and passes `--kv-transfer-config` to vLLM, offloading **full-precision** KV to CPU RAM and re-fetching on prefix reuse. This is **orthogonal to and not stackable with** the on-GPU KV-quant backends: combining `VLLM_LMCACHE_ENABLED` with `VLLM_KV_CACHE_DTYPE=turboquant*`/`rotorquant*` hard-fails in the entrypoint (both hook KV cache I/O). It is a serving/deployment knob only — not a Phase 3 quantization benchmark column. Entrypoint logic is unit-tested via `tests/test_lmcache_entrypoint.py` using the `VLLM_ENTRYPOINT_DRY_RUN` affordance (no GPU/docker required).
```

- [ ] **Step 6: Run the full test file**

Run: `source .venv/bin/activate && pytest tests/test_lmcache_entrypoint.py -v`
Expected: PASS — all entrypoint and docs-parity tests.

- [ ] **Step 7: Commit**

```bash
git add deployments/models/.env.example CLAUDE.md tests/test_lmcache_entrypoint.py
git commit -m "docs(serving): document LMCache CPU offload knob + parity test

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Final verification (manual, on a GPU host)

Not part of the automated suite — for whoever deploys it:

1. `cd deployments/models && cp .env.example .env`, set `HF_TOKEN`, `VLLM_MODEL`, `VLLM_TOOL_CALL_PARSER`, and `VLLM_LMCACHE_ENABLED=1`.
2. `docker compose up` → container logs show a successful `pip install lmcache`, the `Launching: …` line includes `--kv-transfer-config`, and LMCache prints its init / CPU-buffer log line.
3. `curl localhost:8000/health` returns 200. Send two completions sharing a long system prompt; the second shows a lower TTFT (CPU prefix hit).
4. Unset `VLLM_LMCACHE_ENABLED`, `docker compose up` again → the `Launching: …` line is identical to before this change (regression check).
5. Confirm/adjust `VLLM_LMCACHE_VERSION` and `VLLM_LMCACHE_KV_TRANSFER_CONFIG` against the exact `lmcache`↔vLLM 0.20.0 pairing if LMCache fails to initialize.

---

## Self-Review

- **Spec coverage:** Env knobs (Component 1) → Task 1/2 env blocks + Task 3 `.env.example`. Entrypoint logic (Component 2) → Task 1/2. Error handling/constraints (Component 3): install failure via `set -eu` (existing), quant hard-fail (Task 1/2 + tests), cold-start doc (Task 3 `.env.example`). Docs (Component 4) → Task 3. Verification → manual section + automated tests. Open item (version/connector) → documented in `.env.example` (Task 3) + manual step 5. All covered.
- **Placeholder scan:** none — all steps carry concrete code/commands.
- **Type/name consistency:** the five `VLLM_LMCACHE_*` names, `VLLM_ENTRYPOINT_DRY_RUN`, the `LMCACHE_*` runtime exports, and the dry-run output lines (`LMCACHE_MAX_LOCAL_CPU_SIZE=…`, `LMCACHE_CHUNK_SIZE=…`) are identical across compose edits, the entrypoint, and the test assertions. Test helpers (`_extract_entrypoint_script`, `_base_env`, `_run`, `COMPOSE_FILES`, `MODELS_DIR`) are defined once in Task 1 and reused in Task 3.
- **Refinement note vs spec:** the spec's illustrative `${VAR:-{...}}` default is implemented as a safer `if [ -z ]` assignment to avoid a literal-JSON-brace clash inside bash parameter expansion; behavior matches the spec's stated default.
