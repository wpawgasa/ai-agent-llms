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
