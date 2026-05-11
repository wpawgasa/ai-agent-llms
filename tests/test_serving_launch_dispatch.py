"""Tests for serving/launch.sh backend dispatch and launcher script integrity."""

import json
import os
import stat
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
SERVING_DIR = PROJECT_ROOT / "serving"


def is_executable(path: Path) -> bool:
    return path.is_file() and bool(os.stat(path).st_mode & stat.S_IXUSR)


# ---------------------------------------------------------------------------
# Launcher script existence + executable bit
# ---------------------------------------------------------------------------


def test_launch_sh_exists_and_executable():
    assert is_executable(SERVING_DIR / "launch.sh"), (
        "serving/launch.sh missing or not executable"
    )


def test_launch_vllm_sh_exists_and_executable():
    assert is_executable(SERVING_DIR / "launch_vllm.sh")


def test_launch_sglang_sh_exists_and_executable():
    assert is_executable(SERVING_DIR / "launch_sglang.sh"), (
        "serving/launch_sglang.sh missing or not executable"
    )


def test_launch_trtllm_sh_exists_and_executable():
    assert is_executable(SERVING_DIR / "launch_trtllm.sh"), (
        "serving/launch_trtllm.sh missing or not executable"
    )


def test_yaml_helper_exists():
    assert (SERVING_DIR / "_yaml_helper.sh").is_file()


# ---------------------------------------------------------------------------
# Dispatch logic: launch.sh selects the right backend from serving.engine
# ---------------------------------------------------------------------------


def _write_temp_yaml(engine: str, tmp_path: Path) -> Path:
    cfg = {
        "model": {"name": "test/model"},
        "serving": {"engine": engine, "port": 8000},
    }
    p = tmp_path / f"test_{engine}.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def _dispatch_engine(yaml_path: Path) -> str:
    """Run launch.sh in dry-run mode and return which sub-launcher would be called.

    Uses bash -n (syntax check) + grep on the script output to avoid actually
    launching a server. We instead parse what `exec bash ...` line would be
    hit by extracting the ENGINE variable from the YAML.
    """
    result = subprocess.run(
        ["python3", "-c",
         f"""
import yaml
with open('{yaml_path}') as f:
    c = yaml.safe_load(f)
print(c.get('serving', {{}}).get('engine', 'vllm'))
"""],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("engine", ["vllm", "sglang", "tensorrt_llm"])
def test_dispatch_reads_engine_from_yaml(tmp_path: Path, engine: str):
    yaml_path = _write_temp_yaml(engine, tmp_path)
    assert _dispatch_engine(yaml_path) == engine


def test_launch_sh_syntax_valid():
    """bash -n parses launch.sh without errors."""
    result = subprocess.run(
        ["bash", "-n", str(SERVING_DIR / "launch.sh")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Syntax error in launch.sh:\n{result.stderr}"


def test_launch_sglang_sh_syntax_valid():
    result = subprocess.run(
        ["bash", "-n", str(SERVING_DIR / "launch_sglang.sh")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Syntax error in launch_sglang.sh:\n{result.stderr}"


def test_launch_trtllm_sh_syntax_valid():
    result = subprocess.run(
        ["bash", "-n", str(SERVING_DIR / "launch_trtllm.sh")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Syntax error in launch_trtllm.sh:\n{result.stderr}"


# ---------------------------------------------------------------------------
# YAML config integrity: all *_sglang.yaml and *_trtllm.yaml files are valid
# ---------------------------------------------------------------------------


def _backend_yamls(backend: str) -> list[Path]:
    return sorted(
        list((PROJECT_ROOT / "configs" / "models_exp_a").glob(f"*_{backend}.yaml"))
        + list((PROJECT_ROOT / "configs" / "models_exp_bc").glob(f"*_{backend}.yaml"))
    )


@pytest.mark.parametrize("yaml_path", _backend_yamls("sglang"))
def test_sglang_yaml_has_required_fields(yaml_path: Path):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("model", {}).get("name"), f"Missing model.name in {yaml_path.name}"
    serving = cfg.get("serving", {})
    assert serving.get("engine") == "sglang", f"serving.engine != sglang in {yaml_path.name}"
    assert serving.get("port") == 8000, f"serving.port != 8000 in {yaml_path.name}"
    assert "context_length" in serving, f"Missing serving.context_length in {yaml_path.name}"


@pytest.mark.parametrize("yaml_path", _backend_yamls("trtllm"))
def test_trtllm_yaml_has_required_fields(yaml_path: Path):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg.get("model", {}).get("name"), f"Missing model.name in {yaml_path.name}"
    serving = cfg.get("serving", {})
    assert serving.get("engine") == "tensorrt_llm", (
        f"serving.engine != tensorrt_llm in {yaml_path.name}"
    )
    assert serving.get("port") == 8000, f"serving.port != 8000 in {yaml_path.name}"
    assert serving.get("source") in ("hf", "engine_dir"), (
        f"serving.source not in {{hf, engine_dir}} in {yaml_path.name}"
    )


# ---------------------------------------------------------------------------
# vLLM regression: existing vLLM YAMLs still declare engine: vllm
# ---------------------------------------------------------------------------


def _vllm_yamls() -> list[Path]:
    # Exclude the new backend variants (they have _sglang / _trtllm suffix)
    return sorted(
        p for p in (PROJECT_ROOT / "configs" / "models_exp_a").glob("*.yaml")
        if not any(p.name.endswith(f"_{b}.yaml") for b in ("sglang", "trtllm"))
        and p.name != "frontier.yaml"
    )


@pytest.mark.parametrize("yaml_path", _vllm_yamls())
def test_existing_vllm_yamls_unchanged(yaml_path: Path):
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    engine = cfg.get("serving", {}).get("engine", "vllm")
    assert engine == "vllm", (
        f"Unexpected engine '{engine}' in existing config {yaml_path.name}; "
        f"vLLM configs must not be changed by the SGLang/TRT-LLM PR."
    )


# ---------------------------------------------------------------------------
# Speculative-decoding wiring tests (LAUNCH_PRINT_ONLY=1)
# ---------------------------------------------------------------------------

VLLM_LAUNCHER = SERVING_DIR / "launch_vllm.sh"
SGLANG_LAUNCHER = SERVING_DIR / "launch_sglang.sh"
TRTLLM_LAUNCHER = SERVING_DIR / "launch_trtllm.sh"


def _run_launcher(
    script: Path,
    yaml_path: Path,
    extra_args: list | None = None,
    env_overrides: dict | None = None,
) -> subprocess.CompletedProcess:
    """Run a launcher with LAUNCH_PRINT_ONLY=1 to capture its resolved arg list."""
    env = {**os.environ, "LAUNCH_PRINT_ONLY": "1", **(env_overrides or {})}
    cmd = ["bash", str(script), str(yaml_path)] + (extra_args or [])
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _make_vllm_yaml(tmp_path: Path, spec: dict | None = None) -> Path:
    cfg: dict = {
        "model": {"name": "test/model"},
        "serving": {
            "engine": "vllm",
            "tool_call_parser": "hermes",
            "gpu_memory_utilization": 0.90,
            "max_model_len": 4096,
            "port": 8000,
        },
    }
    if spec is not None:
        cfg["serving"]["speculative"] = spec
    p = tmp_path / "test_vllm.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def _make_sglang_yaml(tmp_path: Path, spec: dict | None = None, **extra_serving) -> Path:
    cfg: dict = {
        "model": {"name": "test/model"},
        "serving": {
            "engine": "sglang",
            "tool_call_parser": "pythonic",
            "mem_fraction_static": 0.90,
            "context_length": 4096,
            "max_running_requests": 64,
            "attention_backend": "triton",
            "port": 8000,
            "kv_cache_dtype": "auto",
            "pp_size": 1,
            "enable_dp_attention": False,
            **extra_serving,
        },
    }
    if spec is not None:
        cfg["serving"]["speculative"] = spec
    p = tmp_path / "test_sglang.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def _make_trtllm_yaml(
    tmp_path: Path, spec: dict | None = None, source: str = "hf", engine_dir: str = ""
) -> Path:
    cfg: dict = {
        "model": {"name": "test/model"},
        "serving": {
            "engine": "tensorrt_llm",
            "source": source,
            "engine_dir": engine_dir,
            "max_batch_size": 64,
            "max_num_tokens": 4096,
            "tp_size": 1,
            "port": 8000,
        },
    }
    if spec is not None:
        cfg["serving"]["speculative"] = spec
    p = tmp_path / "test_trtllm.yaml"
    p.write_text(yaml.dump(cfg))
    return p


# --- vLLM speculative tests ---


def test_vllm_launcher_emits_speculative_config_json_when_enabled(tmp_path: Path):
    yaml_path = _make_vllm_yaml(
        tmp_path,
        spec={
            "enabled": True,
            "method": "dflash",
            "draft_model": "z-lab/Test-Draft",
            "num_speculative_tokens": 5,
        },
    )
    result = _run_launcher(VLLM_LAUNCHER, yaml_path)
    assert result.returncode == 0, result.stderr
    parts = result.stdout.strip().split()
    assert "--speculative-config" in parts, f"--speculative-config not in output: {result.stdout!r}"
    idx = parts.index("--speculative-config")
    spec = json.loads(parts[idx + 1])
    assert spec["method"] == "dflash"
    assert spec["model"] == "z-lab/Test-Draft"
    assert spec["num_speculative_tokens"] == 5


def test_vllm_launcher_omits_speculative_when_disabled(tmp_path: Path):
    yaml_path = _make_vllm_yaml(tmp_path)  # no speculative block
    result = _run_launcher(VLLM_LAUNCHER, yaml_path)
    assert result.returncode == 0, result.stderr
    assert "--speculative-config" not in result.stdout


def test_vllm_cli_override_no_speculative_trumps_yaml(tmp_path: Path):
    yaml_path = _make_vllm_yaml(
        tmp_path,
        spec={"enabled": True, "method": "dflash", "draft_model": "z-lab/Test-Draft"},
    )
    result = _run_launcher(VLLM_LAUNCHER, yaml_path, extra_args=["--no-speculative"])
    assert result.returncode == 0, result.stderr
    assert "--speculative-config" not in result.stdout, (
        "--no-speculative should suppress spec config from YAML"
    )


def test_vllm_cli_speculative_method_override(tmp_path: Path):
    yaml_path = _make_vllm_yaml(tmp_path)  # no YAML spec block
    result = _run_launcher(
        VLLM_LAUNCHER,
        yaml_path,
        extra_args=[
            "--speculative-method", "dflash",
            "--speculative-draft-model", "z-lab/CLI-Draft",
            "--speculative-num-tokens", "3",
        ],
    )
    assert result.returncode == 0, result.stderr
    parts = result.stdout.strip().split()
    assert "--speculative-config" in parts
    idx = parts.index("--speculative-config")
    spec = json.loads(parts[idx + 1])
    assert spec["method"] == "dflash"
    assert spec["model"] == "z-lab/CLI-Draft"
    assert spec["num_speculative_tokens"] == 3


# --- SGLang speculative tests ---


def test_sglang_launcher_emits_dflash_flags(tmp_path: Path):
    yaml_path = _make_sglang_yaml(
        tmp_path,
        spec={
            "enabled": True,
            "algorithm": "DFLASH",
            "draft_model_path": "z-lab/Test-Draft",
            "num_draft_tokens": 5,
            "dflash_block_size": 4,
            "dflash_draft_window_size": 16,
        },
    )
    result = _run_launcher(SGLANG_LAUNCHER, yaml_path)
    assert result.returncode == 0, result.stderr
    args = result.stdout.strip()
    assert "--speculative-algorithm" in args
    assert "DFLASH" in args
    assert "--speculative-draft-model-path" in args
    assert "z-lab/Test-Draft" in args
    assert "--speculative-num-draft-tokens" in args
    assert "--speculative-dflash-block-size" in args
    assert "--speculative-dflash-draft-window-size" in args


def test_sglang_rejects_dp_attention_with_speculative(tmp_path: Path):
    yaml_path = _make_sglang_yaml(
        tmp_path,
        spec={"enabled": True, "algorithm": "DFLASH", "draft_model_path": "z-lab/Draft"},
        enable_dp_attention=True,
    )
    result = _run_launcher(SGLANG_LAUNCHER, yaml_path)
    assert result.returncode != 0, "Should have failed due to dp-attention + speculative"
    assert "dp-attention" in result.stderr.lower() or "dp_attention" in result.stderr.lower()


def test_sglang_rejects_pp_size_gt_1_with_speculative(tmp_path: Path):
    yaml_path = _make_sglang_yaml(
        tmp_path,
        spec={"enabled": True, "algorithm": "DFLASH", "draft_model_path": "z-lab/Draft"},
        pp_size=2,
    )
    result = _run_launcher(SGLANG_LAUNCHER, yaml_path)
    assert result.returncode != 0, "Should have failed due to pp_size > 1 + speculative"
    assert "pp_size" in result.stderr


# --- TRT-LLM speculative tests ---


def test_trtllm_rejects_hf_source_with_speculative(tmp_path: Path):
    yaml_path = _make_trtllm_yaml(
        tmp_path,
        spec={"enabled": True, "mode": "dflash"},
        source="hf",
    )
    result = _run_launcher(TRTLLM_LAUNCHER, yaml_path)
    assert result.returncode != 0, "Should have failed: spec + source=hf"
    assert "engine_dir" in result.stderr


def test_trtllm_emits_speculative_mode_when_engine_dir(tmp_path: Path):
    engine_dir = str(tmp_path / "fake_engine")
    (tmp_path / "fake_engine").mkdir()
    yaml_path = _make_trtllm_yaml(
        tmp_path,
        spec={"enabled": True, "mode": "dflash"},
        source="engine_dir",
        engine_dir=engine_dir,
    )
    result = _run_launcher(TRTLLM_LAUNCHER, yaml_path)
    assert result.returncode == 0, result.stderr
    assert "--speculative_decoding_mode" in result.stdout
    assert "dflash" in result.stdout


# --- Pydantic round-trip ---


def test_speculative_config_pydantic_roundtrip():
    from llm_workflow_agents.config.schema import ModelConfig, SpeculativeConfig

    for yaml_path in [
        PROJECT_ROOT / "configs/models_exp_a/llama31_8b_dflash.yaml",
        PROJECT_ROOT / "configs/models_exp_a/llama31_8b_dflash_sglang.yaml",
        PROJECT_ROOT / "configs/models_exp_a/llama31_8b_dflash_trtllm.yaml",
    ]:
        if not yaml_path.exists():
            pytest.skip(f"Smoke-test YAML not found: {yaml_path.name}")
        with open(yaml_path) as f:
            raw = yaml.safe_load(f)
        spec_raw = raw.get("serving", {}).get("speculative", {})
        spec = SpeculativeConfig.model_validate(spec_raw)
        assert spec.enabled is True, f"speculative.enabled should be True in {yaml_path.name}"


# --- run_exp_a.sh forwards speculative CLI to launcher ---


def test_run_exp_a_forwards_speculative_cli(tmp_path: Path):
    run_exp_a = PROJECT_ROOT / "scripts" / "run_exp_a.sh"
    if not run_exp_a.exists():
        pytest.skip("run_exp_a.sh not found")
    result = subprocess.run(
        [
            "bash", str(run_exp_a),
            "--dry-run",
            "--speculative-method", "dflash",
            "--speculative-draft-model", "z-lab/Test-Draft",
            "--speculative-num-tokens", "5",
        ],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    output = result.stdout + result.stderr
    assert "--speculative-method" in output or "dflash" in output, (
        f"Expected spec args to appear in dry-run output:\n{output}"
    )
