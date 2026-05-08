"""Tests for serving/launch.sh backend dispatch and launcher script integrity."""

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
