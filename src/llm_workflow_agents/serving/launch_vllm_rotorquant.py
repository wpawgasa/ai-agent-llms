"""Launch vLLM's OpenAI API server with the project's custom RotorQuant KV cache.

vLLM's CLI parser builds ``--kv-cache-dtype`` choices from a ``Literal[...]``
at import time, so ``"rotorquant"`` is rejected before any project hook can
run. This launcher installs the RotorQuant runtime hooks, appends
``"rotorquant"`` to the argparse action's choices, then runs the server
in-process.

Usage mirrors ``python -m vllm.entrypoints.openai.api_server`` — argv is
forwarded unchanged, except ``--rotorquant-bit-width`` is stripped and
forwarded to ``RotorQuantConfig``.
"""

from __future__ import annotations

import asyncio
import sys

from llm_workflow_agents.quantization.rotorquant.vllm_integration import (
    RotorQuantConfig,
    register_rotorquant_backend,
)


def _extend_kv_cache_choices(parser) -> None:
    for action in parser._actions:
        if "--kv-cache-dtype" in action.option_strings:
            if action.choices and "rotorquant" not in action.choices:
                action.choices = list(action.choices) + ["rotorquant"]
            return
    raise RuntimeError("vLLM parser has no --kv-cache-dtype action")


def _rotorquant_bit_width_from_argv(argv: list[str]) -> int:
    for i, tok in enumerate(argv):
        if tok == "--rotorquant-bit-width" and i + 1 < len(argv):
            argv.pop(i)
            return int(argv.pop(i))
        if tok.startswith("--rotorquant-bit-width="):
            argv.pop(i)
            return int(tok.split("=", 1)[1])
    return 3


def main() -> None:
    argv = sys.argv[1:]
    bit_width = _rotorquant_bit_width_from_argv(argv)
    register_rotorquant_backend(RotorQuantConfig(bit_width=bit_width))

    import vllm.entrypoints.openai.api_server as api_server
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
    except ImportError:
        from vllm.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="vLLM + project RotorQuant KV cache")
    parser = api_server.make_arg_parser(parser)
    _extend_kv_cache_choices(parser)

    args = parser.parse_args(argv)
    api_server.validate_parsed_serve_args(args)

    asyncio.run(api_server.run_server(args))


if __name__ == "__main__":
    main()
