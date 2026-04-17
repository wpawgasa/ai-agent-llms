"""Launch vLLM's OpenAI API server with the project's custom TurboQuant KV cache.

vLLM's CLI parser builds ``--kv-cache-dtype`` choices from a ``Literal[...]``
at import time, so plain ``"turboquant"`` is rejected before any project hook
can run. This launcher installs the TurboQuant runtime hooks, appends
``"turboquant"`` to the argparse action's choices, then runs the server
in-process.

Usage mirrors ``python -m vllm.entrypoints.openai.api_server`` — argv is
forwarded unchanged.
"""

from __future__ import annotations

import asyncio
import sys

from llm_workflow_agents.quantization.turboquant.vllm_integration import (
    TurboQuantConfig,
    register_turboquant_backend,
)


def _extend_kv_cache_choices(parser) -> None:
    for action in parser._actions:
        if "--kv-cache-dtype" in action.option_strings:
            if action.choices and "turboquant" not in action.choices:
                action.choices = list(action.choices) + ["turboquant"]
            return
    raise RuntimeError("vLLM parser has no --kv-cache-dtype action")


def _turboquant_bit_width_from_argv(argv: list[str]) -> int:
    for i, tok in enumerate(argv):
        if tok == "--turboquant-bit-width" and i + 1 < len(argv):
            argv.pop(i)
            return int(argv.pop(i))
        if tok.startswith("--turboquant-bit-width="):
            argv.pop(i)
            return int(tok.split("=", 1)[1])
    return 3


def main() -> None:
    argv = sys.argv[1:]
    bit_width = _turboquant_bit_width_from_argv(argv)
    register_turboquant_backend(TurboQuantConfig(bit_width=bit_width))

    import vllm.entrypoints.openai.api_server as api_server
    from vllm.utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser(description="vLLM + project TurboQuant KV cache")
    parser = api_server.make_arg_parser(parser)
    _extend_kv_cache_choices(parser)

    args = parser.parse_args(argv)
    api_server.validate_parsed_serve_args(args)

    asyncio.run(api_server.run_server(args))


if __name__ == "__main__":
    main()
