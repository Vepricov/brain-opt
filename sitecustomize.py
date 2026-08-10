"""Narrow compatibility fixes for VERL 7aed6b23 with vLLM 0.8.5."""

import os
import sys


for option in ("--logprobs_mode", "--logprobs-mode"):
    while option in sys.argv:
        index = sys.argv.index(option)
        del sys.argv[index : index + 2]

_CONTROL_PLANE_MARKERS = (
    "/ray/dashboard/",
    "/ray/_private/runtime_env/agent/",
)
_is_ray_control_plane = any(
    marker in argument.replace("\\", "/")
    for argument in sys.argv
    for marker in _CONTROL_PLANE_MARKERS
)

if not _is_ray_control_plane:
    try:
        import torch.utils._pytree as _pytree

        if not hasattr(_pytree, "register_pytree_node") and hasattr(_pytree, "_register_pytree_node"):
            _pytree.register_pytree_node = _pytree._register_pytree_node
    except Exception:
        pass

    try:
        import vllm.entrypoints.cli.serve as _serve

        if not hasattr(_serve, "run_headless"):
            import uvloop as _uvloop
            from vllm.entrypoints.openai.api_server import run_server as _run_server

            def run_headless(args):
                _uvloop.run(_run_server(args))

            _serve.run_headless = run_headless
    except Exception:
        pass

    try:
        import ray as _ray

        _ray_init = _ray.init

        def _rl_muon_ray_init(*args, **kwargs):
            runtime_env = kwargs.setdefault("runtime_env", {})
            env_vars = runtime_env.setdefault("env_vars", {})
            env_vars.setdefault("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
            env_vars.setdefault("VLLM_USE_V1", "1")
            env_vars.setdefault("TRITON_LIBCUDA_PATH", "/lib/x86_64-linux-gnu")
            return _ray_init(*args, **kwargs)

        _ray.init = _rl_muon_ray_init
    except Exception:
        pass