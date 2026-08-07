# Project EmbodiedGen
#
# Copyright (c) 2025 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.


import fileinput
import site

import gradio_client.utils as gradio_client_utils


def _patch_gradio_schema_bool_bug() -> None:
    """Patch schema parser for bool-style for gradio<5.33."""
    original_get_type = gradio_client_utils.get_type
    original_json_schema_to_python_type = (
        gradio_client_utils._json_schema_to_python_type
    )

    def _safe_get_type(schema):
        if isinstance(schema, bool):
            return {}
        return original_get_type(schema)

    def _safe_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        return original_json_schema_to_python_type(schema, defs)

    gradio_client_utils.get_type = _safe_get_type
    gradio_client_utils._json_schema_to_python_type = (
        _safe_json_schema_to_python_type
    )


def _patch_open3d_cuda_device_count_bug() -> None:
    """Patch open3d to avoid cuda device count bug."""
    with fileinput.FileInput(
        f'{site.getsitepackages()[0]}/open3d/__init__.py', inplace=True
    ) as file:
        for line in file:
            print(
                line.replace(
                    '_pybind_cuda.open3d_core_cuda_device_count()', '1'
                ),
                end='',
            )


def _neutralize_warp_in_parent() -> None:
    """Prevent NVIDIA Warp from calling cuInit() in the ZeroGPU parent.

    Root cause of @spaces.GPU silent hangs (spaces>=0.50): kaolin imports
    warp at module top-level. When any kaolin module triggers warp.init(),
    Warp's `init_cuda_driver` dlopens libcuda.so + calls cuInit() in the
    parent process. After spaces forks the worker, torch.init(nvidia_uuid)
    in the worker hangs forever because the inherited CUDA driver state is
    poisoned (parent never had a real GPU; ZeroGPU exposes one only post-fork).

    Fix: stub warp.init / warp.context.runtime_init with a pid-aware no-op.
    The parent-resident pid skips init; the forked worker (different pid)
    runs the real init so warp keeps working inside @spaces.GPU code paths.

    Must be called BEFORE any import that pulls kaolin (e.g. embodied_gen.data,
    thirdparty.TRELLIS).
    """
    import os
    import sys

    try:
        import warp  # noqa: F401  -- pure python import, no cuInit
    except ImportError:
        return

    parent_pid = os.getpid()

    def _make_pid_safe(orig):
        def _wrapped(*args, **kwargs):
            if os.getpid() == parent_pid:
                sys.stderr.write(
                    f"[warp-neutralize] skip {orig.__name__} in parent pid={parent_pid}\n"
                )
                sys.stderr.flush()
                return None
            return orig(*args, **kwargs)

        _wrapped.__wrapped__ = orig
        _wrapped.__name__ = getattr(orig, "__name__", "wrapped")
        return _wrapped

    if hasattr(warp, "init") and not hasattr(warp.init, "__wrapped__"):
        warp.init = _make_pid_safe(warp.init)

    try:
        from warp import context as _wctx

        if hasattr(_wctx, "runtime_init") and not hasattr(
            _wctx.runtime_init, "__wrapped__"
        ):
            _wctx.runtime_init = _make_pid_safe(_wctx.runtime_init)
    except Exception:
        pass
