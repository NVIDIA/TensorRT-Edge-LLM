# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Shared harness for Edge-LLM TensorRT plugin unit tests.

Builds a single-plugin TensorRT engine from a declarative IO spec and drives it
using torch CUDA tensors for device memory (``tensor.data_ptr()`` for
``set_tensor_address`` and the current torch CUDA stream for
``execute_async_v3``).

Usage:
    runner = PluginRunner()
    runner.build(
        input_specs=[("x", trt.float16, (-1, 16))],
        output_names=["y"],
        plugin_name="MyPlugin", plugin_version="1",
        plugin_fields=[trt.PluginField("k", np.int32([3]), trt.PluginFieldType.INT32)],
        profiles={"x": ((1, 16), (4, 16), (8, 16))},
    )
    runner.execute({"x": x_gpu, "y": y_gpu})   # input shapes inferred from tensors
"""

import ctypes
import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pytest

# Conditional imports for the GPU/TensorRT dependencies, so the module imports
# cleanly (and its tests skip) on a host where TensorRT or a CUDA-capable torch
# is unavailable, instead of failing collection.
try:
    import tensorrt as trt
    import torch

    DEPENDENCIES_AVAILABLE = torch.cuda.is_available()
    IMPORT_ERROR = None if DEPENDENCIES_AVAILABLE else "CUDA not available for torch"
except ImportError as e:  # pragma: no cover - exercised only without deps
    DEPENDENCIES_AVAILABLE = False
    IMPORT_ERROR = str(e)

    class _DummyModule:

        def __getattr__(self, name):
            return None

    trt = _DummyModule()
    torch = _DummyModule()


def _device_sm() -> int:
    """Compute capability as major*10+minor (0 without a CUDA device)."""
    if not (DEPENDENCIES_AVAILABLE and torch.cuda.is_available()):
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


class PluginUnsupportedError(RuntimeError):
    """The plugin/engine cannot serve this config on this device (build-time).

    The graceful-failure tests pass ``expect_unsupported`` to treat a clean
    build-time rejection as a valid outcome instead of a failure."""


def _fail_unsupported(msg: str):
    """A config the device cannot serve must be gated by an explicit SM/feature
    skipif (or an expect_unsupported graceful test). Reaching this fallback means
    a missing gate or a broken build/env -- fail loudly rather than skip green."""
    pytest.fail(msg + " (gate this config with an explicit SM/feature skipif, "
                "or it is a real build/environment failure)")


# --------------------------------------------------------------------------- #
# Plugin library discovery / registration
# --------------------------------------------------------------------------- #
_PLUGIN_LIB_NAME = "libNvInfer_edgellm_plugin.so"
# Default build directory. Set EDGELLM_PLUGIN_LIB to point at the library
# directly if it was built elsewhere.
_plugins_loaded = False


def find_plugin_library() -> Optional[str]:
    """Locate the Edge-LLM plugin shared library, or return None if missing.

    Honors EDGELLM_PLUGIN_LIB when set, then looks under build/ relative to the
    current working directory and to this file's repository root (the project
    is built into build/, see tests/README).
    """
    env_path = os.environ.get("EDGELLM_PLUGIN_LIB")
    if env_path:
        return os.path.abspath(env_path) if os.path.exists(env_path) else None
    candidates = [
        os.path.join("build", _PLUGIN_LIB_NAME),
        os.path.join(os.path.dirname(__file__), "..", "..", "build",
                     _PLUGIN_LIB_NAME),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def load_edgellm_plugins(logger) -> str:
    """Load the plugin library + initialize the TRT plugin registry (idempotent).

    Returns the resolved library path. Raises RuntimeError if not found.
    """
    global _plugins_loaded
    path = find_plugin_library()
    if path is None:
        raise RuntimeError(
            f"Could not find {_PLUGIN_LIB_NAME}. Build the project first, or "
            f"set EDGELLM_PLUGIN_LIB to the library path.")
    ctypes.CDLL(path)
    trt.init_libnvinfer_plugins(logger, "")
    _plugins_loaded = True
    return path


# --------------------------------------------------------------------------- #
# dtype helpers
# --------------------------------------------------------------------------- #
def _trt_to_torch_dtype_map():
    m = {
        trt.float16: torch.float16,
        trt.float32: torch.float32,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    # Optional dtypes depending on TRT/torch versions.
    if hasattr(trt, "bfloat16") and hasattr(torch, "bfloat16"):
        m[trt.bfloat16] = torch.bfloat16
    if hasattr(trt, "fp8") and hasattr(torch, "float8_e4m3fn"):
        m[trt.fp8] = torch.float8_e4m3fn
    if hasattr(trt, "int64"):
        m[trt.int64] = torch.int64
    return m


def trt_dtype_to_torch(dtype):
    """Map a TensorRT DataType to the matching torch dtype."""
    return _trt_to_torch_dtype_map()[dtype]


def trt_dtype_to_numpy(dtype):
    """Map a TensorRT DataType to the matching numpy dtype (constants)."""
    m = {
        trt.float32: np.float32,
        trt.float16: np.float16,
        trt.int32: np.int32,
        trt.int8: np.int8,
    }
    return m[dtype]


def make_field(name: str, value, field_type) -> "trt.PluginField":
    """Build a trt.PluginField from a python scalar / sequence."""
    np_dtype = {
        trt.PluginFieldType.INT32: np.int32,
        trt.PluginFieldType.FLOAT32: np.float32,
    }[field_type]
    arr = np.asarray(value, dtype=np_dtype).reshape(-1)
    return trt.PluginField(name, arr, field_type)


def pf_int32(name: str, value) -> "trt.PluginField":
    return make_field(name, value, trt.PluginFieldType.INT32)


def pf_float32(name: str, value) -> "trt.PluginField":
    return make_field(name, value, trt.PluginFieldType.FLOAT32)


# --------------------------------------------------------------------------- #
# Engine runner
# --------------------------------------------------------------------------- #
@dataclass
class InputSpec:
    name: str
    dtype: object  # trt DataType
    shape: Tuple[int, ...]  # may contain -1 for dynamic dims


# TensorRT registers the logger of the FIRST builder/runtime globally and
# ignores the ones passed later. A per-runner logger is garbage-collected
# when its test ends while TensorRT still dereferences it, crashing a later
# engine build. Share one process-lifetime logger instead (first caller's
# severity wins).
_LOGGER = None


def _get_logger(verbose: bool):
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = trt.Logger(
            trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    return _LOGGER


class PluginRunner:
    """Builds and executes a single-plugin TensorRT engine with torch buffers."""

    def __init__(self, verbose: bool = False):
        self.logger = _get_logger(verbose)
        if not _plugins_loaded:
            load_edgellm_plugins(self.logger)
        self.engine = None
        self.context = None
        self.device = torch.device("cuda")

    def build(
        self,
        *,
        input_specs: Sequence[Tuple[str, object, Tuple[int, ...]]],
        output_names: Sequence[str],
        plugin_name: str,
        plugin_version: str,
        plugin_fields: Sequence["trt.PluginField"],
        profiles: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...],
                                  Tuple[int, ...]]],
        constant_specs: Optional[Sequence[Tuple[str, object, Tuple[int, ...],
                                                object]]] = None,
        plugin_input_order: Optional[Sequence[str]] = None,
        plugin_namespace: str = "",
        workspace_bytes: int = 1 << 30,
        expect_unsupported: bool = False,
    ):
        """Construct the engine.

        ``profiles`` maps each input name to (min, opt, max) shape tuples.
        ``output_names`` are assigned to plugin outputs 0..N in order.

        ``constant_specs`` optionally declares engine-weight constants wired
        as plugin inputs: (name, trt_dtype, shape, values). ``values`` may be
        a torch tensor, numpy array, or None; a None value or zero-volume
        shape produces a zero-length constant (type-only weights).

        ``plugin_input_order`` optionally lists names from ``input_specs`` and
        ``constant_specs`` defining the plugin input order (so constants can
        interleave with regular inputs). Defaults to all inputs in
        ``input_specs`` order followed by all constants in ``constant_specs``
        order.
        """
        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                     workspace_bytes)
        # Required for in-place / aliased plugin IO (e.g. KV cache update).
        if hasattr(trt.PreviewFeature, "ALIASED_PLUGIN_IO_10_03"):
            config.set_preview_feature(
                trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03, True)

        tensors_by_name = {}
        for name, dtype, shape in input_specs:
            tensors_by_name[name] = network.add_input(name, dtype, shape)

        # trt.Weights does not own memory — keep the numpy buffers alive until
        # the build completes.
        self._constant_arrays = []
        constant_names = []
        for name, dtype, shape, values in (constant_specs or []):
            if values is None or int(np.prod(shape)) == 0:
                # TRT requires values == nullptr when count == 0 — use the
                # type-only Weights constructor.
                weights = trt.Weights(dtype)
            else:
                arr = values
                if not isinstance(arr, np.ndarray):
                    arr = arr.detach().cpu().numpy() if hasattr(
                        arr, "detach") else np.asarray(arr)
                arr = np.ascontiguousarray(
                    arr.astype(trt_dtype_to_numpy(dtype)))
                self._constant_arrays.append(arr)
                weights = trt.Weights(arr)
            const = network.add_constant(shape, weights)
            const.get_output(0).name = name
            tensors_by_name[name] = const.get_output(0)
            constant_names.append(name)

        if plugin_input_order is None:
            plugin_input_order = [s[0] for s in input_specs] + constant_names
        inputs = [tensors_by_name[n] for n in plugin_input_order]

        registry = trt.get_plugin_registry()
        creator = registry.get_creator(plugin_name, plugin_version,
                                       plugin_namespace)
        if creator is None:
            raise RuntimeError(
                f"{plugin_name} v{plugin_version} not found in plugin registry"
            )
        fc = trt.PluginFieldCollection(list(plugin_fields))
        plugin = creator.create_plugin(plugin_name, fc,
                                       trt.TensorRTPhase.BUILD)
        if plugin is None:
            # Not built for this architecture (e.g. a CuTe DSL plugin
            # compiled only for newer SMs) -- expected only when the caller
            # opts in; otherwise a missing gate or a broken build.
            if expect_unsupported:
                raise PluginUnsupportedError(
                    f"{plugin_name} plugin not available in this build")
            _fail_unsupported(
                f"{plugin_name} plugin not available in this build")

        layer = network.add_plugin_v3(inputs, [], plugin)
        for i, oname in enumerate(output_names):
            layer.get_output(i).name = oname
            network.mark_output(layer.get_output(i))

        profile = builder.create_optimization_profile()
        for name, (lo, opt, hi) in profiles.items():
            profile.set_shape(name, lo, opt, hi)
        config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            # Unbuildable on this device (e.g. FP8 KV cache without FP8
            # tensor cores) -- expected only when the caller opts in;
            # otherwise a missing gate or a broken build.
            if expect_unsupported:
                raise PluginUnsupportedError(
                    f"engine build unsupported for {plugin_name} on this device"
                )
            _fail_unsupported(
                f"engine build unsupported for {plugin_name} on this device")
        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized)
        self.context = self.engine.create_execution_context()
        return self

    def execute(self,
                tensors: Dict[str, "torch.Tensor"],
                input_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
                synchronize: bool = True):
        """Bind torch tensors to every IO tensor and run.

        Input shapes default to each input tensor's own shape. Aliased outputs
        are handled by passing the same torch tensor under both binding names.
        All tensors must be CUDA tensors with matching dtype/layout.
        Set ``synchronize=False`` only when the caller supplies the required
        stream ordering, such as inside CUDA graph capture.
        """
        ctx = self.context
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if name not in tensors:
                raise KeyError(f"Missing device tensor for binding '{name}'")
            t = tensors[name]
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = (input_shapes or {}).get(name, tuple(t.shape))
                if not ctx.set_input_shape(name, shape):
                    raise RuntimeError(f"set_input_shape failed for {name}")
            if not ctx.set_tensor_address(name, t.data_ptr()):
                raise RuntimeError(f"set_tensor_address failed for {name}")
        # Execute on torch's current stream so the plugin is ordered after the
        # torch ops that produced the input tensors; a separate stream would not
        # be synchronized with them and could read partially-written inputs.
        stream = torch.cuda.current_stream()
        ok = ctx.execute_async_v3(stream.cuda_stream)
        if synchronize:
            stream.synchronize()
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #
COS_SIM_THRESHOLD = 0.99999
# Element-wise abs/rel tolerance (allclose semantics). Complements the cosine
# check, which is insensitive to a global scale factor.
DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2


def cosine_sim(expected: "torch.Tensor", actual: "torch.Tensor") -> float:
    """Cosine similarity between two tensors (flattened, fp64).

    fp64 is required, not a luxury: an fp32 dot over ~64k elements
    accumulates ~1e-5 of error -- larger than the margin the 0.99999
    threshold leaves -- and the error depends on the platform's reduction
    order (aarch64 torch builds crossed the bar while x86 stayed under)."""
    e = expected.double().flatten().cpu()
    a = actual.double().flatten().cpu()
    en, an = float(e.norm()), float(a.norm())
    if en == 0.0 and an == 0.0:
        return 1.0  # identical all-zero tensors
    if en == 0.0 or an == 0.0:
        return 0.0
    return float(torch.dot(e, a) / (en * an))


def assert_close(name: str,
                 expected: "torch.Tensor",
                 actual: "torch.Tensor",
                 atol: float = DEFAULT_ATOL,
                 rtol: float = DEFAULT_RTOL,
                 cos_threshold: float = COS_SIM_THRESHOLD):
    """Assert the plugin output matches the reference on two criteria:

    1. cosine similarity >= cos_threshold (default 0.99999) -- catches
       structural errors (wrong layout, dropped/extra terms, state errors);
    2. element-wise closeness ``|actual - expected| <= atol + rtol*|expected|``
       (allclose semantics) -- catches magnitude / scale errors that cosine,
       being scale-invariant, would miss.
    """
    e = expected.float()
    a = actual.float()
    for label, t in (("expected", e), ("actual", a)):
        if not torch.isfinite(t).all():
            n_bad = int((~torch.isfinite(t)).sum())
            raise AssertionError(
                f"{name}: {label} has {n_bad}/{t.numel()} non-finite values")
    cos = cosine_sim(e, a)
    diff = (e - a).abs()
    n_viol = int((diff > atol + rtol * e.abs()).sum())
    if cos < cos_threshold or n_viol:
        max_abs = float(diff.max()) if diff.numel() else 0.0
        raise AssertionError(
            f"{name}: cos_sim={cos:.6f} (need >= {cos_threshold}); "
            f"allclose violations={n_viol}/{e.numel()} "
            f"(atol={atol}, rtol={rtol}, max_abs={max_abs:.5f})")


# --------------------------------------------------------------------------- #
# Required batch / sequence-length cases
# --------------------------------------------------------------------------- #
# Each entry is (label, [per-row sequence lengths]), covering batch sizes
# 1/2/3/4/8 with both even and uneven length patterns.
RAGGED_CASES = [
    ("bs1", [1536]),
    ("bs2_even", [1024, 1024]),
    ("bs3_uneven", [10, 2048, 128]),
    ("bs4_even", [512, 512, 512, 512]),
    ("bs8_uneven", [10, 96, 240, 480, 800, 1200, 1664, 2048]),
]
MAX_BATCH = 8
MAX_SEQ = 2048


def poison_padding(tensors, context_lengths, value: float = 1e3):
    """Fill each row's padding region ``[context_lengths[b]:]`` with ``value``
    across all given ``[batch, seq, ...]`` tensors. Any kernel that reads past a
    row's valid length then corrupts the result, catching missing
    context-length masking. ``tensors`` may be a single tensor or an iterable.
    """
    if not isinstance(tensors, (list, tuple)):
        tensors = (tensors, )
    for t in tensors:
        seq = t.shape[1]
        for b in range(t.shape[0]):
            valid = int(context_lengths[b])
            if valid < seq:
                t[b, valid:] = value
