#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Run a CuTe DSL kernel script while injecting common compile options.

This wrapper keeps cross-compile handling centralized instead of adding
--gpu-arch / --host-target plumbing to every kernel script under kernelSrcs/.
"""

from __future__ import annotations

import argparse
import atexit
import ctypes
import functools
import hashlib
import importlib.metadata
import importlib.util
import inspect
import os
from pathlib import Path
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import tempfile


def _install_static_runtime_shim():
    """Build the JIT runtime symbol provider from libcuda_dialect_runtime_static.a.

    AOT export resolves ``_cudaLaunchKernelEx``-family symbols from the shared
    libraries listed in ``CUTE_DSL_LIBS``. By default the package loader points
    that at the large ``libcute_dsl_runtime.so``, which adds a fragile
    dependency: when that library is present but fails to dlopen (wrong-arch
    file, truncated install), the DSL swallows the load error and every export
    later dies with an opaque ``JIT session error: Symbols not found`` +
    ``Failed to dump object file with PIC relocation``.

    Link the small static runtime shim archive (the same objects packed into
    libcutedsl_{arch}.a) into a throwaway shared object instead. Returns the
    shim path, or None when unavailable (multi-flavor install, missing
    binutils, opt-out). Must run before ``import cutlass``.
    """
    def skip(reason: str):
        print(f"[cutedsl_compile_wrapper] static runtime shim NOT installed: {reason}; "
              "falling back to libcute_dsl_runtime.so", file=sys.stderr)
        return None

    if os.environ.get("CUTE_DSL_STATIC_SHIM", "1").strip().lower() in ("0", "off", "false", "no"):
        return skip("disabled via CUTE_DSL_STATIC_SHIM")
    spec = importlib.util.find_spec("nvidia_cutlass_dsl")
    if spec is None or not spec.submodule_search_locations:
        return skip("nvidia_cutlass_dsl package not found")
    pkg_dir = Path(next(iter(spec.submodule_search_locations)))
    archives = sorted(pkg_dir.glob("cu*/lib/libcuda_dialect_runtime_static.a"))
    ld = shutil.which("ld")
    ar = shutil.which("ar")
    if len(archives) != 1:
        # Multi-flavor installs are ambiguous — the shim ABI is per-flavor.
        return skip(f"expected exactly one flavor archive, found {[str(a) for a in archives]}")
    if ld is None or ar is None:
        return skip("binutils (ar/ld) not found on PATH")
    tmp_dir = Path(tempfile.mkdtemp(prefix="cutedsl_static_shim_"))
    shim = tmp_dir / "libcutedsl_static_shim.so"
    try:
        subprocess.run([ar, "x", str(archives[0])], cwd=tmp_dir, check=True, capture_output=True)
        objs = sorted(str(o) for o in tmp_dir.glob("*.o"))
        subprocess.run([ld, "-shared", "-o", str(shim), *objs], check=True, capture_output=True)
        # Verify loadable before advertising it. Lazy binding: the shim's
        # cuda*/cu* references intentionally stay undefined until the JIT
        # resolves them from the process (eager RTLD_NOW would fail here).
        ctypes.CDLL(str(shim), mode=os.RTLD_LAZY)
    except (OSError, subprocess.SubprocessError) as exc:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        detail = getattr(exc, "stderr", b"") or b""
        return skip(f"shim build/load failed: {exc} {detail.decode(errors='replace').strip()}")
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    # The shim's wrappers call real CUDA runtime/driver entry points lazily
    # when helper compiles launch kernels on the build GPU (e.g.
    # cute.testing.convert on FP8 paths). Those callees must be in the
    # process's global symbol scope — cupy and friends load cudart
    # RTLD_LOCAL, so put cudart and the driver there explicitly. Runtime and
    # driver are backward compatible with the flavor's ABI (>= 12.8).
    for soname in ("libcuda.so.1", "libcudart.so.12", "libcudart.so.13", "libcudart.so"):
        try:
            ctypes.CDLL(soname, mode=os.RTLD_GLOBAL)
        except OSError:
            continue
    # Advertise the shim before `import cutlass`: the import-time machinery
    # consumes CUTE_DSL_LIBS and loads the listed libraries into the process,
    # which is where the JIT resolves the runtime symbols from.
    sep = ";" if sys.platform.startswith("win32") else ":"
    existing = os.environ.get("CUTE_DSL_LIBS", "")
    if str(shim) not in (existing.split(sep) if existing else []):
        os.environ["CUTE_DSL_LIBS"] = (existing + sep if existing else "") + str(shim)
    return shim


def _use_static_runtime_only(shim) -> None:
    """Make CUTE_DSL_LIBS reference the static shim only.

    The package loader prepends ``libcute_dsl_runtime.so`` to CUTE_DSL_LIBS
    during ``import cutlass``, so this strip must run after the import (the
    DSL reads the variable later, when the kernel script instantiates the
    DSL). Only applied when the shim was built — otherwise the dynamic
    runtime remains the sole symbol provider.
    """
    if shim is None:
        return
    sep = ";" if sys.platform.startswith("win32") else ":"
    existing = os.environ.get("CUTE_DSL_LIBS", "")
    entries = [e for e in existing.split(sep) if e] if existing else []
    entries = [e for e in entries if Path(e).name != "libcute_dsl_runtime.so"]
    if str(shim) not in entries:
        entries.append(str(shim))
    os.environ["CUTE_DSL_LIBS"] = sep.join(entries)
    # One diagnostic line per compile process: which libraries back the JIT
    # runtime symbols, and whether each actually loads.
    probes = []
    for entry in entries:
        try:
            ctypes.CDLL(entry, mode=os.RTLD_LAZY)
            probes.append(f"{Path(entry).name}:OK")
        except OSError as exc:
            probes.append(f"{Path(entry).name}:FAIL({exc})")
    print(f"[cutedsl_compile_wrapper] static runtime shim active; CUTE_DSL_LIBS = {probes}",
          file=sys.stderr)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", required=True, help="Kernel script to execute.")
    parser.add_argument("--gpu-arch", default="", help="Target GPU arch (e.g. sm_120a).")
    parser.add_argument("--host-target", default="", help="CuTe DSL --host-target option to inject.")
    parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Arguments forwarded to --script.")
    return parser.parse_args()


_args = _parse_args()

# Native builds (no --host-target) must select the GPU arch via the
# CUTE_DSL_ARCH environment variable, NOT via a cute.compile `--gpu-arch`
# option: passing --gpu-arch in options without --host-target routes the DSL
# down a dump path whose JIT session does not register the runtime libraries
# (every export fails with `JIT session error: Symbols not found:
# [_cudaLaunchKernelEx, ...]`), and the environment variable is only honored
# when set before `import cutlass` — later assignment silently compiles for
# the local GPU's arch instead. Cross builds keep per-kernel option
# injection: a process-wide CUTE_DSL_ARCH would leak the target SM into
# library helper compiles that must run on the local build GPU.
if not _args.host_target and _args.gpu_arch:
    os.environ["CUTE_DSL_ARCH"] = _args.gpu_arch

_static_runtime_shim = _install_static_runtime_shim()

import cutlass.cute as cute  # noqa: E402  (the import wires CUTE_DSL_LIBS...)

_use_static_runtime_only(_static_runtime_shim)  # ...and this rewires it to the shim


def _patch_hardware_info_occupancy_probe() -> None:
    """Replace HardwareInfo.get_max_active_clusters with a driver-attribute query.

    The stock implementation compiles a dummy kernel through cute.compile and
    loads it on the local GPU to run the occupancy APIs. With CUTE_DSL_ARCH
    pinned to the target SM (native builds where target != build GPU), that
    dummy cubin targets the wrong arch and every driver handle call fails with
    CUDA_ERROR_INVALID_HANDLE (seen via gemm_blackwell_geforce.py's probe).

    The probe requests the full dynamic-smem opt-in, so it effectively
    measures one resident block per SM: for a given cluster size the result
    is the multiprocessor count divided by the cluster size. Compute exactly
    that from pure driver attributes — no kernel compile, no arch coupling.
    """
    import cutlass.utils as _utils  # noqa: PLC0415

    def get_max_active_clusters(self, cluster_size, stream=None):
        if cluster_size <= 0 or cluster_size > 32:
            raise ValueError(
                f"Cluster size must be between 1 and 32, {cluster_size} is not supported"
            )
        return self.get_device_multiprocessor_count() // cluster_size

    _utils.hardware_info.HardwareInfo.get_max_active_clusters = get_max_active_clusters


if os.environ.get("CUTE_DSL_ARCH"):
    # Only needed when a process-wide target arch is pinned (native builds);
    # cross builds leave helper compiles device-native and the stock probe works.
    _patch_hardware_info_occupancy_probe()


# Per-target shared memory per multiprocessor, in bytes
# (CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR).
_MAX_SMEM_PER_MP_BYTES = {
    80: 167936,
    86: 102400,
    87: 167936,
    89: 102400,
    90: 233472,
    100: 233472,
    101: 233472,
    103: 233472,
    107: 233472,
    109: 233472,
    110: 233472,
    120: 102400,
    121: 102400,
}


def _patch_target_smem_per_mp_query(target_gpu_arch: str) -> None:
    """Answer the min-blocks smem query with the TARGET's static value.

    Kernels launched with ``min_blocks_per_mp > 1`` (int4 GEMV's tuned
    register-budget hints) make the DSL read
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR while tracing.
    That query needs an initialized CUDA driver — impossible on a GPU-less
    build host — and even with a GPU it reports the *build* device, not the
    artifact target. Substitute the target SM's documented value; every
    other attribute keeps the stock driver behavior.
    """
    match = re.search(r"sm_?(\d+)", target_gpu_arch)
    if match is None:
        return
    smem_bytes = _MAX_SMEM_PER_MP_BYTES.get(int(match.group(1)))
    if smem_bytes is None:
        return

    from cutlass.base_dsl.runtime import cuda as _dsl_cuda_helpers

    stock_get_device_attribute = _dsl_cuda_helpers.get_device_attribute
    smem_attr = (
        _dsl_cuda_helpers.cuda.CUdevice_attribute
        .CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
    )

    def get_device_attribute(attribute, device_id: int = 0):
        if attribute == smem_attr:
            return smem_bytes
        return stock_get_device_attribute(attribute, device_id)

    _dsl_cuda_helpers.get_device_attribute = get_device_attribute


if _args.gpu_arch:
    _patch_target_smem_per_mp_query(_args.gpu_arch)


def _install_version_hash_workaround() -> None:
    """Avoid importing duplicate helper modules while computing the cache key.

    Some CuTe DSL installs include both ``cutlass._mlir_helpers`` and
    ``cutlass.base_dsl._mlir_helpers``. The default tree-hash version scan can
    import both and double-register MLIR value casters. Kernel AOT builds only
    need a stable version component in the cache key, so use the installed
    package version instead of scanning the whole package tree.
    """
    import cutlass.cutlass_dsl.cutlass as cutlass_dsl

    @functools.lru_cache(maxsize=1)
    def get_version(_self):
        version_hash = hashlib.sha256()
        version = importlib.metadata.version("nvidia-cutlass-dsl")
        version_hash.update(f"nvidia-cutlass-dsl=={version}".encode())
        return version_hash

    cutlass_dsl.CutlassBaseDSL.get_version = get_version


def _merge_compile_options(existing_options, gpu_arch: str, host_target: str) -> str:
    options = str(existing_options or "").strip()
    extra_options: list[str] = []
    if gpu_arch and "--gpu-arch" not in options:
        extra_options.append(f"--gpu-arch {gpu_arch}")
    if host_target:
        # Quote: the long form ("llvm -mtriple=...") has spaces and the DSL
        # re-tokenizes this string, so an unquoted value would be split apart.
        extra_options.append(f"--host-target {shlex.quote(host_target)}")
    return " ".join([option for option in [options, *extra_options] if option])


def _should_inject_compile_options(compile_target, kernel_src_root: Path) -> bool:
    """Only inject cross options for kernels defined under ``kernelSrcs/``.

    CuTe DSL library helpers (e.g. ``HardwareInfo`` occupancy probes and
    ``cute.testing.convert``) compile dummy kernels that must run on the local
    build GPU. Injecting the cross target (``--gpu-arch`` for a different SM)
    into those would make them launch an incompatible cubin on the build GPU.

    ``@cute.jit`` / ``@cute.kernel`` decorate the user function, so
    ``inspect.getsourcefile`` on the decorated object resolves to the CuTe DSL
    library (``cutlass/base_dsl/dsl.py``) rather than the kernel script. Unwrap
    the decoration chain (``__wrapped__``) so kernels under ``kernelSrcs/`` are
    detected and their cross options are injected.
    """
    candidates = []
    for obj in (compile_target, getattr(compile_target, "__func__", None)):
        if obj is None:
            continue
        candidates.append(obj)
        try:
            unwrapped = inspect.unwrap(obj)
        except ValueError:
            unwrapped = None
        if unwrapped is not None and unwrapped is not obj:
            candidates.append(unwrapped)

    saw_source = False
    for obj in candidates:
        try:
            source_file = inspect.getsourcefile(obj)
        except TypeError:
            source_file = None
        if source_file is None:
            continue
        saw_source = True
        try:
            Path(source_file).resolve().relative_to(kernel_src_root)
            return True
        except ValueError:
            continue

    # No resolvable source anywhere (e.g. builtin/library helpers): preserve the
    # original "inject when origin is unknown" behavior.
    return not saw_source


def main() -> int:
    args = _args

    script = Path(args.script).resolve()
    kernel_src_root = Path(__file__).resolve().parents[1]
    script_args = args.script_args
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    # Compile-option injection is cross-only; native arch selection happened
    # via CUTE_DSL_ARCH before `import cutlass` (see module top).
    inject_gpu_arch = args.gpu_arch if args.host_target else ""

    _install_version_hash_workaround()
    real_compile = cute.compile

    def compile_with_options(*compile_args, **compile_kwargs):
        if (
            (inject_gpu_arch or args.host_target)
            and compile_args
            and _should_inject_compile_options(compile_args[0], kernel_src_root)
        ):
            compile_kwargs["options"] = _merge_compile_options(
                compile_kwargs.get("options"),
                inject_gpu_arch,
                args.host_target,
            )
            print(f"CuTe DSL compile options: {compile_kwargs['options']}")
        return real_compile(*compile_args, **compile_kwargs)

    cute.compile = compile_with_options
    sys.argv = [str(script), *script_args]
    sys.path.insert(0, str(script.parent))
    # Kernel scripts import shared helpers (cutedsl_utils.aot_placeholders)
    # relative to kernelSrcs/, which is not on sys.path under runpy.
    if str(kernel_src_root) not in sys.path:
        sys.path.insert(0, str(kernel_src_root))
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
