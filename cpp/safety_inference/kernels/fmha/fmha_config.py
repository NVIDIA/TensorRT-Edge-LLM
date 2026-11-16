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
"""Build-only configuration for Safety-owned FMHA kernels."""

import pathlib

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute import runtime

THREADS = 128
HEAD_SIZES = (64, 128)
SOURCE_DIR = pathlib.Path(__file__).resolve().parent
SOURCE_FILES = {
    name: SOURCE_DIR / name
    for name in ("safetyFmha.py", "generate.py", "safetyFmhaLaunch.h",
                 "fmha_config.py")
}
NUMERICAL_OPTIONS = {
    "dtype": "float16",
    "skip_rescale": True,
    "hybrid_exp2": False,
    "sliding_window": False,
}


def tile_size(head_size: int) -> int:
    """Return the admitted tile width for one FP16 head size."""
    if head_size not in HEAD_SIZES:
        raise ValueError(f"Unsupported head size: {head_size}")
    return 128 if head_size == 64 else 64


def compile_variant(kernel_type, head_size: int, visual: bool, arch: str):
    """Symbolic tensor descriptors avoid GPU allocation during export."""
    sym = cute.sym_int32
    tile = tile_size(head_size)
    kernel = kernel_type(
        head_size,
        m_block_size=tile,
        n_block_size=tile,
        num_threads=THREADS,
        is_causal=not visual,
        packed_varlen=visual,
        skip_rescale=NUMERICAL_OPTIONS['skip_rescale'],
        hybrid_exp2=NUMERICAL_OPTIONS['hybrid_exp2'],
        use_sliding_window=NUMERICAL_OPTIONS['sliding_window'])
    prefix = (runtime.make_fake_tensor(cutlass.Int32, (1, ), (1, ),
                                       assumed_align=4), )
    if visual:
        q = runtime.make_fake_compact_tensor(cutlass.Float16,
                                             (sym(), sym(), head_size),
                                             stride_order=(2, 1, 0),
                                             assumed_align=16)
        lengths = runtime.make_fake_tensor(cutlass.Int32, (sym(), ), (1, ),
                                           assumed_align=4)
        compiled = cute.compile(kernel.__call_vit__,
                                *prefix,
                                q,
                                q,
                                q,
                                q,
                                lengths,
                                cutlass.Int32(512),
                                cutlass.Float32(1),
                                cutlass.Float32(1),
                                cutlass.Float32(1),
                                cutlass.Int32(1),
                                cuda.CUstream(0),
                                options=f"--gpu-arch={arch}")
    else:
        q = runtime.make_fake_compact_tensor(cutlass.Float16,
                                             (1, sym(), sym(), head_size),
                                             stride_order=(3, 2, 1, 0),
                                             assumed_align=16)
        # Logical BSND view over physical BHSD storage; D remains contiguous.
        kv = runtime.make_fake_compact_tensor(cutlass.Float16,
                                              (1, sym(), sym(), head_size),
                                              stride_order=(3, 1, 2, 0),
                                              assumed_align=16)
        lengths = runtime.make_fake_tensor(cutlass.Int32, (2, ), (1, ),
                                           assumed_align=4)
        compiled = cute.compile(kernel.__call__,
                                *prefix,
                                q,
                                kv,
                                kv,
                                q,
                                lengths,
                                lengths,
                                None,
                                None,
                                cutlass.Float32(1),
                                cutlass.Int32(1),
                                cuda.CUstream(0),
                                options=f"--gpu-arch={arch}")
    return compiled, tile


def compile_blackwell(head_size: int, visual: bool, arch: str):
    """Compile the guarded Safety Blackwell kernel without allocating CUDA resources."""
    import safetyBlackwell

    mask = safetyBlackwell.fmha_utils.MaskEnum
    kernel = safetyBlackwell.SafetyBlackwell(
        cutlass.Float32,
        cutlass.Float32, (128, 128, head_size),
        True,
        mask.RESIDUAL_MASK if visual else mask.WINDOW_MASK_INFERENCE,
        is_causal=not visual)
    sym = cute.sym_int32
    q = runtime.make_fake_compact_tensor(cutlass.Float16,
                                         (sym(), sym(), head_size),
                                         stride_order=(2, 1, 0),
                                         assumed_align=16)
    kv = runtime.make_fake_compact_tensor(cutlass.Float16,
                                          (sym(), sym(), head_size),
                                          stride_order=(2, 1, 0) if visual else
                                          (1, 2, 0),
                                          assumed_align=16)
    lengths = runtime.make_fake_tensor(cutlass.Int32, (sym(), ), (1, ),
                                       assumed_align=4)
    valid = runtime.make_fake_tensor(cutlass.Int32, (1, ), (1, ),
                                     assumed_align=4)
    compiled = cute.compile(kernel,
                            valid,
                            q,
                            kv,
                            kv,
                            q,
                            lengths,
                            lengths,
                            cutlass.Int32(512),
                            cutlass.Float32(1),
                            cutlass.Float32(1),
                            cutlass.Float32(1),
                            cutlass.Int32(1),
                            cuda.CUstream(0),
                            options=f"--gpu-arch={arch}")
    return compiled, 256


def source_files(blackwell: bool):
    """Return the admitted source closure for the selected hardware backend."""
    if not blackwell:
        return SOURCE_FILES
    files = {
        name: SOURCE_DIR / name
        for name in ("safetyBlackwell.py", "generate.py", "safetyFmhaLaunch.h",
                     "safetyBlackwellLaunch.h", "fmha_config.py")
    }
    files["safety_fmha_helpers.py"] = SOURCE_DIR / "safety_fmha_helpers.py"
    return files
