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
"""AOT-export dynamic-sequence video sparse attention for Blackwell GPUs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute

_SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SOURCE_DIR))

from cutedsl_utils import aot_placeholders  # noqa: E402


# Sequence and prefix lengths are runtime dimensions; these only trace the kernel.
EXAMPLE_TOKENS = 34_462
EXAMPLE_PREFIX_TOKENS = 422
HEADS = 56
HEAD_DIM = 128
BLOCK_SIZE = 64
SOFTMAX_SCALE = 1.0 / (HEAD_DIM**0.5)


def _entry_point(backend: str):
    """Import and bind only the kernel implementation being exported."""
    if backend == "sm100":
        from sparse_attn.sm100.mainloop import forward

        @cute.jit
        def dynamic_sparse_attention_sm100(
            q, k, v, output, kc, vc, threshold, conditioning_tokens: cutlass.Int32, stream: cuda.CUstream
        ):
            return forward(
                q,
                k,
                v,
                output,
                kc,
                vc,
                threshold,
                cutlass.Int32(q.shape[1]),
                cutlass.Int32(kc.shape[1]),
                cutlass.Float32(SOFTMAX_SCALE),
                cutlass.Int32(0),
                cutlass.Int32(cute.ceil_div(conditioning_tokens,
                                           BLOCK_SIZE)),
                stream,
            )

        return dynamic_sparse_attention_sm100

    if backend == "sm120":
        from sparse_attn.sm120 import make_kernel

        forward = make_kernel()

        @cute.jit
        def dynamic_sparse_attention_sm120(
            q, k, v, output, kc, vc, threshold, conditioning_tokens: cutlass.Int32, stream: cuda.CUstream
        ):
            return forward(
                q,
                k,
                v,
                output,
                kc,
                vc,
                threshold,
                cutlass.Float32(SOFTMAX_SCALE),
                cutlass.Int32(0),
                cutlass.Int32(cute.ceil_div(conditioning_tokens,
                                           BLOCK_SIZE)),
                stream,
            )

        return dynamic_sparse_attention_sm120

    raise ValueError(f"unsupported backend: {backend}")


def _tensor(dtype, shape):
    tensor = aot_placeholders.make_compact_tensor(dtype,
                                                  shape,
                                                  assumed_align=16)
    return tensor.mark_compact_shape_dynamic(
        mode=1, stride_order=tuple(range(len(shape))))


def export(output_dir: str, file_name: str, function_prefix: str,
           backend: str) -> None:
    tokens = EXAMPLE_TOKENS
    blocks = (tokens + BLOCK_SIZE - 1) // BLOCK_SIZE
    bthd = (1, tokens, HEADS, HEAD_DIM)
    bnhd = (1, blocks, HEADS, HEAD_DIM)
    arguments = (
        _tensor(cutlass.BFloat16, bthd),
        _tensor(cutlass.BFloat16, bthd),
        _tensor(cutlass.BFloat16, bthd),
        _tensor(cutlass.BFloat16, bthd),
        _tensor(cutlass.BFloat16, bnhd),
        _tensor(cutlass.BFloat16, bnhd),
        _tensor(cutlass.Float32, (1, blocks, HEADS)),
        cutlass.Int32(EXAMPLE_PREFIX_TOKENS),
        aot_placeholders.make_stream(),
    )
    options = aot_placeholders.compile_options()
    entry_point = _entry_point(backend)
    compiled = cute.compile(
        entry_point,
        *arguments,
        **({"options": options} if options else {}),
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    compiled.export_to_c(output_dir, file_name, function_prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export_only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--file_name", required=True)
    parser.add_argument("--function_prefix", required=True)
    parser.add_argument("--backend", required=True, choices=("sm100", "sm120"))
    args = parser.parse_args()
    export(args.output_dir, args.file_name, args.function_prefix,
           args.backend)


if __name__ == "__main__":
    main()
