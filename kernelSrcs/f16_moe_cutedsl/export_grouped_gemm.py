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
"""Export one architecture-family FP16 MoE grouped-GEMM AOT module."""

import argparse
import os
import sys

import export_common


def _make_kernel(family: str):
    if family == "ampere":
        from grouped_gemm_ampere import F16MoeGroupedGemmAmpere

        return F16MoeGroupedGemmAmpere()
    if family == "blackwell":
        from grouped_gemm_blackwell import F16MoeGroupedGemmBlackwell

        return F16MoeGroupedGemmBlackwell()
    if family == "blackwell_geforce":
        from grouped_gemm_blackwell_geforce import (
            F16MoeGroupedGemmBlackwellGeforce, )

        return F16MoeGroupedGemmBlackwellGeforce()
    raise ValueError(f"Unsupported f16_moe family: {family}")


def export_grouped_gemm(args: argparse.Namespace) -> tuple[str, str]:
    """Compile and export the selected family with the shared C ABI."""
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda
    from cutedsl_utils import aot_placeholders

    class ExportWrapper:
        """Bind raw pointers plus max_m/max_n/max_k bounds to device layouts.

        The positional ABI then carries group_count, max_active_clusters, and
        the CUDA stream. Exact group shapes and addresses stay device resident.
        """

        def __init__(self, kernel):
            self.kernel = kernel

        @cute.jit
        def wrapper(
            self,
            a_ptr: cute.Pointer,
            b_ptr: cute.Pointer,
            d_ptr: cute.Pointer,
            problem_shapes_ptr: cute.Pointer,
            strides_ptr: cute.Pointer,
            addresses_ptr: cute.Pointer,
            scratch_ptr: cute.Pointer,
            max_m: cutlass.Int32,
            n: cutlass.Int32,
            k: cutlass.Int32,
            group_count: cutlass.Int32,
            max_active_clusters: cutlass.Int32,
            stream: cuda.CUstream,
        ) -> cutlass.Int32:
            # Global bounds keep tiled partitions runtime-sized. Exact
            # per-expert shapes and addresses still come from device metadata.
            initial_a = cute.make_tensor(
                a_ptr,
                layout=cute.make_ordered_layout((max_m, k, 1),
                                                order=(1, 0, 2)),
            )
            initial_b = cute.make_tensor(
                b_ptr,
                layout=cute.make_ordered_layout((n, k, 1),
                                                order=(1, 0, 2)),
            )
            initial_d = cute.make_tensor(
                d_ptr,
                layout=cute.make_ordered_layout((max_m, n, 1),
                                                order=(1, 0, 2)),
            )
            problem_shapes = cute.make_tensor(
                problem_shapes_ptr,
                layout=cute.make_layout((export_common.MAX_NUM_EXPERTS, 4),
                                        stride=(4, 1)),
            )
            strides = cute.make_tensor(
                strides_ptr,
                layout=cute.make_layout((export_common.MAX_NUM_EXPERTS, 3, 2),
                                        stride=(6, 2, 1)),
            )
            addresses = cute.make_tensor(
                addresses_ptr,
                layout=cute.make_layout((export_common.MAX_NUM_EXPERTS, 3),
                                        stride=(3, 1)),
            )
            scratch = cute.make_tensor(
                cute.recast_ptr(scratch_ptr, dtype=cutlass.Uint8),
                layout=cute.make_layout(
                    (
                        export_common.MAX_NUM_EXPERTS,
                        export_common.TENSORMAPS_PER_BLOCK,
                        export_common.BYTES_PER_TENSORMAP,
                    ),
                    stride=(
                        export_common.TENSORMAPS_PER_BLOCK *
                        export_common.BYTES_PER_TENSORMAP,
                        export_common.BYTES_PER_TENSORMAP,
                        1,
                    ),
                ),
            )
            self.kernel(
                initial_a,
                initial_b,
                initial_d,
                problem_shapes,
                strides,
                addresses,
                scratch,
                group_count,
                max_active_clusters,
                stream,
            )
            return cutlass.Int32(0)

    kernel = _make_kernel(args.family)
    export_wrapper = ExportWrapper(kernel)
    # Runtime-typed compile argument: deployment passes the target GPU's
    # persistent block count through the exported ABI.
    max_active_clusters = export_common.runtime_max_active_clusters(args.family)
    pointers = (
        export_common.make_ptr(
            cutlass.Float16,
            export_common.DESCRIPTOR_ALIGNMENT,
        ),
        export_common.make_ptr(
            cutlass.Float16,
            export_common.DESCRIPTOR_ALIGNMENT,
        ),
        export_common.make_ptr(
            cutlass.Float16,
            export_common.DESCRIPTOR_ALIGNMENT,
        ),
        export_common.make_ptr(
            cutlass.Int32,
            export_common.DESCRIPTOR_ALIGNMENT,
        ),
        export_common.make_ptr(
            cutlass.Int32,
            export_common.DESCRIPTOR_ALIGNMENT,
        ),
        export_common.make_ptr(
            cutlass.Int64,
            export_common.DESCRIPTOR_ALIGNMENT,
        ),
        export_common.make_ptr(
            cutlass.Uint8,
            export_common.TENSORMAP_ALIGNMENT,
        ),
    )
    stream = aot_placeholders.make_stream()
    compiled = cute.compile(
        export_wrapper.wrapper,
        *pointers,
        128,
        128,
        128,
        export_common.MAX_NUM_EXPERTS,
        max_active_clusters,
        stream,
        options=aot_placeholders.compile_options(),
    )

    os.makedirs(args.output_dir, exist_ok=True)
    compiled.export_to_c(args.output_dir, args.file_name, args.function_prefix)
    return export_common.verify_export(args.output_dir, args.file_name)


def parse_args() -> argparse.Namespace:
    """Parse the AOT export command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=("ampere", "blackwell", "blackwell_geforce"),
        required=True,
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--file_name", required=True)
    parser.add_argument("--function_prefix", required=True)
    return parser.parse_args()


def main() -> int:
    """Export one grouped-GEMM module and print the generated paths."""
    header, obj = export_grouped_gemm(parse_args())
    print(f"Exported {header}")
    print(f"Exported {obj}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
