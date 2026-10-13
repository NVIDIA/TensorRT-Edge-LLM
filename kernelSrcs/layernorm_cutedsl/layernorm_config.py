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
"""Compile-time launch geometry for the CuTe DSL LayerNorm kernel.

Deliberately free of ``cutlass``, ``cupy`` and ``cuda.bindings`` imports so the
geometry is importable -- and therefore testable -- on a host with no GPU and no
CuTe DSL installation. ``layernorm.py`` is the only consumer.

Two schedules exist:

``W`` (wide)
    One CTA per row, with the CTA sized to the row and capped at 1024 threads.
    This is the schedule vendored from FlashInfer.

``S`` (staged)
    A 128-thread CTA covering one or two rows, with each thread holding several
    vector blocks and the row staged through shared memory. Ported from
    ``kernelSrcs/rmsnorm_cutedsl/rmsnorm.py``.

The schedule is a property of the *target* SM, which must be passed in
explicitly. It is deliberately not derived from the local device: the CuTe DSL
AOT export runs on the build host, which is an x86 GPU for every cross-compiled
aarch64 artifact, so a device query here resolves the wrong architecture. See
``SCHEDULE_BY_SM``.
"""

from dataclasses import dataclass

__all__ = [
    "SUPPORTED_HIDDEN_SIZES",
    "SUPPORTED_TARGET_SMS",
    "SCHEDULE_BY_SM",
    "SHARED_MEMORY_PER_BLOCK_OPTIN",
    "LayerNormConfig",
    "select_schedule",
    "layernorm_config",
]

COPY_BITS = 128
WARP_SIZE = 32

SUPPORTED_HIDDEN_SIZES = (4096, 4097, 5120, 7168, 8192)

# Must stay in sync with _LAYERNORM_SUPPORTED_SMS in kernelSrcs/build_cutedsl.py
# and with the static_assert in cpp/kernels/layerNorm/cuteDslLayerNormRunner.cpp.
SUPPORTED_TARGET_SMS = (80, 86, 87, 90, 100, 101, 110, 120, 121)

# Per-target opt-in shared memory per block, in bytes. Static rather than
# queried: the export runs on the build host, not the target board, so
# cudaDeviceProp here describes the wrong device.
SHARED_MEMORY_PER_BLOCK_OPTIN = {
    80: 163840,
    86: 101376,
    87: 163840,
    90: 233472,
    100: 233472,
    101: 233472,
    110: 233472,
    120: 101376,
    121: 101376,
}

# Which schedule each target SM uses. Every SM is listed explicitly: a rule of
# the form "S unless 121" would silently change the schedule on architectures
# where the kernel has never been measured, and SM 80/100/120 are exactly where
# the L0 numerical tests execute.
#
# The plugin was measured against the decomposed TensorRT graph on the three
# edge targets only. Every other SM keeps the vendored W schedule.
#
# A full-size rerun validated this selection against the decomposed TensorRT
# path. The figures compare the earlier wide-source implementation with the
# selected-source implementation, not a controlled same-source S/W experiment,
# so they are validation outcomes rather than isolated schedule attribution:
#
#   SM87  Orin  board plugin/decomposed 1.095 -> 0.498 with S
#   SM110 Thor  board plugin/decomposed 1.204 -> 0.788 with S
#   SM121 GB10  W 0.929 -> 0.923; retain W
SCHEDULE_BY_SM = {
    80: "W",  # A100 - not measured
    86: "W",  # not measured; not in the layernorm CI group
    87: "S",  # Orin - selected schedule validated over the full matrix
    90: "W",  # H100 - not measured
    100: "W",  # B100 - not measured
    101: "W",  # not measured
    110: "S",  # Thor - selected schedule validated over the full matrix
    120: "W",  # RTX 50xx - not measured
    121: "W",  # GB10 - retained after full-matrix validation
}


@dataclass(frozen=True)
class LayerNormConfig:
    """Resolved launch geometry for one (target SM, dtype, hidden size)."""

    schedule: str
    hidden_size: int
    vec_size: int
    copy_bits: int
    threads_per_row: int
    num_threads: int
    rows_per_block: int
    warps_per_row: int
    num_vec_blocks: int
    cols_per_tile: int
    use_async_copy: bool
    smem_bytes: int

    @property
    def needs_padding(self) -> bool:
        """Whether any lane in the tile falls outside the row."""
        return self.cols_per_tile != self.hidden_size


def select_schedule(target_sm: int, hidden_size: int) -> str:
    """Return "S" or "W" for a target SM and hidden size.

    H=4097 always uses W. Its largest power-of-two divisor is 1, which forces a
    scalar copy; the resulting 16-bit copy width cannot drive the asynchronous
    global-to-shared copy that the staged schedule depends on. The shipped auto
    policy therefore keeps the vendored W schedule for this odd-H variant.
    """
    _validate(target_sm, hidden_size)
    if hidden_size == 4097:
        return "W"
    return SCHEDULE_BY_SM[target_sm]


def layernorm_config(target_sm: int,
                     dtype_width: int,
                     hidden_size: int,
                     schedule: str = None) -> LayerNormConfig:
    """Resolve the launch geometry.

    Args:
        target_sm: SM version of the board the artifact will run on.
        dtype_width: Storage dtype width in bits (16 for FP16 and BF16).
        hidden_size: Normalized axis length.
        schedule: "S" or "W" to override the SM-derived choice. Used to
            validate both schedules on any board; leave unset in the build.
    """
    _validate(target_sm, hidden_size)
    if schedule is None:
        schedule = select_schedule(target_sm, hidden_size)
    elif schedule not in ("S", "W"):
        raise ValueError(f"unknown LayerNorm schedule {schedule!r}")

    max_vec_size = COPY_BITS // dtype_width
    if schedule == "W":
        vec_size = _widest_vec_filling_a_warp(hidden_size, max_vec_size)
        threads_per_row = _round_up_pow2(
            (hidden_size + vec_size - 1) // vec_size)
        num_threads = threads_per_row
        rows_per_block = 1
    else:
        # The staged schedule vectorizes on the row's alignment rather than on
        # divisibility, and sizes the CTA before the row.
        vec_size = min(hidden_size & (-hidden_size), max_vec_size)
        threads_per_row = _staged_threads_per_row(hidden_size)
        num_threads = 128 if hidden_size <= 16384 else 256
        rows_per_block = num_threads // threads_per_row

    warps_per_row = max(threads_per_row // WARP_SIZE, 1)
    num_vec_blocks = max(
        1,
        (hidden_size // vec_size + threads_per_row - 1) // threads_per_row)
    cols_per_tile = vec_size * num_vec_blocks * threads_per_row
    copy_bits = vec_size * dtype_width

    # Both reductions get their own buffer. Sharing one would be a
    # write-after-read race across the mean and variance passes.
    reduction_bytes = 2 * rows_per_block * warps_per_row * 4
    # The staged tile is never zero-filled, so an inexactly covered row would
    # feed uninitialized shared memory into the mean reduction. Staging is
    # therefore restricted to exact coverage rather than relying on the
    # predicate that protects the register-resident path.
    if schedule == "S" and copy_bits >= 32 and cols_per_tile == hidden_size:
        tile_bytes = rows_per_block * cols_per_tile * (dtype_width // 8)
        use_async_copy = (tile_bytes <=
                          SHARED_MEMORY_PER_BLOCK_OPTIN[target_sm] // 2)
    else:
        tile_bytes = 0
        use_async_copy = False
    smem_bytes = (tile_bytes if use_async_copy else 0) + reduction_bytes

    return LayerNormConfig(
        schedule=schedule,
        hidden_size=hidden_size,
        vec_size=vec_size,
        copy_bits=copy_bits,
        threads_per_row=threads_per_row,
        num_threads=num_threads,
        rows_per_block=rows_per_block,
        warps_per_row=warps_per_row,
        num_vec_blocks=num_vec_blocks,
        cols_per_tile=cols_per_tile,
        use_async_copy=use_async_copy,
        smem_bytes=smem_bytes,
    )


def _validate(target_sm: int, hidden_size: int) -> None:
    if target_sm not in SUPPORTED_TARGET_SMS:
        raise ValueError(
            f"unsupported LayerNorm target SM {target_sm}; "
            f"expected one of {list(SUPPORTED_TARGET_SMS)}")
    if hidden_size not in SUPPORTED_HIDDEN_SIZES:
        raise ValueError(
            f"unsupported LayerNorm hidden size {hidden_size}; "
            f"expected one of {list(SUPPORTED_HIDDEN_SIZES)}")


def _widest_vec_filling_a_warp(hidden_size: int, max_vec_size: int) -> int:
    """Widest aligned vector that still gives at least one warp of columns."""
    import math

    for vec_size in (max_vec_size, max_vec_size // 2, max_vec_size // 4,
                     max_vec_size // 8):
        if vec_size < 1:
            continue
        if hidden_size % vec_size != 0:
            continue
        if hidden_size // vec_size >= WARP_SIZE:
            return vec_size
    return math.gcd(max_vec_size, hidden_size)


def _round_up_pow2(threads_needed: int) -> int:
    threads = WARP_SIZE
    while threads < threads_needed and threads < 1024:
        threads *= 2
    return min(threads, 1024)


def _staged_threads_per_row(hidden_size: int) -> int:
    if hidden_size <= 64:
        return 8
    if hidden_size <= 128:
        return 16
    if hidden_size <= 3072:
        return 32
    if hidden_size <= 6144:
        return 64
    if hidden_size <= 16384:
        return 128
    return 256
