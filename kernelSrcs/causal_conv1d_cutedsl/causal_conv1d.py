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

"""AOT-export packed variable-length causal Conv1D kernels."""

import argparse
import os
import sys
from pathlib import Path

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32

_KERNEL_SRCS_DIR = Path(__file__).resolve().parent.parent
if str(_KERNEL_SRCS_DIR) not in sys.path:
    sys.path.insert(0, str(_KERNEL_SRCS_DIR))

from cutedsl_utils import aot_placeholders  # noqa: E402

NUM_THREADS = 256
TOKENS_PER_TILE = 64
AOT_TOTAL_TOKENS = 32
AOT_NUM_SEQUENCES = 2
AOT_DIM = 128
AOT_NUM_SLOTS = 2


class PackedCausalConv1d:
    """Compute packed causal convolution and update the recurrent-state pool."""

    def __init__(self, width: int, sequence_mode: bool):
        if width < 2 or width > 8:
            raise ValueError(f"width must be in [2, 8], got {width}")
        self.width = width
        self.history_len = width - 1
        self.sequence_mode = sequence_mode

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        cu_seqlens: cute.Tensor,
        state_indices: cute.Tensor,
        has_initial_state: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor,
        conv_states: cute.Tensor,
        out: cute.Tensor,
        total_tokens: Int32,
        num_sequences: Int32,
        dim: Int32,
        num_slots: Int32,
        pad_slot_id: Int32,
        has_bias: Int32,
        stream: cuda.CUstream,
    ):
        token_tiles = cute.ceil_div(max(total_tokens, 1), TOKENS_PER_TILE)
        self.packed_kernel(
            x,
            cu_seqlens,
            state_indices,
            has_initial_state,
            weight,
            bias,
            conv_states,
            out,
            total_tokens,
            num_sequences,
            dim,
            num_slots,
            pad_slot_id,
            has_bias,
            self.sequence_mode,
        ).launch(
            grid=(num_sequences if self.sequence_mode else token_tiles, cute.ceil_div(dim, NUM_THREADS), 1),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )

        if not self.sequence_mode:
            self.update_state_kernel(
                x,
                cu_seqlens,
                state_indices,
                has_initial_state,
                conv_states,
                total_tokens,
                num_sequences,
                dim,
                num_slots,
                pad_slot_id,
            ).launch(
                grid=(num_sequences, cute.ceil_div(dim, NUM_THREADS), 1),
                block=(NUM_THREADS, 1, 1),
                stream=stream,
            )

    @cute.kernel
    def packed_kernel(
        self,
        x: cute.Tensor,
        cu_seqlens: cute.Tensor,
        state_indices: cute.Tensor,
        has_initial_state: cute.Tensor,
        weight: cute.Tensor,
        bias: cute.Tensor,
        conv_states: cute.Tensor,
        out: cute.Tensor,
        total_tokens: Int32,
        num_sequences: Int32,
        dim: Int32,
        num_slots: Int32,
        pad_slot_id: Int32,
        has_bias: Int32,
        sequence_mode: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        work_idx, dim_block, _ = cute.arch.block_idx()
        dim_idx = cutlass.Int32(dim_block * NUM_THREADS + tidx)
        if dim_idx < dim:
            if cutlass.const_expr(sequence_mode):
                sequence_idx = work_idx
                if sequence_idx < num_sequences:
                    seq_start = cutlass.Int32(cu_seqlens[sequence_idx])
                    seq_end = cutlass.Int32(cu_seqlens[sequence_idx + 1])
                    if seq_start >= 0 and seq_end >= seq_start and seq_end <= total_tokens:
                        state_slot = cutlass.Int32(state_indices[sequence_idx])
                        valid_slot = state_slot != pad_slot_id and state_slot >= 0 and state_slot < num_slots
                        valid_state = has_initial_state[sequence_idx] != 0 and valid_slot
                        state_offset = (
                            (cutlass.Int64(state_slot) * cutlass.Int64(dim) + cutlass.Int64(dim_idx))
                            * self.history_len
                        )

                        window = cute.make_rmem_tensor(self.width, cutlass.Float32)
                        for tap in cutlass.range_constexpr(self.history_len):
                            window[tap] = cutlass.Float32(0.0)
                            if valid_state:
                                window[tap] = cutlass.Float32(conv_states[state_offset + tap])

                        weight_values = cute.make_rmem_tensor(self.width, cutlass.Float32)
                        for tap in cutlass.range_constexpr(self.width):
                            weight_idx = cutlass.Int64(dim_idx) * self.width + tap
                            weight_values[tap] = cutlass.Float32(weight[weight_idx])

                        bias_value = cutlass.Float32(0.0)
                        if has_bias != 0:
                            bias_value = cutlass.Float32(bias[dim_idx])

                        token = seq_start
                        x_idx = cutlass.Int64(seq_start) * cutlass.Int64(dim) + cutlass.Int64(dim_idx)
                        while token < seq_end:
                            window[self.history_len] = cutlass.Float32(x[x_idx])
                            acc = bias_value
                            for tap in cutlass.range_constexpr(self.width):
                                acc += window[tap] * weight_values[tap]
                            out[x_idx] = cutlass.Float16(acc)
                            for tap in cutlass.range_constexpr(self.history_len):
                                window[tap] = window[tap + 1]
                            token += 1
                            x_idx += cutlass.Int64(dim)

                        if valid_slot:
                            for tap in cutlass.range_constexpr(self.history_len):
                                conv_states[state_offset + tap] = cutlass.Float16(window[tap])
            else:
                token_start = cutlass.Int32(work_idx * TOKENS_PER_TILE)
                sequence_idx = cutlass.Int32(0)
                if token_start < total_tokens:
                    token_end = min(token_start + TOKENS_PER_TILE, total_tokens)
                    weight_values = cute.make_rmem_tensor(self.width, cutlass.Float32)
                    for tap in cutlass.range_constexpr(self.width):
                        weight_idx = cutlass.Int64(dim_idx) * self.width + tap
                        weight_values[tap] = cutlass.Float32(weight[weight_idx])

                    bias_value = cutlass.Float32(0.0)
                    if has_bias != 0:
                        bias_value = cutlass.Float32(bias[dim_idx])

                    lane_idx = cute.arch.lane_idx()
                    low = cutlass.Int32(0)
                    high = num_sequences
                    if lane_idx == 0:
                        while low < high:
                            mid = low + (high - low) // 2
                            if cu_seqlens[mid + 1] <= token_start:
                                low = mid + 1
                            else:
                                high = mid
                    sequence_idx = cutlass.Int32(cute.arch.shuffle_sync(low, 0))
                    seq_start = cutlass.Int32(0)
                    seq_end = cutlass.Int32(0)
                    state_slot = cutlass.Int32(-1)
                    initial_state = cutlass.Int32(0)
                    window = cute.make_rmem_tensor(self.width, cutlass.Float32)
                    for tap in cutlass.range_constexpr(self.history_len):
                        window[tap] = cutlass.Float32(0.0)

                    if lane_idx == 0 and sequence_idx < num_sequences:
                        seq_start = cutlass.Int32(cu_seqlens[sequence_idx])
                        seq_end = cutlass.Int32(cu_seqlens[sequence_idx + 1])
                        state_slot = cutlass.Int32(state_indices[sequence_idx])
                        initial_state = cutlass.Int32(has_initial_state[sequence_idx])
                    seq_start = cutlass.Int32(cute.arch.shuffle_sync(seq_start, 0))
                    seq_end = cutlass.Int32(cute.arch.shuffle_sync(seq_end, 0))
                    state_slot = cutlass.Int32(cute.arch.shuffle_sync(state_slot, 0))
                    initial_state = cutlass.Int32(cute.arch.shuffle_sync(initial_state, 0))
                    valid_state = (
                        initial_state != 0
                        and state_slot != pad_slot_id
                        and state_slot >= 0
                        and state_slot < num_slots
                    )
                    if sequence_idx < num_sequences:
                        for tap in cutlass.range_constexpr(self.history_len):
                            input_pos = token_start - seq_start - self.history_len + tap
                            if input_pos >= 0:
                                x_idx = cutlass.Int64(seq_start + input_pos) * cutlass.Int64(dim) + dim_idx
                                window[tap] = cutlass.Float32(x[x_idx])
                            elif valid_state:
                                state_idx = (
                                    (cutlass.Int64(state_slot) * cutlass.Int64(dim) + dim_idx) * self.history_len
                                    + self.history_len
                                    + input_pos
                                )
                                window[tap] = cutlass.Float32(conv_states[state_idx])

                    token = token_start
                    out_idx = cutlass.Int64(token_start) * cutlass.Int64(dim) + cutlass.Int64(dim_idx)
                    single_sequence_tile = (
                        sequence_idx < num_sequences
                        and seq_start >= 0
                        and seq_start <= token_start
                        and token_end <= seq_end
                        and seq_end <= total_tokens
                    )
                    if single_sequence_tile:
                        while token < token_end:
                            window[self.history_len] = cutlass.Float32(x[out_idx])
                            acc = bias_value
                            for tap in cutlass.range_constexpr(self.width):
                                acc += window[tap] * weight_values[tap]
                            out[out_idx] = cutlass.Float16(acc)
                            for tap in cutlass.range_constexpr(self.history_len):
                                window[tap] = window[tap + 1]
                            token += 1
                            out_idx += cutlass.Int64(dim)
                    else:
                        while token < token_end:
                            while sequence_idx + 1 < num_sequences and token >= cu_seqlens[sequence_idx + 1]:
                                sequence_idx += 1
                                seq_start = cutlass.Int32(cu_seqlens[sequence_idx])
                                seq_end = cutlass.Int32(cu_seqlens[sequence_idx + 1])
                                state_slot = cutlass.Int32(state_indices[sequence_idx])
                                valid_state = (
                                    has_initial_state[sequence_idx] != 0
                                    and state_slot != pad_slot_id
                                    and state_slot >= 0
                                    and state_slot < num_slots
                                )
                                for tap in cutlass.range_constexpr(self.history_len):
                                    window[tap] = cutlass.Float32(0.0)
                                    if valid_state:
                                        state_idx = (
                                            (cutlass.Int64(state_slot) * cutlass.Int64(dim) + dim_idx)
                                            * self.history_len
                                            + tap
                                        )
                                        window[tap] = cutlass.Float32(conv_states[state_idx])

                            output_value = cutlass.Float32(0.0)
                            if sequence_idx < num_sequences:
                                valid_sequence = (
                                    seq_start >= 0
                                    and seq_start <= token
                                    and token < seq_end
                                    and seq_end <= total_tokens
                                )
                                if valid_sequence:
                                    window[self.history_len] = cutlass.Float32(x[out_idx])
                                    acc = bias_value
                                    for tap in cutlass.range_constexpr(self.width):
                                        acc += window[tap] * weight_values[tap]
                                    output_value = acc
                                    for tap in cutlass.range_constexpr(self.history_len):
                                        window[tap] = window[tap + 1]
                            out[out_idx] = cutlass.Float16(output_value)
                            token += 1
                            out_idx += cutlass.Int64(dim)

    @cute.kernel
    def update_state_kernel(
        self,
        x: cute.Tensor,
        cu_seqlens: cute.Tensor,
        state_indices: cute.Tensor,
        has_initial_state: cute.Tensor,
        conv_states: cute.Tensor,
        total_tokens: Int32,
        num_sequences: Int32,
        dim: Int32,
        num_slots: Int32,
        pad_slot_id: Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        sequence_idx, dim_block, _ = cute.arch.block_idx()
        dim_idx = cutlass.Int32(dim_block * NUM_THREADS + tidx)
        if sequence_idx < num_sequences and dim_idx < dim:
            state_slot = cutlass.Int32(state_indices[sequence_idx])
            if state_slot != pad_slot_id and state_slot >= 0 and state_slot < num_slots:
                seq_start = cutlass.Int32(cu_seqlens[sequence_idx])
                seq_end = cutlass.Int32(cu_seqlens[sequence_idx + 1])
                if seq_start >= 0 and seq_end >= seq_start and seq_end <= total_tokens:
                    seq_len = seq_end - seq_start
                    tail_len = min(seq_len, self.history_len)
                    dst_offset = self.history_len - tail_len
                    state_offset = (
                        (cutlass.Int64(state_slot) * cutlass.Int64(dim) + cutlass.Int64(dim_idx)) * self.history_len
                    )

                    t = cutlass.Int32(0)
                    while t < dst_offset:
                        if has_initial_state[sequence_idx] != 0:
                            conv_states[state_offset + t] = conv_states[state_offset + tail_len + t]
                        else:
                            conv_states[state_offset + t] = cutlass.Float16(0.0)
                        t += 1

                    tail_start = seq_len - tail_len
                    t = cutlass.Int32(0)
                    while t < tail_len:
                        src_idx = (
                            cutlass.Int64(seq_start + tail_start + t) * cutlass.Int64(dim) + cutlass.Int64(dim_idx)
                        )
                        conv_states[state_offset + dst_offset + t] = x[src_idx]
                        t += 1


def _dynamic_tensor(dtype, size):
    return aot_placeholders.make_compact_tensor(dtype, (size,), assumed_align=16).mark_compact_shape_dynamic(
        mode=0,
        stride_order=(0,),
        divisibility=1,
    )


def compile_kernel(width: int, mode: str):
    history_len = width - 1
    x = _dynamic_tensor(cutlass.Float16, AOT_TOTAL_TOKENS * AOT_DIM)
    cu_seqlens = _dynamic_tensor(cutlass.Int32, AOT_NUM_SEQUENCES + 1)
    state_indices = _dynamic_tensor(cutlass.Int32, AOT_NUM_SEQUENCES)
    has_initial_state = _dynamic_tensor(cutlass.Int32, AOT_NUM_SEQUENCES)
    weight = _dynamic_tensor(cutlass.Float16, AOT_DIM * width)
    bias = _dynamic_tensor(cutlass.Float16, AOT_DIM)
    conv_states = _dynamic_tensor(cutlass.Float16, AOT_NUM_SLOTS * AOT_DIM * history_len)
    out = _dynamic_tensor(cutlass.Float16, AOT_TOTAL_TOKENS * AOT_DIM)

    return cute.compile(
        PackedCausalConv1d(width, mode == "short"),
        x,
        cu_seqlens,
        state_indices,
        has_initial_state,
        weight,
        bias,
        conv_states,
        out,
        cutlass.Int32(AOT_TOTAL_TOKENS),
        cutlass.Int32(AOT_NUM_SEQUENCES),
        cutlass.Int32(AOT_DIM),
        cutlass.Int32(AOT_NUM_SLOTS),
        cutlass.Int32(-1),
        cutlass.Int32(1),
        aot_placeholders.make_stream(),
        options="--opt-level 3",
    )


def export_kernel(width: int, mode: str, output_dir: str, file_name: str, function_prefix: str):
    compiled = compile_kernel(width, mode)
    os.makedirs(output_dir, exist_ok=True)
    compiled.export_to_c(file_path=output_dir, file_name=file_name, function_prefix=function_prefix)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, choices=range(2, 9), required=True)
    parser.add_argument("--mode", choices=("prefill", "short"), required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--file_name", required=True)
    parser.add_argument("--function_prefix", required=True)
    parser.add_argument("--export_only", action="store_true")
    args = parser.parse_args()
    export_kernel(args.width, args.mode, args.output_dir, args.file_name, args.function_prefix)


if __name__ == "__main__":
    main()
