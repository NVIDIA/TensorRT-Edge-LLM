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
"""NumPy-only conversion of AWQ/GPTQ weights to INT4 plugin layouts."""

from typing import Optional, Tuple

import numpy as np


def _pack_for_gemm(unpacked: np.ndarray, plugin_version: int) -> np.ndarray:
    """Pack biased ``[N,K]`` nibbles for one dense INT4 plugin contract."""
    if plugin_version == 1:
        return pack_intweights(unpacked)
    if plugin_version == 2:
        return pack_cutedsl_fragment(unpacked)
    raise ValueError(f"unsupported INT4 GEMM plugin version {plugin_version}")


def select_column_packed(values: np.ndarray, indices: np.ndarray,
                         channel_to_bit: Tuple[int, ...]) -> np.ndarray:
    """Select output channels from an int32 tensor packed eight per column."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size % 8:
        raise ValueError(
            "packed INT4 output selection requires a multiple of 8")
    source = values.astype(np.int32, copy=False)
    result = np.zeros((source.shape[0], indices.size // 8), dtype=np.int32)
    for target_offset in range(8):
        selected = indices[target_offset::8]
        source_columns = selected // 8
        source_offsets = selected % 8
        source_bits = np.asarray(
            [channel_to_bit[int(offset)] for offset in source_offsets],
            dtype=np.int32)
        unpacked = source[:, source_columns]
        unpacked = (unpacked >> (4 * source_bits.reshape(1, -1))) & 0xF
        result |= unpacked << (4 * channel_to_bit[target_offset])
    return result.astype(values.dtype, copy=False)


def select_pair_packed_rows(values: np.ndarray,
                            indices: np.ndarray) -> np.ndarray:
    """Select output channels from a uint8 tensor packed two per row."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size % 2:
        raise ValueError("pair-packed INT4 selection requires an even size")
    source = values.astype(np.uint8, copy=False)
    unpacked = np.empty((source.shape[0] * 2, source.shape[1]), dtype=np.uint8)
    unpacked[0::2] = source & np.uint8(0xF)
    unpacked[1::2] = source >> np.uint8(4)
    selected = unpacked[indices]
    return np.ascontiguousarray(selected[0::2]
                                | (selected[1::2] << np.uint8(4)))


def pack_intweights(unpacked: np.ndarray) -> np.ndarray:
    """Pack nibbles into legacy ``Int4GroupwiseGemmPlugin`` layout."""
    n_dim, k_dim = unpacked.shape
    if n_dim % 4 or k_dim % 64:
        raise ValueError(
            f"INT4 plugin packing requires N%4=0 and K%64=0, got {(n_dim, k_dim)}"
        )
    packed = unpacked.astype(np.int16, copy=False)
    packed = packed.reshape(n_dim, k_dim // 32, 4, 4, 2)
    packed = packed.transpose(0, 1, 3, 2, 4).reshape(n_dim, k_dim // 32, 32)
    packed = packed.reshape(n_dim, k_dim // 32, 4, 4, 2)
    packed = packed.transpose(0, 1, 2, 4, 3).reshape(n_dim, k_dim)
    packed = packed.reshape(n_dim // 4, 4, k_dim // 64, 64)
    packed = packed.transpose(0, 2, 1, 3).reshape(n_dim // 4, k_dim // 64, 64,
                                                  4)
    packed = (packed[..., 0] | (packed[..., 1] << 4)
              | (packed[..., 2] << 8) | (packed[..., 3] << 12))
    packed16 = packed.reshape(n_dim // 4, k_dim).astype(np.int16)
    return packed16.view(np.int8).reshape(n_dim // 2, k_dim)


def pack_cutedsl_fragment(unpacked: np.ndarray) -> np.ndarray:
    """Pack biased nibbles for ``Int4GroupwiseGemmPluginV2``.

    The V2 plugin consumes an INT8 view of fragment-ordered uint32 words with
    shape ``[ceil(N/128) * ceil(K/64) * 8, 512]``. N is padded to 128 with
    nibble 8 (quantized zero); K must already be 64-aligned.
    """
    n_dim, k_dim = unpacked.shape
    if k_dim % 64:
        raise ValueError(
            f"INT4 V2 fragment packing requires K%64=0, got {(n_dim, k_dim)}")

    n_blocks = (n_dim + 127) // 128
    k_tiles = k_dim // 64
    source = np.asarray(unpacked, dtype=np.uint8) & np.uint8(0xF)
    words = np.empty((n_blocks * k_tiles * 8, 128), dtype=np.uint32)

    for n_block in range(n_blocks):
        for k_tile in range(k_tiles):
            for fragment in range(8):
                k_block, n_pair = divmod(fragment, 2)
                row = (n_block * k_tiles + k_tile) * 8 + fragment
                for thread in range(128):
                    n_lo = n_block * 128 + thread // 4 + 64 * n_pair
                    n_hi = n_lo + 32
                    k0 = k_tile * 64 + 16 * k_block + 2 * (thread % 4)
                    coordinates = (
                        (n_lo, k0),
                        (n_lo, k0 + 8),
                        (n_hi, k0),
                        (n_hi, k0 + 8),
                        (n_lo, k0 + 1),
                        (n_lo, k0 + 9),
                        (n_hi, k0 + 1),
                        (n_hi, k0 + 9),
                    )
                    word = 0
                    for shift, (n_index, k_index) in enumerate(coordinates):
                        value = (source[n_index, k_index]
                                 if n_index < n_dim else np.uint8(8))
                        word |= int(value) << (4 * shift)
                    words[row, thread] = word

    return np.ascontiguousarray(words).view(np.int8).reshape(-1, 512)


def repack_awq(qweight: np.ndarray,
               qzeros: np.ndarray,
               plugin_version: int = 1,
               center_at_eight: bool = False) -> np.ndarray:
    """Convert AWQ weights to plugin bytes, rejecting unrepresentable values."""
    in_features, out_div8 = qweight.shape
    out_features = out_div8 * 8
    group_size = in_features // qzeros.shape[0]
    bit_to_channel = (0, 2, 4, 6, 1, 3, 5, 7)

    weight_nibbles = np.empty((in_features, out_features), dtype=np.int16)
    zero_nibbles = np.empty((qzeros.shape[0], out_features), dtype=np.int16)
    qweight = qweight.astype(np.int32, copy=False)
    qzeros = qzeros.astype(np.int32, copy=False)
    for bit, channel in enumerate(bit_to_channel):
        weight_nibbles[:, channel::8] = (qweight >> (4 * bit)) & 0xF
        zero_nibbles[:, channel::8] = (qzeros >> (4 * bit)) & 0xF
    expanded_zeros = np.repeat(zero_nibbles, group_size, axis=0)
    adjusted = weight_nibbles if center_at_eight else weight_nibbles - expanded_zeros + 8
    if adjusted.size and (adjusted.min() < 0 or adjusted.max() > 15):
        raise ValueError("AWQ weights exceed the signed INT4 range after "
                         "subtracting zero points; repacking would lose data")
    return _pack_for_gemm(np.ascontiguousarray(adjusted.T), plugin_version)


def awq_zero_correction(qzeros: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Return FP32 groupwise coefficients for midpoint-centered AWQ weights."""
    if qzeros.ndim != 2 or scales.shape != (qzeros.shape[0],
                                            qzeros.shape[1] * 8):
        raise ValueError(
            "AWQ qzeros and scales must have matching group and output dimensions"
        )
    zeros = np.empty(scales.shape, dtype=np.int16)
    qzeros = qzeros.astype(np.int32, copy=False)
    for bit, channel in enumerate((0, 2, 4, 6, 1, 3, 5, 7)):
        zeros[:, channel::8] = (qzeros >> (4 * bit)) & 0xF
    return np.ascontiguousarray((8 - zeros).astype(np.float32) *
                                scales.astype(np.float16).astype(np.float32))


def repack_gptq(
    qweight: np.ndarray,
    qzeros: np.ndarray,
    g_idx: Optional[np.ndarray] = None,
    zero_point_offset: int = 1,
    plugin_version: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert row-packed GPTQ int32 weights to plugin bytes and K permutation."""
    in_div8, out_features = qweight.shape
    in_features = in_div8 * 8
    qweight = qweight.astype(np.int32, copy=False)
    qzeros = qzeros.astype(np.int32, copy=False)
    symmetric = qzeros.size == 0
    if symmetric:
        if qzeros.ndim and qzeros.shape[0] > 0:
            num_groups = qzeros.shape[0]
        elif g_idx is not None and g_idx.size:
            num_groups = int(g_idx.max()) + 1
        else:
            num_groups = 1
    else:
        num_groups = qzeros.shape[0]
    group_size = in_features // num_groups

    weight_nibbles = np.empty((in_features, out_features), dtype=np.int16)
    for bit in range(8):
        weight_nibbles[bit::8] = (qweight >> (4 * bit)) & 0xF
    if symmetric:
        zero_nibbles = np.full((num_groups, out_features),
                               8 - int(zero_point_offset),
                               dtype=np.int16)
    else:
        zero_nibbles = np.empty((num_groups, out_features), dtype=np.int16)
        for bit in range(8):
            zero_nibbles[:, bit::8] = (qzeros >> (4 * bit)) & 0xF
    if g_idx is None:
        g_idx = np.arange(in_features, dtype=np.int32) // group_size
    else:
        g_idx = g_idx.astype(np.int32, copy=False)
    expanded_zeros = zero_nibbles[g_idx.astype(np.int64)]
    adjusted = np.clip(
        weight_nibbles - expanded_zeros - int(zero_point_offset) + 8, 0, 15)
    permutation = np.concatenate(
        [np.flatnonzero(g_idx == group) for group in range(num_groups)])
    adjusted = adjusted[permutation]
    return (_pack_for_gemm(np.ascontiguousarray(adjusted.T),
                           plugin_version), permutation.astype(np.int64))


def repack_modelopt_awq(weight: np.ndarray,
                        plugin_version: int = 1) -> np.ndarray:
    """Convert ModelOpt packed uint8 W4A16 weights to plugin bytes."""
    out_half, in_features = weight.shape
    out_features = out_half * 2
    weight16 = weight.astype(np.int16, copy=False)
    nibbles = np.empty((out_features, in_features), dtype=np.int16)
    nibbles[0::2] = weight16 & 0xF
    nibbles[1::2] = (weight16 >> 4) & 0xF
    nibbles = (nibbles + 8) % 16
    return _pack_for_gemm(nibbles, plugin_version)


def _unpack_gptq_rows(qweight: np.ndarray) -> np.ndarray:
    """Unpack GPTQ ``[K//8,N]`` int32 into unsigned ``[K,N]`` nibbles."""
    qweight = qweight.astype(np.int32, copy=False)
    unpacked = np.empty((qweight.shape[0] * 8, qweight.shape[1]),
                        dtype=np.int16)
    for bit in range(8):
        unpacked[bit::8] = (qweight >> (4 * bit)) & 0xF
    return unpacked


def extract_gptq_for_moe(
    qweight: np.ndarray,
    qzeros: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    zero_point_offset: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return Marlin-source nibbles/scales as ``[N,K]`` and ``[N,G]``."""
    unpacked = _unpack_gptq_rows(qweight)
    if qzeros.size:
        qzeros = qzeros.astype(np.int32, copy=False)
        zeros = np.empty((qzeros.shape[0], qzeros.shape[1] * 8),
                         dtype=np.int16)
        for bit in range(8):
            zeros[:, bit::8] = (qzeros >> (4 * bit)) & 0xF
        group_ids = np.arange(unpacked.shape[0]) // group_size
        expanded = zeros[np.minimum(group_ids, zeros.shape[0] - 1)]
        unpacked = np.clip(unpacked - expanded - zero_point_offset + 8, 0, 15)
    return (np.ascontiguousarray(unpacked.T),
            np.ascontiguousarray(scales.astype(np.float16).T))


def _marlin_indices() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pack_index = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int32)
    output_index = np.concatenate(
        [np.arange(offset, 128, 4) for offset in range(4)]).astype(np.int32)
    row_pattern = np.array([
        [0, 1, 8, 9, 0, 1, 8, 9],
        [2, 3, 10, 11, 2, 3, 10, 11],
        [4, 5, 12, 13, 4, 5, 12, 13],
        [6, 7, 14, 15, 6, 7, 14, 15],
    ],
                           dtype=np.int32)
    row_index = np.tile(row_pattern, (32, 1))
    column_rows = []
    for base in (0, 16, 32, 48):
        for offset in range(8):
            row = [base + offset] * 4 + [base + offset + 8] * 4
            column_rows.extend([row] * 4)
    column_index = np.asarray(column_rows, dtype=np.int32)
    return pack_index, output_index, row_index, column_index


def _permute_marlin_scales(scales: np.ndarray, size_k: int, size_n: int,
                           group_size: int) -> np.ndarray:
    scale_permutation = np.array(
        [index + 8 * block for index in range(8) for block in range(8)])
    single_permutation = np.array([
        2 * index + offset for index in range(4)
        for offset in (0, 1, 8, 9, 16, 17, 24, 25)
    ])
    permutation = (scale_permutation if group_size < size_k
                   and group_size != -1 else single_permutation)
    return scales.reshape(-1,
                          len(permutation))[:,
                                            permutation].reshape(-1, size_n)


def pack_moe_marlin(weights: np.ndarray, scales: np.ndarray,
                    group_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pack ``[E,N,K]`` GPTQ nibbles and ``[E,N,G]`` scales for Int4MoE."""
    num_experts, n_dim, k_dim = weights.shape
    if k_dim % 16 or n_dim % 64:
        raise ValueError(
            f"Marlin MoE packing requires K%16=0 and N%64=0, got {(k_dim, n_dim)}"
        )
    pack_index, output_index, row_index, column_index = _marlin_indices()
    packed_experts = []
    for expert_index in range(num_experts):
        weight_kn = np.ascontiguousarray(weights[expert_index].T,
                                         dtype=np.uint32)
        k_tiles, n_tiles = k_dim // 16, n_dim // 64
        tiles = weight_kn.reshape(k_tiles, 16, n_tiles,
                                  64).transpose(0, 2, 1, 3)
        gathered = tiles[:, :, row_index,
                         column_index][:, :, :, pack_index].astype(np.uint32)
        packed = sum(gathered[:, :, :, index] << (4 * index)
                     for index in range(8))
        output = np.zeros((k_tiles, n_tiles * 128), dtype=np.uint32)
        for tile_index in range(n_tiles):
            output[:, tile_index * 128 + output_index] = packed[:,
                                                                tile_index, :]
        packed_experts.append(output.view(np.int32))
    packed_weights = np.stack(packed_experts).view(np.int8)
    packed_scales = np.ascontiguousarray(scales.transpose(0, 2, 1))
    for expert_index in range(num_experts):
        packed_scales[expert_index] = _permute_marlin_scales(
            packed_scales[expert_index], k_dim, n_dim, group_size)
    return np.ascontiguousarray(packed_weights), packed_scales.astype(
        np.float16)
