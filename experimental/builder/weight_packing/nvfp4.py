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
"""NumPy helpers for decoding checkpoint NVFP4 tensors and packing MoE
weights into FC1/FC2 FP4 buffers and six-dimensional block-scale layouts."""

import numpy as np

from ..core.numpy_dtypes import (FP4_E2M1_LEVELS, f32_to_fp8_e4m3_bytes,
                                 fp8_e4m3_bytes_to_f32, round_f32_to_bf16)
from .int4 import pack_moe_marlin

__all__ = [
    "decode_modelopt_nvfp4",
    "swizzle_nvfp4_mma_scales",
    "pack_nvfp4_moe_weight",
    "pack_nvfp4_a16_linear",
    "pack_nvfp4_a16_blackwell_linear",
    "pack_nvfp4_a16_nongated_experts",
    "pack_gated_nvfp4_experts",
]

# Midpoints between consecutive E2M1 levels (for searchsorted quantization).
_E2M1_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                        dtype=np.float32)
_NVFP4_GROUP_SIZE = 16
_NVFP4_MARLIN_GLOBAL_SCALE_EXP = 7


def _unpack_nvfp4_codes(packed: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2] = packed & np.uint8(0x0F)
    codes[:, 1::2] = (packed >> np.uint8(4)) & np.uint8(0x0F)
    return codes


def _marlin_permute_nvfp4_scales(scale_bytes: np.ndarray) -> np.ndarray:
    scales = np.ascontiguousarray(scale_bytes, dtype=np.uint8).view(np.int8)
    n_padded, num_groups = scales.shape
    if n_padded % 64:
        raise ValueError(f"N_pad={n_padded} must be divisible by 64")
    permutation = []
    for output_column in range(64):
        row, column = divmod(output_column, 8)
        swapped_row = ((row & 0x4) | ((row & 0x1) << 1) | ((row >> 1) & 0x1))
        permutation.append(8 * column + swapped_row)
    full_permutation = np.concatenate([
        block * 64 + np.asarray(permutation, dtype=np.int64)
        for block in range(n_padded // 64)
    ])
    result = np.empty((1, num_groups, n_padded), dtype=np.int8)
    result[0][:, full_permutation] = scales.T
    return np.ascontiguousarray(result)


def pack_nvfp4_a16_linear(packed: np.ndarray,
                          scale_bytes: np.ndarray,
                          global_scale: float,
                          pad_n_to: int = 128):
    """Repack one ModelOpt W4A16 linear for the Marlin A16 plugin."""
    codes = _unpack_nvfp4_codes(packed)
    n_logical, k_dim = codes.shape
    if k_dim % _NVFP4_GROUP_SIZE:
        raise ValueError(f"K={k_dim} must be divisible by 16")
    scales = np.ascontiguousarray(scale_bytes, dtype=np.uint8)
    expected_scales = (n_logical, k_dim // _NVFP4_GROUP_SIZE)
    if scales.shape != expected_scales:
        raise ValueError(
            f"weight scale shape {scales.shape} != {expected_scales}")
    n_padded = ((n_logical + pad_n_to - 1) // pad_n_to) * pad_n_to
    if n_padded != n_logical:
        codes = np.pad(codes, ((0, n_padded - n_logical), (0, 0)))
        scales = np.pad(scales, ((0, n_padded - n_logical), (0, 0)))
    dummy_scales = np.ones((1, n_padded, k_dim // _NVFP4_GROUP_SIZE),
                           dtype=np.float16)
    qweights, _ = pack_moe_marlin(codes[np.newaxis], dummy_scales,
                                  _NVFP4_GROUP_SIZE)
    block_scales = _marlin_permute_nvfp4_scales(scales)
    plugin_scale = np.asarray(
        [float(global_scale) * 2**_NVFP4_MARLIN_GLOBAL_SCALE_EXP],
        dtype=np.float16)
    return qweights, block_scales, plugin_scale, n_logical, n_padded


def pack_nvfp4_a16_blackwell_linear(packed: np.ndarray,
                                    scale_bytes: np.ndarray,
                                    global_scale: float,
                                    pad_n_to: int = 128):
    """Repack one ModelOpt W4A16 linear for the SM110 A16 plugin."""
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scale_bytes, dtype=np.uint8).view(np.int8)
    n_logical, k_half = packed.shape
    k_dim = k_half * 2
    if pad_n_to <= 0 or pad_n_to % 128:
        raise ValueError("pad_n_to must be a positive multiple of 128")
    if k_dim <= 0 or k_dim % 64:
        raise ValueError(f"K={k_dim} must be a positive multiple of 64")
    expected_scales = (n_logical, k_dim // _NVFP4_GROUP_SIZE)
    if scales.shape != expected_scales:
        raise ValueError(
            f"weight scale shape {scales.shape} != {expected_scales}")
    n_padded = ((n_logical + pad_n_to - 1) // pad_n_to) * pad_n_to
    if n_padded != n_logical:
        packed = np.pad(packed, ((0, n_padded - n_logical), (0, 0)))
        scales = np.pad(scales, ((0, n_padded - n_logical), (0, 0)))
    n_tiles = n_padded // 128
    k_tiles = k_dim // 64
    qweights = packed.reshape(n_tiles, 128, k_tiles,
                              32).transpose(0, 2, 1, 3).copy().view(np.int8)
    block_scales = scales.reshape(n_tiles, 128, k_tiles,
                                  4).transpose(0, 2, 1, 3).copy()
    return (np.ascontiguousarray(qweights), np.ascontiguousarray(block_scales),
            np.asarray([global_scale], dtype=np.float32), n_logical, n_padded)


def pack_nvfp4_a16_nongated_experts(load_expert, num_experts: int,
                                    hidden_size: int, intermediate_size: int):
    """Stack non-gated ModelOpt W4A16 experts for Nvfp4A16MoePlugin."""
    padded_intermediate = ((intermediate_size + 127) // 128) * 128
    fc1_qweights, fc1_scales, fc1_global = [], [], []
    fc2_qweights, fc2_scales, fc2_global = [], [], []
    for expert_index in range(num_experts):
        expert = load_expert(expert_index)
        qweight, scales, global_scale, _, padded = pack_nvfp4_a16_linear(
            expert["up_packed"], expert["up_sf"], expert["up_alpha"])
        if padded != padded_intermediate:
            raise ValueError(f"FC1 padded N {padded} != {padded_intermediate}")
        fc1_qweights.append(qweight[0])
        fc1_scales.append(scales[0])
        fc1_global.append(global_scale[0])

        down_packed = np.ascontiguousarray(expert["down_packed"],
                                           dtype=np.uint8)
        down_scales = np.ascontiguousarray(expert["down_sf"], dtype=np.uint8)
        if padded_intermediate != intermediate_size:
            down_packed = np.pad(
                down_packed,
                ((0, 0), (0, (padded_intermediate - intermediate_size) // 2)))
            down_scales = np.pad(
                down_scales,
                ((0, 0), (0, (padded_intermediate - intermediate_size) // 16)))
        qweight, scales, global_scale, logical, padded = \
            pack_nvfp4_a16_linear(down_packed, down_scales,
                                   expert["down_alpha"])
        if logical != hidden_size or padded != hidden_size:
            raise ValueError("FC2 hidden size must already be 128-aligned")
        fc2_qweights.append(qweight[0])
        fc2_scales.append(scales[0])
        fc2_global.append(global_scale[0])
    return {
        "fc1_qweights": np.stack(fc1_qweights),
        "fc1_block_scales": np.stack(fc1_scales),
        "fc1_global_scales": np.asarray(fc1_global, dtype=np.float16),
        "fc2_qweights": np.stack(fc2_qweights),
        "fc2_block_scales": np.stack(fc2_scales),
        "fc2_global_scales": np.asarray(fc2_global, dtype=np.float16),
        "padded_intermediate": padded_intermediate,
    }


def decode_modelopt_nvfp4(packed: np.ndarray,
                          scale_fp8_bytes: np.ndarray,
                          weight_scale_2: float,
                          group_size: int = 16) -> np.ndarray:
    """Dequantize one ModelOpt NVFP4 weight tensor to dense fp32 ``[out, in]``.

    * ``packed`` -- uint8 ``[out, in//2]`` (two FP4 E2M1 nibbles per byte).
    * ``scale_fp8_bytes`` -- raw FP8 E4M3 bytes ``[out, in//group_size]``.
    * ``weight_scale_2`` -- per-tensor scalar (fp32).
    """
    w = np.ascontiguousarray(packed, dtype=np.uint8)
    out_f, half = w.shape
    lo = w & np.uint8(0x0F)
    hi = (w >> np.uint8(4)) & np.uint8(0x0F)
    nibbles = np.empty((out_f, half * 2), dtype=np.uint8)
    nibbles[:, 0::2] = lo
    nibbles[:, 1::2] = hi
    sign = (nibbles & np.uint8(0x08)) != 0
    magnitude = nibbles & np.uint8(0x07)
    values = FP4_E2M1_LEVELS[magnitude]
    values = np.where(sign, -values, values).astype(np.float32)

    ws_fp32 = fp8_e4m3_bytes_to_f32(
        np.ascontiguousarray(scale_fp8_bytes, dtype=np.uint8))
    num_groups = ws_fp32.shape[-1]
    in_f = num_groups * group_size
    if values.shape != (out_f, in_f):
        raise ValueError(f"nibble shape {values.shape} != (out={out_f}, "
                         f"num_groups*group_size={in_f})")
    grouped = values.reshape(out_f, num_groups, group_size)
    dense = grouped * ws_fp32[..., np.newaxis]
    dense = dense.reshape(out_f, in_f) * float(weight_scale_2)
    return dense.astype(np.float32)


def swizzle_nvfp4_mma_scales(scale_bytes: np.ndarray, m_dim: int,
                             k_sf_dim: int) -> np.ndarray:
    """Swizzle linear FP8 block scales to the six-dimensional MMA layout."""
    sf = np.ascontiguousarray(scale_bytes, dtype=np.uint8)
    if sf.shape != (m_dim, k_sf_dim):
        raise ValueError(f"scale shape {sf.shape} != ({m_dim}, {k_sf_dim})")
    m_tiles = (m_dim + 127) // 128
    k_tiles = (k_sf_dim + 3) // 4
    padded_m = m_tiles * 128
    padded_k_sf = k_tiles * 4
    sf_padded = np.zeros((padded_m, padded_k_sf), dtype=np.uint8)
    sf_padded[:m_dim, :k_sf_dim] = sf
    sf_5d = sf_padded.reshape(m_tiles, 4, 32, k_tiles, 4)
    return sf_5d.transpose(0, 3, 2, 1, 4).copy().view(np.int8)


def pack_nvfp4_moe_weight(dense_w_mk: np.ndarray, group_size: int = 16):
    """Pack dense ``[M, K]`` fp32 weights for ``Nvfp4MoePlugin``.

    Returns ``(qweights int8 [M, K/2], blocks_scale int8 [m_tiles,k_tiles,32,4,4])``.
    """
    if group_size != 16:
        raise NotImplementedError("Nvfp4MoePlugin requires group_size=16")
    m_dim, k_dim = dense_w_mk.shape
    if k_dim % group_size != 0 or k_dim % 2 != 0:
        raise ValueError(f"K ({k_dim}) must be a multiple of {group_size} "
                         "and even")
    dense = round_f32_to_bf16(np.ascontiguousarray(dense_w_mk, np.float32))
    k_sf_dim = k_dim // group_size
    dense_blocks = dense.reshape(m_dim, k_sf_dim, group_size)
    block_scales = np.maximum(np.abs(dense_blocks).max(axis=-1) / 6.0,
                              1e-12).astype(np.float32)

    scaled = (dense_blocks / block_scales[..., np.newaxis]).clip(-6.0, 6.0)
    abs_idx = np.searchsorted(_E2M1_BOUNDS, np.abs(scaled)).astype(np.uint8)
    sign_bit = (scaled < 0).astype(np.uint8) << np.uint8(3)
    nibbles = (abs_idx | sign_bit).reshape(m_dim, k_dim)

    lo = nibbles[:, 0::2]
    hi = nibbles[:, 1::2]
    qweights = (lo | (hi << np.uint8(4))).astype(np.uint8).view(np.int8)

    sf_bytes = f32_to_fp8_e4m3_bytes(block_scales)
    blocks_scale = swizzle_nvfp4_mma_scales(sf_bytes, m_dim, k_sf_dim)
    return np.ascontiguousarray(qweights), np.ascontiguousarray(blocks_scale)


def _gated_fc1_rows(up: np.ndarray, gate: np.ndarray,
                    layout: str) -> np.ndarray:
    """Arrange matching UP/GATE rows for one gated MoE FC1 input."""
    if up.shape != gate.shape:
        raise ValueError(f"UP/GATE shape mismatch: {up.shape} vs {gate.shape}")
    if layout == "concat":
        return np.ascontiguousarray(np.concatenate((up, gate), axis=0))
    if layout != "interleave":
        raise ValueError(f"unsupported gated FC1 layout {layout!r}")
    intermediate, width = up.shape
    if intermediate % 64:
        raise ValueError(
            "interleaved NVFP4 FC1 requires intermediate_size % 64 == 0")
    chunks = intermediate // 64
    return np.ascontiguousarray(
        np.stack(
            (up.reshape(chunks, 64, width), gate.reshape(chunks, 64, width)),
            axis=1).reshape(2 * intermediate, width))


def pack_gated_nvfp4_experts(load_expert,
                             num_experts: int,
                             hidden_size: int,
                             intermediate_size: int,
                             group_size: int,
                             fc1_layout: str,
                             plugin_intermediate_size: int | None = None):
    """Normalize and arrange provider-packed NVFP4 gated experts.

    ``load_expert(index)`` returns raw ModelOpt FP4 bytes, FP8 scale bytes,
    and scalar weight scales for gate/up/down. Requantization absorbs those
    per-projection scales into the plugin block scales and emits unit alphas.
    """
    if group_size != 16:
        raise NotImplementedError("Nvfp4MoePlugin requires group_size=16")
    plugin_intermediate_size = (intermediate_size if plugin_intermediate_size
                                is None else plugin_intermediate_size)
    if (plugin_intermediate_size < intermediate_size
            or plugin_intermediate_size % 64):
        raise ValueError(
            "plugin intermediate size must be at least the checkpoint size "
            "and divisible by 64")

    fc1_weights = []
    fc1_scales = []
    fc1_alpha = []
    fc2_weights = []
    fc2_scales = []
    fc2_alpha = []
    for expert_index in range(num_experts):
        expert = load_expert(expert_index)
        gate = expert["gate"]
        up = expert["up"]
        down = expert["down"]
        expected_fc1_weight = (intermediate_size, hidden_size // 2)
        expected_fc1_scale = (intermediate_size, hidden_size // group_size)
        expected_fc2_weight = (hidden_size, intermediate_size // 2)
        expected_fc2_scale = (hidden_size, intermediate_size // group_size)
        for projection, expected_weight, expected_scale in (
            (gate, expected_fc1_weight, expected_fc1_scale),
            (up, expected_fc1_weight, expected_fc1_scale),
            (down, expected_fc2_weight, expected_fc2_scale),
        ):
            if projection["packed"].shape != expected_weight:
                raise ValueError(
                    f"NVFP4 weight shape {projection['packed'].shape} != "
                    f"{expected_weight}")
            if projection["sf"].shape != expected_scale:
                raise ValueError(
                    f"NVFP4 scale shape {projection['sf'].shape} != "
                    f"{expected_scale}")
        padding = plugin_intermediate_size - intermediate_size
        up_dense = decode_modelopt_nvfp4(up["packed"], up["sf"], up["alpha"],
                                         group_size)
        gate_dense = decode_modelopt_nvfp4(gate["packed"], gate["sf"],
                                           gate["alpha"], group_size)
        up_dense = np.pad(up_dense, ((0, padding), (0, 0)))
        gate_dense = np.pad(gate_dense, ((0, padding), (0, 0)))
        fc1_dense = _gated_fc1_rows(up_dense, gate_dense, fc1_layout)
        fc1_weight, fc1_scale = pack_nvfp4_moe_weight(fc1_dense, group_size)
        fc1_weights.append(fc1_weight)
        fc1_scales.append(fc1_scale)
        fc1_alpha.append(np.float32(1.0))

        down_dense = decode_modelopt_nvfp4(down["packed"], down["sf"],
                                           down["alpha"], group_size)
        down_dense = np.pad(
            down_dense,
            ((0, 0), (0, padding)),
        )
        fc2_weight, fc2_scale = pack_nvfp4_moe_weight(down_dense, group_size)
        fc2_weights.append(fc2_weight)
        fc2_scales.append(fc2_scale)
        fc2_alpha.append(np.float32(1.0))

    return (
        np.stack(fc1_weights),
        np.stack(fc1_scales),
        np.asarray(fc1_alpha, dtype=np.float32),
        np.stack(fc2_weights),
        np.stack(fc2_scales),
        np.asarray(fc2_alpha, dtype=np.float32),
    )


# Re-expose decoders used elsewhere.
_ = fp8_e4m3_bytes_to_f32
