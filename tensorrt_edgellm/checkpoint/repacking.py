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
Post-load weight repacking for quantized linear layers.

Transforms checkpoint weight formats (AWQ column-packed int32, GPTQ row-packed
int32, ModelOpt uint8, ModelOpt NVFP4) into the per-plugin layout expected by
TensorRT (int4 GEMM plugin or NVFP4 MoE plugin).  All functions operate
in-place on ``module._buffers`` or return new tensors; they are called by
:func:`~loader.load_weights` after all checkpoint tensors have been assigned.
"""

import logging
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

__all__ = [
    "repack_awq_to_plugin",
    "repack_gptq_to_plugin",
    "decode_modelopt_nvfp4",
    "repack_nvfp4_qwen3_moe_experts_thor",
    "repack_nvfp4_qwen3_moe_experts_geforce",
    "repack_nvfp4_expert_up",
    "repack_nvfp4_expert_down",
    "repack_nvfp4_expert_up_prefill_raw",
    "repack_nvfp4_expert_down_prefill_raw",
]

# ---------------------------------------------------------------------------
# AWQ weight swizzle
# ---------------------------------------------------------------------------


def repack_awq_to_plugin(qweight: torch.Tensor,
                         qzeros: torch.Tensor) -> torch.Tensor:
    """Repack AWQ qweight from [in, out//8] int32 to [out//2, in] int8.

    AWQ packs 8 int4 nibbles per int32 along the output axis::

        int32 = (n7 << 28) | (n6 << 24) | ... | (n0 << 0)
        where n_k = output channel (8*col + k), value in [0, 15]

    The int4 GEMM kernel uses ``(nibble - 8) * scale``.
    AWQ dequantizes as ``(nibble - qzero) * scale``.
    So we adjust each nibble: ``adjusted = nibble - qzero + 8``, baking the
    per-group zero-point into the weights before packing.

    Output ``[out//2, in]`` int8: K-block permute, even/odd shuffle within 8,
    N-row interleave, then four nibbles per int16 (viewed as two int8 rows).
    """
    in_features, out_div8 = qweight.shape
    out_features = out_div8 * 8
    group_size = in_features // qzeros.shape[0]

    qw = qweight.cpu().to(torch.int32)
    qz = qzeros.cpu().to(torch.int32)

    # AutoAWQ packs 8 nibbles per int32 in non-sequential output-channel order.
    # Bit position k within each packed int32 stores the value for output channel
    # _AWQ_BIT_TO_CH[k] within that group of 8, derived from AutoAWQ's
    # AWQ_REVERSE_ORDER = [0,4,1,5,2,6,3,7] (packing_utils.py): the inverse
    # permutation gives the output channel encoded at each bit position.
    # Without this reorder, output channels within each group of 8 are scrambled.
    _AWQ_BIT_TO_CH = [0, 2, 4, 6, 1, 3, 5, 7]

    # Extract weight nibbles: nibbles[in, out] = uint4 value in [0, 15]
    nibbles = torch.zeros(in_features, out_features, dtype=torch.int32)
    for k in range(8):
        nibbles[:, _AWQ_BIT_TO_CH[k]::8] = (qw >> (4 * k)) & 0xF

    # Extract zero-point nibbles: zeros[in//g, out] = uint4 in [0, 15]
    zeros = torch.zeros(in_features // group_size,
                        out_features,
                        dtype=torch.int32)
    for k in range(8):
        zeros[:, _AWQ_BIT_TO_CH[k]::8] = (qz >> (4 * k)) & 0xF

    # Expand zeros from [in//g, out] -> [in, out] by repeating each row group_size times
    zeros_expanded = zeros.repeat_interleave(group_size, dim=0)  # [in, out]

    # Adjust nibbles: kernel does (nibble - 8) * scale; AWQ does (nibble - qzero) * scale
    # So adjusted = nibble - qzero + 8 -> kernel result = (adjusted - 8) = (nibble - qzero)
    nibbles = (nibbles - zeros_expanded + 8).clamp(0, 15)

    # Transpose [in, out] -> [out, in] = [N, K] for pack_intweights
    nibbles_nk = nibbles.t().contiguous().numpy().astype(np.int16)  # [N, K]

    packed_int16 = _pack_intweights(nibbles_nk)  # [N//4, K] int16
    packed_int8 = packed_int16.view(np.int8).reshape(
        packed_int16.shape[0] * 2, packed_int16.shape[1])  # [N//2, K]

    return torch.tensor(packed_int8, dtype=torch.int8).to(qweight.device)


def _pack_intweights(unpacked_qweight: np.ndarray) -> np.ndarray:
    """Pack nibbles ``[N, K]`` int16 in ``[0, 15]`` to ``[N//4, K]`` int16 (int4 GEMM layout).

    Steps: permute within each 32-wide K block; even/odd reorder within each 8;
    interleave every four N rows across 64-wide K stripes; pack four nibbles per int16.
    """
    interleave = 4
    kstride = 64
    N, K = unpacked_qweight.shape

    # Step 1: Permute within K-blocks of 32
    # np.arange(32).reshape(4,4,2).transpose(1,0,2) -> [0,1,8,9,16,17,24,25,...]
    pk = unpacked_qweight.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 3, 2, 4)
    pk = pk.reshape(N, K // 32, 32)

    # Step 2: Within each group of 8, reorder [0,1,2,3,4,5,6,7] -> [0,2,4,6,1,3,5,7]
    pk = pk.reshape(N, K // 32, 4, 4, 2).transpose(0, 1, 2, 4, 3)
    pk = pk.reshape(N, K)

    # Step 3: Interleave every 4 rows (N dimension) across K-blocks of 64
    # Pad N to a multiple of interleave if necessary (e.g. router layers with N < 4)
    if N % interleave != 0:
        pad_n = interleave - (N % interleave)
        pk = np.pad(pk, ((0, pad_n), (0, 0)), mode='constant')
        N_padded = N + pad_n
    else:
        N_padded = N
    pk = pk.reshape(N_padded // interleave, interleave, K // kstride, kstride)
    pk = pk.transpose(0, 2, 1, 3)  # [N_padded//4, K//64, 4, 64]
    pk = pk.reshape(N_padded // interleave, K // kstride, kstride, interleave)

    # Step 4: Pack 4 nibbles per int16 (little-endian nibble order)
    pk = (pk[..., 0]
          | (pk[..., 1] << 4)
          | (pk[..., 2] << 8)
          | (pk[..., 3] << 12))
    packed = pk.reshape(N_padded // interleave, K).astype(np.int16)
    return packed


def _gather_rows_by_gidx_order(
    weight: torch.Tensor,
    g_idx: torch.Tensor,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reorder rows of ``weight`` (K major) so channels with the same ``g_idx`` group are contiguous."""
    group_num = int(weight.shape[0] / group_size)
    gmax = int(torch.max(g_idx).item())
    assert group_num == gmax + 1, (
        f"Group number {group_num} != max(g_idx)+1 ({gmax + 1})")
    indices_list = []
    for i in range(group_num):
        indices = torch.nonzero(g_idx == i, as_tuple=False).squeeze(1)
        indices_list.append(indices)
    permute_idx = torch.cat(indices_list, dim=0)
    new_weight = weight.index_select(0, permute_idx)
    assert new_weight.shape[0] == weight.shape[0]
    return new_weight, permute_idx


# ---------------------------------------------------------------------------
# GPTQ weight swizzle
# ---------------------------------------------------------------------------


def repack_gptq_to_plugin(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    g_idx: Optional[torch.Tensor] = None,
    zero_point_offset: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Repack GPTQ ``qweight`` ``[in//8, out]`` int32 to plugin ``[out//2, in]`` int8.

    Unpacks eight nibbles per int32 along K, applies GPTQ zero-point offset,
    optionally reorders K rows by ``g_idx`` (``desc_act``), transposes to ``[N, K]``,
    then :func:`_pack_intweights`.

    Returns:
        ``(qweight_out, int4_act_perm)`` — permute activations with
        ``x.index_select(-1, int4_act_perm)`` before the int4 GEMM op when non-trivial.
    """
    in_div8, out_features = qweight.shape
    in_features = in_div8 * 8

    qw = qweight.cpu().to(torch.int32)
    qz = qzeros.cpu().to(torch.int32)

    # Some symmetric GPTQ checkpoints (e.g. Qwen3.5 int4) omit zero points
    # entirely, storing ``qzeros`` as an empty ``[num_groups, 0]`` tensor.
    # Treat these as symmetric quantization with the implicit midpoint zero (8).
    symmetric = qz.numel() == 0
    if symmetric:
        if qz.dim() >= 1 and qz.shape[0] > 0:
            num_groups = qz.shape[0]
        elif g_idx is not None and g_idx.numel() > 0:
            num_groups = int(g_idx.max().item()) + 1
        else:
            num_groups = 1
    else:
        num_groups = qz.shape[0]
    group_size = in_features // num_groups

    # Extract weight nibbles: nibbles[in, out] = uint4 value in [0, 15]
    # GPTQ row-packs: bit k of column `in` is in row `in//8`, bit position 4*k
    nibbles = torch.zeros(in_features, out_features, dtype=torch.int32)
    for k in range(8):
        nibbles[k::8, :] = (qw >> (4 * k)) & 0xF

    # Extract zero-point nibbles: zeros[group, out] = uint4 in [0, 15]
    # qzeros is [num_groups, out//8] -- same column packing as AWQ qzeros
    if symmetric:
        # No stored zeros: actual_zero is the 4-bit midpoint 8, so stored_zero =
        # 8 - zero_point_offset makes the offset adjustment below a no-op.
        zeros = torch.full((num_groups, out_features),
                           8 - int(zero_point_offset),
                           dtype=torch.int32)
    else:
        zeros = torch.zeros(num_groups, out_features, dtype=torch.int32)
        for k in range(8):
            zeros[:, k::8] = (qz >> (4 * k)) & 0xF

    if g_idx is None:
        g_idx_t = torch.arange(in_features, dtype=torch.int32) // group_size
    else:
        g_idx_t = g_idx.cpu().to(torch.int32)
    # Expand zeros from [num_groups, out] -> [in, out] using per-channel group ids.
    zeros_expanded = zeros[g_idx_t.to(torch.int64)]  # [in, out]

    # GPTQ checkpoints differ on whether qzeros stores zero or zero-1.
    # actual_zero = stored_zero + zero_point_offset.
    # Adjust nibbles: kernel does (nibble - 8) * scale; GPTQ does (nibble - actual_zero) * scale
    # -> repacked = nibble - (stored_zero + zero_point_offset) + 8
    nibbles = (nibbles - zeros_expanded - int(zero_point_offset) + 8).clamp(
        0, 15)

    # Gather K rows by group (identity order when ``g_idx`` is sequential).
    nibbles, permute_idx = _gather_rows_by_gidx_order(nibbles, g_idx_t,
                                                      group_size)

    # Transpose [in, out] -> [out, in] = [N, K] for pack_intweights
    nibbles_nk = nibbles.t().contiguous().numpy().astype(np.int16)
    packed_int16 = _pack_intweights(nibbles_nk)
    packed_int8 = packed_int16.view(np.int8).reshape(packed_int16.shape[0] * 2,
                                                     packed_int16.shape[1])

    qw_out = torch.tensor(packed_int8, dtype=torch.int8).to(qweight.device)
    perm = permute_idx.to(torch.int64)
    return qw_out, perm


# ---------------------------------------------------------------------------
# Post-load cast / format fixups (called by load_weights)
# ---------------------------------------------------------------------------


def apply_all_repacking(model: nn.Module) -> None:
    """Apply all quantization repacking passes after checkpoint load.

    MoE expert stacking runs FIRST because it needs the original GPTQ int32
    weights (before regular repacking converts them to the swizzled plugin
    format).  After stacking, the per-expert GPTQLinear modules have their
    qweight set to None so ``_repack_gptq_weights`` skips them.
    """
    _stack_moe_experts(model)
    _repack_awq_weights(model)
    _repack_gptq_weights(model)
    _cast_modelopt_awq_prepacked(model)
    _cast_fp8_linear_scales(model)
    _cast_nvfp4_weights(model)


def _cast_modelopt_awq_prepacked(model: nn.Module) -> None:
    """Post-process W4A16 prepacked AWQ linear buffers after load.

    1. Unpack ``[N//2, K] uint8`` (two nibbles per byte) to nibbles, apply
       :func:`_pack_intweights`, store ``[N//2, K] int8``.
    2. Cast optional ``pre_quant_scale`` to float16 so forward() Mul stays in fp16.
    3. Transpose scales to ``[K//g, N]`` float16 for the int4 GEMM custom op.
    """
    from ..models.linear import ModelOptAWQPrepackedLinear  # local import
    for module in model.modules():
        if isinstance(module, ModelOptAWQPrepackedLinear):
            # 1. Repack weight: ModelOpt uint8[N//2, K] -> swizzled int8[N//2, K]
            w = module._buffers.get("weight")
            if w is not None and w.dtype == torch.uint8:
                w_cpu = w.cpu()
                N_half, K = w_cpu.shape
                N = N_half * 2
                # Unpack 2 nibbles per byte: low nibble -> even N rows, high -> odd N rows
                # ModelOpt pack_int4_in_uint8 stores weights using two's complement masking:
                # s in [-8,7] -> u = s & 0xF (so s=-8 -> u=8, s=0 -> u=0, s=7 -> u=7)
                # The plugin kernel uses (nibble - 8) * scale, so nibble must be s+8 in [0,15].
                # Convert: plugin_nibble = (u + 8) % 16
                w_i16 = w_cpu.to(torch.int16)
                nibbles = torch.zeros(N, K, dtype=torch.int16)
                nibbles[0::2] = w_i16 & 0xF  # even N channels = low nibble
                nibbles[1::2] = (
                    w_i16 >> 4) & 0xF  # odd N channels = high nibble
                nibbles = (nibbles +
                           8) % 16  # two's complement -> plugin convention
                nibbles_np = nibbles.numpy().astype(np.int16)
                packed_int16 = _pack_intweights(nibbles_np)  # [N//4, K] int16
                packed_int8 = packed_int16.view(np.int8).reshape(
                    packed_int16.shape[0] * 2, packed_int16.shape[1])
                module._buffers["weight"] = torch.tensor(packed_int8,
                                                         dtype=torch.int8).to(
                                                             w.device)

            sc = module._buffers.get("weight_scale")
            if sc is None:
                continue

            # 2. Cast pre_quant_scale to float16 so forward() Mul stays in fp16.
            # pre_quant_scale is an AWQ activation smoothing scale applied as
            # x_smooth = x * pqs before the GEMM (matches the reference pipeline
            # where DQ+MatMul patterns include a leading Mul(x, pqs) node).
            pqs = module._buffers.get("pre_quant_scale")
            sc_f32 = sc.to(torch.float32)  # work in fp32 for precision
            if pqs is not None and pqs.dtype != torch.float16:
                module._buffers["pre_quant_scale"] = pqs.to(torch.float16)

            # 3. Transpose [N, K//g] -> [K//g, N] and cast to float16
            module._buffers["weight_scale"] = sc_f32.t().contiguous().to(
                torch.float16)


def _cast_fp8_linear_scales(model: nn.Module) -> None:
    """Cast FP8Linear ``input_scale`` / ``weight_scale`` to float16 if needed."""
    from ..models.linear import FP8Linear  # local import to avoid circular dep
    for module in model.modules():
        if not isinstance(module, FP8Linear):
            continue
        for name in ("weight_scale", "input_scale"):
            t = module._buffers.get(name)
            if t is None or t.dtype == torch.float16:
                continue
            module._buffers[name] = t.to(torch.float16)


def _cast_nvfp4_weights(model: nn.Module) -> None:
    """View-cast NVFP4Linear weight buffers from uint8 to int8 in-place.

    Packed FP4 nibbles have the same bit pattern in both types.
    Some ONNX importers mishandle UINT8 weight initializers for block DQ; int8 works.
    """
    from ..models.linear import \
        NVFP4Linear  # local import to avoid circular dep
    for module in model.modules():
        if isinstance(module, NVFP4Linear):
            w = module._buffers.get("weight")
            if w is not None and w.dtype == torch.uint8:
                module._buffers["weight"] = w.view(torch.int8)


def _repack_awq_weights(model: nn.Module) -> None:
    """Swizzle ``AWQLinear.qweight`` after load (fold zeros; pack to int8 layout).

    Scales should already be ``[K//g, N]``; cast to float16 if needed.
    """
    from ..models.linear import AWQLinear  # local import to avoid circular dep
    for module in model.modules():
        if isinstance(module, AWQLinear):
            qw = module._buffers.get("qweight")
            qz = module._buffers.get("qzeros")
            if qw is not None and qw.dtype == torch.int32 and qz is not None:
                module._buffers["qweight"] = repack_awq_to_plugin(qw, qz)
                logger.debug("Repacked AWQ qweight: %s -> %s", list(qw.shape),
                             list(module._buffers["qweight"].shape))
            sc = module._buffers.get("scales")
            if sc is not None and sc.dtype != torch.float16:
                module._buffers["scales"] = sc.to(torch.float16)


def _repack_gptq_weights(model: nn.Module) -> None:
    """Swizzle ``GPTQLinear.qweight`` after load; set ``int4_act_perm`` for ``desc_act``."""
    from ..models.linear import \
        GPTQLinear  # local import to avoid circular dep
    for module in model.modules():
        if isinstance(module, GPTQLinear):
            qw = module._buffers.get("qweight")
            qz = module._buffers.get("qzeros")
            if qw is not None and qw.dtype == torch.int32 and qz is not None:
                logger.info("Repacking GPTQ module: %s, qweight=%s, qzeros=%s",
                            type(module).__name__, list(qw.shape), list(qz.shape))
                g_idx_buf = module._buffers.get("g_idx")
                packed, perm = repack_gptq_to_plugin(
                    qw, qz, g_idx_buf, getattr(module, "zero_point_offset", 1))
                module._buffers["qweight"] = packed
                module._buffers["int4_act_perm"] = perm
                logger.debug("Repacked GPTQ qweight: %s -> %s", list(qw.shape),
                             list(packed.shape))
            sc = module._buffers.get("scales")
            if sc is not None and sc.dtype != torch.float16:
                module._buffers["scales"] = sc.to(torch.float16)
    logger.info("Repacked GPTQ weights")


def _stack_moe_experts(model: nn.Module) -> None:
    """Stack per-expert weights into the layout required by the active MoE plugin.

    Walks every ``nn.Module`` in *model* and invokes ``_prepare_moe_weights``
    on every block that defines it. Each block decides its own packing path
    (Marlin for ``Int4MoePlugin``; CuTeDSL N-major for ``Nvfp4MoePlugin``;
    CuTeDSL 6D MMA for ``NvFP4MoEPluginGeforce``); this helper is
    backend-agnostic.

    Must run BEFORE ``_repack_gptq_weights`` because the GPTQ path needs
    the original int32-packed weights.  After extracting, per-expert
    qweight buffers are set to ``None`` so the regular GPTQ repacking
    skips them.
    """
    count = 0
    for module in model.modules():
        if hasattr(module, "_prepare_moe_weights"):
            module._prepare_moe_weights()
            count += 1
    if count:
        logger.info("Stacked expert weights for %d MoE block(s)", count)


# ---------------------------------------------------------------------------
# Marlin INT4 repacking for MoE experts
# ---------------------------------------------------------------------------
# Adapted from tensorrt_edgellm/llm_models/layers/int4_moe_plugin.py.
# These functions convert GPTQ int32-packed weights → Marlin layout consumed
# by trt_edgellm::Int4MoePlugin.
# ---------------------------------------------------------------------------


def _unpack_int4_gptq(qweight: torch.Tensor) -> torch.Tensor:
    """Unpack GPTQ ``[K//8, N]`` int32 → ``[K, N]`` int16 nibbles."""
    pack_factor = 8
    wf = torch.tensor(list(range(0, 32, 4)),
                      dtype=torch.int32).unsqueeze(0).to(qweight.device)
    weight = torch.bitwise_and(
        torch.bitwise_right_shift(
            qweight.unsqueeze(1).expand(-1, pack_factor, -1),
            wf.unsqueeze(-1).to(qweight.device)).to(torch.int16), 15)
    return weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])


def _unpack_qzeros_moe(qzeros: torch.Tensor) -> torch.Tensor:
    """Unpack GPTQ qzeros ``[num_groups, N//8]`` → ``[num_groups, N]``."""
    device = qzeros.device
    wf = torch.tensor([0, 4, 8, 12, 16, 20, 24, 28],
                      dtype=torch.int64,
                      device=device).view(1, 1, -1)
    z = qzeros.unsqueeze(2).expand(-1, -1, 8).to(torch.int64)
    return torch.bitwise_and(torch.bitwise_right_shift(z, wf),
                             15).reshape(qzeros.shape[0], -1)


def _extract_gptq_for_marlin(
    proj: nn.Module,
    group_size: int,
    zero_point_offset: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(weights [N, K] int16, scales [N, num_groups] fp16)`` from a
    GPTQ linear module, remapping zero-points so Marlin's ``(q - 8) * scale``
    equals GPTQ's ``(q - zero) * scale``.

    GPTQ checkpoints differ on whether qzeros stores ``zero_point`` or
    ``zero_point - 1``.  The adjustment is therefore::

        q_marlin = q - (stored_zero + zero_point_offset) + 8
    """
    unpacked = _unpack_int4_gptq(proj.qweight)  # [K, N]

    # Symmetric GPTQ checkpoints may omit zero points (``qzeros`` is ``None`` or
    # an empty ``[num_groups, 0]`` tensor); the implicit midpoint zero (8)
    # already matches Marlin's ``(q - 8) * scale``, so no remapping is needed.
    qzeros = getattr(proj, "qzeros", None)
    if qzeros is not None and qzeros.numel() > 0:
        zeros = _unpack_qzeros_moe(qzeros)  # [num_groups, N]
        K, N = unpacked.shape
        group_ids = torch.arange(K, device=unpacked.device) // group_size
        zeros_expanded = zeros[group_ids.clamp(max=zeros.shape[0] - 1)]
        # actual_zero = stored_zero + zero_point_offset
        unpacked = torch.clamp(
            unpacked.to(torch.int32) - zeros_expanded.to(torch.int32) -
            int(zero_point_offset) + 8, 0, 15).to(torch.int16)

    weights = unpacked.transpose(0, 1).contiguous()  # [N, K]
    scales = proj.scales.data.to(torch.float16).transpose(0, 1).contiguous()
    return weights, scales


# Pre-computed Marlin tensor core layout indices (from int4_moe_plugin.py).
_MARLIN_PACK_IDX = np.array([0, 2, 4, 6, 1, 3, 5, 7], dtype=np.int32)

# fmt: off
_MARLIN_OUT_IDX = np.array([
    0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60,
    64, 68, 72, 76, 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 124,
    1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61,
    65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125,
    2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62,
    66, 70, 74, 78, 82, 86, 90, 94, 98, 102, 106, 110, 114, 118, 122, 126,
    3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63,
    67, 71, 75, 79, 83, 87, 91, 95, 99, 103, 107, 111, 115, 119, 123, 127
], dtype=np.int32)

_ROW_PATTERN = np.array([
    [0, 1, 8, 9, 0, 1, 8, 9], [2, 3, 10, 11, 2, 3, 10, 11],
    [4, 5, 12, 13, 4, 5, 12, 13], [6, 7, 14, 15, 6, 7, 14, 15]
], dtype=np.int32)
_MARLIN_ROW_IDX = np.tile(_ROW_PATTERN, (32, 1))

_MARLIN_COL_IDX = np.array([
    [0,0,0,0,8,8,8,8],[0,0,0,0,8,8,8,8],[0,0,0,0,8,8,8,8],[0,0,0,0,8,8,8,8],
    [1,1,1,1,9,9,9,9],[1,1,1,1,9,9,9,9],[1,1,1,1,9,9,9,9],[1,1,1,1,9,9,9,9],
    [2,2,2,2,10,10,10,10],[2,2,2,2,10,10,10,10],[2,2,2,2,10,10,10,10],[2,2,2,2,10,10,10,10],
    [3,3,3,3,11,11,11,11],[3,3,3,3,11,11,11,11],[3,3,3,3,11,11,11,11],[3,3,3,3,11,11,11,11],
    [4,4,4,4,12,12,12,12],[4,4,4,4,12,12,12,12],[4,4,4,4,12,12,12,12],[4,4,4,4,12,12,12,12],
    [5,5,5,5,13,13,13,13],[5,5,5,5,13,13,13,13],[5,5,5,5,13,13,13,13],[5,5,5,5,13,13,13,13],
    [6,6,6,6,14,14,14,14],[6,6,6,6,14,14,14,14],[6,6,6,6,14,14,14,14],[6,6,6,6,14,14,14,14],
    [7,7,7,7,15,15,15,15],[7,7,7,7,15,15,15,15],[7,7,7,7,15,15,15,15],[7,7,7,7,15,15,15,15],
    [16,16,16,16,24,24,24,24],[16,16,16,16,24,24,24,24],[16,16,16,16,24,24,24,24],[16,16,16,16,24,24,24,24],
    [17,17,17,17,25,25,25,25],[17,17,17,17,25,25,25,25],[17,17,17,17,25,25,25,25],[17,17,17,17,25,25,25,25],
    [18,18,18,18,26,26,26,26],[18,18,18,18,26,26,26,26],[18,18,18,18,26,26,26,26],[18,18,18,18,26,26,26,26],
    [19,19,19,19,27,27,27,27],[19,19,19,19,27,27,27,27],[19,19,19,19,27,27,27,27],[19,19,19,19,27,27,27,27],
    [20,20,20,20,28,28,28,28],[20,20,20,20,28,28,28,28],[20,20,20,20,28,28,28,28],[20,20,20,20,28,28,28,28],
    [21,21,21,21,29,29,29,29],[21,21,21,21,29,29,29,29],[21,21,21,21,29,29,29,29],[21,21,21,21,29,29,29,29],
    [22,22,22,22,30,30,30,30],[22,22,22,22,30,30,30,30],[22,22,22,22,30,30,30,30],[22,22,22,22,30,30,30,30],
    [23,23,23,23,31,31,31,31],[23,23,23,23,31,31,31,31],[23,23,23,23,31,31,31,31],[23,23,23,23,31,31,31,31],
    [32,32,32,32,40,40,40,40],[32,32,32,32,40,40,40,40],[32,32,32,32,40,40,40,40],[32,32,32,32,40,40,40,40],
    [33,33,33,33,41,41,41,41],[33,33,33,33,41,41,41,41],[33,33,33,33,41,41,41,41],[33,33,33,33,41,41,41,41],
    [34,34,34,34,42,42,42,42],[34,34,34,34,42,42,42,42],[34,34,34,34,42,42,42,42],[34,34,34,34,42,42,42,42],
    [35,35,35,35,43,43,43,43],[35,35,35,35,43,43,43,43],[35,35,35,35,43,43,43,43],[35,35,35,35,43,43,43,43],
    [36,36,36,36,44,44,44,44],[36,36,36,36,44,44,44,44],[36,36,36,36,44,44,44,44],[36,36,36,36,44,44,44,44],
    [37,37,37,37,45,45,45,45],[37,37,37,37,45,45,45,45],[37,37,37,37,45,45,45,45],[37,37,37,37,45,45,45,45],
    [38,38,38,38,46,46,46,46],[38,38,38,38,46,46,46,46],[38,38,38,38,46,46,46,46],[38,38,38,38,46,46,46,46],
    [39,39,39,39,47,47,47,47],[39,39,39,39,47,47,47,47],[39,39,39,39,47,47,47,47],[39,39,39,39,47,47,47,47],
    [48,48,48,48,56,56,56,56],[48,48,48,48,56,56,56,56],[48,48,48,48,56,56,56,56],[48,48,48,48,56,56,56,56],
    [49,49,49,49,57,57,57,57],[49,49,49,49,57,57,57,57],[49,49,49,49,57,57,57,57],[49,49,49,49,57,57,57,57],
    [50,50,50,50,58,58,58,58],[50,50,50,50,58,58,58,58],[50,50,50,50,58,58,58,58],[50,50,50,50,58,58,58,58],
    [51,51,51,51,59,59,59,59],[51,51,51,51,59,59,59,59],[51,51,51,51,59,59,59,59],[51,51,51,51,59,59,59,59],
    [52,52,52,52,60,60,60,60],[52,52,52,52,60,60,60,60],[52,52,52,52,60,60,60,60],[52,52,52,52,60,60,60,60],
    [53,53,53,53,61,61,61,61],[53,53,53,53,61,61,61,61],[53,53,53,53,61,61,61,61],[53,53,53,53,61,61,61,61],
    [54,54,54,54,62,62,62,62],[54,54,54,54,62,62,62,62],[54,54,54,54,62,62,62,62],[54,54,54,54,62,62,62,62],
    [55,55,55,55,63,63,63,63],[55,55,55,55,63,63,63,63],[55,55,55,55,63,63,63,63],[55,55,55,55,63,63,63,63],
], dtype=np.int32)
# fmt: on


def _marlin_permute_scales(s, size_k, size_n, group_size):
    """Permute scale columns for Marlin kernel shared-memory read pattern."""
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend(
            [2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape((-1, size_n)).contiguous()


def pack_int4_awq_marlin(
    weights_q: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack INT4 ``[E, N, K]`` weights + ``[E, N, num_groups]`` scales to Marlin.

    Returns ``(weights_marlin [E, K//16, 2*N] int32,
               scales_marlin  [E, num_groups, N] fp16)``.
    """
    num_experts, N, K = weights_q.shape
    device = weights_q.device
    weights_marlin_list = []

    for expert_id in range(num_experts):
        w_np = weights_q[expert_id].transpose(
            0, 1).contiguous().cpu().numpy().astype(np.uint32)  # [K, N]

        k_tiles, n_tiles = K // 16, N // 64
        tiles = w_np.reshape(k_tiles, 16, n_tiles, 64).transpose(0, 2, 1, 3)
        gathered = tiles[:, :, _MARLIN_ROW_IDX,
                         _MARLIN_COL_IDX][:, :, :,
                                          _MARLIN_PACK_IDX].astype(np.uint32)

        packed_out = (gathered[:, :, :, 0] | (gathered[:, :, :, 1] << 4)
                      | (gathered[:, :, :, 2] << 8)
                      | (gathered[:, :, :, 3] << 12)
                      | (gathered[:, :, :, 4] << 16)
                      | (gathered[:, :, :, 5] << 20)
                      | (gathered[:, :, :, 6] << 24)
                      | (gathered[:, :, :, 7] << 28))

        out = np.zeros((k_tiles, n_tiles * 128), dtype=np.uint32)
        for n_tile_id in range(n_tiles):
            out[:,
                n_tile_id * 128 + _MARLIN_OUT_IDX] = packed_out[:,
                                                                n_tile_id, :]
        weights_marlin_list.append(
            torch.from_numpy(out.view(np.int32)).to(device))

    weights_marlin = torch.stack(weights_marlin_list, dim=0)

    scales_marlin = scales.transpose(1, 2).contiguous()  # [E, num_groups, N]
    for e in range(num_experts):
        scales_marlin[e] = _marlin_permute_scales(scales_marlin[e], K, N,
                                                  group_size)

    return weights_marlin, scales_marlin


# ---------------------------------------------------------------------------
# NVFP4 MoE Marlin tile pack
# ---------------------------------------------------------------------------

_FP8_MAX = 448.0
_FP4_E2M1_POSITIVE_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                     dtype=np.float32)
# Midpoints between consecutive E2M1 levels (for searchsorted-based quantization).
_E2M1_BOUNDS = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
                        dtype=np.float32)


def decode_modelopt_nvfp4(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    group_size: int = 16,
) -> np.ndarray:
    """Dequantize one ModelOpt NVFP4 weight tensor to dense fp32 ``[out, in]``.

    ``weight`` is ``[out, in//2]`` int8/uint8 with two FP4 E2M1 nibbles per
    byte (low nibble = even index).  ``weight_scale`` is ``[out, in//group_size]``
    FP8 E4M3 (accepts ``float8_e4m3fn``, an int8 view of it, or a float cast).
    ``weight_scale_2`` is ``[1]`` fp32 per-tensor scale-of-scale.
    """
    w = weight.detach().cpu().numpy()
    if w.dtype == np.int8:
        w = w.view(np.uint8)
    if w.dtype != np.uint8:
        raise TypeError(f"unexpected weight dtype {w.dtype}")
    out_f, half = w.shape

    lo = w & np.uint8(0x0F)
    hi = (w >> np.uint8(4)) & np.uint8(0x0F)
    nibbles = np.empty((out_f, half * 2), dtype=np.uint8)
    nibbles[:, 0::2] = lo
    nibbles[:, 1::2] = hi
    sign = (nibbles & np.uint8(0x08)) != 0
    magnitude = nibbles & np.uint8(0x07)
    values = _FP4_E2M1_POSITIVE_LEVELS[magnitude]
    values = np.where(sign, -values, values).astype(np.float32)

    if weight_scale.dtype == torch.float8_e4m3fn:
        ws_fp32 = weight_scale.detach().to(torch.float32).cpu().numpy()
    elif weight_scale.dtype == torch.int8:
        ws_fp32 = (weight_scale.detach().view(torch.float8_e4m3fn).to(
            torch.float32).cpu().numpy())
    elif weight_scale.dtype in (torch.float32, torch.float16, torch.bfloat16):
        ws_fp32 = weight_scale.detach().to(torch.float32).cpu().numpy()
    else:
        raise TypeError(f"unsupported weight_scale dtype {weight_scale.dtype}")

    ws2 = float(weight_scale_2.detach().reshape(-1)[0].item())

    num_groups = ws_fp32.shape[-1]
    in_f = num_groups * group_size
    if values.shape != (out_f, in_f):
        raise ValueError(f"nibble shape {values.shape} does not match "
                         f"(out={out_f}, num_groups*group_size={in_f})")
    values_grouped = values.reshape(out_f, num_groups, group_size)
    dense = values_grouped * ws_fp32[..., np.newaxis]
    dense = dense.reshape(out_f, in_f)
    dense *= ws2
    return dense.astype(np.float32)


def _round_dense_to_bf16(dense: np.ndarray) -> np.ndarray:
    """Round a dense fp32 weight through BF16 while returning fp32 storage."""
    dense_t = torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32))
    return dense_t.to(torch.bfloat16).to(torch.float32).cpu().numpy()


def _swizzle_nvfp4_geforce_mma_scales(scale_bytes: np.ndarray, m_dim: int,
                                      k_sf_dim: int) -> np.ndarray:
    """Swizzle linear FP8 block scales to CuTeDSL's 6D MMA layout."""
    if scale_bytes.dtype == np.int8:
        sf = scale_bytes.view(np.uint8)
    elif scale_bytes.dtype == np.uint8:
        sf = scale_bytes
    else:
        raise TypeError(f"unexpected scale dtype {scale_bytes.dtype}")
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


def _pack_nvfp4_geforce_moe_weight(
        dense_w_mk: np.ndarray,
        group_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack dense ``[M, K]`` weights for ``NvFP4MoEPluginGeforce``.

    Returns ``(qweights [M, K/2] int8,
    blocks_scale [m_tiles, k_tiles, 32, 4, 4] int8)``.  The scale tensor
    stores raw FP8 E4M3 block scales in the physical CuTeDSL MMA layout.
    """
    if group_size != 16:
        raise NotImplementedError(
            "NvFP4MoEPluginGeforce requires group_size=16")

    m_dim, k_dim = dense_w_mk.shape
    if k_dim % group_size != 0 or k_dim % 2 != 0:
        raise ValueError(
            f"K ({k_dim}) must be a multiple of {group_size} and even")

    dense = _round_dense_to_bf16(dense_w_mk)
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

    sf_bytes = torch.from_numpy(block_scales.copy()).to(
        torch.float8_e4m3fn).view(torch.uint8).cpu().numpy()
    blocks_scale = _swizzle_nvfp4_geforce_mma_scales(sf_bytes, m_dim, k_sf_dim)

    return (torch.from_numpy(qweights.copy()),
            torch.from_numpy(blocks_scale.copy()))


def repack_nvfp4_qwen3_moe_experts_geforce(
    experts: Iterable[nn.Module],
    hidden_size: int,
    moe_inter_size: int,
    group_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack Qwen3 NVFP4 experts for ``NvFP4MoEPluginGeforce``.

    Each expert is expected to contain ModelOpt ``NVFP4Linear`` gate/up/down
    projection tensors.  Dense weights are decoded, rounded through BF16, and
    repacked for the CuTeDSL GeForce plugin.  FC1 uses SwiGLU order
    ``[up, gate]``.
    """
    from ..models.linear import \
        NVFP4Linear  # local import to avoid circular dep

    fc1_qweights = []
    fc1_blocks_scale = []
    fc2_qweights = []
    fc2_blocks_scale = []

    for expert in experts:
        gate = expert.gate_proj
        up = expert.up_proj
        down = expert.down_proj
        if not (isinstance(gate, NVFP4Linear) and isinstance(up, NVFP4Linear)
                and isinstance(down, NVFP4Linear)):
            raise TypeError("Qwen3 NVFP4 MoE experts must use NVFP4Linear")

        gate_dense = decode_modelopt_nvfp4(gate.weight, gate.weight_scale,
                                           gate.weight_scale_2, group_size)
        up_dense = decode_modelopt_nvfp4(up.weight, up.weight_scale,
                                         up.weight_scale_2, group_size)
        down_dense = decode_modelopt_nvfp4(down.weight, down.weight_scale,
                                           down.weight_scale_2, group_size)

        if gate_dense.shape != (moe_inter_size, hidden_size):
            raise ValueError(f"gate dense shape {gate_dense.shape} != "
                             f"({moe_inter_size}, {hidden_size})")
        if up_dense.shape != (moe_inter_size, hidden_size):
            raise ValueError(f"up dense shape {up_dense.shape} != "
                             f"({moe_inter_size}, {hidden_size})")
        if down_dense.shape != (hidden_size, moe_inter_size):
            raise ValueError(f"down dense shape {down_dense.shape} != "
                             f"({hidden_size}, {moe_inter_size})")

        fc1_dense = np.concatenate([up_dense, gate_dense], axis=0)
        fc1_qw, fc1_sf = _pack_nvfp4_geforce_moe_weight(fc1_dense, group_size)
        fc2_qw, fc2_sf = _pack_nvfp4_geforce_moe_weight(down_dense, group_size)
        fc1_qweights.append(fc1_qw)
        fc1_blocks_scale.append(fc1_sf)
        fc2_qweights.append(fc2_qw)
        fc2_blocks_scale.append(fc2_sf)

    return (torch.stack(fc1_qweights,
                        dim=0), torch.stack(fc1_blocks_scale, dim=0),
            torch.stack(fc2_qweights,
                        dim=0), torch.stack(fc2_blocks_scale, dim=0))


def repack_nvfp4_qwen3_moe_experts_thor(
    experts: Iterable[nn.Module],
    hidden_size: int,
    moe_inter_size: int,
    group_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack Qwen3 NVFP4 gated experts for ``Nvfp4MoePlugin`` (CuTeDSL N-major path, SM100/101/110).

    Both the prefill grouped GEMM (``NvFP4MoEContiguousGemmRunner`` →
    ``cute_dsl_nvfp4_moe_fc{1,2}_*`` AOT cubins) and the decode GEMV
    (``CuteDslDecodeGemvRunner`` → ``gemv_up_swiglu`` / ``gemv_dn_none``
    AOT cubins) read from a single shared weight buffer in the N-major
    byte layout — N axis innermost, two FP4 nibbles packed per byte along
    N. (The byte layout's name is historical: it was originally chosen to
    match the now-deleted Marlin decode kernel, but every kernel the
    plugin actually launches today is CuTeDSL.)

    The plugin runs gated SwiGLU as a single fused FC1 over 32-column
    interleaved ``(up, gate)`` chunks with ``moe_inter_size = 2 * I`` and
    ``activation_type = 1`` (SiLU). Per expert we

    1. decode each NVFP4 weight to dense FP32 (:func:`decode_modelopt_nvfp4`),
    2. interleave matching 32-output-column chunks as ``[up_chunk, gate_chunk]``
       along the FC1 N axis, then transpose to ``[H, 2*I]``,
    3. run :func:`_nvfp4_pack_n_major` on the merged FC1 weight ``[H, 2*I]``
       and on the transposed down weight ``[I, H]``.

    A dequant-then-requant round-trip is unavoidable because gate and up
    have independent ``weight_scale_2`` values in the checkpoint, which makes
    the layout-only ``_pack_nvfp4_raw_n_major`` path unsuitable for the
    merged FC1 weight.

    :return: 8 stacked tensors mapping to the ``Nvfp4MoePlugin`` v1 input
        slots ``[3..10]``:

        * ``fc_up_qweights``               ``[E, H, I]``  int8
        * ``fc_up_blocks_scale``           ``[E, padUp(2*I, 128), padUp(H/16, 4)]``  int8 (atom layout, raw FP8 E4M3)
        * ``fc_up_global_scale``           ``[E]``        float32
        * ``fc_down_qweights``             ``[E, I, H/2]``  int8
        * ``fc_down_blocks_scale``         ``[E, padUp(H, 128), padUp(I/16, 4)]``    int8 (atom layout, raw FP8 E4M3)
        * ``fc_down_global_scale``         ``[E]``        float32
        * ``fc_up_blocks_scale_decode``    ``[E, H/16, 2*I]``  int8 (decode slot; INT8-only validated by the plugin)
        * ``fc_down_blocks_scale_decode``  ``[E, I/16, H]``    int8 (decode slot; INT8-only validated by the plugin)
    """
    from ..models.linear import \
        NVFP4Linear  # local import to avoid circular dep

    if group_size != 16:
        raise NotImplementedError(
            "Nvfp4MoePlugin requires quantization_group_size == 16; "
            f"got {group_size}")
    if hidden_size % 64 != 0:
        raise ValueError(
            f"hidden_size must be a multiple of 64, got {hidden_size}")
    if moe_inter_size % 64 != 0:
        raise ValueError(
            f"moe_inter_size must be a multiple of 64, got {moe_inter_size}")

    fc1_qw_list, fc1_pref_sf_list, fc1_dec_sf_list, fc1_gs_list = ([], [], [],
                                                                   [])
    fc2_qw_list, fc2_pref_sf_list, fc2_dec_sf_list, fc2_gs_list = ([], [], [],
                                                                   [])

    # CuTeDSL FC1 SwiGLU prefill kernel expects gate/up *interleaved at 32-col
    # granularity along the N axis* — NOT plain ``[gate, up]`` halves. See
    # ``kernelSrcs/nvfp4_moe_cutedsl/README.md:260-262``:
    #
    #   "SwiGLU weights: Must be preprocessed with
    #    interleave_linear_and_gate(weight, group_size=32, dim=1).
    #    Plain ``[up..., gate...]`` concatenation produces wrong results
    #    silently."
    #
    # The kernel epilogue (``blockscaled_contiguous_grouped_gemm_n_major.py``
    # line 2011 in the SwiGLU branch) pairs TMEM tiles ``(2i, 2i+1)`` as
    # ``(up, gate)``, with each tile = 32 output-N cols. So for output
    # position ``p`` (in ``[0, I)``):
    #
    #   chunk_idx = p // 32   (tile-pair index)
    #   in_chunk_offset = p % 32
    #   up_proj_weight @ p   ↔ B-N[chunk_idx * 64 +  0 + in_chunk_offset]
    #   gate_proj_weight @ p ↔ B-N[chunk_idx * 64 + 32 + in_chunk_offset]
    SWIGLU_INTERLEAVE_GROUP = 32
    _GS = SWIGLU_INTERLEAVE_GROUP
    if moe_inter_size % _GS != 0:
        raise ValueError(
            f"moe_inter_size ({moe_inter_size}) must be a multiple of "
            f"{_GS} for SwiGLU prefill kernel's interleave layout")

    for expert in experts:
        gate, up, down = expert.gate_proj, expert.up_proj, expert.down_proj
        if not (isinstance(gate, NVFP4Linear) and isinstance(up, NVFP4Linear)
                and isinstance(down, NVFP4Linear)):
            raise TypeError("Qwen3 NVFP4 MoE experts must use NVFP4Linear")

        gate_dense = decode_modelopt_nvfp4(gate.weight, gate.weight_scale,
                                           gate.weight_scale_2, group_size)
        up_dense = decode_modelopt_nvfp4(up.weight, up.weight_scale,
                                         up.weight_scale_2, group_size)
        down_dense = decode_modelopt_nvfp4(down.weight, down.weight_scale,
                                           down.weight_scale_2, group_size)

        if gate_dense.shape != (moe_inter_size, hidden_size):
            raise ValueError(f"gate dense shape {gate_dense.shape} != "
                             f"({moe_inter_size}, {hidden_size})")
        if up_dense.shape != (moe_inter_size, hidden_size):
            raise ValueError(f"up dense shape {up_dense.shape} != "
                             f"({moe_inter_size}, {hidden_size})")
        if down_dense.shape != (hidden_size, moe_inter_size):
            raise ValueError(f"down dense shape {down_dense.shape} != "
                             f"({hidden_size}, {moe_inter_size})")

        # Interleave (up, gate) at SWIGLU_INTERLEAVE_GROUP-col granularity along
        # the N axis. ``gate_dense`` and ``up_dense`` are ``[I, H]`` (NVFP4Linear
        # convention: out × in). For each ``chunk_idx``, B-N gets the
        # corresponding 32-col slice of up_dense then gate_dense.
        n_chunks = moe_inter_size // _GS
        # Stack into [I/32, 2, 32, H] then flatten to [2*I, H]:
        # axis 0 = chunk_idx, axis 1 = (up=0, gate=1), axis 2 = within-chunk offset.
        up_chunks = up_dense.reshape(n_chunks, _GS, hidden_size)
        gate_chunks = gate_dense.reshape(n_chunks, _GS, hidden_size)
        # Pair = [up_chunk_i, gate_chunk_i], stacked → [chunks, 2, 32, H]
        paired = np.stack([up_chunks, gate_chunks], axis=1)
        # Flatten the first three axes (chunks × 2 × 32) → 2*I along N
        fc1_dense_NK = paired.reshape(2 * moe_inter_size, hidden_size)
        # Transpose to K-major [H, 2*I] for _nvfp4_pack_n_major.
        fc1_dense_KN = np.ascontiguousarray(fc1_dense_NK.T)
        fc1_qw, fc1_pref_sf, fc1_dec_sf, fc1_gs = _nvfp4_pack_n_major(
            fc1_dense_KN)
        # FC2: down dense is [H, I]; transpose to [I, H] (K=I, N=H).
        fc2_dense_KN = np.ascontiguousarray(down_dense.T)
        fc2_qw, fc2_pref_sf, fc2_dec_sf, fc2_gs = _nvfp4_pack_n_major(
            fc2_dense_KN)

        fc1_qw_list.append(torch.from_numpy(fc1_qw))
        fc1_pref_sf_list.append(torch.from_numpy(fc1_pref_sf))
        fc1_dec_sf_list.append(torch.from_numpy(fc1_dec_sf))
        fc1_gs_list.append(fc1_gs)
        fc2_qw_list.append(torch.from_numpy(fc2_qw))
        fc2_pref_sf_list.append(torch.from_numpy(fc2_pref_sf))
        fc2_dec_sf_list.append(torch.from_numpy(fc2_dec_sf))
        fc2_gs_list.append(fc2_gs)

    return (
        torch.stack(fc1_qw_list, dim=0),
        torch.stack(fc1_pref_sf_list, dim=0),
        torch.tensor(fc1_gs_list, dtype=torch.float32),
        torch.stack(fc2_qw_list, dim=0),
        torch.stack(fc2_pref_sf_list, dim=0),
        torch.tensor(fc2_gs_list, dtype=torch.float32),
        torch.stack(fc1_dec_sf_list, dim=0),
        torch.stack(fc2_dec_sf_list, dim=0),
    )


def _atom_sf_offsets(M: int, num_sf_cols: int) -> np.ndarray:
    """Byte offsets for the 128x4 atom-layout scale-factor swizzle; matches ``MarlinConverter.atom_sf_offset``."""
    m = np.arange(M, dtype=np.int64)[:, None]
    k = np.arange(num_sf_cols, dtype=np.int64)[None, :]
    num_k_tiles = (num_sf_cols + 3) // 4
    return (m // 128 * num_k_tiles * 512 + k // 4 * 512 + m % 32 * 16 +
            (m % 128) // 32 * 4 + k % 4)


def _nvfp4_pack_n_major(
    dense_w_kn: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Pack one ``[K, N]`` dense weight to ``(weights_i8 [K, N//2], prefill_sf_i8 [padUp(N,128), padUp(K//16,4)], decode_sf_i8 [K//16, N], global_scale_f32)`` — N-major NVFP4 layout.

    Scheme-B per-``(k//16, n)`` block scales. Prefill SF is the 128×4 atom-layout
    swizzle (M=N, K=K/16) with raw IEEE FP8 E4M3 bytes; M / K_sf are padded to
    128 / 4 so the last partial atom tile is fully addressable. Decode SF is
    row-major ``[K/16, N]`` FP16→FP8 Marlin-projected bytes. Global scale is ``s_max/448``.
    """
    K, N = dense_w_kn.shape
    if K % 16 or N % 16:
        raise ValueError(
            f"K ({K}) must be a multiple of 16; N ({N}) must be a multiple of 16 "
            f"(Nvfp4MoePlugin's prefill atom-layout SF is read by the kernel "
            f"at granularity 16 along N; see `dense_weights_from_nvfp4_nmajor_buffers`)"
        )
    w = np.ascontiguousarray(dense_w_kn, dtype=np.float32)

    # The Nvfp4MoePlugin's prefill atom-layout SF is read by the CuTeDSL kernel
    # at *one byte per (K-group of 16, N-block of 16)* (i.e., the kernel rounds
    # the N coordinate down to a multiple of 16 before looking up the SF byte;
    # see `moe_decode_gemv.py:gate_leader = (n_base >> 4) << 4` and the unit
    # test reference `dense_weights_from_nvfp4_nmajor_buffers._read_prefill_sf`
    # which reads SF at `m_idx = c * 16`).  We must therefore use a single SF
    # per 16-K × 16-N tile here, not per-N: otherwise the 15 SF bytes per block
    # that the kernel never reads silently take 15/16 of every weight along
    # with them and the gated FC1 reconstruction error is ~50%.
    K_sf = K // 16
    N_blocks = N // 16
    # max |w| per 16-K × 16-N tile  → shape [K_sf, N_blocks]
    tile_max = np.abs(w.reshape(K_sf, 16, N_blocks, 16)).max(axis=(1, 3))
    # broadcast the tile max back to [K_sf, N] so every element in a tile uses
    # the same group_scale during FP4 nibble quantization.
    group_scales = np.maximum(tile_max / 6.0, 1e-12)
    group_scales_full = np.repeat(group_scales, 16, axis=1)  # [K_sf, N]
    s_max = max(float(np.abs(w).max()) / 6.0, 1e-12)

    w_scaled = (w / np.repeat(group_scales_full, 16, axis=0)).clip(-6.0, 6.0)
    abs_idx = np.searchsorted(_E2M1_BOUNDS, np.abs(w_scaled)).astype(np.uint8)
    sign_bit = (w_scaled < 0).astype(np.uint8) << np.uint8(3)
    nibbles = (abs_idx | sign_bit) & np.uint8(0xF)

    # byte[k, j] = nibble[k, 2j] | (nibble[k, 2j+1] << 4)
    lo = nibbles[:, 0::2]
    hi = nibbles[:, 1::2]
    weights_int8 = (lo | (hi << np.uint8(4))).astype(np.uint8).view(
        np.int8).reshape(K, N // 2).copy()

    # SF targets are the *tile-level* values broadcast to per-N (granularity 1)
    # so the atom-layout scatter is unchanged; the kernel only reads positions
    # M=0,16,32,... but every position within a 16-N block now holds the same
    # value, so the granularity-1 scatter and the granularity-16 read agree
    # bit-exactly.
    sf_targets = (group_scales_full / s_max * _FP8_MAX).astype(np.float32)
    sf_fp8_nk = torch.from_numpy(sf_targets.T.copy()).to(
        torch.float8_e4m3fn).view(torch.uint8).numpy()
    padded_N = ((N + 127) // 128) * 128
    padded_K_sf = ((K_sf + 3) // 4) * 4
    prefill_flat = np.zeros(padded_N * padded_K_sf, dtype=np.uint8)
    prefill_flat[_atom_sf_offsets(N, K_sf)] = sf_fp8_nk
    prefill_sf = prefill_flat.view(np.int8).reshape(padded_N,
                                                    padded_K_sf).copy()

    h16 = sf_targets.astype(np.float16).view(np.uint16).astype(np.uint32)
    decode_sf = (((
        (h16 << np.uint32(1)) & np.uint32(0xFF00)) >> np.uint32(8)).astype(
            np.uint8).view(np.int8).reshape(K_sf, N).copy())

    return weights_int8, prefill_sf, decode_sf, float(s_max / _FP8_MAX)


def repack_nvfp4_expert_up(
    dense_w_hi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Pack dense up_proj ``[H, I]`` to ``(weights [H, I/2], prefill_sf [padUp(I, 128), padUp(H/16, 4)], decode_sf [H/16, I], global_scale)``."""
    return _nvfp4_pack_n_major(dense_w_hi)


def repack_nvfp4_expert_down(
    dense_w_ih: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Pack dense down_proj ``[I, H]`` to ``(weights [I, H/2], prefill_sf [padUp(H, 128), padUp(I/16, 4)], decode_sf [I/16, H], global_scale)``."""
    return _nvfp4_pack_n_major(dense_w_ih)


# ---------------------------------------------------------------------------
# Raw-aligned prefill repack (layout-only, matches vLLM's weight-scale handling)
# ---------------------------------------------------------------------------
#
# The functions below produce the **prefill-path** plugin buffers
# (``fc_up_qweights`` / ``fc_up_blocks_scale`` / ``fc_up_global_scale`` and the
# symmetric down variants) by applying a pure layout transform to the raw
# ModelOpt checkpoint tensors — no dequantize/requantize round-trip. This
# matches vLLM's approach (see ``swizzle_blockscale`` in
# ``vllm/model_executor/layers/quantization/utils/nvfp4_utils.py``): preserve
# the checkpoint FP4 weight nibbles, FP8 E4M3 block scales, and FP32
# ``weight_scale_2`` bit-exact; change only the on-device layout to what the
# kernel expects.
#
# The decode-path SF (``fc_*_blocks_scale_decode``) is a separate buffer and is
# still produced by the dequant-then-requant path — see ``_nvfp4_pack_n_major``.


def _nibble_transpose_fp4(w_kmajor: np.ndarray) -> np.ndarray:
    """Transpose packed FP4 weights from K-major to N-major, bit-exact.

    Input ``w_kmajor`` has shape ``[out, in/2]`` (in innermost) with the byte
    convention ``byte[o, j] = fp4[o, 2j] | (fp4[o, 2j+1] << 4)`` — ModelOpt's
    checkpoint layout (also used by :func:`decode_modelopt_nvfp4`).

    Output has shape ``[in, out/2]`` (out innermost) using the same packing
    convention on the swapped axes. Every FP4 nibble is preserved; no
    dequant/requant.
    """
    if w_kmajor.dtype == np.int8:
        w = w_kmajor.view(np.uint8)
    elif w_kmajor.dtype == np.uint8:
        w = w_kmajor
    else:
        raise TypeError(f"unexpected weight dtype {w_kmajor.dtype}")
    out_f, half_in = w.shape
    in_f = half_in * 2
    if out_f % 2 != 0:
        raise ValueError(
            f"out ({out_f}) must be even for N-major nibble packing")
    # Unpack K-major bytes -> [out, in] nibbles.
    lo = w & np.uint8(0x0F)
    hi = (w >> np.uint8(4)) & np.uint8(0x0F)
    nibbles = np.empty((out_f, in_f), dtype=np.uint8)
    nibbles[:, 0::2] = lo
    nibbles[:, 1::2] = hi
    # Transpose to [in, out], repack along out (innermost).
    nibbles_t = np.ascontiguousarray(nibbles.T)  # shape [in, out]
    lo_t = nibbles_t[:, 0::2]
    hi_t = nibbles_t[:, 1::2]
    packed = (lo_t | (hi_t << np.uint8(4))).astype(np.uint8)
    return packed.view(np.int8).reshape(in_f, out_f // 2).copy()


def _atom_swizzle_raw_sf(raw_sf_bytes: np.ndarray, M: int,
                         K_sf: int) -> np.ndarray:
    """Atom-layout (128×4) swizzle for raw FP8 block-scale bytes.

    Input ``raw_sf_bytes`` has shape ``[M, K_sf]`` uint8/int8 holding FP8 E4M3
    bytes (per-(N, K-group) block scales from the checkpoint). Output is the
    ``[padUp(M, 128), padUp(K_sf, 4)]`` int8 array with bytes placed at the
    :func:`_atom_sf_offsets` positions — the same atom layout
    :func:`_nvfp4_pack_n_major` emits for its prefill SF, and bit-identical to
    vLLM's ``swizzle_blockscale`` permutation.
    """
    if raw_sf_bytes.dtype == np.int8:
        sf = raw_sf_bytes.view(np.uint8)
    elif raw_sf_bytes.dtype == np.uint8:
        sf = raw_sf_bytes
    else:
        raise TypeError(f"unexpected sf dtype {raw_sf_bytes.dtype}")
    if sf.shape != (M, K_sf):
        raise ValueError(f"raw sf shape {sf.shape} != ({M}, {K_sf})")
    padded_M = ((M + 127) // 128) * 128
    padded_K_sf = ((K_sf + 3) // 4) * 4
    flat = np.zeros(padded_M * padded_K_sf, dtype=np.uint8)
    # ``_atom_sf_offsets`` returns a 2-D ``[M, K_sf]`` index; assign the raw
    # 2-D sf bytes directly so shapes broadcast correctly (mirrors
    # ``_nvfp4_pack_n_major``'s scatter of ``sf_fp8_nk``).
    flat[_atom_sf_offsets(M, K_sf)] = sf
    return flat.view(np.int8).reshape(padded_M, padded_K_sf).copy()


def _sf_bytes_from_checkpoint(raw_sf: torch.Tensor) -> np.ndarray:
    """Extract raw FP8-E4M3 bytes from a checkpoint ``weight_scale`` tensor."""
    if raw_sf.dtype == torch.float8_e4m3fn:
        return raw_sf.detach().cpu().view(torch.uint8).numpy()
    if raw_sf.dtype == torch.int8:
        return raw_sf.detach().cpu().view(torch.uint8).numpy()
    if raw_sf.dtype in (torch.float32, torch.float16, torch.bfloat16):
        # Float-cast fallback (non-ModelOpt checkpoints). Go through FP8 E4M3.
        return raw_sf.detach().to(torch.float8_e4m3fn).cpu().view(
            torch.uint8).numpy()
    raise TypeError(f"unsupported weight_scale dtype {raw_sf.dtype}")


def _marlin_project_raw_fp8(raw_sf_bytes: np.ndarray, out_f: int,
                            K_sf: int) -> np.ndarray:
    """Marlin (top-8-bit-of-FP16) projection applied to **raw** FP8 bytes.

    ``_nvfp4_pack_n_major``'s decode SF is derived from the requantized
    normalized ``sf_targets = group_scale / s_max * 448`` so that, at decode
    runtime, ``marlin_unproject(byte) * (s_max/448)`` recovers the group
    scale in real magnitude. When the weight global scale is instead the raw
    checkpoint ``weight_scale_2``, the decode SF must be projected from the
    raw FP8 values directly so that ``marlin_unproject(byte) * ws2_raw ≈
    raw_fp8 * ws2_raw = group_scale`` — i.e. the per-expert global scale
    agrees with both the prefill and decode SFB conventions.

    Input bytes ``[out, K_sf]`` FP8 E4M3 bytes (checkpoint layout). Output
    shape ``[K_sf, out]`` int8 (decode SF layout of :func:`_nvfp4_pack_n_major`).
    """
    if raw_sf_bytes.dtype == np.int8:
        sf = raw_sf_bytes.view(np.uint8)
    elif raw_sf_bytes.dtype == np.uint8:
        sf = raw_sf_bytes
    else:
        raise TypeError(f"unexpected sf dtype {raw_sf_bytes.dtype}")
    if sf.shape != (out_f, K_sf):
        raise ValueError(f"sf shape {sf.shape} != ({out_f}, {K_sf})")
    # FP8 E4M3 -> FP32 (lossless cast).
    sf_fp32 = torch.from_numpy(sf.copy()).view(torch.float8_e4m3fn).to(
        torch.float32).numpy()
    # Transpose to [K_sf, out] to match the decode SF layout.
    sf_fp32_kn = np.ascontiguousarray(sf_fp32.T)
    # FP16 top-8-bit projection, identical to _nvfp4_pack_n_major's decode
    # path (applied here to raw FP8 values rather than to re-normalised
    # sf_targets).
    h16 = sf_fp32_kn.astype(np.float16).view(np.uint16).astype(np.uint32)
    return (((
        (h16 << np.uint32(1)) & np.uint32(0xFF00)) >> np.uint32(8)).astype(
            np.uint8).view(np.int8).reshape(K_sf, out_f).copy())


def _pack_nvfp4_raw_n_major(
    raw_weight: torch.Tensor,
    raw_sf_fp8: torch.Tensor,
    raw_ws2: torch.Tensor,
    out_f: int,
    in_f: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Layout-only repack of raw ModelOpt NVFP4 tensors to N-major plugin layout.

    Produces **both** the prefill and the decode SF buffers from the **same**
    raw checkpoint FP8 block scales + per-tensor ``weight_scale_2``. No
    dequantize/requantize round-trip; every FP4 nibble and every FP8 block-
    scale byte is preserved from the checkpoint.

    The prefill SF is 128×4 atom-swizzled; the decode SF is FP16-top-8-bit
    Marlin-projected from the same raw FP8 bytes (not from the requant-
    normalised ``sf_targets`` that :func:`_nvfp4_pack_n_major` uses). With the
    per-expert global scale set to the checkpoint's ``weight_scale_2``, both
    the prefill kernel (``fp4 × prefill_sfb × ws2``) and the decode kernel
    (``fp4 × marlin_unproject(decode_sfb) × ws2``) recover the same real
    group scale ``raw_fp8 × ws2`` — guaranteeing prefill/decode consistency.

    :param raw_weight: checkpoint ``weight``, shape ``[out, in/2]`` uint8/int8
        (two FP4 nibbles per byte along ``in``).
    :param raw_sf_fp8: checkpoint ``weight_scale``, shape ``[out, in/16]``
        FP8-E4M3 (or an ``int8`` / float cast of the same).
    :param raw_ws2: checkpoint ``weight_scale_2`` scalar FP32 (``[1]`` tensor).
    :param out_f: output dimension.
    :param in_f: input dimension; must be a multiple of 16.
    :return: ``(weights_i8 [in, out/2],
              prefill_sf_i8 [padUp(out,128), padUp(in/16,4)],
              decode_sf_i8 [in/16, out],
              global_scale)``.
    """
    if in_f % 16 != 0 or out_f % 2 != 0:
        raise ValueError(
            f"in ({in_f}) must be multiple of 16; out ({out_f}) must be even")
    if tuple(raw_weight.shape) != (out_f, in_f // 2):
        raise ValueError(f"raw_weight shape {tuple(raw_weight.shape)} != "
                         f"({out_f}, {in_f // 2})")
    if tuple(raw_sf_fp8.shape) != (out_f, in_f // 16):
        raise ValueError(f"raw_sf_fp8 shape {tuple(raw_sf_fp8.shape)} != "
                         f"({out_f}, {in_f // 16})")

    w_bytes = raw_weight.detach().cpu().numpy()
    weights_int8 = _nibble_transpose_fp4(w_bytes)

    sf_bytes = _sf_bytes_from_checkpoint(raw_sf_fp8)
    prefill_sf = _atom_swizzle_raw_sf(sf_bytes, out_f, in_f // 16)
    decode_sf = _marlin_project_raw_fp8(sf_bytes, out_f, in_f // 16)

    ws2 = float(raw_ws2.detach().reshape(-1)[0].item())
    return weights_int8, prefill_sf, decode_sf, ws2


def repack_nvfp4_expert_up_prefill_raw(
    raw_weight: torch.Tensor,
    raw_sf_fp8: torch.Tensor,
    raw_ws2: torch.Tensor,
    hidden_size: int,
    moe_inter_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Raw-aligned ``up_proj`` repack: checkpoint ``[I, H/2]`` → plugin ``[H, I/2]``.

    Returns ``(weights, prefill_sf, decode_sf, global_scale)`` where both SF
    buffers come from the **same** raw FP8 block scales (different layouts:
    prefill is atom-swizzled; decode is Marlin FP16-top-8-bit projected).
    See :func:`_pack_nvfp4_raw_n_major` for semantics.
    """
    return _pack_nvfp4_raw_n_major(
        raw_weight,
        raw_sf_fp8,
        raw_ws2,
        out_f=moe_inter_size,
        in_f=hidden_size,
    )


def repack_nvfp4_expert_down_prefill_raw(
    raw_weight: torch.Tensor,
    raw_sf_fp8: torch.Tensor,
    raw_ws2: torch.Tensor,
    hidden_size: int,
    moe_inter_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Raw-aligned ``down_proj`` repack: checkpoint ``[H, I/2]`` → plugin ``[I, H/2]``.

    Returns ``(weights, prefill_sf, decode_sf, global_scale)``; both SF buffers
    come from the **same** raw FP8 block scales. See :func:`_pack_nvfp4_raw_n_major`.
    """
    return _pack_nvfp4_raw_n_major(
        raw_weight,
        raw_sf_fp8,
        raw_ws2,
        out_f=hidden_size,
        in_f=moe_inter_size,
    )
