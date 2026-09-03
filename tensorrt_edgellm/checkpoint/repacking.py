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

from ..models.ops import int4_gemm_plugin_version

logger = logging.getLogger(__name__)

__all__ = [
    "NVFP4_MOE_INTERLEAVE_SIZE_ALIGNMENT",
    "NVFP4_MOE_INTERMEDIATE_SIZE_ALIGNMENT",
    "repack_awq_to_plugin",
    "repack_gptq_to_plugin",
    "decode_modelopt_nvfp4",
    "unpack_nvfp4_codes",
    "repack_nvfp4_a16_marlin_linear",
    "repack_nvfp4_a16_marlin_moe_experts",
    "repack_nvfp4_a16_marlin_gated_moe_experts",
    "repack_nvfp4_gated_moe_experts",
    "repack_nvfp4_moe_experts",
    "repack_fp16_moe_experts",
]

# Fp16MoePlugin FC1_N must be a multiple of 128 (kROW_ALIGNMENT); ReLU2 sets
# FC1_N = padded_inter, so the intermediate size is padded to 128.
FP16_MOE_INTERMEDIATE_SIZE_ALIGNMENT = 128

NVFP4_MOE_INTERLEAVE_SIZE_ALIGNMENT = 64
NVFP4_MOE_INTERMEDIATE_SIZE_ALIGNMENT = 128

# ---------------------------------------------------------------------------
# AWQ weight transform
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

    if int4_gemm_plugin_version() == 2:
        packed_int8 = repack_to_cutedsl_fragment(nibbles_nk)  # [rows, 512]
    else:
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
    pk = pk.reshape(N // interleave, interleave, K // kstride, kstride)
    pk = pk.transpose(0, 2, 1, 3)  # [N//4, K//64, 4, 64]
    pk = pk.reshape(N // interleave, K // kstride, kstride, interleave)

    # Step 4: Pack 4 nibbles per int16 (little-endian nibble order)
    pk = (pk[..., 0]
          | (pk[..., 1] << 4)
          | (pk[..., 2] << 8)
          | (pk[..., 3] << 12))
    return pk.reshape(N // interleave, K).astype(np.int16)


# ---------------------------------------------------------------------------
# cuteDSL fragment layout (Int4GroupwiseGemmPluginV2)
# ---------------------------------------------------------------------------
# The cuteDSL W4A16 kernel consumes a *fragment-order* weight buffer.  With bN
# pinned to 128 (and bK=64 fixed) the layout is tile-independent, so it can be
# baked once at export.  This mirrors
# ``kernelSrcs/int4_fp16_gemm_cutedsl/int4_reference.py::repack_b_for_tile``
# (bN=128, bK=64) but operates directly on biased nibbles [N, K].
_CUTEDSL_BN = 128
_CUTEDSL_BK = 64
_CUTEDSL_THREADS = 128
# (bit shift, hi-N selector, K offset) for the 8 nibbles of one 32-bit word.
_CUTEDSL_NIBBLES = (
    (0, False, 0),
    (4, False, 8),
    (8, True, 0),
    (12, True, 8),
    (16, False, 1),
    (20, False, 9),
    (24, True, 1),
    (28, True, 9),
)


def repack_to_cutedsl_fragment(nibbles_nk: np.ndarray) -> np.ndarray:
    """Biased INT4 nibbles ``[N, K]`` (in ``[0, 15]``) -> cuteDSL fragment-order
    INT8 buffer ``[rows, 512]``, where ``rows = ceil(N/128)*ceil(K/64)*8`` and
    each row is 128 uint32 words (viewed as 512 int8 bytes, little-endian).

    Bit-for-bit identical to ``repack_b_for_tile(bN=128, bK=64)``. Out-of-range
    (padding) nibbles are biased ``8`` so they dequantize to 0. Requires
    ``N % 64 == 0`` and ``K % 64 == 0``.
    """
    bN, bK = _CUTEDSL_BN, _CUTEDSL_BK
    n, k = int(nibbles_nk.shape[0]), int(nibbles_nk.shape[1])
    if n % 64 != 0 or k % 64 != 0:
        raise ValueError(
            f"cuteDSL fragment repack requires N%64==0 and K%64==0, got N={n}, K={k}. "
            "Small/misaligned projections should be left in fp16 (skipped) at "
            "quantize time rather than int4-quantized.")
    k_blocks = bK // 16  # 4
    n_pairs = bN // 64  # 2
    kn = k_blocks * n_pairs  # 8
    num_n_blocks = (n + bN - 1) // bN
    num_k_tiles = (k + bK - 1) // bK
    n_pad, k_pad = num_n_blocks * bN, num_k_tiles * bK

    b = np.full((n_pad, k_pad), 8, dtype=np.int64)
    b[:n, :k] = nibbles_nk.astype(np.int64) & 0xF

    nb = np.arange(num_n_blocks).reshape(-1, 1)  # (num_n_blocks, 1)
    kt = np.arange(num_k_tiles).reshape(1, -1)  # (1, num_k_tiles)
    out = np.zeros((num_n_blocks * num_k_tiles * kn, _CUTEDSL_THREADS),
                   dtype=np.int64)

    for t in range(_CUTEDSL_THREADS):
        n_base = t // 4
        kb = 2 * (t % 4)
        for kbl in range(k_blocks):
            for p in range(n_pairs):
                idx = kbl * n_pairs + p
                n_lo = nb * bN + (n_base + 64 * p)  # (num_n_blocks, 1)
                n_hi = n_lo + 32
                k0 = kt * bK + (16 * kbl + kb)  # (1, num_k_tiles)
                word = np.zeros((num_n_blocks, num_k_tiles), dtype=np.int64)
                for shift, hi, koff in _CUTEDSL_NIBBLES:
                    nrow = np.broadcast_to(n_hi if hi else n_lo,
                                           (num_n_blocks, num_k_tiles))
                    kcol = np.broadcast_to(k0 + koff,
                                           (num_n_blocks, num_k_tiles))
                    word = word | (b[nrow, kcol] << shift)
                rows = ((nb * num_k_tiles + kt) * kn + idx).reshape(-1)
                out[rows, t] = word.reshape(-1)

    out32 = np.ascontiguousarray(out.astype(np.uint32))  # [rows, 128] uint32
    return out32.view(np.int8).reshape(out32.shape[0], out32.shape[1] * 4)


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
# GPTQ weight transform
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
    if int4_gemm_plugin_version() == 2:
        n_dim, k_dim = nibbles_nk.shape
        if n_dim % 64 != 0 or k_dim % 64 != 0:
            raise ValueError(
                f"GPTQ projection [N={n_dim}, K={k_dim}] is not 64-aligned, "
                f"which the cuteDSL Int4GroupwiseGemmPluginV2 fragment layout "
                f"requires (N%64==0 and K%64==0).")
        packed_int8 = repack_to_cutedsl_fragment(nibbles_nk)  # [rows, 512]
    else:
        packed_int16 = _pack_intweights(nibbles_nk)
        packed_int8 = packed_int16.view(np.int8).reshape(
            packed_int16.shape[0] * 2, packed_int16.shape[1])

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
    _repack_nvfp4_a16_marlin_linears(model)


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
                if int4_gemm_plugin_version() == 2:
                    n_dim, k_dim = nibbles_np.shape
                    if n_dim % 64 != 0 or k_dim % 64 != 0:
                        raise ValueError(
                            f"ModelOpt AWQ-prepacked projection [N={n_dim}, "
                            f"K={k_dim}] is not 64-aligned, which the cuteDSL "
                            f"Int4GroupwiseGemmPluginV2 fragment layout requires "
                            f"(N%64==0 and K%64==0).")
                    packed_int8 = repack_to_cutedsl_fragment(nibbles_np)
                else:
                    packed_int16 = _pack_intweights(
                        nibbles_np)  # [N//4, K] int16
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


def _repack_nvfp4_a16_marlin_linears(model: nn.Module) -> None:
    """Transform NVFP4A16MarlinLinear checkpoint buffers into Marlin layout.

    Replaces the raw ModelOpt ``weight`` / ``weight_scale`` / ``weight_scale_2``
    buffers with the plugin buffers ``qweight`` / ``block_scales`` /
    ``global_scale`` and records ``n_padded`` for the forward slice.
    """
    from ..models.linear import NVFP4A16MarlinLinear  # local import
    for module in model.modules():
        if not isinstance(module, NVFP4A16MarlinLinear):
            continue
        # Routed MoE experts are stacked into the MoE plugin at export time
        # (repack_nvfp4_a16_marlin_moe_experts), so leave their raw buffers.
        if getattr(module, "_skip_dense_a16_repack", False):
            continue
        wp = module._buffers.get("weight")
        ws = module._buffers.get("weight_scale")
        wg = module._buffers.get("weight_scale_2")
        if wp is None:
            logger.warning(
                "NVFP4A16MarlinLinear missing weight; skipping repack")
            continue
        if wp.dtype in (torch.float16, torch.bfloat16, torch.float32):
            logger.warning(
                "NVFP4A16MarlinLinear has dense %s weight; refusing to "
                "quantize in-export. Checkpoint must provide packed NVFP4 "
                "(uint8 weight + e4m3 scales). Skipping repack.", wp.dtype)
            continue
        if ws is None or wg is None:
            logger.warning("NVFP4A16MarlinLinear missing packed buffers; "
                           "skipping repack")
            continue
        qweight, block_scales, global_scale, _, n_padded = (
            repack_nvfp4_a16_marlin_linear(wp, ws, wg, pad_n_to=128))
        # Drop the raw checkpoint buffers and install the plugin buffers.
        for name in ("weight", "weight_scale", "weight_scale_2"):
            module._buffers.pop(name, None)
        module.register_buffer("qweight", qweight)
        module.register_buffer("block_scales", block_scales)
        module.register_buffer("global_scale", global_scale)
        module.n_padded = int(n_padded)


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
    """View-cast NVFP4 weight buffers from uint8 to int8 in-place.

    Packed FP4 nibbles have the same bit pattern in both types.
    Some ONNX importers mishandle UINT8 weight initializers for block DQ; int8 works.
    """
    from ..models.linear import \
        is_nvfp4_linear  # local import to avoid circular dep
    for module in model.modules():
        if is_nvfp4_linear(module):
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
    (Marlin for ``Int4MoePlugin``; CuTeDSL 6D MMA for
    ``Nvfp4MoePlugin``); this helper is backend-agnostic.

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


def _extract_awq_for_marlin(
    proj: nn.Module,
    group_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract ``(weights [N, K] int16, scales [N, num_groups] fp16)`` from a
    ``ModelOptAWQPrepackedLinear`` module by folding ``pre_quant_scale`` into
    the weight and re-quantizing symmetric INT4 per-group.

    AWQ math::

        y = W @ (s_pqs ⊙ x) = (W ⊙ s_pqs[None, :]) @ x

    The Marlin MoE plugin has no ``pre_quant_scale`` input, so we fold it
    into the weight here and emit the same ``([N, K] int16, [N, G] fp16)``
    format that ``_extract_gptq_for_marlin`` produces — downstream Marlin
    packing in ``_prepare_moe_weights`` is unchanged.

    Inputs (raw modelopt buffers, BEFORE ``_cast_modelopt_awq_prepacked``):
        proj.weight:          uint8/int8 [N//2, K]  modelopt 2's-complement packed
        proj.weight_scale:    fp32       [N, K//g]
        proj.pre_quant_scale: fp16       [K]        (ones if absent — no-op)
    """
    w_u8 = proj.weight  # [N//2, K]
    sc = proj.weight_scale.data.to(torch.float32)  # [N, K//g]
    _pqs = getattr(proj, "pre_quant_scale", None)
    pqs = (_pqs.data.to(torch.float32) if _pqs is not None else torch.ones(
        w_u8.shape[1], dtype=torch.float32, device=w_u8.device))  # [K]

    N_half, K = w_u8.shape
    N = N_half * 2
    assert K % group_size == 0, (
        f"K={K} not divisible by group_size={group_size}")
    num_groups = K // group_size

    # 1. Unpack 2 nibbles per byte. modelopt convention:
    #    signed s in [-8, 7] -> u = s & 0xF; even N rows = low nibble,
    #    odd N rows = high nibble (see _cast_modelopt_awq_prepacked).
    w_u16 = w_u8.to(torch.int16) & 0xFF
    nibbles = torch.zeros(N, K, dtype=torch.int16, device=w_u8.device)
    nibbles[0::2] = w_u16 & 0xF
    nibbles[1::2] = (w_u16 >> 4) & 0xF
    # Convert 2's-complement nibble -> signed [-8, 7]
    s_signed = torch.where(nibbles < 8, nibbles, nibbles - 16)

    # 2. Dequantize symmetrically: W_fp32 = s * weight_scale (no zero point)
    sc_expanded = sc.repeat_interleave(group_size, dim=1)  # [N, K]
    w_fp32 = s_signed.to(torch.float32) * sc_expanded  # [N, K]

    # 3. Fold pre_quant_scale into weight (per-K-channel): W_eff = W * diag(pqs)
    w_folded = w_fp32 * pqs.unsqueeze(0)  # [N, K]

    # 4. Re-quantize per-group symmetric INT4. Scale = absmax / 7 so signed
    #    values land in [-7, 7]; plugin convention is (q_marlin - 8) * scale,
    #    so we shift to [1, 15] below.
    w_folded_grouped = w_folded.view(N, num_groups, group_size)  # [N, G, g]
    new_scale = w_folded_grouped.abs().amax(dim=-1) / 7.0  # [N, G]
    new_scale = new_scale.clamp(min=1e-10)
    new_q_signed = torch.round(
        w_folded_grouped / new_scale.unsqueeze(-1)).clamp(-7,
                                                          7).to(torch.int16)
    new_q_marlin = (new_q_signed + 8).view(N,
                                           K).contiguous()  # [N, K] in [1, 15]
    return new_q_marlin, new_scale.to(torch.float16).contiguous()


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


# Global-scale exponent for Marlin's skip-flop E2M1 conversion. The activation
# is FP16 (exp_bias 15), so the per-tensor global scale absorbs 2^(15-8)=2^7;
# the block E4M3 scale supplies the remaining 2^7.
_NVFP4_MARLIN_GLOBAL_SCALE_EXP = 7
_NVFP4_GROUP_SIZE = 16


def _nvfp4_marlin_e4m3_scale_perm() -> "torch.Tensor":
    """Within-64-column N permutation for E4M3 block scales in the FE2M1 path.

    The NVFP4 Marlin kernel reads block scales with ``s_gl_stride = N / 16`` (16
    E4M3 bytes per 16-byte load) rather than the fp16 ``N / 8``, so the fp16
    :func:`_marlin_permute_scales` layout does not match. Empirically (recovered
    against the on-device kernel), output column ``m`` of each 64-column N-block
    reads storage position ``8*(m % 8) + swap_low2bits(m // 8)`` — a 64x64
    transpose composed with a swap of the two low bits of the row index.
    """
    perm = []
    for m in range(64):
        r, c = divmod(m, 8)
        r_sw = (r & 0x4) | ((r & 0x1) << 1) | ((r >> 1) & 0x1)
        perm.append(8 * c + r_sw)
    return torch.tensor(perm, dtype=torch.long)


def _marlin_permute_nvfp4_e4m3_scales(ws_i8: torch.Tensor,
                                      n_padded: int) -> torch.Tensor:
    """Permute ``[N_pad, num_groups]`` E4M3 scale bytes to ``[1, num_groups, N_pad]``.

    Places output column ``m``'s scale byte at the storage position the kernel
    reads it from (see :func:`_nvfp4_marlin_e4m3_scale_perm`).
    """
    if n_padded % 64 != 0:
        raise ValueError(f"N_pad={n_padded} must be divisible by 64")
    num_groups = ws_i8.shape[1]
    n_blocks = n_padded // 64
    perm64 = _nvfp4_marlin_e4m3_scale_perm()
    full_perm = (torch.arange(n_blocks).repeat_interleave(64) * 64 +
                 perm64.repeat(n_blocks))
    block_scales = torch.empty((1, num_groups, n_padded), dtype=torch.int8)
    block_scales[0, :, full_perm] = ws_i8.t().contiguous()
    return block_scales.contiguous()


def unpack_nvfp4_codes(weight_packed: torch.Tensor) -> torch.Tensor:
    """Unpack ``[N, K//2]`` packed NVFP4 bytes to ``[N, K]`` uint8 E2M1 codes.

    Two E2M1 codes per byte with the ModelOpt / compressed-tensors convention:
    the low nibble is the even K index, the high nibble the odd K index (matches
    :func:`decode_modelopt_nvfp4`).
    """
    w = weight_packed
    if w.dtype == torch.int8:
        w = w.view(torch.uint8)
    if w.dtype != torch.uint8:
        raise TypeError(f"unexpected weight_packed dtype {w.dtype}")
    n, half = w.shape
    codes = torch.empty((n, half * 2), dtype=torch.uint8, device=w.device)
    codes[:, 0::2] = w & 0x0F
    codes[:, 1::2] = (w >> 4) & 0x0F
    return codes


def repack_nvfp4_a16_marlin_linear(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    pad_n_to: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Repack one ModelOpt NVFP4 (W4A16) linear into dense Marlin layout.

    The E2M1 codes pass through the same nibble bit-shuffle / permutation as
    AWQ int4 (:func:`pack_int4_awq_marlin`) — no zero-point remap — because
    Marlin treats the 4-bit field as an E2M1 code, not an unsigned int. The
    E4M3 block scales get the standard Marlin scale permutation; the per-tensor
    global scale is cast to FP16 and pre-multiplied by the skip-flop factor
    ``2**7``.

    ModelOpt stores ``weight_scale_2`` directly as the per-tensor *multiplier*
    ``amax / (FP4_MAX * FP8_MAX)`` (a small number that dequant multiplies by),
    which is exactly what Marlin wants — so it is applied as-is (no inversion).

    Args:
      weight_packed:  ``[N, K//2]`` uint8/int8, two E2M1 codes per byte.
      weight_scale:   ``[N, K//16]`` float8_e4m3fn (or its int8 view).
      weight_scale_2: scalar ModelOpt per-tensor multiplier (fp32).
      pad_n_to:       zero-pad N up to this multiple (Marlin needs N % 128 == 0).

    Returns:
      qweights:     ``[1, K//16, 8*N_pad]`` int8 (INT32-viewable Marlin codes)
      block_scales: ``[1, K//16, N_pad]``  int8 (raw E4M3 bytes, Marlin-permuted)
      global_scale: ``[1]`` float16 (weight_scale_2 * 2**7)
      n_logical:    original N
      n_padded:     N rounded up to ``pad_n_to``
    """
    codes = unpack_nvfp4_codes(weight_packed)  # [N, K] uint8
    n_logical, k = codes.shape
    if k % _NVFP4_GROUP_SIZE != 0:
        raise ValueError(f"K={k} must be divisible by {_NVFP4_GROUP_SIZE}")
    num_groups = k // _NVFP4_GROUP_SIZE

    ws = weight_scale
    if ws.dtype == torch.float8_e4m3fn:
        ws_i8 = ws.view(torch.int8)
    elif ws.dtype in (torch.int8, torch.uint8):
        ws_i8 = ws.view(torch.int8)
    else:
        raise TypeError(f"unexpected weight_scale dtype {ws.dtype}")
    if ws_i8.shape != (n_logical, num_groups):
        raise ValueError(f"weight_scale shape {tuple(ws_i8.shape)} != "
                         f"({n_logical}, {num_groups})")

    n_padded = ((n_logical + pad_n_to - 1) // pad_n_to) * pad_n_to
    if n_padded != n_logical:
        pad_rows = n_padded - n_logical
        codes = torch.cat(
            [codes, torch.zeros((pad_rows, k), dtype=torch.uint8)], dim=0)
        ws_i8 = torch.cat(
            [ws_i8,
             torch.zeros((pad_rows, num_groups), dtype=torch.int8)],
            dim=0)

    # Marlin-repack the E2M1 codes via the dtype-agnostic int4 bit-shuffle.
    dummy_scales = torch.ones((1, n_padded, num_groups), dtype=torch.float16)
    weights_marlin, _ = pack_int4_awq_marlin(
        codes[None].to(torch.int32),
        dummy_scales,
        group_size=_NVFP4_GROUP_SIZE)  # [1, K//16, 2*N_pad] int32
    qweights = weights_marlin.view(
        torch.int8).contiguous()  # [1,K//16,8*N_pad]

    # E4M3 block scales -> [1, K//16, N_pad] Marlin-permuted int8 bytes.
    block_scales = _marlin_permute_nvfp4_e4m3_scales(ws_i8, n_padded)

    # Apply the skip-flop rule to the ModelOpt per-tensor multiplier. Compute
    # the product in fp64 to avoid an intermediate low-precision round.
    g = float(weight_scale_2.detach().reshape(-1)[0].item())
    mult = float(2**_NVFP4_MARLIN_GLOBAL_SCALE_EXP)
    global_scale = torch.tensor([g * mult],
                                dtype=torch.float64).to(torch.float16)

    return qweights, block_scales, global_scale, n_logical, n_padded


def _pad_nvfp4_linear_k(weight_packed: torch.Tensor,
                        weight_scale: torch.Tensor, k_padded: int):
    """Zero-pad a packed NVFP4 linear along the input dimension K.

    ``weight_packed`` [N, K/2] uint8 -> [N, k_padded/2]; ``weight_scale``
    [N, K/16] e4m3 -> [N, k_padded/16]. The extra K columns are zero codes /
    zero scale bytes, which are exact and contribute nothing once the padded
    activation columns (also zero) multiply them.
    """
    wp = weight_packed
    if wp.dtype == torch.int8:
        wp = wp.view(torch.uint8)
    n, half = wp.shape
    k = half * 2
    if k_padded < k or k_padded % 16 != 0:
        raise ValueError(f"k_padded={k_padded} invalid for K={k}")
    if k_padded == k:
        return weight_packed, weight_scale
    wp_pad = torch.zeros((n, k_padded // 2), dtype=torch.uint8)
    wp_pad[:, :half] = wp
    ws_i8 = weight_scale.view(
        torch.int8) if weight_scale.dtype in (torch.float8_e4m3fn,
                                              torch.uint8) else weight_scale
    ws_pad = torch.zeros((n, k_padded // 16), dtype=torch.int8)
    ws_pad[:, :k // 16] = ws_i8
    return wp_pad, ws_pad


def repack_nvfp4_a16_marlin_moe_experts(
    fc1_packed: "list",
    fc1_scale: "list",
    fc1_global: "list",
    fc2_packed: "list",
    fc2_scale: "list",
    fc2_global: "list",
    moe_inter_padded: int = 1920,
) -> Tuple[torch.Tensor, ...]:
    """Stack + repack per-expert NVFP4 (W4A16) MoE weights for ``Nvfp4A16MoePlugin``.

    Non-gated (ReLU2) contract, matching the pinned SGLang
    ``prepare_moe_nvfp4_layer_for_marlin``:

      FC1 (up_proj):   per expert N=moe_inter -> pad to ``moe_inter_padded``
                       (Marlin N%128); K=hidden.
      FC2 (down_proj): per expert N=hidden; K=moe_inter -> pad to
                       ``moe_inter_padded`` (Marlin K%64).

    Each list holds the ``E`` per-expert checkpoint tensors. Returns
    ``(fc1_qweight [E,H/16,8*I_pad], fc1_block_scales [E,H/16,I_pad], fc1_global [E],
       fc2_qweight [E,I_pad/16,8*H], fc2_block_scales [E,I_pad/16,H], fc2_global [E])``.
    """
    num_experts = len(fc1_packed)
    fc1_q, fc1_bs, fc1_g = [], [], []
    fc2_q, fc2_bs, fc2_g = [], [], []
    for e in range(num_experts):
        # FC1: pad N (moe_inter) up to a 128 multiple (== moe_inter_padded).
        q1, s1, g1, _, n1 = repack_nvfp4_a16_marlin_linear(fc1_packed[e],
                                                           fc1_scale[e],
                                                           fc1_global[e],
                                                           pad_n_to=128)
        if n1 != moe_inter_padded:
            raise ValueError(
                f"FC1 padded N {n1} != moe_inter_padded {moe_inter_padded}")
        fc1_q.append(q1)
        fc1_bs.append(s1)
        fc1_g.append(g1)
        # FC2: pad K (moe_inter) up to moe_inter_padded; N (hidden) is unpadded.
        wp2, ws2 = _pad_nvfp4_linear_k(fc2_packed[e], fc2_scale[e],
                                       moe_inter_padded)
        q2, s2, g2, _, _ = repack_nvfp4_a16_marlin_linear(wp2,
                                                          ws2,
                                                          fc2_global[e],
                                                          pad_n_to=128)
        fc2_q.append(q2)
        fc2_bs.append(s2)
        fc2_g.append(g2)
    return (
        torch.cat(fc1_q, dim=0),
        torch.cat(fc1_bs, dim=0),
        torch.cat(fc1_g, dim=0),
        torch.cat(fc2_q, dim=0),
        torch.cat(fc2_bs, dim=0),
        torch.cat(fc2_g, dim=0),
    )


def repack_nvfp4_a16_marlin_gated_moe_experts(
    gate_packed: "list",
    gate_scale: "list",
    gate_global: "list",
    up_packed: "list",
    up_scale: "list",
    up_global: "list",
    down_packed: "list",
    down_scale: "list",
    down_global: "list",
    moe_inter_padded: int,
) -> Tuple[torch.Tensor, ...]:
    """Stack + repack SwiGLU NVFP4 (W4A16) experts for ``Nvfp4A16MoePlugin``.

    FC1 is two independently Marlin-packed projections concatenated on N
    (``[gate | up]``, each padded to ``moe_inter_padded``). The plugin takes
    one FC1 global scale; gate/up must share ``weight_scale_2`` (ModelOpt
    official Qwen3.6 NVFP4 does). FC2 matches the non-gated helper.
    """
    num_experts = len(gate_packed)
    fc1_q, fc1_bs, fc1_g = [], [], []
    fc2_q, fc2_bs, fc2_g = [], [], []
    for e in range(num_experts):
        q_gate, s_gate, g_gate, _, n_gate = repack_nvfp4_a16_marlin_linear(
            gate_packed[e], gate_scale[e], gate_global[e], pad_n_to=128)
        q_up, s_up, g_up, _, n_up = repack_nvfp4_a16_marlin_linear(
            up_packed[e], up_scale[e], up_global[e], pad_n_to=128)
        if n_gate != moe_inter_padded or n_up != moe_inter_padded:
            raise ValueError(f"SwiGLU FC1 padded N gate={n_gate} up={n_up} != "
                             f"moe_inter_padded {moe_inter_padded}")
        if not torch.equal(g_gate.reshape(-1), g_up.reshape(-1)):
            raise ValueError(
                "Nvfp4A16MoePlugin SwiGLU needs one FC1 global scale; "
                f"expert {e} gate/up weight_scale_2 differ")
        fc1_q.append(torch.cat([q_gate, q_up], dim=-1))
        fc1_bs.append(torch.cat([s_gate, s_up], dim=-1))
        fc1_g.append(g_gate)
        wp2, ws2 = _pad_nvfp4_linear_k(down_packed[e], down_scale[e],
                                       moe_inter_padded)
        q2, s2, g2, _, _ = repack_nvfp4_a16_marlin_linear(wp2,
                                                          ws2,
                                                          down_global[e],
                                                          pad_n_to=128)
        fc2_q.append(q2)
        fc2_bs.append(s2)
        fc2_g.append(g2)
    return (
        torch.cat(fc1_q, dim=0),
        torch.cat(fc1_bs, dim=0),
        torch.cat(fc1_g, dim=0),
        torch.cat(fc2_q, dim=0),
        torch.cat(fc2_bs, dim=0),
        torch.cat(fc2_g, dim=0),
    )


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


def _decode_or_passthrough_nvfp4(proj: nn.Module,
                                 group_size: int = 16) -> np.ndarray:
    """Return a gated-expert projection as dense fp32 ``[out, in]``.

    Pre-quantized NVFP4 MoE checkpoints keep small gate/router-style
    projections in float16/bfloat16 while packing the heavy experts to NVFP4.
    Such a weight loads into an NVFP4-typed linear (so ``is_nvfp4_linear`` is
    True) but its buffer stays float, not packed int8/uint8. Decode only the
    genuinely packed weights; pass a float weight through unchanged so it is
    re-packed downstream by ``_pack_nvfp4_moe_weight``. Any other dtype is
    unexpected and raised so the failure surfaces here rather than as a
    downstream precision/shape mismatch.
    """
    w = proj.weight
    if w.dtype in (torch.int8, torch.uint8):
        return decode_modelopt_nvfp4(w, proj.weight_scale, proj.weight_scale_2,
                                     group_size)
    if w.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(
            f"unexpected gated-MoE projection weight dtype {w.dtype}; expected "
            "packed int8/uint8 or unquantized float16/bfloat16")
    return w.detach().to(torch.float32).cpu().numpy()


def _round_dense_to_bf16(dense: np.ndarray) -> np.ndarray:
    """Round a dense fp32 weight through BF16 while returning fp32 storage."""
    dense_t = torch.from_numpy(np.ascontiguousarray(dense, dtype=np.float32))
    return dense_t.to(torch.bfloat16).to(torch.float32).cpu().numpy()


def _swizzle_nvfp4_mma_scales(scale_bytes: np.ndarray, m_dim: int,
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


def _pack_nvfp4_moe_weight(
        dense_w_mk: np.ndarray,
        group_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack dense ``[M, K]`` weights for ``Nvfp4MoePlugin``.

    Returns ``(qweights [M, K/2] int8,
    blocks_scale [m_tiles, k_tiles, 32, 4, 4] int8)``.  The scale tensor
    stores raw FP8 E4M3 block scales in the physical CuTeDSL MMA layout.
    """
    if group_size != 16:
        raise NotImplementedError("Nvfp4MoePlugin requires group_size=16")

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
    blocks_scale = _swizzle_nvfp4_mma_scales(sf_bytes, m_dim, k_sf_dim)

    return (torch.from_numpy(qweights.copy()),
            torch.from_numpy(blocks_scale.copy()))


def _interleave_gated_moe_fc1(
    gate_dense: np.ndarray,
    up_dense: np.ndarray,
    hidden_size: int,
    moe_inter_size: int,
) -> np.ndarray:
    """Build FC1 dense weight as 64-row interleaved up/gate chunks.

    Layout: ``[up_chunk(64), gate_chunk(64), up_chunk(64), gate_chunk(64), ...]``
    along the M axis. Consumed natively by the SM100/101/110 ``Nvfp4MoePlugin`` split
    FC1 kernel.
    """
    if gate_dense.shape != up_dense.shape:
        raise ValueError(
            f"gate dense shape {gate_dense.shape} != up dense shape {up_dense.shape}"
        )
    expected_shape = (moe_inter_size, hidden_size)
    if gate_dense.shape != expected_shape:
        raise ValueError(
            f"gate/up dense shape {gate_dense.shape} != {expected_shape}")

    if moe_inter_size % NVFP4_MOE_INTERLEAVE_SIZE_ALIGNMENT != 0:
        raise ValueError(
            f"moe_inter_size ({moe_inter_size}) must be a multiple of "
            f"{NVFP4_MOE_INTERLEAVE_SIZE_ALIGNMENT} for the interleaved gated "
            "FC1 layout")

    n_chunks = moe_inter_size // NVFP4_MOE_INTERLEAVE_SIZE_ALIGNMENT
    up_chunks = up_dense.reshape(n_chunks, NVFP4_MOE_INTERLEAVE_SIZE_ALIGNMENT,
                                 hidden_size)
    gate_chunks = gate_dense.reshape(n_chunks,
                                     NVFP4_MOE_INTERLEAVE_SIZE_ALIGNMENT,
                                     hidden_size)
    return np.stack([up_chunks, gate_chunks],
                    axis=1).reshape(2 * moe_inter_size, hidden_size)


def _concat_gated_moe_fc1(
    gate_dense: np.ndarray,
    up_dense: np.ndarray,
    hidden_size: int,
    moe_inter_size: int,
) -> np.ndarray:
    """Build FC1 dense weight as plain ``[up_all, gate_all]`` concat.

    Layout: all ``moe_inter_size`` up rows followed by all ``moe_inter_size``
    gate rows along the M axis. Consumed natively by the SM12x
    ``NvFP4MoEPluginGeforce`` fused kernel.
    """
    if gate_dense.shape != up_dense.shape:
        raise ValueError(
            f"gate dense shape {gate_dense.shape} != up dense shape {up_dense.shape}"
        )
    expected_shape = (moe_inter_size, hidden_size)
    if gate_dense.shape != expected_shape:
        raise ValueError(
            f"gate/up dense shape {gate_dense.shape} != {expected_shape}")
    if moe_inter_size % NVFP4_MOE_INTERMEDIATE_SIZE_ALIGNMENT != 0:
        raise ValueError(
            f"moe_inter_size ({moe_inter_size}) must be a multiple of "
            f"{NVFP4_MOE_INTERMEDIATE_SIZE_ALIGNMENT} for the concatenated "
            "gated FC1 layout")
    return np.concatenate([up_dense, gate_dense],
                          axis=0).reshape(2 * moe_inter_size, hidden_size)


def _pad_nvfp4_gated_moe_dense_weights(
    gate_dense: np.ndarray,
    up_dense: np.ndarray,
    down_dense: np.ndarray,
    hidden_size: int,
    moe_inter_size: int,
    padded_moe_inter_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-pad gated MoE dense weights along the intermediate dimension."""
    if gate_dense.shape != up_dense.shape:
        raise ValueError(
            f"gate dense shape {gate_dense.shape} != up dense shape "
            f"{up_dense.shape}")
    if gate_dense.shape != (moe_inter_size, hidden_size):
        raise ValueError(f"gate/up dense shape {gate_dense.shape} != "
                         f"({moe_inter_size}, {hidden_size})")
    if down_dense.shape != (hidden_size, moe_inter_size):
        raise ValueError(f"down dense shape {down_dense.shape} != "
                         f"({hidden_size}, {moe_inter_size})")
    if padded_moe_inter_size < moe_inter_size:
        raise ValueError(
            f"padded_moe_inter_size ({padded_moe_inter_size}) must be >= "
            f"moe_inter_size ({moe_inter_size})")
    if padded_moe_inter_size == moe_inter_size:
        return gate_dense, up_dense, down_dense

    padded_gate_dense = np.zeros((padded_moe_inter_size, hidden_size),
                                 dtype=gate_dense.dtype)
    padded_up_dense = np.zeros((padded_moe_inter_size, hidden_size),
                               dtype=up_dense.dtype)
    padded_down_dense = np.zeros((hidden_size, padded_moe_inter_size),
                                 dtype=down_dense.dtype)
    padded_gate_dense[:moe_inter_size] = gate_dense
    padded_up_dense[:moe_inter_size] = up_dense
    padded_down_dense[:, :moe_inter_size] = down_dense
    return padded_gate_dense, padded_up_dense, padded_down_dense


def repack_nvfp4_gated_moe_experts(
    experts: Iterable[nn.Module],
    hidden_size: int,
    moe_inter_size: int,
    group_size: int = 16,
    fc1_layout: str = "interleave",
    moe_inter_size_alignment: int = NVFP4_MOE_INTERLEAVE_SIZE_ALIGNMENT,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack gated NVFP4 experts for the active NVFP4 MoE plugin.

    Each expert is expected to contain ModelOpt NVFP4 gate/up/down
    projection tensors.  Dense weights are decoded, rounded through BF16, and
    repacked for the CuTeDSL plugin.

    Args:
        experts: per-expert ``nn.Module`` containers exposing
            ``gate_proj`` / ``up_proj`` / ``down_proj``.
        hidden_size: model hidden size ``H``.
        moe_inter_size: per-expert intermediate size ``I``.
        group_size: NVFP4 K-axis group size (must be ``16``).
        fc1_layout: gated FC1 row layout.
            * ``"interleave"`` (default) -- ``Nvfp4MoePlugin`` (SM100/101/110): 64-row
              up/gate interleaved chunks along the M axis.
            * ``"concat"`` -- ``NvFP4MoEPluginGeforce`` (SM12x): plain
              ``[up_all, gate_all]`` concat along the M axis.
        moe_inter_size_alignment: physical intermediate-size alignment. Use
            :data:`NVFP4_MOE_INTERLEAVE_SIZE_ALIGNMENT` for
            ``"interleave"`` and
            :data:`NVFP4_MOE_INTERMEDIATE_SIZE_ALIGNMENT` for ``"concat"``.
    """
    from ..models.linear import \
        is_nvfp4_linear  # local import to avoid circular dep

    if fc1_layout == "interleave":
        build_fc1_dense = _interleave_gated_moe_fc1
        layout_alignment = NVFP4_MOE_INTERLEAVE_SIZE_ALIGNMENT
    elif fc1_layout == "concat":
        build_fc1_dense = _concat_gated_moe_fc1
        layout_alignment = NVFP4_MOE_INTERMEDIATE_SIZE_ALIGNMENT
    else:
        raise ValueError(
            f"fc1_layout={fc1_layout!r} not recognized; use "
            "'interleave' (SM100/101/110 Nvfp4MoePlugin) or 'concat' (SM12x "
            "NvFP4MoEPluginGeforce)")

    if moe_inter_size_alignment <= 0:
        raise ValueError(
            f"moe_inter_size_alignment ({moe_inter_size_alignment}) must be "
            ">= 1")
    if moe_inter_size_alignment % layout_alignment != 0:
        raise ValueError(
            f"moe_inter_size_alignment ({moe_inter_size_alignment}) must be "
            f"a multiple of {layout_alignment} for "
            f"fc1_layout={fc1_layout!r}")
    padded_moe_inter_size = (
        (moe_inter_size + moe_inter_size_alignment - 1) //
        moe_inter_size_alignment) * moe_inter_size_alignment
    if padded_moe_inter_size % layout_alignment != 0:
        raise ValueError(
            f"padded_moe_inter_size ({padded_moe_inter_size}) must be a "
            f"multiple of {layout_alignment} for fc1_layout={fc1_layout!r}")
    if padded_moe_inter_size % group_size != 0:
        raise ValueError(
            f"padded_moe_inter_size ({padded_moe_inter_size}) must be a "
            f"multiple of group_size ({group_size})")

    fc1_qweights = []
    fc1_blocks_scale = []
    fc2_qweights = []
    fc2_blocks_scale = []

    for expert in experts:
        gate = expert.gate_proj
        up = expert.up_proj
        down = expert.down_proj
        if not (is_nvfp4_linear(gate) and is_nvfp4_linear(up)
                and is_nvfp4_linear(down)):
            raise TypeError("Gated NVFP4 MoE experts must use NVFP4 quant")

        gate_dense = _decode_or_passthrough_nvfp4(gate, group_size)
        up_dense = _decode_or_passthrough_nvfp4(up, group_size)
        down_dense = _decode_or_passthrough_nvfp4(down, group_size)

        gate_dense, up_dense, down_dense = _pad_nvfp4_gated_moe_dense_weights(
            gate_dense, up_dense, down_dense, hidden_size, moe_inter_size,
            padded_moe_inter_size)

        fc1_dense = build_fc1_dense(gate_dense, up_dense, hidden_size,
                                    padded_moe_inter_size)
        fc1_qw, fc1_sf = _pack_nvfp4_moe_weight(fc1_dense, group_size)
        fc2_qw, fc2_sf = _pack_nvfp4_moe_weight(down_dense, group_size)
        fc1_qweights.append(fc1_qw)
        fc1_blocks_scale.append(fc1_sf)
        fc2_qweights.append(fc2_qw)
        fc2_blocks_scale.append(fc2_sf)

    return (torch.stack(fc1_qweights,
                        dim=0), torch.stack(fc1_blocks_scale, dim=0),
            torch.stack(fc2_qweights,
                        dim=0), torch.stack(fc2_blocks_scale, dim=0))


def repack_nvfp4_moe_experts(
    experts: Iterable[nn.Module],
    hidden_size: int,
    moe_inter_size: int,
    group_size: int = 16,
    hidden_size_alignment: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, int, int]:
    """Pack pre-quantized, ungated NVFP4 experts.

    Preserve checkpoint FP4 weights and global scales while padding the
    intermediate and optional hidden dimensions and swizzling block scales.
    Returns packed FC1/FC2 weights, scales, alphas, and the padded sizes.
    """
    from ..models.linear import \
        is_nvfp4_linear  # local import to avoid circular dep

    fc1_qweights = []
    fc1_blocks_scale = []
    fc1_alpha = []
    fc2_qweights = []
    fc2_blocks_scale = []
    fc2_alpha = []
    padded_inter_size = (
        (moe_inter_size + NVFP4_MOE_INTERMEDIATE_SIZE_ALIGNMENT - 1) //
        NVFP4_MOE_INTERMEDIATE_SIZE_ALIGNMENT
    ) * NVFP4_MOE_INTERMEDIATE_SIZE_ALIGNMENT
    if hidden_size_alignment <= 0:
        raise ValueError(
            f"hidden_size_alignment ({hidden_size_alignment}) must be >= 1")
    padded_hidden_size = ((hidden_size + hidden_size_alignment - 1) //
                          hidden_size_alignment) * hidden_size_alignment
    if padded_hidden_size % group_size != 0:
        raise ValueError(
            f"padded_hidden_size ({padded_hidden_size}) must be a multiple of "
            f"group_size ({group_size})")
    h_padded_k = padded_hidden_size // 2  # FC1 K dim (FP4 packed)
    h_padded_sf = padded_hidden_size // group_size  # FC1/FC2 SF count along H

    for expert in experts:
        up = expert.up_proj
        down = expert.down_proj
        if not (is_nvfp4_linear(up) and is_nvfp4_linear(down)):
            raise TypeError("NVFP4 MoE experts must use NVFP4 quant")

        if tuple(up.weight.shape) != (moe_inter_size, hidden_size // 2):
            raise ValueError(f"up weight shape {tuple(up.weight.shape)} != "
                             f"({moe_inter_size}, {hidden_size // 2})")
        if tuple(up.weight_scale.shape) != (moe_inter_size,
                                            hidden_size // group_size):
            raise ValueError(
                f"up scale shape {tuple(up.weight_scale.shape)} != "
                f"({moe_inter_size}, {hidden_size // group_size})")
        if tuple(down.weight.shape) != (hidden_size, moe_inter_size // 2):
            raise ValueError(
                f"down weight shape {tuple(down.weight.shape)} != "
                f"({hidden_size}, {moe_inter_size // 2})")
        if tuple(down.weight_scale.shape) != (hidden_size,
                                              moe_inter_size // group_size):
            raise ValueError(
                f"down scale shape {tuple(down.weight_scale.shape)} != "
                f"({hidden_size}, {moe_inter_size // group_size})")

        up_weight = up.weight.detach().cpu()
        down_weight = down.weight.detach().cpu()
        if up_weight.dtype == torch.uint8:
            up_weight = up_weight.view(torch.int8)
        if down_weight.dtype == torch.uint8:
            down_weight = down_weight.view(torch.int8)
        if up_weight.dtype != torch.int8 or down_weight.dtype != torch.int8:
            raise TypeError("NVFP4 MoE weights must be int8/uint8")

        needs_pad = (padded_inter_size != moe_inter_size
                     or padded_hidden_size != hidden_size)
        if needs_pad:
            padded_up_weight = torch.zeros((padded_inter_size, h_padded_k),
                                           dtype=torch.int8)
            padded_up_weight[:moe_inter_size, :hidden_size // 2] = up_weight
            up_weight = padded_up_weight

            padded_down_weight = torch.zeros(
                (padded_hidden_size, padded_inter_size // 2), dtype=torch.int8)
            padded_down_weight[:hidden_size, :moe_inter_size //
                               2] = down_weight
            down_weight = padded_down_weight

            up_sf_bytes = _sf_bytes_from_checkpoint(up.weight_scale)
            padded_up_sf = np.zeros((padded_inter_size, h_padded_sf),
                                    dtype=np.uint8)
            padded_up_sf[:moe_inter_size, :hidden_size //
                         group_size] = up_sf_bytes

            down_sf_bytes = _sf_bytes_from_checkpoint(down.weight_scale)
            padded_down_sf = np.zeros(
                (padded_hidden_size, padded_inter_size // group_size),
                dtype=np.uint8)
            padded_down_sf[:hidden_size, :moe_inter_size //
                           group_size] = down_sf_bytes
        else:
            padded_up_sf = _sf_bytes_from_checkpoint(up.weight_scale)
            padded_down_sf = _sf_bytes_from_checkpoint(down.weight_scale)

        up_sf = _swizzle_nvfp4_mma_scales(padded_up_sf, padded_inter_size,
                                          h_padded_sf)
        down_sf = _swizzle_nvfp4_mma_scales(padded_down_sf, padded_hidden_size,
                                            padded_inter_size // group_size)

        fc1_qweights.append(up_weight.contiguous())
        fc1_blocks_scale.append(torch.from_numpy(up_sf))
        fc1_alpha.append(float(up.weight_scale_2.detach().reshape(-1)[0]))
        fc2_qweights.append(down_weight.contiguous())
        fc2_blocks_scale.append(torch.from_numpy(down_sf))
        fc2_alpha.append(float(down.weight_scale_2.detach().reshape(-1)[0]))

    return (torch.stack(fc1_qweights,
                        dim=0), torch.stack(fc1_blocks_scale, dim=0),
            torch.tensor(fc1_alpha,
                         dtype=torch.float32), torch.stack(fc2_qweights,
                                                           dim=0),
            torch.stack(fc2_blocks_scale,
                        dim=0), torch.tensor(fc2_alpha, dtype=torch.float32),
            padded_inter_size, padded_hidden_size)


def repack_fp16_moe_experts(
    experts: Iterable[nn.Module],
    hidden_size: int,
    moe_inter_size: int,
    activation_type: int,
    dtype: torch.dtype = torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Stack unquantized experts into ``Fp16MoePlugin`` FC1/FC2 buffers.

    Each expert exposes ``up_proj``/``down_proj`` (and ``gate_proj`` for
    SwiGLU) ``.weight`` tensors. The intermediate dimension is zero-padded to a
    multiple of 128 so the plugin's ``FC1_N % 128 == 0`` contract holds; the
    padding is inert (ReLU²(0)=0 and the matching FC2 columns are zero).

    Returns ``(fc1_weights, fc2_weights, padded_inter_size)`` where, with
    ``I`` the padded intermediate size:

    * ReLU² (``activation_type == 4``): ``fc1 = [E, I, H]`` from ``up_proj``.
    * SwiGLU (``activation_type == 2``): ``fc1 = [E, 2*I, H]`` with the 64-row
      up/gate interleave the split kernel expects.
    * ``fc2 = [E, H, I]`` from ``down_proj``.
    """
    _ACT_SWIGLU = 2
    _ACT_RELU2 = 4
    if activation_type not in (_ACT_SWIGLU, _ACT_RELU2):
        raise ValueError(
            f"activation_type must be 2 (SwiGLU) or 4 (ReLU2), got {activation_type}"
        )

    padded_inter_size = (
        (moe_inter_size + FP16_MOE_INTERMEDIATE_SIZE_ALIGNMENT - 1) //
        FP16_MOE_INTERMEDIATE_SIZE_ALIGNMENT
    ) * FP16_MOE_INTERMEDIATE_SIZE_ALIGNMENT

    fc1_list = []
    fc2_list = []
    for expert in experts:
        up_w = expert.up_proj.weight.detach().to(dtype)  # [inter, H]
        down_w = expert.down_proj.weight.detach().to(dtype)  # [H, inter]
        if tuple(up_w.shape) != (moe_inter_size, hidden_size):
            raise ValueError(f"up weight shape {tuple(up_w.shape)} != "
                             f"({moe_inter_size}, {hidden_size})")
        if tuple(down_w.shape) != (hidden_size, moe_inter_size):
            raise ValueError(f"down weight shape {tuple(down_w.shape)} != "
                             f"({hidden_size}, {moe_inter_size})")

        up_pad = torch.zeros((padded_inter_size, hidden_size), dtype=dtype)
        up_pad[:moe_inter_size] = up_w
        down_pad = torch.zeros((hidden_size, padded_inter_size), dtype=dtype)
        down_pad[:, :moe_inter_size] = down_w

        if activation_type == _ACT_RELU2:
            fc1_list.append(up_pad)  # [I, H], ungated
        else:
            gate_w = expert.gate_proj.weight.detach().to(dtype)  # [inter, H]
            if tuple(gate_w.shape) != (moe_inter_size, hidden_size):
                raise ValueError(f"gate weight shape {tuple(gate_w.shape)} != "
                                 f"({moe_inter_size}, {hidden_size})")
            gate_pad = torch.zeros((padded_inter_size, hidden_size),
                                   dtype=dtype)
            gate_pad[:moe_inter_size] = gate_w
            # 64-row up/gate interleave: [up0:64, gate0:64, up64:128, ...].
            chunk = 64
            n_chunks = padded_inter_size // chunk
            up_chunks = up_pad.reshape(n_chunks, chunk, hidden_size)
            gate_chunks = gate_pad.reshape(n_chunks, chunk, hidden_size)
            fc1_list.append(
                torch.stack([up_chunks, gate_chunks],
                            dim=1).reshape(2 * padded_inter_size, hidden_size))
        fc2_list.append(down_pad)  # [H, I]

    fc1_weights = torch.stack(fc1_list, dim=0).contiguous()
    fc2_weights = torch.stack(fc2_list, dim=0).contiguous()
    return fc1_weights, fc2_weights, padded_inter_size


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
