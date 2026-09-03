# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Quantized linear modules: checkpoint buffer names match safetensors keys.


Forwards call :mod:`ops` stubs; ONNX export maps them via registered symbolics.

Layouts (buffers): FP8 — ``weight`` ``[out,in]`` fp8, scales fp16; NVFP4 —
packed uint8 weights and fp8 block scales; AWQ — ``qweight`` ``[in,out//8]``,
``qzeros``, ``scales``; GPTQ — ``qweight`` ``[in//8,out]``, ``qzeros``,
``scales``, ``g_idx``; W4A16 prepacked — ``weight`` ``[out//2,in]`` uint8,
``weight_scale``, optional ``pre_quant_scale``; FP16 — ``weight`` ``[out,in]``.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (QUANT_FP8, QUANT_FP16, QUANT_INT4_AWQ,
                      QUANT_INT4_AWQ_MODELOPT, QUANT_INT4_GPTQ, QUANT_INT8_SQ,
                      QUANT_MXFP8, QUANT_NVFP4, QUANT_NVFP4_A16, Mapping,
                      ModelConfig, module_quant_type)
from .ops import (fp8_dequantize, fp8_quantize, fused_nvfp4_gemm_allreduce,
                  int4_gemm_plugin_version, int4_groupwise_gemm,
                  int4_groupwise_gemm_v2, int8_sq_act_qdq, int8_sq_weight_dq,
                  mxfp8_act_qdq, mxfp8_weight_dq, nvfp4_a16_gemm,
                  nvfp4_act_qdq, nvfp4_dequantize)

logger = logging.getLogger(__name__)


def _require_fp16_input(hidden_states: torch.Tensor, layer_name: str) -> None:
    if hidden_states.dtype != torch.float16:
        raise TypeError(
            f"{layer_name} expects float16 input, got {hidden_states.dtype}")


def _int4_groupwise_gemm_dispatch(hidden_states, qweight, scales, gemm_n,
                                  gemm_k, group_size):
    """Emit the INT4 groupwise GEMM custom op for the selected plugin backend.

    ``int4_gemm_plugin_version()`` (CLI ``--int4-gemm-plugin-version``) picks the cuteDSL
    ``Int4GroupwiseGemmPluginV2`` (default) or the legacy AWQ
    ``Int4GroupwiseGemmPlugin``. The weight repack in ``checkpoint/repacking.py``
    reads the same selector so the emitted op matches the weight layout.
    """
    if int4_gemm_plugin_version() == 2:
        return int4_groupwise_gemm_v2(hidden_states, qweight, scales, gemm_n,
                                      gemm_k, group_size)
    return int4_groupwise_gemm(hidden_states, qweight, scales, gemm_n, gemm_k,
                               group_size)


__all__ = [
    "LinearBase",
    "LinearMethodBase",
    "NVFP4LinearMethod",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "is_nvfp4_linear",
    "is_int4_linear",
    "FP16Linear",
    "FP8Linear",
    "MXFP8Linear",
    "AWQLinear",
    "ModelOptAWQPrepackedLinear",
    "NVFP4A16MarlinLinear",
    "GPTQLinear",
    "INT8SQLinear",
    "TPMode",
    "make_linear",
]


class TPMode(str, Enum):
    """Tensor-parallel sharding mode tag.
    """
    REPLICATED = "replicated"
    COL = "col"
    ROW = "row"


# ---------------------------------------------------------------------------
# LinearBase
# ---------------------------------------------------------------------------


class LinearBase(nn.Module):
    """Common base for quantized / TP-aware linear layers."""

    def tp_split_dim(self, attr: str) -> Optional[int]:
        """Axis to shard *attr* along under TP, or None if replicated.

        Default: every attribute is replicated. Subclasses override
        (directly or via composed :class:`LinearMethodBase`). The base
        loader handles the actual slice
        (:func:`checkpoint.loader._shard_for_module`).
        """
        return None


# ---------------------------------------------------------------------------
# LinearMethodBase
# ---------------------------------------------------------------------------


class LinearMethodBase(ABC):
    """Per-quant-format extension point.

    Owns buffer allocation, forward kernel, per-rank shard, and any
    quant-specific post-processing. The Linear class (Col / Row) owns the
    TP mode and forward shell (which ``apply_*`` to call).
    """

    @abstractmethod
    def create_weights(self, module: "LinearBase", in_features: int,
                       out_features: int, bias: bool,
                       dtype: torch.dtype) -> None:
        ...

    @abstractmethod
    def apply(self, module: "LinearBase", x: torch.Tensor) -> torch.Tensor:
        ...

    def apply_linear_allreduce(self, module: "LinearBase",
                               x: torch.Tensor) -> torch.Tensor:
        """Forward for row-parallel layers (local GEMM + AllReduce).

        Subclasses with a fused gemm+allreduce kernel override.
        Default raises so missing-row-parallel-support is loud, not silent.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support row-parallel forward")

    def shardable_attrs(self, tp_mode: TPMode) -> set:
        """Buffer names that participate in TP under the given tp_mode."""
        return set()


# ---------------------------------------------------------------------------
# FP16Linear
# ---------------------------------------------------------------------------


class FP16Linear(LinearBase):
    """Plain float16 linear (embed_tokens, lm_head, non-quantised).

    Activations must be float16 (no bfloat16/float32 inputs for now).
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features,
                                               in_features,
                                               dtype=torch.float16),
                                   requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features,
                                                 dtype=torch.float16),
                                     requires_grad=False)
        else:
            self.register_parameter("bias", None)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "FP16Linear")
        bias = self.bias if self.bias is not None else None
        return F.linear(hidden_states, self.weight, bias)


# ---------------------------------------------------------------------------
# FP8Linear
# ---------------------------------------------------------------------------


class FP8Linear(LinearBase):
    """FP8 E4M3 linear.

    Export emits standard ONNX ``QuantizeLinear`` (``output_dtype=float8e4m3fn``)
    and ``DequantizeLinear`` with per-tensor **float16** scales, then ``MatMul``
    (via ``F.linear``).  NVFP4 is unchanged elsewhere in this package.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty(out_features, in_features))
        # Per-tensor scales declared as 0-dim scalars to match ModelOpt's
        # unified-checkpoint layout (``shape=[]``).  Avoids the rank mismatch
        # that previously forced visual ``_load_weights`` callers to reshape
        # source tensors with ``view(1)`` before assignment.
        self.register_buffer("weight_scale", torch.ones((),
                                                        dtype=torch.float16))
        self.register_buffer("input_scale", torch.ones((),
                                                       dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "FP8Linear")
        # Activation: ONNX QuantizeLinear -> FP8 + DequantizeLinear -> FP16
        hidden_states_q = fp8_quantize(hidden_states, self.input_scale)
        hidden_states_dq = fp8_dequantize(hidden_states_q, self.input_scale)
        # Weight: DQ FP8 -> FP16 (standard ONNX DequantizeLinear)
        w_fp16 = fp8_dequantize(self.weight, self.weight_scale)
        bias = self.bias.to(torch.float16) if self.bias is not None else None
        return F.linear(hidden_states_dq, w_fp16, bias)


# ---------------------------------------------------------------------------
# NVFP4LinearMethod
# ---------------------------------------------------------------------------


class NVFP4LinearMethod(LinearMethodBase):
    """NVFP4 (FP4 E2M1) with FP8 per-group scales.

    Owns the NVFP4 buffer layout (weight + per-group fp8 scale +
    global scale + activation scale), the dequant+matmul forward, the
    FusedNvfp4GemmAllReduce row-parallel forward, and the per-mode shard
    declaration. Used by both :class:`ColumnParallelLinear` and
    :class:`RowParallelLinear` via composition.
    """

    def __init__(self, group_size: int = 16) -> None:
        self.group_size: int = group_size

    def create_weights(self, module: "LinearBase", in_features: int,
                       out_features: int, bias: bool,
                       dtype: torch.dtype) -> None:
        num_groups = in_features // self.group_size
        # forward path needs group_size on the module
        module.group_size = self.group_size
        # weight: packed fp4 stored as int8 (same bits as uint8, TRT needs int8)
        module.register_buffer(
            "weight",
            torch.empty(out_features, in_features // 2, dtype=torch.int8))
        # Per-group fp8 scale [out, num_groups]
        module.register_buffer(
            "weight_scale",
            torch.ones(out_features, num_groups, dtype=torch.float32))
        module.register_buffer("weight_scale_2", torch.ones(1))
        module.register_buffer("input_scale", torch.ones(1))
        if bias:
            module.register_buffer("bias", torch.empty(out_features))
        else:
            module.bias = None

    def apply(self, module: "LinearBase", x: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(x, type(module).__name__)
        # Activation: DynQ + 2x trt::DQ -> float16 activations
        x_dq = nvfp4_act_qdq(x, module.input_scale)
        # Weight: 2xstandard-ONNX DQ -> w_dq (float16)
        w_dq = nvfp4_dequantize(module.weight, module.weight_scale,
                                module.weight_scale_2, module.group_size)
        bias = module.bias.to(
            torch.float16) if module.bias is not None else None
        return F.linear(x_dq, w_dq, bias)

    def apply_linear_allreduce(self, module: "LinearBase",
                               x: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(x, type(module).__name__)
        # Single op: TRT_FP4DynamicQuantize + DequantizeLinear +
        # FusedNvfp4GemmAllReducePlugin. Output is FP16, already AllReduced.
        out = fused_nvfp4_gemm_allreduce(
            x,
            module.input_scale,
            module.weight,
            module.weight_scale,
            module.weight_scale_2,
            tp_size=module.tp_size,
        )
        if module.bias is not None:
            out = out + module.bias.to(torch.float16)
        return out

    def shardable_attrs(self, tp_mode: TPMode) -> set:
        if tp_mode == TPMode.COL:
            return {"weight", "weight_scale", "bias"}
        if tp_mode == TPMode.ROW:
            return {"weight", "weight_scale"}
        return set()


# ---------------------------------------------------------------------------
# ColumnParallelLinear / RowParallelLinear  (TP-aware Linear classes)
# ---------------------------------------------------------------------------


class ColumnParallelLinear(LinearBase):
    """Column-parallel or replicated linear.

    Shards the weight [out_features, in_features] along dim 0, so each rank owns
    ``out_features / tp_size`` output channels. No collective on the
    forward path because the output is already per-rank.

    ``out_features`` is the per-rank output size. The caller is expected
    to pass values already adjusted by :meth:`ModelConfig.for_rank` when
    ``tp_size>1``.
    When ``tp_mode=REPLICATED`` (or ``tp_size==1``), this class is the
    degenerate non-TP case, which is what :class:`ReplicatedLinear`
    selects.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 dtype: torch.dtype,
                 mapping: Mapping,
                 quant_method: LinearMethodBase,
                 tp_mode: TPMode = TPMode.COL) -> None:
        super().__init__()
        self.mapping = mapping or Mapping()
        self.tp_mode = TPMode(tp_mode)
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.in_features = in_features
        self.out_features = out_features
        self.quant_method = quant_method
        self.quant_method.create_weights(self, in_features, out_features, bias,
                                         dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x)

    def tp_split_dim(self, attr: str) -> Optional[int]:
        return 0 if attr in self.quant_method.shardable_attrs(
            self.tp_mode) else None


class ReplicatedLinear(ColumnParallelLinear):
    """Non-TP linear. Same compute path as ColumnParallelLinear with
    ``tp_mode=REPLICATED``. The full weight is loaded on every rank and
    no sharding happens at load time (the empty
    :meth:`LinearMethodBase.shardable_attrs` for REPLICATED makes
    :func:`tp_split_dim` return ``None`` for every attr).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool,
                 dtype: torch.dtype, mapping: Mapping,
                 quant_method: LinearMethodBase) -> None:
        super().__init__(in_features,
                         out_features,
                         bias,
                         dtype,
                         mapping,
                         quant_method,
                         tp_mode=TPMode.REPLICATED)


class RowParallelLinear(LinearBase):
    """Row-parallel linear with AllReduce on output.

    Shards the weight [out_features, in_features] along dim 1: each rank owns
    ``in_features / tp_size`` input channels and computes a partial sum.
    An AllReduce across ranks turns the partial sums into the full
    output.
    ``in_features`` is the per-rank input size. The caller is expected
    to pass values already adjusted by :meth:`ModelConfig.for_rank` when
    ``tp_size>1``.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool,
                 dtype: torch.dtype, mapping: Mapping,
                 quant_method: LinearMethodBase) -> None:
        super().__init__()
        self.mapping = mapping
        self.tp_mode = TPMode.ROW
        self.tp_size = self.mapping.tp_size
        self.tp_rank = self.mapping.tp_rank
        self.in_features = in_features
        self.out_features = out_features
        self.quant_method = quant_method
        self.quant_method.create_weights(self, in_features, out_features, bias,
                                         dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply_linear_allreduce(self, x)

    def tp_split_dim(self, attr: str) -> Optional[int]:
        return 1 if attr in self.quant_method.shardable_attrs(
            TPMode.ROW) else None


def is_nvfp4_linear(module: nn.Module) -> bool:
    """True if module is an NVFP4-quantized linear (col or row parallel).

    Replaces ``isinstance(module, NVFP4Linear)`` call-sites after the
    LinearMethodBase migration.
    """
    return (isinstance(module, LinearBase) and isinstance(
        getattr(module, "quant_method", None), NVFP4LinearMethod))


# ---------------------------------------------------------------------------
# MXFP8Linear
# ---------------------------------------------------------------------------


class MXFP8Linear(LinearBase):
    """MXFP8 E4M3 linear with E8M0 per-block scales (block_size=32).

    Expects a **unified checkpoint** from ``modelopt.torch.export.export_hf_checkpoint``
    (ModelOpt >= 0.42.0) where weights are already FP8E4M3 and E8M0 scales
    are pre-computed.

    Activation is dynamically quantized at runtime via
    ``TRT_MXFP8DynamicQuantize`` -> ``TRT_MXFP8DequantizeLinear``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 32,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        num_blocks = in_features // block_size

        # Weight: FP8E4M3 [out, in] — loaded from unified checkpoint.
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))
        # E8M0 scale: UINT8 [out, in // block_size]
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, num_blocks, dtype=torch.uint8))
        if bias:
            self.register_buffer("bias", torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "MXFP8Linear")
        # Activation: DynQ + DQ -> float16
        hidden_states_dq = mxfp8_act_qdq(hidden_states)
        # Weight: DQ FP8+E8M0 -> float16
        w_dq = mxfp8_weight_dq(self.weight, self.weight_scale, self.block_size)
        bias = self.bias.to(torch.float16) if self.bias is not None else None
        return F.linear(hidden_states_dq, w_dq, bias)


# ---------------------------------------------------------------------------
# AWQLinear
# ---------------------------------------------------------------------------


class AWQLinear(LinearBase):
    """INT4 AWQ linear (column-packed checkpoint layout).

    Checkpoints use ``qweight`` ``[in, out//8]`` int32; :func:`loader.load_weights`
    repacks to ``[out//2, in]`` int8 for the custom int4 GEMM op.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        # Initialized in AWQ layout [in, out//8] int32; loader repacks to
        # [out//2, in] int8 (swizzled plugin layout) before inference/export.
        self.register_buffer(
            "qweight",
            torch.zeros(in_features, out_features // 8, dtype=torch.int32))
        self.register_buffer(
            "qzeros",
            torch.zeros(in_features // group_size,
                        out_features // 8,
                        dtype=torch.int32))
        self.register_buffer(
            "scales",
            torch.ones(in_features // group_size,
                       out_features,
                       dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "AWQLinear")
        out = _int4_groupwise_gemm_dispatch(
            hidden_states,
            self.qweight,
            self.scales.to(torch.float16),
            self.out_features,
            self.in_features,
            self.group_size,
        )
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# ModelOptAWQPrepackedLinear
# ---------------------------------------------------------------------------


class ModelOptAWQPrepackedLinear(LinearBase):
    """W4A16 AWQ linear with prepacked uint8 weights.

    Checkpoint: ``weight`` ``[out//2, in]`` uint8, ``weight_scale`` ``[out, in//g]``
    float32, optional ``pre_quant_scale`` ``[in]`` float16. The loader repacks
    weights to int8 and transposes scales to ``[in//g, out]`` float16.

    Buffers: ``weight``, ``weight_scale``, ``pre_quant_scale`` (ones if absent).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        # Loaded from checkpoint as uint8; cast to int8 by the loader.
        self.register_buffer(
            "weight",
            torch.zeros(out_features // 2, in_features, dtype=torch.int8))
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features,
                       in_features // group_size,
                       dtype=torch.float16))
        # Optional per-input-channel AWQ smoothing scale.  The loader assigns
        # it from the checkpoint when present; the default of ones is a no-op.
        self.register_buffer("pre_quant_scale",
                             torch.ones(in_features, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "ModelOptAWQPrepackedLinear")
        hidden_states = hidden_states * self.pre_quant_scale
        out = _int4_groupwise_gemm_dispatch(
            hidden_states,
            self.weight,
            self.weight_scale,
            self.out_features,
            self.in_features,
            self.group_size,
        )
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# NVFP4A16MarlinLinear  (compressed-tensors NVFP4 W4A16, dense Marlin)
# ---------------------------------------------------------------------------


class NVFP4A16MarlinLinear(LinearBase):
    """Dense NVFP4 (W4A16) linear backed by the Marlin ``Nvfp4A16GemmPlugin``.

    Checkpoint buffers (ModelOpt ``W4A16_NVFP4``) load raw and are transformed
    in place by :func:`repacking.repack_nvfp4_a16_marlin_linear`:

      ``weight``         [N, K//2]  uint8  -> ``qweight``      [1, K//16, 8*N_pad] int8
      ``weight_scale``   [N, K//16] f8e4m3 -> ``block_scales`` [1, K//16, N_pad]    int8
      ``weight_scale_2`` scalar     f32    -> ``global_scale`` [1]                   fp16

    ``out_features`` is the logical N; the plugin emits the padded width and the
    forward slices back to ``out_features``.

    Activations stay FP16 end-to-end: the Marlin kernel has an FP16 (half2)
    E2M1 path, so no BF16 cast is needed. ``global_scale`` is repacked as FP16
    (pre-scaled by ``2**7``).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 16,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        # Raw ModelOpt checkpoint buffers (populated by the loader, then repacked).
        self.register_buffer(
            "weight",
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8))
        self.register_buffer(
            "weight_scale",
            torch.zeros(out_features,
                        in_features // group_size,
                        dtype=torch.float8_e4m3fn))
        self.register_buffer("weight_scale_2",
                             torch.ones(1, dtype=torch.float32))
        # Padded output width, overwritten by the repacking step.
        self.n_padded = out_features
        if bias:
            self.register_buffer("bias", torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "NVFP4A16MarlinLinear")
        out = nvfp4_a16_gemm(
            hidden_states,
            self.qweight,
            self.block_scales,
            self.global_scale,
            self.n_padded,
            self.in_features,
        )
        if self.n_padded != self.out_features:
            out = out[..., :self.out_features]
        if self.bias is not None:
            out = out + self.bias.to(torch.float16)
        return out


# ---------------------------------------------------------------------------
# GPTQLinear
# ---------------------------------------------------------------------------


class GPTQLinear(LinearBase):
    """GPTQ INT4 linear (symmetric or asymmetric, group-quantized) -- int4_gptq format.

    GPTQ packs 8 int4 nibbles per int32 along the *input* axis (row-packed),
    giving ``qweight [in//8, out]`` -- the opposite orientation from AWQ's
    ``[in, out//8]``.  After loading, :func:`repacking.repack_gptq_to_plugin`
    repacks to ``[out//2, in]`` int8 (the same plugin layout as AWQLinear).

    After load, ``int4_act_perm`` orders hidden dim for the int4 GEMM when
    ``desc_act`` reorders groups (otherwise identity).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 128,
        zero_point_offset: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.zero_point_offset = int(zero_point_offset)
        # Initialized in GPTQ layout [in//8, out] int32; loader repacks to
        # [out//2, in] int8 (swizzled plugin layout) before inference/export.
        self.register_buffer(
            "qweight",
            torch.zeros(in_features // 8, out_features, dtype=torch.int32))
        self.register_buffer(
            "qzeros",
            torch.zeros(in_features // group_size,
                        out_features // 8,
                        dtype=torch.int32))
        self.register_buffer(
            "scales",
            torch.ones(in_features // group_size,
                       out_features,
                       dtype=torch.float16))
        # g_idx maps each input channel to its group; used when repacking (``desc_act``).
        self.register_buffer(
            "g_idx",
            torch.arange(in_features, dtype=torch.int32) // group_size)
        self.register_buffer("int4_act_perm",
                             torch.arange(in_features, dtype=torch.int64))
        if bias:
            self.register_buffer("bias", torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "GPTQLinear")
        perm = self.int4_act_perm
        if perm is not None:
            perm_d = perm.to(device=hidden_states.device, dtype=torch.int64)
            hidden_states = hidden_states.index_select(-1, perm_d)
        out = _int4_groupwise_gemm_dispatch(
            hidden_states,
            self.qweight,
            self.scales.to(torch.float16),
            self.out_features,
            self.in_features,
            self.group_size,
        )
        if self.bias is not None:
            out = out + self.bias
        return out


# ---------------------------------------------------------------------------
# INT8SQLinear
# ---------------------------------------------------------------------------


class INT8SQLinear(LinearBase):
    """INT8 SmoothQuant W8A8 linear (per-channel weight, per-tensor activation).

    Checkpoint layout (keys match exactly):
        weight          [out, in]   int8   -- symmetric per-channel INT8
        weight_scale    [out]       float32 -- per-channel dequant scale
        input_scale     []          float32 -- per-tensor activation scale
        pre_quant_scale [in]        float16 -- SmoothQuant activation smoother

    Forward ONNX pattern (TRT recognises Q-DQ-MatMul for INT8 GEMM fusion)::

        x_smooth = x16 * pre_quant_scale                      # Mul
        x_dq = QDQ(x_smooth, input_scale)                     # Q + DQ + Cast
        w_dq = DequantizeLinear(weight, weight_scale, axis=0)  # DQ + Cast
        output = F.linear(x_dq, w_dq)                         # MatMul
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale",
                             torch.ones(out_features, dtype=torch.float32))
        self.register_buffer("input_scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer("pre_quant_scale",
                             torch.ones(in_features, dtype=torch.float16))
        if bias:
            self.register_buffer("bias", torch.empty(out_features))
        else:
            self.bias = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _require_fp16_input(hidden_states, "INT8SQLinear")
        # SmoothQuant activation smoothing (Mul in ONNX)
        hidden_states_smooth = hidden_states * self.pre_quant_scale
        # Activation QDQ: QuantizeLinear + DequantizeLinear (per-tensor INT8)
        hidden_states_dq = int8_sq_act_qdq(hidden_states_smooth,
                                           self.input_scale)
        # Weight dequantize: DequantizeLinear (per-channel, axis=0)
        w_dq = int8_sq_weight_dq(self.weight, self.weight_scale)
        bias = self.bias.to(torch.float16) if self.bias is not None else None
        return F.linear(hidden_states_dq, w_dq, bias)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def is_int4_linear(module: nn.Module) -> bool:
    """Return whether *module* emits an INT4 groupwise GEMM plugin."""
    return isinstance(module,
                      (AWQLinear, ModelOptAWQPrepackedLinear, GPTQLinear))


def make_linear(
    config: ModelConfig,
    in_features: int,
    out_features: int,
    bias: bool = False,
    module_name: str = "",
    tp_mode: TPMode = TPMode.REPLICATED,
) -> nn.Module:
    """Return the right linear layer class for *module_name* under *config*.

    Effective quant type is resolved by :func:`config.module_quant_type` —
    that helper encodes the full decision tree (``quant.excluded`` forces
    FP16; tied ``lm_head`` with no ``layer_overrides`` entry and an
    unquantized backbone is FP16; otherwise ``layer_overrides`` takes
    precedence over the dominant ``quant_type``).  Effective type is always
    a concrete quant string (``fp8``, ``nvfp4``, ...), never
    ``mixed_precision``.

    Args:
        config:        Model config (provides quant type and group_size).
        in_features:   Input feature dimension.
        out_features:  Output feature dimension.
        bias:          Include bias term.
        module_name:   Module path relative to the model root (e.g. ``"lm_head"``).
        tp_mode:       Tensor-parallel sharding mode: ``'col'`` (output dim
                       sharded, no comm), ``'row'`` (input dim sharded,
                       AllReduce on output) or ``'none'`` (replicated).
                       Only takes effect when ``config.tp_size > 1``.
    """
    quant_type = module_quant_type(module_name, config)

    # NVFP4 routes through the new composition design.
    tp_mode = TPMode(tp_mode)
    if quant_type == QUANT_NVFP4:
        method = NVFP4LinearMethod(group_size=config.quant.group_size)
        if config.tp_size == 1:
            return ReplicatedLinear(in_features, out_features, bias,
                                    torch.float16, config.mapping, method)
        if tp_mode == TPMode.ROW:
            return RowParallelLinear(in_features, out_features, bias,
                                     torch.float16, config.mapping, method)
        return ColumnParallelLinear(in_features,
                                    out_features,
                                    bias,
                                    torch.float16,
                                    config.mapping,
                                    method,
                                    tp_mode=tp_mode)

    if quant_type == QUANT_FP16:
        layer = FP16Linear(in_features, out_features, bias)
    elif quant_type == QUANT_FP8:
        layer = FP8Linear(in_features, out_features, bias)
    elif quant_type == QUANT_MXFP8:
        layer = MXFP8Linear(in_features, out_features, config.quant.group_size,
                            bias)
    elif quant_type == QUANT_INT4_AWQ:
        layer = AWQLinear(in_features, out_features, config.quant.group_size,
                          bias)
    elif quant_type == QUANT_INT4_AWQ_MODELOPT:
        layer = ModelOptAWQPrepackedLinear(in_features, out_features,
                                           config.quant.group_size, bias)
    elif quant_type == QUANT_NVFP4_A16:
        layer = NVFP4A16MarlinLinear(in_features, out_features,
                                     config.quant.group_size, bias)
    elif quant_type == QUANT_INT4_GPTQ:
        layer = GPTQLinear(in_features, out_features, config.quant.group_size,
                           config.quant.gptq_zero_point_offset, bias)
    elif quant_type == QUANT_INT8_SQ:
        layer = INT8SQLinear(in_features, out_features, bias)
    else:
        raise ValueError(f"Unknown quant_type: {quant_type!r}")

    # Tag with TP sharding mode so the checkpoint loader can shard on assignment.
    layer.tp_mode = tp_mode if config.tp_size > 1 else TPMode.REPLICATED
    return layer
