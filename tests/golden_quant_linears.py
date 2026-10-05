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
"""Standalone fake-quant linears for the few-layer numeric golden.

Each ``_Golden*Linear`` mirrors one EdgeLLM quant recipe's forward (NVFP4 / FP8 /
MXFP8 / INT8-SmoothQuant / AWQ / GPTQ) by reusing the exact ``trt::*`` ops the
export emits, so the PyTorch golden reproduces the engine's quantized numerics
rather than a weight-only FP16 dequant. Consumed by ``golden_layer_dump.py``'s
``_load_quantized_state``; kept separate to keep that file focused on
orchestration."""

import torch

# ---------------------------------------------------------------------------
# Quantized (NVFP4 / FP8) golden support
#
# For a compressed checkpoint we keep HF's architecture (attention / cache / rope
# / decode) and only swap the recipe-quantized projections for fake-quant linears
# that reuse EdgeLLM's exact ops (nvfp4_act_qdq / nvfp4_dequantize for NVFP4,
# fp8_quantize / fp8_dequantize for FP8, mxfp8_act_qdq / mxfp8_weight_dq for
# MXFP8), so the golden matches the engine's quantized numerics rather than a
# weight-only FP16 dequant.
# ---------------------------------------------------------------------------


class _GoldenLinearBase(torch.nn.Module):
    """Base for the fake-quant goldens, mirroring ``models.linear.LinearBase``.

    ``_load_quantized_state`` overwrites :attr:`quantize_activations` per
    instance the same way ``make_linear`` does on the export side, so the two
    graphs stay in the same regime.
    """

    quantize_activations: bool = True


class _GoldenNVFP4Linear(_GoldenLinearBase):
    """Standalone NVFP4 fake-quant linear (mirrors NVFP4LinearMethod.apply).

    Holds the compressed buffers (packed fp4 weight + fp8 per-group scale +
    per-tensor scales) and, in forward, fake-quantizes the activation and
    dequantizes the weight via the same ``trt::nvfp4_*`` ops the export emits.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 has_bias: bool,
                 group_size: int = 16) -> None:
        super().__init__()
        self.group_size = group_size
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features // 2, dtype=torch.int8))
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features,
                       in_features // group_size,
                       dtype=torch.float8_e4m3fn))
        # scale-2 / input_scale are per-tensor scalars stored 0-dim in the ckpt.
        self.register_buffer("weight_scale_2",
                             torch.ones((), dtype=torch.float32))
        self.register_buffer("input_scale", torch.ones((),
                                                       dtype=torch.float32))
        if has_bias:
            self.register_buffer(
                "bias", torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from tensorrt_edgellm.models.ops import nvfp4_act_qdq, nvfp4_dequantize
        x = x.to(torch.float16)
        x_dq = nvfp4_act_qdq(
            x, self.input_scale) if self.quantize_activations else x
        w_dq = nvfp4_dequantize(self.weight, self.weight_scale,
                                self.weight_scale_2, self.group_size)
        return torch.nn.functional.linear(x_dq, w_dq, self.bias)


class _GoldenFP8Linear(_GoldenLinearBase):
    """Standalone FP8 (E4M3) fake-quant linear (mirrors FP8Linear.forward).

    Activation: fp8_quantize -> fp8_dequantize (per-tensor); weight: fp8_dequantize.
    Reuses the same ``trt::fp8_*`` ops the export emits.
    """

    def __init__(self, in_features: int, out_features: int,
                 has_bias: bool) -> None:
        super().__init__()
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))
        # Per-tensor scales (stored 0-dim fp32 in the ckpt).
        self.register_buffer("weight_scale", torch.ones((),
                                                        dtype=torch.float32))
        self.register_buffer("input_scale", torch.ones((),
                                                       dtype=torch.float32))
        if has_bias:
            self.register_buffer(
                "bias", torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from tensorrt_edgellm.models.ops import fp8_dequantize, fp8_quantize
        x_dq = x.to(torch.float16)
        if self.quantize_activations:
            x_dq = fp8_dequantize(fp8_quantize(x_dq, self.input_scale),
                                  self.input_scale)
        w_dq = fp8_dequantize(self.weight, self.weight_scale)
        return torch.nn.functional.linear(x_dq, w_dq, self.bias)


class _GoldenMXFP8Linear(_GoldenLinearBase):
    """Standalone MXFP8 fake-quant linear (mirrors MXFP8Linear.forward).

    Weight is FP8 E4M3 with a per-block (block_size=32) E8M0 scale; the activation
    is dynamically quantized at runtime (no stored scale). Reuses the same
    ``trt::mxfp8_*`` ops the export emits.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 has_bias: bool,
                 block_size: int = 32) -> None:
        super().__init__()
        self.block_size = block_size
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn))
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features,
                       in_features // block_size,
                       dtype=torch.uint8))
        if has_bias:
            self.register_buffer(
                "bias", torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from tensorrt_edgellm.models.ops import mxfp8_act_qdq, mxfp8_weight_dq
        x_dq = x.to(torch.float16)
        if self.quantize_activations:
            x_dq = mxfp8_act_qdq(x_dq)
        w_dq = mxfp8_weight_dq(self.weight, self.weight_scale, self.block_size)
        return torch.nn.functional.linear(x_dq, w_dq, self.bias)


class _GoldenINT8SQLinear(_GoldenLinearBase):
    """INT8 SmoothQuant W8A8 fake-quant linear (mirrors INT8SQLinear.forward).

    Per-channel symmetric INT8 weight, per-tensor symmetric INT8 activation, with a
    SmoothQuant per-input-channel ``pre_quant_scale`` smoother applied first. Reuses
    the same ``trt::int8_sq_*`` ops the export emits.
    """

    def __init__(self, in_features: int, out_features: int,
                 has_bias: bool) -> None:
        super().__init__()
        self.register_buffer(
            "weight", torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale",
                             torch.ones(out_features, dtype=torch.float32))
        self.register_buffer("input_scale", torch.ones((),
                                                       dtype=torch.float32))
        self.register_buffer("pre_quant_scale",
                             torch.ones(in_features, dtype=torch.float16))
        if has_bias:
            self.register_buffer(
                "bias", torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from tensorrt_edgellm.models.ops import (int8_sq_act_qdq,
                                                 int8_sq_weight_dq)

        # The smoother stays in both regimes: the weight carries its inverse.
        x_dq = x.to(torch.float16) * self.pre_quant_scale
        if self.quantize_activations:
            x_dq = int8_sq_act_qdq(x_dq, self.input_scale)
        w_dq = int8_sq_weight_dq(self.weight, self.weight_scale)
        return torch.nn.functional.linear(x_dq, w_dq, self.bias)


# AutoAWQ packs 8 output channels per int32 in this bit-position -> channel order
# (mirrors checkpoint/repacking.py:_AWQ_BIT_TO_CH). GPTQ packs sequentially.
_AWQ_BIT_TO_CH = [0, 2, 4, 6, 1, 3, 5, 7]


def _awq_dequantize(qweight: torch.Tensor, qzeros: torch.Tensor,
                    scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """AutoAWQ INT4 -> fp16 weight [out, in]. w = (nibble - zero) * scale per group.

    qweight [in, out//8] int32 (8 nibbles/int32 along out, AWQ bit order); qzeros
    [in//g, out//8] int32 (same order); scales [in//g, out] fp16. Mirrors
    checkpoint/repacking.repack_awq_to_plugin's decode.
    """
    dev = qweight.device
    in_f, out_div8 = qweight.shape
    out_f = out_div8 * 8
    qw = qweight.to(torch.int32)
    qz = qzeros.to(torch.int32)
    nib = torch.zeros(in_f, out_f, dtype=torch.int32, device=dev)
    zero = torch.zeros(qzeros.shape[0], out_f, dtype=torch.int32, device=dev)
    for k in range(8):
        nib[:, _AWQ_BIT_TO_CH[k]::8] = (qw >> (4 * k)) & 0xF
        zero[:, _AWQ_BIT_TO_CH[k]::8] = (qz >> (4 * k)) & 0xF
    zero_exp = zero.repeat_interleave(group_size, dim=0).to(torch.float32)
    scale_exp = scales.to(torch.float32).repeat_interleave(group_size, dim=0)
    w = (nib.to(torch.float32) - zero_exp) * scale_exp  # [in, out]
    return w.t().contiguous().to(torch.float16)  # [out, in]


def _gptq_dequantize(qweight: torch.Tensor,
                     qzeros: torch.Tensor,
                     scales: torch.Tensor,
                     g_idx: "torch.Tensor | None",
                     group_size: int,
                     zero_point_offset: int = 1) -> torch.Tensor:
    """GPTQ INT4 -> fp16 weight [out, in]. w = (nibble - zero - offset) * scale.

    qweight [in//8, out] int32 (8 nibbles/int32 along in, sequential); qzeros
    [in//g, out//8] int32 (sequential); scales [in//g, out]; g_idx [in] maps each
    input channel to its group (desc_act). Mirrors repack_gptq_to_plugin's decode;
    the weight stays in original input-channel order so no activation permute is
    needed (unlike the plugin path).
    """
    dev = qweight.device
    in_div8, out_f = qweight.shape
    in_f = in_div8 * 8
    qw = qweight.to(torch.int32)
    qz = qzeros.to(torch.int32)
    nib = torch.zeros(in_f, out_f, dtype=torch.int32, device=dev)
    zero = torch.zeros(qzeros.shape[0], out_f, dtype=torch.int32, device=dev)
    for k in range(8):
        nib[k::8, :] = (qw >> (4 * k)) & 0xF
        zero[:, k::8] = (qz >> (4 * k)) & 0xF
    if g_idx is None:
        g_idx = torch.arange(in_f, device=dev) // group_size
    g = g_idx.to(torch.int64).to(dev)
    zero_exp = zero[g].to(torch.float32)  # [in, out]
    scale_exp = scales.to(torch.float32)[g]  # [in, out]
    w = (nib.to(torch.float32) - zero_exp - zero_point_offset) * scale_exp
    return w.t().contiguous().to(torch.float16)  # [out, in]


class _GoldenAWQLinear(_GoldenLinearBase):
    """AutoAWQ INT4 W4A16 fake-quant linear. Dequantizes the weight to fp16 and does
    a plain fp16 matmul (AWQ does not quantize activations)."""

    def __init__(self, in_features: int, out_features: int, group_size: int,
                 has_bias: bool) -> None:
        super().__init__()
        self.group_size = group_size
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
        if has_bias:
            self.register_buffer(
                "bias", torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _awq_dequantize(self.qweight, self.qzeros, self.scales,
                            self.group_size)
        return torch.nn.functional.linear(x.to(torch.float16), w, self.bias)


def _detect_gptq_zero_point_offset(qzeros: torch.Tensor) -> int:
    """GPTQ qzeros stores either ``zero`` or ``zero - 1`` (real zero point 8).

    Self-describing from the packed nibbles (mirrors config._detect_gptq_zero_point_offset):
    all-8 (sym, packed 0x888...) -> offset 0; otherwise (sym-stored-7 or asym) -> 1.
    """
    qz = qzeros.flatten()[:1024].to(torch.int64)
    nibs = torch.cat([(qz >> (4 * i)) & 0xF for i in range(8)])
    return 0 if bool((nibs == 8).all()) else 1


class _GoldenGPTQLinear(_GoldenLinearBase):
    """GPTQ INT4 W4A16 fake-quant linear (weight-only, no activation quant)."""

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 group_size: int,
                 has_bias: bool,
                 zero_point_offset: int = 1) -> None:
        super().__init__()
        self.group_size = group_size
        self.zero_point_offset = zero_point_offset
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
        self.register_buffer(
            "g_idx",
            torch.arange(in_features, dtype=torch.int32) // group_size)
        if has_bias:
            self.register_buffer(
                "bias", torch.empty(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = _gptq_dequantize(self.qweight, self.qzeros, self.scales,
                             self.g_idx, self.group_size,
                             self.zero_point_offset)
        return torch.nn.functional.linear(x.to(torch.float16), w, self.bias)
