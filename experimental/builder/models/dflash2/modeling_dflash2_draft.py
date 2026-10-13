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
"""Qwen3.8 DFlash2 draft graph with production TensorRT plugins."""

import numpy as np
import tensorrt as trt

from ...ops import Linear, Module, RMSNorm
from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs
from ..dflash.modeling_dflash_draft import (DFlashDecoderLayer,
                                            DFlashDraftModel,
                                            DFlashTargetProjection)

_MAX_BLOCK_SIZE = 16


def _resolve_target_lm_head(ctx, lm_head):
    """Use an injected paired head or the quantized head stored in the draft."""
    if lm_head is not None:
        return lm_head
    if (ctx.weights.has("lm_head.weight")
            or ctx.weights.has("lm_head.qweight")):
        return Linear(ctx, ctx.weights.causal_lm_head_prefix())
    raise ValueError(
        "DFlash2 requires the paired target lm_head inside the draft engine")


def _drop_anchor_token(hidden, hidden_size: int):
    shape = F.shape_of(hidden)
    batch = shape[0:1]
    block = shape[1:2]
    one = F.constant(np.asarray([1], dtype=np.int32), "one")
    return F.dynamic_slice(hidden, (0, one, 0),
                           (batch, block - one, hidden_size))


def _to_block_major(hidden, query_lengths, hidden_size: int):
    physical_tokens = F.shape_of(hidden)[0:1]
    batch = F.shape_of(query_lengths)[0:1]
    block = physical_tokens / batch
    return F.dynamic_reshape(hidden, (batch, block, hidden_size))


class DFlash2GroupedConv(Module):
    """One shared coefficient GEMM and two optimized convolution applies."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        cfg = ctx.cfg
        self.block_size = _MAX_BLOCK_SIZE
        self.kernel_size = cfg.dflash2_conv_kernel_size
        self.group_size = cfg.dflash2_conv_group_size
        if cfg.hidden_size % self.group_size:
            raise ValueError("DFlash2 conv group_size must divide hidden_size")
        self.num_groups = cfg.hidden_size // self.group_size
        self.kernel_projection = Linear(ctx,
                                        self.key("kernel_projection"),
                                        tensor_parallel=False)
        base = self.weights.f16(self.key("base_kernel"))
        expected = (2, self.kernel_size, cfg.hidden_size)
        if tuple(base.shape) != expected:
            raise ValueError(
                f"{self.key('base_kernel')} shape must be {expected}, got {base.shape}"
            )
        self.pre_base = base[0]
        self.post_base = base[1]

    def _apply(self, hidden, delta, base, side: str, residual=None):
        return F.dflash2_grouped_dynamic_conv(
            hidden,
            delta,
            F.constant(base, f"{self.prefix}.{side}_base_kernel"),
            residual,
            block_size=self.block_size,
            kernel_size=self.kernel_size,
            group_size=self.group_size)

    def prepare(self, hidden):
        coefficients = self.kernel_projection(hidden)
        side_size = self.kernel_size * self.num_groups
        pre = F.slice_last_dim(coefficients, 0, side_size, 3).reshape(
            (0, 0, self.kernel_size, self.num_groups))
        post = F.slice_last_dim(coefficients, side_size, side_size, 3).reshape(
            (0, 0, self.kernel_size, self.num_groups))
        return self._apply(hidden, pre, self.pre_base, "pre"), post

    def finish(self, hidden, post, residual):
        return self._apply(hidden, post, self.post_base, "post", residual)


class DFlash2DecoderLayer(DFlashDecoderLayer):
    """DFlash layer with dynamic conv around attention and MLP only."""

    def __init__(self, ctx, prefix: str) -> None:
        super().__init__(ctx, prefix)
        self.attention_conv = DFlash2GroupedConv(ctx,
                                                 self.key("attention_conv"))
        self.mlp_conv = DFlash2GroupedConv(ctx, self.key("mlp_conv"))

    def forward(self, hidden, hidden_delta, past, rope, ragged, delta_rope,
                delta_positions, delta_token_to_sequence, attention_mask,
                attention_pos_id):
        hidden_size = self.cfg.hidden_size
        hidden = _to_block_major(hidden, ragged.query_lengths, hidden_size)
        attention_input = self.input_norm(hidden)
        attention_input, attention_post = self.attention_conv.prepare(
            attention_input)
        attention, present = self.attention(
            F.dynamic_reshape(attention_input, (-1, hidden_size)),
            hidden_delta, past, rope, ragged, delta_rope, delta_positions,
            delta_token_to_sequence, attention_mask, attention_pos_id)
        attention = _to_block_major(attention, ragged.query_lengths,
                                    hidden_size)
        hidden = self.attention_conv.finish(attention, attention_post, hidden)

        mlp_input = self.post_norm(hidden)
        mlp_input, mlp_post = self.mlp_conv.prepare(mlp_input)
        feed_forward = self.mlp(mlp_input)
        hidden = self.mlp_conv.finish(feed_forward, mlp_post, hidden)
        return F.dynamic_reshape(hidden, (-1, hidden_size)), present


class DFlash2DraftModel(DFlashDraftModel):
    """DFlash2 backbone plus target head and TopK selector inputs."""

    def __init__(self, ctx, lm_head=None) -> None:
        # DFlashDraftModel.from_config owns target lm_head loading and calls this
        # constructor, but the DFlash2 layer/head topology is intentionally
        # explicit rather than conditionally changing the DFlash v1 graph.
        Module.__init__(self, ctx)
        self._base_weights = None
        self.fc = DFlashTargetProjection(ctx, "fc")
        self.hidden_norm = RMSNorm(ctx, "hidden_norm", ctx.cfg.rms_norm_eps)
        self.layers = [
            DFlash2DecoderLayer(ctx, f"layers.{index}")
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx, "norm", ctx.cfg.rms_norm_eps)
        self.lm_head = _resolve_target_lm_head(ctx, lm_head)
        self.selector_projection = Linear(
            ctx, "candidate_selector.hidden_projection", tensor_parallel=False)

    def forward(self, **io):
        # The official BF16 checkpoint's residual stream exceeds FP16 range.
        # Keep it in FP32; RMSNorm returns FP16 activations for the heavy ops.
        hidden = io["inputs_embeds"].cast(trt.float32)
        delta = self.hidden_norm(self.fc(io["base_hidden"]))
        ragged = RaggedDecoderInputs.from_dict(io)
        present = []
        for index, layer in enumerate(self.layers):
            hidden, cache = layer(hidden, delta, io["past_key_values"][index],
                                  io["rope"], ragged, io["delta_rope"],
                                  io["delta_positions"],
                                  io["delta_token_to_sequence"],
                                  io["attention_mask"], io["attention_pos_id"])
            present.append(cache)

        hidden = _to_block_major(self.norm(hidden), ragged.query_lengths,
                                 self.cfg.hidden_size)
        prediction_hidden = _drop_anchor_token(hidden, self.cfg.hidden_size)
        projected = self.selector_projection(prediction_hidden).cast(
            trt.float16)
        unary_logits = self.lm_head(prediction_hidden).cast(trt.float32)
        unary_values, candidate_ids = F.topk(unary_logits,
                                             self.cfg.dflash2_selector_top_k,
                                             2)
        outputs = {
            "spec_proposal_support_ids": candidate_ids,
            "spec_proposal_unary_values": unary_values,
            "spec_proposal_projected_hidden": projected,
        }
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
