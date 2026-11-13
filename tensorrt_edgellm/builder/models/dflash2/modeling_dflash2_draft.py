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


def _target_logit_transform(values, cfg):
    """Apply the target's output multiplier and logit softcap to the support values.

    Both maps are monotonic, so applying them after top-k keeps the support ids.
    """
    if cfg.dflash2_output_multiplier != 1.0:
        values = values * np.float32(cfg.dflash2_output_multiplier)
    if cfg.dflash2_final_logit_softcapping > 0.0:
        cap = np.float32(cfg.dflash2_final_logit_softcapping)
        values = (values / cap).tanh() * cap
    return values


_VOCAB_TOPK_CHUNK_RANGE = (2048, 8192)


def _vocab_topk(logits, k: int):
    """TopK over the last axis of ``[batch, rows, vocab]`` logits, per vocabulary chunk first.

    TensorRT runs a vocabulary-wide TopK as one 128-thread block per row. The per-chunk winners contain the global
    top-k, so the values equal a single TopK's; at exact ties on the k boundary another tied id may be chosen.
    """
    vocab_size = logits.shape[-1]
    low, high = _VOCAB_TOPK_CHUNK_RANGE
    chunk = next((c for c in range(max(low, k),
                                   min(high, vocab_size // 2) + 1)
                  if vocab_size % c == 0), None)
    if chunk is None:
        return F.topk(logits, k, 2)
    chunks = vocab_size // chunk
    shape = F.shape_of(logits)
    lead = (shape[0:1], shape[1:2])
    values, local = F.topk(F.dynamic_reshape(logits, (*lead, chunks, chunk)),
                           k, 3)
    offsets = (np.arange(chunks, dtype=np.int32) * chunk).reshape(
        1, 1, chunks, 1)
    ids = F.dynamic_reshape(
        local.cast(trt.int32) + offsets, (*lead, chunks * k))
    top_values, winners = F.topk(
        F.dynamic_reshape(values, (*lead, chunks * k)), k, 2)
    return top_values, F.gather_elements(ids, winners, 2)


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


class _BlockView:
    """Token and [batch, block] extents of the draft input.

    Every token-major <-> block-major reshape uses these input-derived extents. Deriving the block from each
    intermediate tensor instead lets TensorRT's interval bounds grow by max/min batch per round trip, which at
    maxBatchSize 8 pushes the per-sequence token bound to about 2^24 and breaks plugin workspace sizing.
    """

    def __init__(self, inputs_embeds, query_lengths) -> None:
        self.tokens = F.shape_of(inputs_embeds)[0:1]
        self.batch = F.shape_of(query_lengths)[0:1]
        self.block = self.tokens / self.batch

    def block_major(self, hidden, hidden_size: int):
        return F.dynamic_reshape(hidden, (self.batch, self.block, hidden_size))

    def token_major(self, hidden, hidden_size: int):
        return F.dynamic_reshape(hidden, (self.tokens, hidden_size))


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

    def __init__(self, ctx, prefix: str, layer_index: int) -> None:
        super().__init__(ctx, prefix, layer_index)
        self.attention_conv = DFlash2GroupedConv(ctx,
                                                 self.key("attention_conv"))
        self.mlp_conv = DFlash2GroupedConv(ctx, self.key("mlp_conv"))

    def forward(self, hidden, hidden_delta, past, rope, ragged, delta_rope,
                delta_positions, delta_token_to_sequence, attention_mask,
                attention_pos_id, view: _BlockView):
        hidden_size = self.cfg.hidden_size
        hidden = view.block_major(hidden, hidden_size)
        attention_input = self.input_norm(hidden)
        attention_input, attention_post = self.attention_conv.prepare(
            attention_input)
        attention, present = self.attention(
            view.token_major(attention_input, hidden_size), hidden_delta, past,
            rope, ragged, delta_rope, delta_positions, delta_token_to_sequence,
            attention_mask, attention_pos_id)
        attention = view.block_major(attention, hidden_size)
        hidden = self._finish(self.attention_conv, attention, attention_post,
                              hidden)

        mlp_input = self.post_norm(hidden)
        mlp_input, mlp_post = self.mlp_conv.prepare(mlp_input)
        feed_forward = self.mlp(mlp_input)
        hidden = self._finish(self.mlp_conv, feed_forward, mlp_post, hidden)
        return view.token_major(hidden, hidden_size), present

    def _finish(self, conv, output, post, residual):
        return conv.finish(output, post, residual)


class DFlash2DraftModel(DFlashDraftModel):
    """DFlash2 backbone plus target head and TopK selector inputs."""

    layer_class = DFlash2DecoderLayer

    @classmethod
    def from_config(cls, ctx):
        # A LiLiCorr checkpoint is a DFlash2 draft with a different projector.
        if cls is DFlash2DraftModel and ctx.cfg.is_lilicorr:
            from ..lilicorr.modeling_lilicorr_draft import LiLiCorrDraftModel
            return LiLiCorrDraftModel.from_config(ctx)
        return super().from_config(ctx)

    def __init__(self, ctx, lm_head=None) -> None:
        # DFlashDraftModel.from_config owns target lm_head loading and calls this
        # constructor, but the DFlash2 layer/head topology is intentionally
        # explicit rather than conditionally changing the DFlash v1 graph.
        Module.__init__(self, ctx)
        self._base_weights = None
        self.fc = self._target_projection(ctx)
        self.hidden_norm = RMSNorm(ctx, "hidden_norm", ctx.cfg.rms_norm_eps)
        self.layers = [
            self.layer_class(ctx, f"layers.{index}", index)
            for index in range(ctx.cfg.num_hidden_layers)
        ]
        self.norm = RMSNorm(ctx, "norm", ctx.cfg.rms_norm_eps)
        self.lm_head = _resolve_target_lm_head(ctx, lm_head)
        self._init_projector(ctx)

    def _target_projection(self, ctx):
        return DFlashTargetProjection(ctx, "fc", widen_at_load=True)

    def _init_projector(self, ctx) -> None:
        self.selector_projection = Linear(
            ctx, "candidate_selector.hidden_projection", tensor_parallel=False)

    def _prediction_hidden(self, io):
        """Return ``(prediction_hidden [B, block - 1, H], delta, present KV)``."""
        # The official BF16 checkpoint's residual stream exceeds FP16 range.
        # Keep it in FP32; RMSNorm returns FP16 activations for the heavy ops.
        hidden = io["inputs_embeds"].cast(trt.float32)
        delta = self.hidden_norm(self.fc(io["base_hidden"]))
        ragged = RaggedDecoderInputs.from_dict(io)
        view = _BlockView(io["inputs_embeds"], ragged.query_lengths)
        present = []
        for index, layer in enumerate(self.layers):
            hidden, cache = layer(hidden,
                                  delta,
                                  io["past_key_values"][index],
                                  io["rope"],
                                  ragged,
                                  io["delta_rope"],
                                  io["delta_positions"],
                                  io["delta_token_to_sequence"],
                                  io["attention_mask"],
                                  io["attention_pos_id"],
                                  view=view)
            present.append(cache)

        hidden = view.block_major(self.norm(hidden), self.cfg.hidden_size)
        return _drop_anchor_token(hidden, self.cfg.hidden_size), delta, present

    def _proposal_outputs(self, prediction_hidden, delta, io):
        del delta, io
        projected = self.selector_projection(prediction_hidden).cast(
            trt.float16)
        unary_logits = self.lm_head(prediction_hidden).cast(trt.float32)
        unary_values, candidate_ids = _vocab_topk(
            unary_logits, self.cfg.dflash2_selector_top_k)
        unary_values = _target_logit_transform(unary_values, self.cfg)
        return {
            "spec_proposal_support_ids": candidate_ids,
            "spec_proposal_unary_values": unary_values,
            "spec_proposal_projected_hidden": projected,
        }

    def forward(self, **io):
        prediction_hidden, delta, present = self._prediction_hidden(io)
        outputs = self._proposal_outputs(prediction_hidden, delta, io)
        for index, tensor in enumerate(present):
            outputs[f"present_key_values_{index}"] = tensor
        return outputs
