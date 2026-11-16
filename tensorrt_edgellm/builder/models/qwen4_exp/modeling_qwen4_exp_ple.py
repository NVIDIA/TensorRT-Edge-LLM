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
"""Qwen4-Exp hashed n-gram injection (PLE).

Row IDs are a function of ``input_ids`` alone, so the runtime hashes them before the engine runs
and hands in a row map into its embedding pool; the table itself stays in the checkpoint, which is
what keeps a table larger than the rest of the model out of the engine. The lookup is in the graph:
``ple_pool_gather`` reads the quantized rows, dequantizes them and concatenates the heads. What
remains is the decision of how much of each token's n-gram vector to write into each residual
stream, and a dilated depthwise convolution that widens that decision over a few tokens of lexical
context.
"""

import math

import numpy as np

from ...ops import BuildContext, Linear, Module, RaggedDecoderInputs, Tensor
from ...ops import functional as F
from .modeling_qwen4_exp_residual import Qwen4ExpGroupedRMSNorm

__all__ = ["Qwen4ExpPle"]

#: Subnormal in FP16, so the floor only guards the sqrt against a true zero.
_GATE_EPSILON = 1e-6


class Qwen4ExpPle(Module):
    """Gate and dilated convolution written into the residual streams.

    The gate is asymmetric: one key per stream ("how much"), one value shared
    by all of them ("what"). :meth:`forward` returns the injection, the next
    convolution state, and for a speculative base the per-token states; the
    caller adds the injection to the streams with a bare addition, because the
    per-stream gate is already inside it.

    ``prefix`` is ``model.layers.{i}.ple``.
    """

    def __init__(self, ctx: BuildContext, prefix: str) -> None:
        super().__init__(ctx, prefix)
        cfg = ctx.cfg
        self.conv_kernel = cfg.ple_conv_kernel_size
        # Adjacent n-grams overlap, so dilating by the n-gram order samples
        # non-overlapping windows.
        self.conv_dilation = cfg.ngram_size
        self.key_proj = Linear(ctx,
                               self.key("key_proj"),
                               tensor_parallel=False)
        self.value_proj = Linear(ctx,
                                 self.key("value_proj"),
                                 tensor_parallel=False)
        self.norm_key = Qwen4ExpGroupedRMSNorm(ctx, self.key("norm_key"),
                                               cfg.hidden_size)
        self.norm_query = Qwen4ExpGroupedRMSNorm(ctx, self.key("norm_query"),
                                                 cfg.hidden_size)
        self.norm_conv = Qwen4ExpGroupedRMSNorm(ctx, self.key("norm_conv"),
                                                cfg.hidden_size)

    def _conv_weight(self) -> Tensor:
        cfg = self.cfg
        weight = self.weights.f16(self.key("conv1d.weight"))
        expected = (cfg.hc_hidden_size, 1, self.conv_kernel)
        if weight.shape != expected:
            raise ValueError(f"{self.key('conv1d.weight')} must have shape "
                             f"{expected}, got {weight.shape}")
        return F.constant(weight, "ple_conv_weight")

    def _table_scale(self) -> Tensor:
        """Per-tensor dequantization scale, read from the table's own checkpoint entry."""
        return F.constant(
            self.weights.f16(
                self.key("ple_embedding.ngram_embedding.weight_scale")),
            "ple_table_scale")

    def forward(self,
                hyper_states: Tensor,
                ple_cache_pool: Tensor,
                ple_row_map: Tensor,
                conv_state: Tensor,
                ragged: RaggedDecoderInputs,
                tree_parent_ids: Tensor = None,
                tree_depths: Tensor = None,
                collect_intermediate: bool = False):
        cfg = self.cfg
        ngram_embeds = F.ple_pool_gather(ple_cache_pool, ple_row_map,
                                         self._table_scale())
        streams = (0, cfg.hc_count, cfg.hidden_size)
        key_normed = self.norm_key(
            self.key_proj(ngram_embeds)).reshape(streams)
        value = self.value_proj(ngram_embeds)
        query_normed = self.norm_query(hyper_states).reshape(streams)

        gate = (key_normed * query_normed).sum(-1, keepdim=True) / np.float16(
            math.sqrt(cfg.hidden_size))
        # Signed square root: magnitude compression, so the sigmoid below is
        # not saturated.
        gate = gate.abs().maximum(
            np.float16(_GATE_EPSILON)).sqrt() * gate.sign()
        gated_value = gate.sigmoid() * value.unsqueeze(1, 2)
        gated_value = gated_value.reshape((0, cfg.hc_hidden_size))
        gated_value_normed = self.norm_conv(gated_value)

        # No padding mask: the token-major decoder ABI carries only real
        # tokens, and the convolution reads sequence boundaries out of the
        # ragged metadata rather than a padded layout.
        convolved, conv_state_out, intermediate = F.causal_conv1d(
            gated_value_normed,
            self._conv_weight(),
            F.constant(np.zeros(cfg.hc_hidden_size, np.float16),
                       "ple_conv_bias"),
            conv_state,
            ragged,
            cfg.hc_hidden_size, (self.conv_kernel - 1) * self.conv_dilation,
            dilation=self.conv_dilation,
            tree_parent_ids=tree_parent_ids,
            tree_depths=tree_depths,
            use_intermediate=collect_intermediate)
        return gated_value + convolved.silu(), conv_state_out, intermediate
