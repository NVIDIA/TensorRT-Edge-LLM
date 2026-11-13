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
"""Qwen vision checkpoint-direct graph."""

import numpy as np
import tensorrt as trt

from ...core import contracts
from ...ops import (LayerNorm, Linear, Module, NetworkModule,
                    PackedVisionAttention, PatchEmbedding, VisionPatchMerger)
from ...ops import functional as F


class Qwen25VLPatchEmbedding(PatchEmbedding):
    """Qwen2.5-VL bias-free patch projection."""

    def __init__(self, ctx, hidden_size: int) -> None:
        super().__init__(ctx, hidden_size, bias=False)


class Qwen25VLMerger(VisionPatchMerger):
    """Qwen2.5-VL merger using the provider's ``ln_q``/``mlp`` keys."""

    def __init__(self, ctx, prefix: str, hidden_size: int,
                 merge_unit: int) -> None:
        super().__init__(ctx,
                         prefix,
                         hidden_size,
                         merge_unit,
                         False,
                         norm_name="ln_q",
                         fc1_name="mlp.0",
                         fc2_name="mlp.2")


class Qwen25VLVisionMLP(Module):
    """Qwen2.5-VL visual feed-forward module."""

    def __init__(self, ctx, prefix: str, hidden_act: str) -> None:
        super().__init__(ctx, prefix)
        self.hidden_act = hidden_act
        self.gate = Linear(ctx,
                           self.key("gate_proj"),
                           rank=2,
                           tensor_parallel=False)
        self.up = Linear(ctx,
                         self.key("up_proj"),
                         rank=2,
                         tensor_parallel=False)
        self.down = Linear(ctx,
                           self.key("down_proj"),
                           rank=2,
                           tensor_parallel=False)

    def forward(self, hidden):
        gate = self.gate(hidden).activation(self.hidden_act)
        return self.down(gate * self.up(hidden))


class Qwen25VLVisionBlock(Module):
    """Qwen2.5-VL pre-normalized visual transformer block."""

    def __init__(self, ctx, prefix: str, hidden_size: int, num_heads: int,
                 hidden_act: str) -> None:
        super().__init__(ctx, prefix)
        self.norm1 = LayerNorm(ctx, self.key("norm1"), 1e-6, 2)
        self.norm2 = LayerNorm(ctx, self.key("norm2"), 1e-6, 2)
        self.attn = PackedVisionAttention(ctx, self.key("attn"), hidden_size,
                                          num_heads)
        self.mlp = Qwen25VLVisionMLP(ctx, self.key("mlp"), hidden_act)

    def forward(self, hidden, rotary, cu_seqlens, max_seqlen):
        hidden = hidden
        attention = self.attn(self.norm1(hidden), rotary, cu_seqlens,
                              max_seqlen)
        hidden = hidden + attention
        return hidden + self.mlp(self.norm2(hidden))


class Qwen25VLVisualEncoder(NetworkModule):
    """Qwen2.5-VL vision transformer."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx, "visual")
        self.visual = bundle.component_dict(contracts.Component.VISUAL)
        self.hidden_size = int(self.visual["hidden_size"])
        self.num_heads = int(self.visual["num_heads"] if "num_heads" in self.
                             visual else self.visual["num_attention_heads"])
        self.patch_size = int(self.visual.get("patch_size", 14))
        self.temporal_patch_size = int(
            self.visual.get("temporal_patch_size", 2))
        self.channels = int(self.visual.get("in_channels", 3))
        self.merge_unit = int(self.visual.get("spatial_merge_size", 2))**2
        hidden_act = str(self.visual.get("hidden_act", "silu"))
        self.patch_embed = Qwen25VLPatchEmbedding(ctx, self.hidden_size)
        self.blocks = [
            Qwen25VLVisionBlock(ctx, prefix, self.hidden_size, self.num_heads,
                                hidden_act)
            for prefix in self.weights.layer_prefixes((
                r"(.+\.blocks\.\d+)\.norm1\.weight$", ))
        ]
        self.full_attention = set(
            self.visual.get("fullatt_block_indexes", range(len(self.blocks))))
        self.final_merger = Qwen25VLMerger(
            ctx, self.patch_embed.visual_prefix + ".merger", self.hidden_size,
            self.merge_unit)

    def input_tensors(self):
        input_size = (self.channels * self.temporal_patch_size *
                      self.patch_size * self.patch_size)
        pixels = self.add_input("input", trt.float16, (-1, input_size))
        rotary = self.add_input("rotary_pos_emb", trt.float32,
                                (-1, self.hidden_size // self.num_heads // 2))
        cu_seqlens = self.add_input("cu_seqlens", trt.int32, (-1, ))
        max_seqlen = self.add_input("max_seqlen_carrier", trt.int32, (-1, ))
        return {
            "pixels":
            pixels,
            "rotary":
            rotary,
            "cu_seqlens":
            cu_seqlens,
            "max_seqlen":
            max_seqlen,
            "cu_window_seqlens":
            self.add_input("cu_window_seqlens", trt.int32, (-1, )),
            "window_index":
            self.add_input("window_index", trt.int64, (-1, )),
            "reverse_window_index":
            self.add_input("reverse_window_index", trt.int64, (-1, )),
        }

    def _windowed_inputs(self, hidden, rotary, io):
        hidden = hidden.reshape(
            (-1, self.merge_unit,
             self.hidden_size)).gather(io["window_index"], 0).reshape(
                 (-1, self.hidden_size))
        rotary_width = self.hidden_size // self.num_heads // 2
        rotary = rotary.reshape(
            (-1, self.merge_unit, rotary_width)).gather(io["window_index"],
                                                        0).reshape(
                                                            (-1, rotary_width))
        return hidden, rotary

    def forward(self, **io):
        hidden = self.patch_embed(io["pixels"])
        rotary = io["rotary"]
        hidden, rotary = self._windowed_inputs(hidden, rotary, io)

        window_size = int(self.visual.get("window_size", 0))
        window_max_seqlen = ((window_size //
                              self.patch_size)**2 if window_size else 1)
        window_max_carrier = F.constant(
            np.zeros(window_max_seqlen, dtype=np.int32), "window_max_seqlen")
        for index, block in enumerate(self.blocks):
            block_cu_seqlens = io["cu_seqlens"]
            block_max_seqlen = io["max_seqlen"]
            if index not in self.full_attention:
                block_cu_seqlens = io["cu_window_seqlens"]
                block_max_seqlen = window_max_carrier
            hidden = block(hidden, rotary, block_cu_seqlens, block_max_seqlen)

        merged = self.final_merger(hidden)
        merged = merged.gather(io["reverse_window_index"], 0)
        return {"output": merged}
