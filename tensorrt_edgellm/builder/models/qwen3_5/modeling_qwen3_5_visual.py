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

import tensorrt as trt

from ...core import contracts
from ...ops import (FastPositionEmbedding, NetworkModule, PatchEmbedding,
                    VisionPatchMerger, VisionTransformerBlock)


class Qwen3_5VisionPatchMerger(VisionPatchMerger):
    """Qwen3.5 patch-merger extension point."""


class Qwen3_5VisionBlock(VisionTransformerBlock):
    """Qwen3.5 visual-block extension point."""


class Qwen3_5VisionModel(NetworkModule):
    """Qwen3.5 vision transformer."""

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
        self.hidden_act = str(
            self.visual.get("hidden_act", "gelu_pytorch_tanh"))
        self.patch_embed = PatchEmbedding(ctx, self.hidden_size)
        self.position_embedding = FastPositionEmbedding(ctx)
        self.blocks = [
            Qwen3_5VisionBlock(ctx, prefix, self.hidden_size, self.num_heads,
                               self.hidden_act)
            for prefix in self.weights.layer_prefixes((
                r"(.+\.blocks\.\d+)\.norm1\.weight$", ))
        ]
        self.merger = Qwen3_5VisionPatchMerger(
            ctx, self.patch_embed.visual_prefix + ".merger", self.hidden_size,
            self.merge_unit, False)

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
            "fast_pos_embed_idx":
            self.add_input("fast_pos_embed_idx", trt.int64, (4, -1)),
            "fast_pos_embed_weight":
            self.add_input("fast_pos_embed_weight", trt.float16, (4, -1)),
        }

    def forward(self, **io):
        hidden = self.patch_embed(io["pixels"])
        hidden = self.position_embedding(hidden, io["fast_pos_embed_idx"],
                                         io["fast_pos_embed_weight"])
        rotary = io["rotary"]
        for block in self.blocks:
            hidden = block(hidden, rotary, io["cu_seqlens"], io["max_seqlen"])
        return {"output": self.merger(hidden)}
