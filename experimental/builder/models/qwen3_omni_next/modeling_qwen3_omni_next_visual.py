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
"""Checkpoint-direct Qwen3-Omni-Next visual encoder."""

import tensorrt as trt

from ...core import contracts
from ...ops import (FastPositionEmbedding, NetworkModule, PatchEmbedding,
                    VisionPatchMerger, VisionTransformerBlock)

__all__ = [
    "Qwen3OmniNextVisualMerger",
    "Qwen3OmniNextVisualBlock",
    "Qwen3OmniNextVisualEncoder",
]


class Qwen3OmniNextVisualMerger(VisionPatchMerger):
    """Qwen3-Omni-Next visual patch merger."""


class Qwen3OmniNextVisualBlock(VisionTransformerBlock):
    """Next visual transformer block."""


class Qwen3OmniNextVisualEncoder(NetworkModule):
    """Packed Qwen3-VL-style visual graph with Next-owned weight names."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx, "visual")
        self.visual_config = bundle.component_dict(contracts.Component.VISUAL)
        self.hidden_size = int(self.visual_config["hidden_size"])
        heads = self.visual_config.get("num_heads")
        if heads is None:
            heads = self.visual_config["num_attention_heads"]
        self.num_heads = int(heads)
        self.patch_size = int(self.visual_config.get("patch_size", 14))
        self.temporal_patch_size = int(
            self.visual_config.get("temporal_patch_size", 2))
        self.channels = int(self.visual_config.get("in_channels", 3))
        self.merge_unit = int(self.visual_config.get("spatial_merge_size",
                                                     2))**2
        self.hidden_act = str(
            self.visual_config.get("hidden_act", "gelu_pytorch_tanh"))
        self.patch_embed = PatchEmbedding(ctx, self.hidden_size)
        self.fast_position = FastPositionEmbedding(ctx)
        self.blocks = [
            Qwen3OmniNextVisualBlock(ctx, prefix, self.hidden_size,
                                     self.num_heads, self.hidden_act)
            for prefix in self.weights.layer_prefixes((
                r"(.+\.blocks\.\d+)\.norm1\.weight$", ))
        ]
        deepstack_indexes = list(
            self.visual_config.get("deepstack_visual_indexes", ()))
        self.deepstack_mergers = {
            layer_index:
            Qwen3OmniNextVisualMerger(
                ctx, f"{self.patch_embed.visual_prefix}.merger_list.{index}",
                self.hidden_size, self.merge_unit, True)
            for index, layer_index in enumerate(deepstack_indexes)
        }
        self.final_merger = Qwen3OmniNextVisualMerger(
            ctx, self.patch_embed.visual_prefix + ".merger", self.hidden_size,
            self.merge_unit, False)

    def input_tensors(self):
        input_size = (self.channels * self.temporal_patch_size *
                      self.patch_size * self.patch_size)
        return {
            "pixels":
            self.add_input("input", trt.float16, (-1, input_size)),
            "rotary":
            self.add_input("rotary_pos_emb", trt.float32,
                           (-1, self.hidden_size // self.num_heads // 2)),
            "cu_seqlens":
            self.add_input("cu_seqlens", trt.int32, (-1, )),
            "max_seqlen":
            self.add_input("max_seqlen_carrier", trt.int32, (-1, )),
            "fast_pos_embed_idx":
            self.add_input("fast_pos_embed_idx", trt.int64, (4, -1)),
            "fast_pos_embed_weight":
            self.add_input("fast_pos_embed_weight", trt.float16, (4, -1)),
        }

    def forward(self, **io):
        hidden_states = self.patch_embed(io["pixels"])
        hidden_states = self.fast_position(hidden_states,
                                           io["fast_pos_embed_idx"],
                                           io["fast_pos_embed_weight"])
        deepstack = []
        for index, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, io["rotary"],
                                  io["cu_seqlens"], io["max_seqlen"])
            if index in self.deepstack_mergers:
                deepstack.append(self.deepstack_mergers[index](hidden_states))
        outputs = {"output": self.final_merger(hidden_states)}
        outputs.update({
            f"deepstack_features_{index}": feature
            for index, feature in enumerate(deepstack)
        })
        return outputs
