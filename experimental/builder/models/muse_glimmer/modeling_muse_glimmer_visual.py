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
"""Muse-Glimmer vision encoder for the checkpoint-direct frontend."""

import numpy as np
import tensorrt as trt

from ...core import contracts
from ...ops import LayerNorm, Linear, Module, NetworkModule
from ...ops import functional as F


class MuseGlimmerVisionAttention(Module):
    """Ragged self-attention: separate biased q/k/v/proj + 2D RoPE."""

    def __init__(self, ctx, prefix: str, hidden_size: int,
                 num_heads: int) -> None:
        super().__init__(ctx, prefix)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.q_proj = Linear(ctx,
                             self.key("q_proj"),
                             rank=2,
                             tensor_parallel=False)
        self.k_proj = Linear(ctx,
                             self.key("k_proj"),
                             rank=2,
                             tensor_parallel=False)
        self.v_proj = Linear(ctx,
                             self.key("v_proj"),
                             rank=2,
                             tensor_parallel=False)
        self.proj = Linear(ctx,
                           self.key("proj"),
                           rank=2,
                           tensor_parallel=False)

    def forward(self, hidden_states, rotary, cu_seqlens, max_seqlen):
        query = self.q_proj(hidden_states).reshape(
            (0, self.num_heads, self.head_size))
        key = self.k_proj(hidden_states).reshape(
            (0, self.num_heads, self.head_size))
        value = self.v_proj(hidden_states).reshape(
            (0, self.num_heads, self.head_size))
        query, key = F.apply_rope(query, key, rotary, self.num_heads,
                                  self.head_size)
        output = F.vit_attention(query, key, value, cu_seqlens, max_seqlen,
                                 self.num_heads, self.head_size)
        return self.proj(output.reshape((0, self.hidden_size)))


class MuseGlimmerVisionMLP(Module):
    """Two-layer visual feed-forward block: fc1 -> gelu -> fc2."""

    def __init__(self, ctx, prefix: str, hidden_act: str) -> None:
        super().__init__(ctx, prefix)
        self.hidden_act = hidden_act
        self.fc1 = Linear(ctx, self.key("fc1"), rank=2, tensor_parallel=False)
        self.fc2 = Linear(ctx, self.key("fc2"), rank=2, tensor_parallel=False)

    def forward(self, hidden_states):
        return self.fc2(self.fc1(hidden_states).activation(self.hidden_act))


class MuseGlimmerVisionBlock(Module):
    """Pre-normalized visual transformer block (LayerNorm, eps 1e-5)."""

    def __init__(self, ctx, prefix: str, hidden_size: int, num_heads: int,
                 hidden_act: str, layer_norm_eps: float) -> None:
        super().__init__(ctx, prefix)
        self.norm1 = LayerNorm(ctx, self.key("norm1"), layer_norm_eps, 2)
        self.norm2 = LayerNorm(ctx, self.key("norm2"), layer_norm_eps, 2)
        self.attn = MuseGlimmerVisionAttention(ctx, self.key("attn"),
                                               hidden_size, num_heads)
        self.mlp = MuseGlimmerVisionMLP(ctx, self.key("mlp"), hidden_act)

    def forward(self, hidden_states, rotary, cu_seqlens, max_seqlen):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), rotary, cu_seqlens, max_seqlen)
        return hidden_states + self.mlp(self.norm2(hidden_states))


class MuseGlimmerVisualModel(NetworkModule):
    """Muse-Glimmer vision encoder: tower -> pixel-shuffle -> adapter -> proj."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx, "visual")
        root = bundle.root
        visual = bundle.component_dict(contracts.Component.VISUAL)
        self.hidden_size = int(visual["hidden_size"])
        self.num_heads = int(visual["num_attention_heads"])
        self.head_size = self.hidden_size // self.num_heads
        self.num_layers = int(visual["num_hidden_layers"])
        self.intermediate_size = int(visual["intermediate_size"])
        self.patch_size = int(visual.get("patch_size", 14))
        self.patch_temporal = int(visual.get("patch_temporal", 2))
        self.num_channels = int(visual.get("num_channels", 3))
        self.merge_size = int(visual.get("merge_size", 2))
        self.merge_unit = self.merge_size * self.merge_size
        self.merged_size = self.hidden_size * self.merge_unit
        self.pos_emb_height = int(visual.get("pos_emb_height", 32))
        self.pos_emb_width = int(visual.get("pos_emb_width", 32))
        self.layer_norm_eps = float(visual.get("layer_norm_eps", 1e-5))
        hidden_act = str(visual.get("hidden_act", "gelu"))
        self.in_features = (self.patch_temporal * self.num_channels *
                            self.patch_size * self.patch_size)
        self.rotary_dim = self.head_size // 2

        # Window blocks reorder at spatial_merge_size=1 with a
        # ``pos_emb_height`` x ``patch_size`` window; max tokens per window is
        # ``pos_emb_height**2``.
        self.window_max_seqlen = self.pos_emb_height * self.pos_emb_height

        layer_types = visual.get("layer_types") or [
            "full_attention" if (index + 1) % 4 == 0
            or index == self.num_layers - 1 else "window_attention"
            for index in range(self.num_layers)
        ]
        self.layer_types = list(layer_types)

        # Patch projection (bias-free) and learned position table.
        self.patch_embedding = Linear(
            ctx,
            "vision_tower.patch_embedder.patch_embedding",
            rank=2,
            tensor_parallel=False)
        self.position_key = (
            "vision_tower.patch_embedder.position_embedding_table.weight")
        self.ln_pre = LayerNorm(ctx, "vision_tower.ln_pre",
                                self.layer_norm_eps, 2)
        self.blocks = [
            MuseGlimmerVisionBlock(ctx, f"vision_tower.layers.{index}",
                                   self.hidden_size, self.num_heads,
                                   hidden_act, self.layer_norm_eps)
            for index in range(self.num_layers)
        ]
        self.ln_post = LayerNorm(ctx, "vision_tower.ln_post",
                                 self.layer_norm_eps, 2)

        # Adapter (bias-free fc1 -> gelu -> fc2 -> gelu), projection to the LLM
        # width, and the scaleless perception-embedding RMSNorm.
        self.out_hidden_size = int(
            root.get("out_hidden_size", self.merged_size))
        self.projector_hidden_size = int(
            root.get("projector_hidden_size", 4096))
        text_config = root.get("text_config", {})
        self.text_hidden_size = int(text_config.get("hidden_size", 6656))
        self.perception_eps = float(text_config.get("rms_norm_eps", 1e-5))
        self.adapter_fc1 = Linear(ctx,
                                  "vision_adapter.fc1",
                                  rank=2,
                                  tensor_parallel=False)
        self.adapter_fc2 = Linear(ctx,
                                  "vision_adapter.fc2",
                                  rank=2,
                                  tensor_parallel=False)
        self.vision_projection = Linear(ctx,
                                        "vision_projection",
                                        rank=2,
                                        tensor_parallel=False)

    def input_tensors(self):
        return {
            "pixels":
            self.add_input("input", trt.float16, (-1, self.in_features)),
            "rotary":
            self.add_input("rotary_pos_emb", trt.float32,
                           (-1, self.rotary_dim)),
            "cu_seqlens":
            self.add_input("cu_seqlens", trt.int32, (-1, )),
            "cu_window_seqlens":
            self.add_input("cu_window_seqlens", trt.int32, (-1, )),
            "window_index":
            self.add_input("window_index", trt.int64, (-1, )),
            "reverse_window_index":
            self.add_input("reverse_window_index", trt.int64, (-1, )),
            "fast_pos_embed_idx":
            self.add_input("fast_pos_embed_idx", trt.int64, (4, -1)),
            "fast_pos_embed_weight":
            self.add_input("fast_pos_embed_weight", trt.float16, (4, -1)),
            "max_seqlen":
            self.add_input("max_seqlen_carrier", trt.int32, (-1, )),
        }

    def _patch_embed(self, pixels, idx, blend_weights):
        hidden_states = self.patch_embedding(pixels)
        table = F.constant(self.weights.f16(self.position_key),
                           "visual_pos_embed")
        # idx / blend_weights are [4, T]; gather -> [4, T, H], blend, sum.
        position = table.gather(idx, 0) * blend_weights.reshape((4, -1, 1))
        return hidden_states + position.sum(dim=0)

    def forward(self, **io):
        hidden = self._patch_embed(io["pixels"], io["fast_pos_embed_idx"],
                                   io["fast_pos_embed_weight"])
        hidden = self.ln_pre(hidden)

        # Per-token (spatial_merge_size=1) window reorder; rotary follows.
        hidden = hidden.gather(io["window_index"], 0)
        rotary = io["rotary"].gather(io["window_index"], 0)

        window_max_carrier = F.constant(
            np.zeros(self.window_max_seqlen, dtype=np.int32),
            "window_max_seqlen")
        for index, block in enumerate(self.blocks):
            if self.layer_types[index] == "full_attention":
                cu_now, max_now = io["cu_seqlens"], io["max_seqlen"]
            else:
                cu_now, max_now = io["cu_window_seqlens"], window_max_carrier
            hidden = block(hidden, rotary, cu_now, max_now)

        # Undo the window order and group tokens into 2x2 merge blocks
        # (reverse_window_index is host-precomputed to do both), then ln_post.
        hidden = hidden.gather(io["reverse_window_index"], 0)
        hidden = self.ln_post(hidden)

        # Pixel-shuffle (merge_size 2): [T, H] -> [G, merge_unit, H]
        # -> [G, H, merge_unit] -> [G, H * merge_unit].
        hidden = hidden.reshape((-1, self.merge_unit, self.hidden_size))
        hidden = hidden.transpose((0, 2, 1))
        hidden = hidden.reshape((-1, self.merged_size))

        # Adapter (fc1 -> gelu -> fc2 -> gelu), projection, scaleless norm.
        hidden = self.adapter_fc1(hidden).gelu()
        hidden = self.adapter_fc2(hidden).gelu()
        hidden = self.vision_projection(hidden)
        hidden = F.rms_norm(hidden,
                            np.ones(self.text_hidden_size, dtype=np.float16),
                            self.perception_eps, 2)
        return {"output": hidden}
