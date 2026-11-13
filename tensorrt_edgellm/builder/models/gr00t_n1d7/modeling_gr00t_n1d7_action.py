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
"""GR00T N1.7 action head: one AlternateVLDiT flow-matching step. The graph
emits the velocity; the runtime owns the Euler update. The timestep-only inputs
(AdaLayerNorm modulation, action time bias) are constants from :mod:`.artifacts`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
import tensorrt as trt

from tensorrt_edgellm.models.gr00t_n1d7.geometry import (BLOCK_NORM_EPS,
                                                         OUT_NORM_EPS)

from ...ops import BuildContext, Linear, Module, NetworkModule, Tensor
from ...ops import functional as F
from .runtime_config import Geometry, geometry_from_bundle

if TYPE_CHECKING:
    from ...core.builder import BuildArgs
    from ...core.bundle import BundleConfig


def _affine_free_layer_norm(hidden_states: Tensor,
                            width: int,
                            eps: float,
                            rank: int = 3) -> Tensor:
    """LayerNorm with ``elementwise_affine=False``."""
    return F.layer_norm(hidden_states, np.ones(width, dtype=np.float32),
                        np.zeros(width, dtype=np.float32), eps, rank)


class BakedCategoryLinear(Module):
    """One embodiment's slice of a ``CategorySpecificLinear`` ``W [categories, in, out]``,
    selected at build time because the id is fixed for a deployed policy."""

    def __init__(self, ctx: BuildContext, prefix: str,
                 embodiment_id: int) -> None:
        super().__init__(ctx, prefix)
        stacked = self.weights.f16(self.key("W"))
        bias = self.weights.f16(self.key("b"))
        if not 0 <= embodiment_id < stacked.shape[0]:
            raise ValueError(
                f"embodiment id {embodiment_id} is outside {prefix}.W with "
                f"{stacked.shape[0]} categories")
        self.weight = np.ascontiguousarray(stacked[embodiment_id].T)
        self.bias = np.ascontiguousarray(bias[embodiment_id])

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.linear_with_weights(hidden_states,
                                     self.weight,
                                     self.bias,
                                     rank=3)


class BakedCategoryMLP(Module):
    """``CategorySpecificMLP``: ``layer2(relu(layer1(x)))``."""

    def __init__(self, ctx: BuildContext, prefix: str,
                 embodiment_id: int) -> None:
        super().__init__(ctx, prefix)
        self.layer1 = BakedCategoryLinear(ctx, self.key("layer1"),
                                          embodiment_id)
        self.layer2 = BakedCategoryLinear(ctx, self.key("layer2"),
                                          embodiment_id)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.layer2(self.layer1(hidden_states).relu())


class ActionEncoder(Module):
    """``MultiEmbodimentActionEncoder`` with the timestep half of ``W2`` folded:
    ``W2 @ concat(a, tau) + b == W2_a @ a + (W2_tau @ tau + b)``, and the second term is
    schedule-only, so it is bound as ``action_time_bias``."""

    def __init__(self, ctx: BuildContext, prefix: str, embodiment_id: int,
                 geometry: Geometry) -> None:
        super().__init__(ctx, prefix)
        self.geometry = geometry
        self.w1 = BakedCategoryLinear(ctx, self.key("W1"), embodiment_id)
        self.w3 = BakedCategoryLinear(ctx, self.key("W3"), embodiment_id)
        stacked = self.weights.f16(self.key("W2.W"))[embodiment_id]
        width = geometry.inner_dim
        if stacked.shape != (2 * width, width):
            raise ValueError(f"{prefix}.W2.W has shape {stacked.shape}, "
                             f"expected {(2 * width, width)}")
        self.w2_action = np.ascontiguousarray(stacked[:width].T)

    def forward(self, actions: Tensor, time_bias: Tensor) -> Tensor:
        action_embed = F.linear_with_weights(self.w1(actions),
                                             self.w2_action,
                                             None,
                                             rank=3)
        gated = (action_embed + time_bias.reshape(
            (1, 1, self.geometry.inner_dim))).silu()
        return self.w3(gated)


class AdaLayerNormModulation(Module):
    """Split one hoisted modulation row and apply it to an affine-free norm."""

    def __init__(self, ctx: BuildContext, width: int, eps: float,
                 shift_first: bool) -> None:
        super().__init__(ctx)
        self.width = width
        self.eps = eps
        self.shift_first = shift_first

    def forward(self, hidden_states: Tensor, modulation_row: Tensor) -> Tensor:
        first = modulation_row.slice_last_dim(0, self.width, 3)
        second = modulation_row.slice_last_dim(self.width, self.width, 3)
        shift, scale = ((first, second) if self.shift_first else
                        (second, first))
        normed = _affine_free_layer_norm(hidden_states, self.width, self.eps)
        return normed * (scale + np.float16(1.0)) + shift


class DiTBlock(Module):
    """One AlternateVLDiT block: ada-norm attention then a feed-forward."""

    def __init__(self, ctx: BuildContext, index: int,
                 geometry: Geometry) -> None:
        super().__init__(ctx, f"model.transformer_blocks.{index}")
        self.index = index
        self.geometry = geometry
        self.is_cross = index % 2 == 0
        self.num_heads = geometry.num_heads
        self.head_dim = geometry.head_dim
        self.inner_dim = geometry.inner_dim
        self.norm1 = AdaLayerNormModulation(ctx,
                                            self.inner_dim,
                                            BLOCK_NORM_EPS,
                                            shift_first=False)
        self.to_q = Linear(ctx, self.key("attn1.to_q"), tensor_parallel=False)
        self.to_out = Linear(ctx,
                             self.key("attn1.to_out.0"),
                             tensor_parallel=False)
        self.ff_in = Linear(ctx,
                            self.key("ff.net.0.proj"),
                            tensor_parallel=False)
        self.ff_out = Linear(ctx, self.key("ff.net.2"), tensor_parallel=False)
        if not self.is_cross:
            self.to_k = Linear(ctx,
                               self.key("attn1.to_k"),
                               tensor_parallel=False)
            self.to_v = Linear(ctx,
                               self.key("attn1.to_v"),
                               tensor_parallel=False)

    def forward(self, hidden_states: Tensor, modulation_row: Tensor,
                cross_key: Optional[Tensor], cross_value: Optional[Tensor],
                cross_mask: Optional[Tensor]) -> Tensor:
        normed = self.norm1(hidden_states, modulation_row)
        shape = (0, 0, self.num_heads, self.head_dim)
        query = self.to_q(normed).reshape(shape).transpose((0, 2, 1, 3))
        if self.is_cross:
            key = cross_key.transpose((0, 2, 1, 3))
            value = cross_value.transpose((0, 2, 1, 3))
            mask = cross_mask
        else:
            key = self.to_k(normed).reshape(shape).transpose((0, 2, 1, 3))
            value = self.to_v(normed).reshape(shape).transpose((0, 2, 1, 3))
            mask = None
        attended = F.scaled_dot_product_attention(query,
                                                  key,
                                                  value,
                                                  mask=mask,
                                                  scale=self.head_dim**-0.5)
        attended = attended.transpose((0, 2, 1, 3)).reshape(
            (0, 0, self.inner_dim))
        hidden_states = hidden_states + self.to_out(attended)
        normed = _affine_free_layer_norm(hidden_states, self.inner_dim,
                                         BLOCK_NORM_EPS)
        return hidden_states + self.ff_out(self.ff_in(normed).gelu_tanh())


class Gr00tActionModel(NetworkModule):
    """One flow-matching denoise step of the GR00T N1.7 action head."""

    @classmethod
    def from_config(cls, ctx: BuildContext) -> Gr00tActionModel:
        return cls(ctx, ctx.bundle, ctx.args)

    def __init__(self, ctx: BuildContext, bundle: BundleConfig,
                 args: BuildArgs) -> None:
        super().__init__(ctx)
        self.geometry = geometry_from_bundle(bundle, args)
        geometry = self.geometry
        embodiment = geometry.embodiment_id
        self.state_encoder = BakedCategoryMLP(ctx, "state_encoder", embodiment)
        self.action_encoder = ActionEncoder(ctx, "action_encoder", embodiment,
                                            geometry)
        self.action_decoder = BakedCategoryMLP(ctx, "action_decoder",
                                               embodiment)
        self.blocks = [
            DiTBlock(ctx, index, geometry)
            for index in range(geometry.num_layers)
        ]
        self.norm_out = AdaLayerNormModulation(ctx,
                                               geometry.inner_dim,
                                               OUT_NORM_EPS,
                                               shift_first=True)
        self.proj_out_2 = Linear(ctx,
                                 "model.proj_out_2",
                                 tensor_parallel=False)
        self.position_embedding = (self.weights.f16(
            "position_embedding.weight")[:geometry.action_horizon]
                                   if geometry.add_pos_embed else None)

    def input_tensors(self) -> Dict[str, Tensor]:
        geometry = self.geometry
        tensors = {
            "noise_trajectory":
            self.add_input("noise_trajectory", trt.float32,
                           (-1, geometry.action_horizon, geometry.action_dim)),
            "state":
            self.add_input(
                "state", trt.float32,
                (-1, geometry.state_history_length, geometry.state_dim)),
            "adanorm_modulation":
            self.add_input(
                "adanorm_modulation", trt.float16,
                (geometry.num_modulation_sites, geometry.modulation_dim)),
            "action_time_bias":
            self.add_input("action_time_bias", trt.float16,
                           (geometry.inner_dim, )),
            "text_attention_mask":
            self.add_input("text_attention_mask", trt.bool, (-1, 1, 1, -1)),
            "image_attention_mask":
            self.add_input("image_attention_mask", trt.bool, (-1, 1, 1, -1)),
        }
        heads, head_dim = geometry.num_heads, geometry.head_dim
        tensors["cross_key"] = {
            index:
            self.add_input(f"cross_k_layer{index:02d}", trt.float16,
                           (-1, -1, heads, head_dim))
            for index in geometry.cross_blocks
        }
        tensors["cross_value"] = {
            index:
            self.add_input(f"cross_v_layer{index:02d}", trt.float16,
                           (-1, -1, heads, head_dim))
            for index in geometry.cross_blocks
        }
        return tensors

    def forward(self, **io: Tensor) -> Dict[str, Tensor]:
        geometry = self.geometry
        state = io["state"].cast(trt.float16).reshape(
            (0, 1, geometry.state_history_length * geometry.state_dim))
        state_features = self.state_encoder(state)

        actions = io["noise_trajectory"].cast(trt.float16)
        action_features = self.action_encoder(actions, io["action_time_bias"])
        if self.position_embedding is not None:
            action_features = action_features + F.constant(
                self.position_embedding.reshape(1, geometry.action_horizon,
                                                geometry.inner_dim),
                "action_position_embedding")

        hidden_states = F.concatenate([state_features, action_features], dim=1)
        modulation = io["adanorm_modulation"]
        text_blocks = set(geometry.text_blocks)
        for index, block in enumerate(self.blocks):
            row = modulation.slice_axis(0, index, 1, 2).reshape(
                (1, 1, geometry.modulation_dim))
            if block.is_cross:
                mask = (io["text_attention_mask"] if index in text_blocks else
                        io["image_attention_mask"])
                hidden_states = block(hidden_states, row,
                                      io["cross_key"][index],
                                      io["cross_value"][index], mask)
            else:
                hidden_states = block(hidden_states, row, None, None, None)

        final_row = modulation.slice_axis(0, geometry.num_layers, 1,
                                          2).reshape(
                                              (1, 1, geometry.modulation_dim))
        hidden_states = self.norm_out(hidden_states, final_row)
        hidden_states = self.proj_out_2(hidden_states)
        hidden_states = hidden_states.slice_axis(1, 1, geometry.action_horizon,
                                                 3)
        velocity = self.action_decoder(hidden_states)
        return {"velocity": velocity.cast(trt.float32)}
