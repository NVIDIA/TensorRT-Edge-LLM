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
"""Static request geometry shared by one GR00T engine bundle."""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_MAX_VL_LEN = 1024
DEFAULT_MAX_IMAGE_LEN = 256
DEFAULT_EMBODIMENT_ID = 22
# diffusers BasicTransformerBlock / DiT norm defaults carried by this checkpoint.
BLOCK_NORM_EPS = 1e-5
OUT_NORM_EPS = 1e-6


@dataclass(frozen=True)
class Geometry:
    """Static request geometry shared by one GR00T engine bundle."""

    action_horizon: int = 40
    action_dim: int = 132
    state_dim: int = 132
    state_history_length: int = 1
    num_denoise_steps: int = 4
    num_timestep_buckets: int = 1000
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 48
    inner_dim: int = 1536
    output_dim: int = 1024
    backbone_dim: int = 2048
    vl_sa_layers: int = 4
    vl_sa_heads: int = 32
    vl_sa_head_dim: int = 64
    max_vl_len: int = DEFAULT_MAX_VL_LEN
    max_image_len: int = DEFAULT_MAX_IMAGE_LEN
    embodiment_id: int = DEFAULT_EMBODIMENT_ID
    add_pos_embed: bool = True
    use_percentiles: bool = True

    @property
    def max_text_len(self) -> int:
        return max(1, self.max_vl_len - self.max_image_len)

    @property
    def num_modulation_sites(self) -> int:
        """One AdaLayerNorm per block, plus the final ``proj_out_1``."""
        return self.num_layers + 1

    @property
    def modulation_dim(self) -> int:
        return 2 * self.inner_dim

    @property
    def cross_blocks(self) -> list:
        return [index for index in range(self.num_layers) if index % 2 == 0]

    @property
    def text_blocks(self) -> list:
        return [index for index in self.cross_blocks if index % 4 == 0]

    @property
    def image_blocks(self) -> list:
        return [index for index in self.cross_blocks if index % 4 != 0]

    @classmethod
    def from_policy(cls, policy: dict, *, dit_inner_dim: int,
                    embodiment_id: int, max_vl_len: int,
                    max_image_len: int) -> "Geometry":
        """Resolve the geometry from the checkpoint's policy ``config.json``."""
        diffusion = policy.get("diffusion_model_cfg") or {}
        vl_sa = policy.get("vl_self_attention_cfg") or {}
        if not policy.get("use_alternate_vl_dit", True):
            raise ValueError("GR00T support requires use_alternate_vl_dit")
        heads = int(diffusion.get("num_attention_heads", 32))
        head_dim = int(diffusion.get("attention_head_dim", 48))
        geometry = cls(
            action_horizon=int(policy.get("action_horizon", 40)),
            action_dim=int(policy.get("max_action_dim", 132)),
            state_dim=int(policy.get("max_state_dim", 132)),
            num_denoise_steps=int(policy.get("num_inference_timesteps", 4)),
            num_timestep_buckets=int(policy.get("num_timestep_buckets", 1000)),
            num_layers=int(diffusion.get("num_layers", 32)),
            num_heads=heads,
            head_dim=head_dim,
            inner_dim=int(dit_inner_dim or heads * head_dim),
            output_dim=int(diffusion.get("output_dim", 1024)),
            backbone_dim=int(policy.get("backbone_embedding_dim", 2048)),
            vl_sa_layers=int(vl_sa.get("num_layers", 4)),
            vl_sa_heads=int(vl_sa.get("num_attention_heads", 32)),
            vl_sa_head_dim=int(vl_sa.get("attention_head_dim", 64)),
            max_vl_len=int(max_vl_len),
            max_image_len=int(max_image_len),
            embodiment_id=int(embodiment_id),
            add_pos_embed=bool(policy.get("add_pos_embed", True)),
            use_percentiles=bool(policy.get("use_percentiles", True)),
        )
        geometry.validate()
        return geometry

    def validate(self) -> None:
        if self.num_layers % 2:
            raise ValueError("GR00T alternating DiT needs an even block count")
        if self.inner_dim != self.num_heads * self.head_dim:
            raise ValueError(f"GR00T DiT inner dim {self.inner_dim} != "
                             f"{self.num_heads} heads x {self.head_dim}")
        if self.num_denoise_steps <= 0:
            raise ValueError("GR00T num_inference_timesteps must be positive")
        if self.vl_sa_heads * self.vl_sa_head_dim != self.backbone_dim:
            raise ValueError(
                f"GR00T VL self-attention {self.vl_sa_heads} heads x "
                f"{self.vl_sa_head_dim} != backbone dim {self.backbone_dim}")
        if not 0 < self.max_image_len < self.max_vl_len:
            raise ValueError(
                "GR00T image span must be shorter than the VL context")
        if not 0 <= self.embodiment_id < 32:
            raise ValueError("GR00T embodiment id must be in [0, 32)")
