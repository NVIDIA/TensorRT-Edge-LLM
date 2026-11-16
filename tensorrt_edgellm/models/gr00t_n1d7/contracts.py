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
"""GR00T N1.7 runtime component contracts: the ``config.json`` each engine
directory carries. Both build routes write these, so the runtime reads one shape.
"""

from __future__ import annotations

from .configuration import (IMAGE_TOKEN_ID, VIDEO_TOKEN_ID,
                            VISION_END_TOKEN_ID, VISION_START_TOKEN_ID)
from .geometry import Geometry

CONTRACT_VERSION = 1


def visual_runtime_config(visual: dict, text: dict, geometry: Geometry, *,
                          min_image_tokens: int, max_image_tokens: int,
                          max_image_tokens_per_image: int) -> dict:
    patch = int(visual.get("patch_size", 16))
    temporal = int(visual.get("temporal_patch_size", 2))
    channels = int(visual.get("in_channels", 3))
    return {
        # The shared Qwen ViT runner validates these against the checkpoint it
        # was built from, so the visual contract carries the VL token ids.
        "model_type":
        "qwen3_vl",
        "text_config": {
            "hidden_size": int(geometry.backbone_dim),
            "vocab_size": int(text["vocab_size"]),
            "rope_theta": float(text.get("rope_theta", 5000000.0)),
            "rope_scaling": dict(text.get("rope_scaling") or {}),
        },
        "image_token_id":
        int(visual.get("image_token_id", IMAGE_TOKEN_ID)),
        "video_token_id":
        VIDEO_TOKEN_ID,
        "vision_start_token_id":
        VISION_START_TOKEN_ID,
        "vision_end_token_id":
        VISION_END_TOKEN_ID,
        "num_position_embeddings":
        int(visual.get("num_position_embeddings", 2304)),
        # The shared runner reads the tower's own geometry from a nested
        # vision_config, matching the checkpoint layout it was written against.
        "vision_config": {
            "num_position_embeddings":
            int(visual.get("num_position_embeddings", 2304)),
            "deepstack_visual_indexes":
            list(visual.get("deepstack_visual_indexes", [])),
        },
        "component":
        "visual",
        "engine_filename":
        "visual.engine",
        "hidden_size":
        int(visual["hidden_size"]),
        "out_hidden_size":
        int(visual.get("out_hidden_size", 2048)),
        "num_heads":
        int(visual["num_heads"]),
        "patch_size":
        patch,
        "temporal_patch_size":
        temporal,
        "spatial_merge_size":
        int(visual.get("spatial_merge_size", 2)),
        "deepstack_visual_indexes":
        list(visual.get("deepstack_visual_indexes", [])),
        "patch_input_size":
        channels * temporal * patch * patch,
        "tensor_contract": {
            "inputs": {
                "input": ["patches", "patch_input_size"],
                "rotary_pos_emb": ["patches", "head_dim_half"],
                "cu_seqlens": ["frames_plus_one"],
                "max_seqlen_carrier": ["max_seqlen"],
                "fast_pos_embed_idx": [4, "patches"],
                "fast_pos_embed_weight": [4, "patches"],
            },
            "outputs": {
                "output": ["patches_over_merge", "out_hidden_size"],
                "deepstack_features_N":
                ["patches_over_merge", "out_hidden_size"],
            },
        },
        "builder_config": {
            "min_image_tokens": int(min_image_tokens),
            "max_image_tokens": int(max_image_tokens),
            "max_image_tokens_per_image": int(max_image_tokens_per_image),
            "max_vl_len": geometry.max_vl_len,
        },
    }


def vl_backbone_runtime_config(text: dict, geometry: Geometry, *,
                               num_deepstack: int, max_batch: int) -> dict:
    hidden = int(text["hidden_size"])
    head_dim = int(text["head_dim"])
    max_vl = geometry.max_vl_len

    profile = {
        "inputs_embeds": {
            "min": [1, 1, hidden],
            "opt": [max_batch, max(1, max_vl // 2), hidden],
            "max": [max_batch, max_vl, hidden],
        },
        "rope_rotary_cos_sin": {
            "min": [1, 1, head_dim],
            "opt": [max_batch, max(1, max_vl // 2), head_dim],
            "max": [max_batch, max_vl, head_dim],
        },
        "attention_pos_id": {
            "min": [1, 1],
            "opt": [max_batch, max(1, max_vl // 2)],
            "max": [max_batch, max_vl],
        },
    }
    outputs = {}
    for index in geometry.cross_blocks:
        outputs[f"cross_k_layer{index:02d}"] = [
            "batch", "vl_len", "dit_heads", "dit_head_dim"
        ]
        outputs[f"cross_v_layer{index:02d}"] = [
            "batch", "vl_len", "dit_heads", "dit_head_dim"
        ]
    inputs = {
        "inputs_embeds": ["batch", "vl_len", "hidden_size"],
        "rope_rotary_cos_sin": ["batch", "vl_len", "head_dim"],
        "attention_pos_id": ["batch", "vl_len"],
    }
    for index in range(num_deepstack):
        name = f"deepstack_embeds_{index}"
        inputs[name] = ["batch", "vl_len", "hidden_size"]
        profile[name] = dict(profile["inputs_embeds"])
    return {
        "component": "vl_backbone",
        "engine_filename": "vl_backbone.engine",
        "num_hidden_layers": int(text["num_hidden_layers"]),
        "hidden_size": hidden,
        "num_attention_heads": int(text["num_attention_heads"]),
        "num_key_value_heads": int(text["num_key_value_heads"]),
        "head_dim": head_dim,
        "rms_norm_eps": float(text.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(text.get("rope_theta", 5000000.0)),
        "rope_scaling": dict(text.get("rope_scaling") or {}),
        "vl_self_attention_layers": geometry.vl_sa_layers,
        "dit_num_heads": geometry.num_heads,
        "dit_head_dim": geometry.head_dim,
        "text_cross_blocks": geometry.text_blocks,
        "image_cross_blocks": geometry.image_blocks,
        "num_deepstack_features": num_deepstack,
        "optimization_profile": profile,
        "tensor_contract": {
            "inputs": inputs,
            "outputs": outputs,
        },
        "builder_config": {
            "max_batch_size": max_batch,
            "max_vl_len": max_vl,
            "max_text_len": geometry.max_text_len,
            "max_image_len": geometry.max_image_len,
        },
    }


def action_runtime_config(geometry: Geometry, *, max_batch: int) -> dict:

    def fixed(shape):
        return {
            "min": [1, *shape],
            "opt": [max_batch, *shape],
            "max": [max_batch, *shape],
        }

    profile = {
        "noise_trajectory":
        fixed([geometry.action_horizon, geometry.action_dim]),
        "state": fixed([geometry.state_history_length, geometry.state_dim]),
    }
    inputs = {
        "noise_trajectory": ["batch", "action_horizon", "action_dim"],
        "state": ["batch", "state_history", "state_dim"],
        "adanorm_modulation": ["sites", "modulation_dim"],
        "action_time_bias": ["inner_dim"],
        "text_attention_mask": ["batch", 1, 1, "vl_len"],
        "image_attention_mask": ["batch", 1, 1, "vl_len"],
    }
    cross = {
        "min": [1, 1, geometry.num_heads, geometry.head_dim],
        "opt": [
            max_batch,
            max(1, geometry.max_vl_len // 2), geometry.num_heads,
            geometry.head_dim
        ],
        "max": [
            max_batch, geometry.max_vl_len, geometry.num_heads,
            geometry.head_dim
        ],
    }
    mask = {
        "min": [1, 1, 1, 1],
        "opt": [max_batch, 1, 1,
                max(1, geometry.max_vl_len // 2)],
        "max": [max_batch, 1, 1, geometry.max_vl_len],
    }
    profile["text_attention_mask"] = dict(mask)
    profile["image_attention_mask"] = dict(mask)
    for index in geometry.cross_blocks:
        for half in ("k", "v"):
            name = f"cross_{half}_layer{index:02d}"
            profile[name] = dict(cross)
            inputs[name] = ["batch", "vl_len", "dit_heads", "dit_head_dim"]
    return {
        "component": "action",
        "engine_filename": "action.engine",
        "action_horizon": geometry.action_horizon,
        "action_dim": geometry.action_dim,
        "state_dim": geometry.state_dim,
        "state_history_length": geometry.state_history_length,
        "num_denoise_steps": geometry.num_denoise_steps,
        "num_timestep_buckets": geometry.num_timestep_buckets,
        "num_layers": geometry.num_layers,
        "num_heads": geometry.num_heads,
        "head_dim": geometry.head_dim,
        "inner_dim": geometry.inner_dim,
        "output_dim": geometry.output_dim,
        "backbone_dim": geometry.backbone_dim,
        "num_modulation_sites": geometry.num_modulation_sites,
        "modulation_dim": geometry.modulation_dim,
        "embodiment_id": geometry.embodiment_id,
        "use_percentiles": geometry.use_percentiles,
        "text_cross_blocks": geometry.text_blocks,
        "image_cross_blocks": geometry.image_blocks,
        "modulation_filename": "adanorm_modulation.safetensors",
        "time_bias_filename": "action_time_bias.safetensors",
        "optimization_profile": profile,
        "tensor_contract": {
            "inputs": inputs,
            "outputs": {
                "velocity": ["batch", "action_horizon", "action_dim"],
            },
        },
        "builder_config": {
            "max_batch_size": max_batch,
            "max_text_len": geometry.max_text_len,
            "max_image_len": geometry.max_image_len,
        },
    }
