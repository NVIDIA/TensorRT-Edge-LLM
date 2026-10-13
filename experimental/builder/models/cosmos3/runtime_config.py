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
"""Runtime contracts for Cosmos3 policy engines."""

from __future__ import annotations

from .configuration import Cosmos3PolicyGeometry


def _component_name(component) -> str:
    return str(getattr(component, "value", component)).replace("-", "_")


def update_llm_config(config: dict, root: dict, cfg, args) -> None:
    """Add Cosmos3 reasoning metadata consumed by the shared LLM runtime."""
    del cfg, args
    config["model_type"] = "cosmos3_edge_text"
    config["qk_norm_for_text"] = False
    text = root.get("text_config") or {}
    if "rope_scaling" in text:
        config["rope_scaling"] = text["rope_scaling"]


def _visual_config(bundle, args) -> dict:
    root = bundle.root
    visual = dict(root.get("vision_config") or {})
    projector = dict(root.get("projector_config") or {})
    visual["model_type"] = "cosmos3_edge_vision"
    result = {
        "model_type": "cosmos3_edge_vision",
        "vision_config": visual,
        "projector_config": projector,
        "builder_config": {
            "min_image_tokens": int(args.min_image_tokens),
            "max_image_tokens": int(args.max_image_tokens),
            "max_image_tokens_per_image": int(args.max_image_tokens_per_image),
        },
    }
    for key in ("image_token_id", "video_token_id", "vision_start_token_id",
                "vision_end_token_id", "projector_hidden_size"):
        if key in root:
            result[key] = root[key]
    if isinstance(root.get("text_config"), dict):
        result["text_config"] = dict(root["text_config"])
    return result


def _transformer_contract(bundle, args) -> tuple[dict, Cosmos3PolicyGeometry]:
    config = bundle.root["_direct_transformer_config"]
    geometry = Cosmos3PolicyGeometry.from_bundle(bundle, args)
    return config, geometry


def _und_prefill_config(bundle, args) -> dict:
    config, geometry = _transformer_contract(bundle, args)
    hidden_size = int(config["hidden_size"])
    head_dim = int(config["head_dim"])
    max_batch = int(args.max_batch_size)
    max_und = geometry.max_und_len
    return {
        "component": "und_prefill",
        "engine_filename": "und_prefill.engine",
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "hidden_size": hidden_size,
        "num_attention_heads": int(config["num_attention_heads"]),
        "num_key_value_heads": int(config["num_key_value_heads"]),
        "head_dim": head_dim,
        "rope_theta": float(config["rope_theta"]),
        "rope_scaling": dict(config.get("rope_scaling") or {}),
        "optimization_profile": {
            "inputs_embeds": {
                "min": [1, 1, hidden_size],
                "opt": [max_batch,
                        max(1, max_und // 2), hidden_size],
                "max": [max_batch, max_und, hidden_size],
            },
            "rope_rotary_cos_sin": {
                "min": [1, 1, head_dim],
                "opt": [max_batch, max(1, max_und // 2), head_dim],
                "max": [max_batch, max_und, head_dim],
            },
            "attention_position_ids": {
                "min": [1, 1],
                "opt": [max_batch, max(1, max_und // 2)],
                "max": [max_batch, max_und],
            },
        },
        "tensor_contract": {
            "inputs": {
                "inputs_embeds": ["batch", "und_len", "hidden_size"],
                "rope_rotary_cos_sin": ["batch", "und_len", "head_dim"],
                "attention_position_ids": ["batch", "und_len"],
            },
            "outputs": {
                "und_k_layerNN":
                ["batch", "und_len", "num_key_value_heads", "head_dim"],
                "und_v_layerNN":
                ["batch", "und_len", "num_key_value_heads", "head_dim"],
                "hidden_states": ["batch", "und_len", "hidden_size"],
            },
        },
        "builder_config": {
            "max_batch_size": int(args.max_batch_size),
            "max_und_len": geometry.max_und_len,
        },
    }


def _gen_config(bundle, args) -> dict:
    config, geometry = _transformer_contract(bundle, args)
    patch = int(config.get("latent_patch_size", 2))
    latent_channel = int(config.get("latent_channel", 48))
    latent_patch_h = (geometry.latent_h + patch - 1) // patch
    latent_patch_w = (geometry.latent_w + patch - 1) // patch
    video_tokens = geometry.latent_t * latent_patch_h * latent_patch_w
    max_batch = int(args.max_batch_size)
    head_dim = int(config["head_dim"])
    kv_heads = int(config["num_key_value_heads"])
    action_dim = int(config.get("max_action_dim", 64))
    action_tokens = geometry.action_token_count
    gen_tokens = video_tokens + action_tokens

    def fixed(shape):
        return {
            "min": [1, *shape],
            "opt": [max_batch, *shape],
            "max": [max_batch, *shape],
        }

    profile = {
        "video_latent":
        fixed([
            latent_channel, geometry.latent_t, geometry.latent_h,
            geometry.latent_w
        ]),
        "action_latent":
        fixed([action_tokens, action_dim]),
        "timestep":
        fixed([]),
        "token_noisy_mask":
        fixed([video_tokens, 1]),
        "action_noisy_mask":
        fixed([action_tokens, 1]),
        "rope_rotary_cos_sin":
        fixed([gen_tokens, head_dim]),
        "attention_position_ids":
        fixed([gen_tokens]),
    }
    for index in range(int(config["num_hidden_layers"])):
        und_profile = {
            "min": [1, 1, kv_heads, head_dim],
            "opt":
            [max_batch,
             max(1, geometry.max_und_len // 2), kv_heads, head_dim],
            "max": [max_batch, geometry.max_und_len, kv_heads, head_dim],
        }
        profile[f"und_k_layer{index:02d}"] = und_profile
        profile[f"und_v_layer{index:02d}"] = und_profile
    return {
        "component":
        "gen",
        "engine_filename":
        "gen.engine",
        "num_hidden_layers":
        int(config["num_hidden_layers"]),
        "hidden_size":
        int(config["hidden_size"]),
        "intermediate_size":
        int(config["intermediate_size"]),
        "num_attention_heads":
        int(config["num_attention_heads"]),
        "num_key_value_heads":
        int(config["num_key_value_heads"]),
        "head_dim":
        head_dim,
        "rms_norm_eps":
        float(config.get("rms_norm_eps", 1e-6)),
        "hidden_act":
        str(config["hidden_act"]),
        "rope_theta":
        float(config["rope_theta"]),
        "rope_scaling":
        dict(config.get("rope_scaling") or {}),
        "latent_channel":
        latent_channel,
        "latent_patch_size":
        patch,
        "num_video_tokens":
        video_tokens,
        "action_chunk_size":
        geometry.action_chunk_size,
        "action_token_count":
        action_tokens,
        "state_rows":
        geometry.state_rows,
        "history_length":
        geometry.state_rows,
        "use_state":
        geometry.use_state,
        "raw_action_dim":
        geometry.raw_action_dim,
        "max_action_dim":
        action_dim,
        "num_embodiment_domains":
        int(config.get("num_embodiment_domains", 32)),
        "domain":
        "droid_lerobot",
        "domain_id":
        geometry.domain_id,
        "timestep_scale":
        float(config.get("timestep_scale", 0.001)),
        "num_inference_steps":
        4,
        "flow_shift":
        5.0,
        "video_latent_frames":
        geometry.latent_t,
        "fps":
        geometry.fps,
        "base_fps":
        float(config.get("base_fps", 24.0)),
        "temporal_compression_factor":
        4,
        "temporal_modality_margin":
        int(config.get("unified_3d_mrope_temporal_modality_margin", 15000)),
        "action_start_frame_offset":
        0 if geometry.use_state else 1,
        "optimization_profile":
        profile,
        "tensor_contract": {
            "inputs": {
                "video_latent": ["batch", latent_channel, "t", "h", "w"],
                "action_latent": ["batch", action_tokens, "max_action_dim"],
                "und_k_layerNN":
                ["batch", "und_len", "num_key_value_heads", "head_dim"],
                "und_v_layerNN":
                ["batch", "und_len", "num_key_value_heads", "head_dim"],
            },
            "outputs": {
                "video_pred": ["batch", latent_channel, "t", "h", "w"],
                "action_pred": ["batch", action_tokens, "max_action_dim"],
            },
        },
        "builder_config": {
            "max_batch_size": int(args.max_batch_size),
            "max_und_len": geometry.max_und_len,
            "height": geometry.height,
            "width": geometry.width,
            "content_height": geometry.content_height,
            "content_width": geometry.content_width,
            "num_frames": geometry.num_frames,
        },
    }


def _vae_encoder_config(bundle, args) -> dict:
    geometry = Cosmos3PolicyGeometry.from_bundle(bundle, args)
    config = bundle.root["_direct_vae_config"]
    max_batch = int(args.max_batch_size)
    pixel_shape = [
        3, geometry.conditioning_frames, geometry.height, geometry.width
    ]
    return {
        "component": "vae_encoder",
        "engine_filename": "vae_encoder.engine",
        "z_dim": int(config["z_dim"]),
        "patch_size": int(config["patch_size"]),
        "latents_mean": list(config["latents_mean"]),
        "latents_std": list(config["latents_std"]),
        "optimization_profile": {
            "pixel_values": {
                "min": [1, *pixel_shape],
                "opt": [max_batch, *pixel_shape],
                "max": [max_batch, *pixel_shape],
            },
        },
        "tensor_contract": {
            "inputs": {
                "pixel_values": [
                    "batch", 3, geometry.conditioning_frames, geometry.height,
                    geometry.width
                ],
            },
            "outputs": {
                "cond_latent": ["batch", "z_dim", "t", "h", "w"],
            },
        },
        "builder_config": {
            "max_batch_size": int(args.max_batch_size),
            "height": geometry.height,
            "width": geometry.width,
            "content_height": geometry.content_height,
            "content_width": geometry.content_width,
            "num_frames": geometry.conditioning_frames,
        },
    }


def component_runtime_config(bundle, component, args):
    """Return the exact model-specific runtime contract."""
    name = _component_name(component)
    if name == "visual":
        return _visual_config(bundle, args)
    if name == "und_prefill":
        return _und_prefill_config(bundle, args)
    if name == "gen":
        return _gen_config(bundle, args)
    if name == "vae_encoder":
        return _vae_encoder_config(bundle, args)
    return None
