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
"""Framework-independent pi0.5 component and runtime contracts."""

from __future__ import annotations

from collections.abc import Mapping

from .policy_assets import MAX_CAMERA_SLOTS, MAX_TOKEN_LEN

CONTRACT_VERSION = 1
DEFAULT_MAX_TOKEN_LEN = -(-MAX_TOKEN_LEN // 16) * 16
DEFAULT_NUM_DENOISE_STEPS = 10
MAX_DENOISE_STEPS = 64
TOKENS_PER_VIEW = 256
KV_TOKENS_PER_PAGE = 128
XQA_CACHE_TILE = 256
DIFFUSION_DENOISE_PHASE = 6

GEMMA_VARIANTS = {
    "gemma_2b": {
        "hidden_size": 2048,
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "intermediate_size": 16384,
    },
    "gemma_300m": {
        "hidden_size": 1024,
        "num_hidden_layers": 18,
        "num_attention_heads": 8,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "intermediate_size": 4096,
    },
}


def config_field(config: dict, names: tuple[str, ...], label: str):
    for name in names:
        if name in config:
            return config[name]
    raise KeyError(
        f"pi0.5 config.json has no {label} field (tried {list(names)})")


def gemma_variant(name: str) -> dict:
    try:
        return dict(GEMMA_VARIANTS[name])
    except KeyError as error:
        raise ValueError(
            f"unsupported pi0.5 Gemma variant {name!r}; expected one of "
            f"{sorted(GEMMA_VARIANTS)}") from error


def default_max_prefix_len(max_token_len: int = DEFAULT_MAX_TOKEN_LEN) -> int:
    return MAX_CAMERA_SLOTS * TOKENS_PER_VIEW + max_token_len


def minimum_prefix_len(policy: dict) -> int:
    return len(policy["cameras"]) * TOKENS_PER_VIEW + MAX_TOKEN_LEN


def validate_max_prefix_len(policy_name: str, policy: dict,
                            max_prefix_len: int) -> None:
    minimum = minimum_prefix_len(policy)
    if max_prefix_len < minimum:
        raise ValueError(
            f"pi0.5 {policy_name} requires --pi05-max-prefix-len >= {minimum} "
            f"({len(policy['cameras'])} camera slots x {TOKENS_PER_VIEW} + "
            f"{MAX_TOKEN_LEN} tokenizer tokens), got {max_prefix_len}")


def resolve_num_denoise_steps(config: dict, requested: int | None) -> int:
    steps = requested
    if steps is None:
        steps = int(
            config.get("num_inference_steps", DEFAULT_NUM_DENOISE_STEPS))
    steps = int(steps)
    if not 1 <= steps <= MAX_DENOISE_STEPS:
        raise ValueError(
            f"pi0.5 num_denoise_steps must be in [1, {MAX_DENOISE_STEPS}], "
            f"got {steps}")
    return steps


def xqa_cache_capacity(max_prefix_len: int, action_horizon: int) -> int:
    return (-(-(max_prefix_len + action_horizon) // XQA_CACHE_TILE) *
            XQA_CACHE_TILE)


def kv_pages_per_seq(max_prefix_len: int, action_horizon: int) -> int:
    return (xqa_cache_capacity(max_prefix_len, action_horizon) //
            KV_TOKENS_PER_PAGE)


def packed_mask_words(action_horizon: int) -> int:
    return -(-action_horizon // 32)


def num_adarms_sites(num_hidden_layers: int) -> int:
    return 2 * num_hidden_layers + 1


def _value(config, name: str, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _shape(minimum, optimum, maximum) -> dict:
    return {
        "min": list(minimum),
        "opt": list(optimum),
        "max": list(maximum),
    }


def _with_onnx_filename(config: dict, include_onnx: bool) -> dict:
    if include_onnx:
        config["onnx_filename"] = "model.onnx"
    return config


def make_visual_config(config,
                       num_views: int,
                       max_views: int,
                       *,
                       include_onnx: bool = True) -> dict:
    image_size = int(_value(config, "image_size", 224))
    patch_size = int(_value(config, "patch_size", 14))
    hidden_size = int(_value(config, "hidden_size", 1152))
    image = [3, image_size, image_size]
    return _with_onnx_filename(
        {
            "component":
            "visual",
            "engine_filename":
            "visual.engine",
            "batch_axis": {
                "pixel_values": 0
            },
            "optimization_profile": {
                "pixel_values":
                _shape([1] + image, [num_views] + image, [max_views] + image),
            },
            "tensor_contract": {
                "inputs": {
                    "pixel_values": {
                        "dtype": "fp16",
                        "rank": 4,
                        "axes": ["views", "channels", "height", "width"],
                    },
                },
                "outputs": {
                    "image_features": {
                        "dtype": "fp16",
                        "rank": 3,
                        "axes": ["views", "tokens", "hidden_size"],
                    },
                },
            },
            "builder_config": {
                "max_views": max_views
            },
            "hidden_size":
            hidden_size,
            "projection_dim":
            int(_value(config, "projection_dim", 2048)),
            "image_size":
            image_size,
            "patch_size":
            patch_size,
            "num_image_tokens":
            int(_value(config, "num_positions",
                       (image_size // patch_size)**2)),
        }, include_onnx)


def make_prefix_config(config,
                       max_prefix_len: int,
                       max_batch_size: int,
                       *,
                       include_onnx: bool = True) -> dict:
    num_layers = int(_value(config, "num_hidden_layers"))
    head_dim = int(_value(config, "head_dim"))
    hidden_size = int(_value(config, "hidden_size"))
    optimum = min(1024, max_prefix_len)
    kv_axes = ["batch", "prefix_len", "num_kv_heads", "head_dim"]
    return _with_onnx_filename(
        {
            "component": "prefix",
            "engine_filename": "prefix.engine",
            "batch_axis": {
                "inputs_embeds": 0,
                "rope_rotary_cos_sin": 0,
                "attention_pos_id": 0,
            },
            "optimization_profile": {
                "inputs_embeds":
                _shape([1, 2, hidden_size], [1, optimum, hidden_size],
                       [max_batch_size, max_prefix_len, hidden_size]),
                "rope_rotary_cos_sin":
                _shape([1, 2, head_dim], [1, optimum, head_dim],
                       [max_batch_size, max_prefix_len, head_dim]),
                "attention_pos_id":
                _shape([1, 2], [1, optimum], [max_batch_size, max_prefix_len]),
            },
            "tensor_contract": {
                "inputs": {
                    "inputs_embeds": {
                        "dtype": "fp16",
                        "rank": 3,
                        "axes": ["batch", "prefix_len", "hidden_size"],
                    },
                    "rope_rotary_cos_sin": {
                        "dtype": "fp32",
                        "rank": 3,
                        "axes": ["batch", "prefix_len", "head_dim"],
                    },
                    "attention_pos_id": {
                        "dtype": "int32",
                        "rank": 2,
                        "axes": ["batch", "prefix_len"],
                    },
                },
                "outputs": {
                    "k_layerNN": {
                        "dtype": "fp16",
                        "rank": 4,
                        "axes": kv_axes,
                    },
                    "v_layerNN": {
                        "dtype": "fp16",
                        "rank": 4,
                        "axes": kv_axes,
                    },
                },
            },
            "builder_config": {
                "max_batch_size": max_batch_size,
                "max_prefix_len": max_prefix_len,
            },
            "num_hidden_layers": num_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": int(_value(config, "num_attention_heads")),
            "num_key_value_heads": int(_value(config, "num_key_value_heads")),
            "head_dim": head_dim,
            "intermediate_size": int(_value(config, "intermediate_size")),
            "rms_norm_eps": float(_value(config, "rms_norm_eps", 1e-6)),
            "vocab_size": int(_value(config, "vocab_size", 257152)),
            "rope_theta": float(_value(config, "rope_theta", 10000.0)),
        }, include_onnx)


def make_action_config(config,
                       max_prefix_len: int,
                       max_batch_size: int,
                       num_denoise_steps: int,
                       hoist_cond: bool = False,
                       *,
                       include_onnx: bool = True) -> dict:
    num_layers = int(_value(config, "num_hidden_layers"))
    head_dim = int(_value(config, "head_dim"))
    kv_heads = int(_value(config, "num_key_value_heads"))
    hidden_size = int(_value(config, "hidden_size"))
    horizon = int(_value(config, "action_horizon"))
    action_dim = int(_value(config, "action_dim"))
    sites = num_adarms_sites(num_layers)
    modulation_dim = hidden_size * 3
    profile = {
        "noise_trajectory":
        _shape([1, horizon, action_dim], [1, horizon, action_dim],
               [max_batch_size, horizon, action_dim]),
        "timestep":
        _shape([1], [1], [max_batch_size]),
        "rope_rotary_cos_sin":
        _shape([horizon, head_dim], [horizon, head_dim],
               [max_batch_size * horizon, head_dim]),
        "attention_pos_id":
        _shape([horizon], [horizon], [max_batch_size * horizon]),
    }
    condition_contract = {
        "timestep": {
            "dtype": "fp32",
            "rank": 1,
            "axes": ["batch"],
        },
    }
    if hoist_cond:
        del profile["timestep"]
        profile["adarms_modulation"] = _shape(
            [1, sites, modulation_dim], [1, sites, modulation_dim],
            [max_batch_size, sites, modulation_dim])
        condition_contract = {
            "adarms_modulation": {
                "dtype": "fp16",
                "rank": 3,
                "axes": ["batch", "adarms_sites", "scale_shift_gate"],
            },
        }
    batch_axis = {name: 0 for name in profile}
    batch_stride = {
        "rope_rotary_cos_sin": horizon,
        "attention_pos_id": horizon,
    }
    batch_bias = {}
    capacity = xqa_cache_capacity(max_prefix_len, horizon)
    pages = kv_pages_per_seq(max_prefix_len, horizon)
    words = packed_mask_words(horizon)
    for name in ("query_lengths", "attention_sequence_lengths",
                 "kvcache_start_index"):
        profile[name] = _shape([1], [1], [max_batch_size])
        batch_axis[name] = 0
    profile["query_start_offsets"] = _shape([2], [2], [max_batch_size + 1])
    batch_axis["query_start_offsets"] = 0
    batch_bias["query_start_offsets"] = 1
    profile["execution_phase_marker"] = _shape([DIFFUSION_DENOISE_PHASE],
                                               [DIFFUSION_DENOISE_PHASE],
                                               [DIFFUSION_DENOISE_PHASE])
    batch_axis["execution_phase_marker"] = None
    profile["context_sequence_count_carrier"] = _shape([0], [0], [0])
    batch_axis["context_sequence_count_carrier"] = None
    profile["kv_page_table"] = _shape([1, 2, pages], [1, 2, pages],
                                      [max_batch_size, 2, pages])
    batch_axis["kv_page_table"] = 0
    profile["attention_mask"] = _shape([horizon, words], [horizon, words],
                                       [max_batch_size * horizon, words])
    batch_axis["attention_mask"] = 0
    batch_stride["attention_mask"] = horizon
    pool = _shape(
        [2, pages, KV_TOKENS_PER_PAGE, kv_heads, head_dim],
        [2, max_batch_size * pages, KV_TOKENS_PER_PAGE, kv_heads, head_dim],
        [2, max_batch_size * pages, KV_TOKENS_PER_PAGE, kv_heads, head_dim])
    for index in range(num_layers):
        name = f"kv_cache_layer{index:02d}"
        profile[name] = pool
        batch_axis[name] = 1
        batch_stride[name] = pages
    pool_axes = ["k_then_v", "num_pages", "page", "kv_heads", "head_dim"]
    kv_contract = {
        "query_lengths": {
            "dtype": "int32",
            "rank": 1,
            "axes": ["batch"],
        },
        "attention_sequence_lengths": {
            "dtype": "int32",
            "rank": 1,
            "axes": ["batch"],
        },
        "query_start_offsets": {
            "dtype": "int32",
            "rank": 1,
            "axes": ["batch_plus_one"],
        },
        "execution_phase_marker": {
            "dtype": "int32",
            "rank": 1,
            "axes": ["execution_phase"],
        },
        "context_sequence_count_carrier": {
            "dtype": "int32",
            "rank": 1,
            "axes": ["context_sequences"],
        },
        "kvcache_start_index": {
            "dtype": "int32",
            "rank": 1,
            "axes": ["batch"],
        },
        "kv_page_table": {
            "dtype": "int32",
            "rank": 3,
            "axes": ["batch", "k_then_v", "pages_per_seq"],
        },
        "attention_mask": {
            "dtype": "int32",
            "rank": 2,
            "axes": ["execution_tokens", "packed_mask_words"],
        },
        "kv_cache_layerNN": {
            "dtype": "fp16",
            "rank": 5,
            "axes": pool_axes,
        },
    }
    return _with_onnx_filename(
        {
            "component": "action",
            "engine_filename": "action.engine",
            "batch_axis": batch_axis,
            "batch_stride": batch_stride,
            "batch_bias": batch_bias,
            "optimization_profile": profile,
            "tensor_contract": {
                "inputs": {
                    "noise_trajectory": {
                        "dtype": "fp32",
                        "rank": 3,
                        "axes": ["batch", "action_horizon", "action_dim"],
                    },
                    **condition_contract,
                    "rope_rotary_cos_sin": {
                        "dtype": "fp32",
                        "rank": 2,
                        "axes": ["execution_tokens", "head_dim"],
                    },
                    "attention_pos_id": {
                        "dtype": "int32",
                        "rank": 1,
                        "axes": ["execution_tokens"],
                    },
                    **kv_contract,
                },
                "outputs": {
                    "action_pred": {
                        "dtype": "fp32",
                        "rank": 3,
                        "axes": ["batch", "action_horizon", "action_dim"],
                    },
                    "present_kv_cache_layerNN": {
                        "dtype": "fp16",
                        "rank": 5,
                        "axes": pool_axes,
                    },
                },
            },
            "builder_config": {
                "max_batch_size": max_batch_size,
                "max_prefix_len": max_prefix_len,
                "kv_cache_capacity": capacity,
                "paged_kv_cache": True,
            },
            "num_hidden_layers": num_layers,
            "hidden_size": hidden_size,
            "num_attention_heads": int(_value(config, "num_attention_heads")),
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "intermediate_size": int(_value(config, "intermediate_size")),
            "rms_norm_eps": float(_value(config, "rms_norm_eps", 1e-6)),
            "action_dim": action_dim,
            "action_horizon": horizon,
            "num_denoise_steps": num_denoise_steps,
            "hoisted_adarms_cond": hoist_cond,
            "num_adarms_sites": sites,
        }, include_onnx)


def make_cond_config(config,
                     num_denoise_steps: int,
                     *,
                     include_onnx: bool = True) -> dict:
    hidden_size = int(_value(config, "hidden_size"))
    sites = num_adarms_sites(int(_value(config, "num_hidden_layers")))
    modulation_dim = hidden_size * 3
    return _with_onnx_filename(
        {
            "component": "cond",
            "engine_filename": "cond.engine",
            "batch_axis": {
                "timestep": None
            },
            "optimization_profile": {
                "timestep": _shape([1], [num_denoise_steps],
                                   [MAX_DENOISE_STEPS]),
            },
            "tensor_contract": {
                "inputs": {
                    "timestep": {
                        "dtype": "fp32",
                        "rank": 1,
                        "axes": ["num_steps"],
                    },
                },
                "outputs": {
                    "adarms_modulation": {
                        "dtype": "fp16",
                        "rank": 3,
                        "axes":
                        ["num_steps", "adarms_sites", "scale_shift_gate"],
                    },
                },
            },
            "builder_config": {
                "max_denoise_steps": MAX_DENOISE_STEPS
            },
            "hidden_size": hidden_size,
            "num_adarms_sites": sites,
            "modulation_dim": modulation_dim,
            "num_denoise_steps": num_denoise_steps,
        }, include_onnx)
