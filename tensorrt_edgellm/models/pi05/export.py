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
"""pi0.5 policy component export (library; no standalone CLI).

Invoked through the unified ``tensorrt-edgellm-export`` entry point, which
routes pi0.5 checkpoints here. Exports the policy components -- ``visual``,
``prefix``, ``action`` and, unless hoisting is turned off, ``cond`` -- plus the
runtime sidecars from a converted openpi PyTorch checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from typing import Dict

import torch

from ..._version import __version__
from ...onnx.export import _strip_attention_plugin_optional_inputs
from ...onnx.export_encoder import _run_dynamo_export
from .modeling_pi05_action import (DIFFUSION_DENOISE_PHASE, KV_TOKENS_PER_PAGE,
                                   Pi05Action, Pi05ActionConfig,
                                   build_pi05_action, build_pi05_cond,
                                   kv_pages_per_seq, num_adarms_sites,
                                   packed_mask_words, xqa_cache_capacity)
from .modeling_pi05_prefix import (Pi05Prefix, Pi05PrefixConfig,
                                   build_pi05_prefix)
from .modeling_pi05_visual import (Pi05Visual, Pi05VisualConfig,
                                   build_pi05_visual)
from .policy_assets import (MAX_CAMERA_SLOTS, MAX_TOKEN_LEN,
                            OPENPI_POLICY_CONTRACTS, POLICY_CONTRACT_FILENAME,
                            TEXT_TOKENIZER_DIRNAME, build_policy_contract,
                            checkpoint_fingerprint, load_json_if_present,
                            policy_config_name, policy_semantics, sha256_file,
                            stage_text_tokenizer, write_policy_contract)
from .weights import (is_pi05_weights, load_checkpoint_weights,
                      load_pi05_config, split_pi05_weights)

logger = logging.getLogger(__name__)

PI05_COMPONENTS = ("visual", "prefix", "action")
# ``cond`` is emitted by the action export under hoisting, not selected directly.
ALL_COMPONENTS = PI05_COMPONENTS + ("cond", )
# The artifact ABI of a pi0.5 bundle. Bump only when an existing export stops being
# buildable, and only alongside an edgellm release -- not for ordinary export changes.
CONTRACT_VERSION = 1

# openpi ``models/gemma.py`` variant tables. The converted config.json records
# only the variant names, so the shapes are resolved here.
_GEMMA_VARIANTS = {
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

# Prompt capacity the prefix is built for: openpi's ``max_token_len`` rounded up to a
# multiple of 16, as the reference engine build does.
DEFAULT_MAX_TOKEN_LEN = -(-MAX_TOKEN_LEN // 16) * 16
_TOKENS_PER_VIEW = 256

# Ceiling on the runtime-selectable denoise step count in a hoisted-cond export:
# it sizes the cond graph's profile and the modulation buffer the runtime slices.
MAX_DENOISE_STEPS = 64
DEFAULT_NUM_DENOISE_STEPS = 10


def _config_field(config: dict, names: "tuple[str, ...]", label: str):
    """Read the first present key among \\p names.

    Two checkpoint schemas carry the same pi0.5 weights: openpi's
    ``convert_jax_model_to_pytorch.py`` writes ``action_dim`` /
    ``action_horizon``, while the LeRobot HF releases write ``max_action_dim`` /
    ``n_action_steps``.
    """
    for name in names:
        if name in config:
            return config[name]
    raise KeyError(
        f"pi0.5 config.json has no {label} field (tried {list(names)})")


def _variant(name: str) -> dict:
    if name not in _GEMMA_VARIANTS:
        raise ValueError(f"Unsupported pi0.5 gemma variant {name!r}; "
                         f"expected one of {sorted(_GEMMA_VARIANTS)}")
    return _GEMMA_VARIANTS[name]


def default_max_prefix_len(max_token_len: int = DEFAULT_MAX_TOKEN_LEN) -> int:
    """Prefix capacity for every configuration: all image slots plus the prompt.

    Sized for ``MAX_CAMERA_SLOTS`` whatever the configuration feeds, so the canonical
    full-slot shape the accuracy and profiling harnesses use always fits.
    """
    return MAX_CAMERA_SLOTS * _TOKENS_PER_VIEW + max_token_len


def _write_component_config(out_dir: str, config: dict,
                            export_id: str) -> None:
    """Stamp contract/package versions and the export identity, and write it."""
    config = dict(config)
    config["contract_version"] = CONTRACT_VERSION
    config["edgellm_version"] = __version__
    config["export_id"] = export_id
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def _export_id(fingerprint: str, options: dict) -> str:
    """Identity stamped on every component of one export.

    Derived rather than random: re-exporting a single component from the same
    checkpoint and options must reproduce it, which is what makes a partial
    re-export into an existing directory provable rather than merely plausible.
    """
    payload = json.dumps(
        {
            "contract_version": CONTRACT_VERSION,
            "edgellm_version": __version__,
            "checkpoint_fingerprint": fingerprint,
            "options": options,
        },
        sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _present_components(output_dir: str) -> "list[str]":
    return [
        c for c in ALL_COMPONENTS
        if os.path.isfile(os.path.join(output_dir, c, "config.json"))
    ]


def _validate_existing_export(output_dir: str, fingerprint: str,
                              export_id: str) -> None:
    """Refuse to write ONNX components beside ones from a different export.

    A partial export reuses whatever is already in ``output_dir``, so without this
    a directory can end up holding components from two checkpoints or two sets of
    export options. Identity is the ``export_id``, which hashes the checkpoint
    fingerprint together with the options. Runs before anything is written, so a
    rejected export leaves the directory exactly as it found it.
    """
    conflicts = []
    manifest = load_json_if_present(
        os.path.join(output_dir, POLICY_CONTRACT_FILENAME))
    if manifest is not None:
        for field, expected in (("checkpoint_fingerprint", fingerprint),
                                ("export_id", export_id)):
            if manifest.get(field) != expected:
                conflicts.append((POLICY_CONTRACT_FILENAME, field,
                                  manifest.get(field), expected))
    for component in _present_components(output_dir):
        config = load_json_if_present(
            os.path.join(output_dir, component, "config.json")) or {}
        if config.get("export_id") != export_id:
            conflicts.append((f"{component}/config.json", "export_id",
                              config.get("export_id"), export_id))
    if not conflicts:
        # assets/ and text_tokenizer/ carry no export identity, so a directory
        # holding only those cannot be told apart from an earlier export's
        # leftovers and must not be merged into.
        if (manifest is None and not _present_components(output_dir)
                and os.path.isdir(output_dir) and os.listdir(output_dir)):
            raise ValueError(
                f"{output_dir} is not empty but holds no pi0.5 export: "
                f"{sorted(os.listdir(output_dir))}. Export into an empty "
                "directory, so no untagged leftover of an earlier export "
                "survives beside it.")
        return
    detail = "; ".join(f"{path} has {field}={found!r}, this export has "
                       f"{expected!r}"
                       for path, field, found, expected in conflicts)
    raise ValueError(
        f"{output_dir} already holds a different pi0.5 export: {detail}. "
        "Export into an empty directory, or re-export every component from the "
        "same checkpoint and options.")


def _validate_policy_semantics(manifest_path: str, contract: dict) -> None:
    """Refuse a partial re-export that would redefine the policy in place.

    ``export_id`` covers the weights and the export options, not the policy
    semantics, the feature contract or the tokenizer the manifest is derived
    from. A re-export rewrites the manifest but leaves the components it was not
    asked for, so without this those declarations can move under components built
    against the old ones. Runs before any output artifact is mutated and before the
    checkpoint weights are loaded.
    """
    previous = load_json_if_present(manifest_path)
    if previous is None:
        return
    was, now = policy_semantics(previous), policy_semantics(contract)
    changed = sorted(k for k in now if was.get(k) != now[k])
    if not changed:
        return
    detail = "; ".join(f"{k}: {was.get(k)!r} -> {now[k]!r}" for k in changed)
    raise ValueError(
        f"{manifest_path} describes a different policy than this export would: "
        f"{detail}. Re-export every component, into an empty directory.")


def _discard(paths: "list[str]") -> None:
    """Drop what a failed run wrote, so no half-updated export looks complete."""
    for path in paths:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)
        else:
            continue
        logger.error("Removed %s left behind by the failed pi0.5 export", path)


def _shape(min_s: list, opt_s: list, max_s: list) -> dict:
    return {"min": min_s, "opt": opt_s, "max": max_s}


# ---------------------------------------------------------------------------
# Component configs
# ---------------------------------------------------------------------------


def make_visual_config(cfg: Pi05VisualConfig, num_views: int,
                       max_views: int) -> dict:
    hw = [3, cfg.image_size, cfg.image_size]
    return {
        "component": "visual",
        "onnx_filename": "model.onnx",
        "engine_filename": "visual.engine",
        "batch_axis": {
            "pixel_values": 0
        },
        "optimization_profile": {
            "pixel_values": _shape([1] + hw, [num_views] + hw,
                                   [max_views] + hw),
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
        "hidden_size": cfg.hidden_size,
        "projection_dim": cfg.projection_dim,
        "image_size": cfg.image_size,
        "patch_size": cfg.patch_size,
        "num_image_tokens": cfg.num_positions,
    }


def make_prefix_config(cfg: Pi05PrefixConfig, max_prefix_len: int,
                       max_batch_size: int) -> dict:
    num_layers, head_dim, hidden = cfg.num_hidden_layers, cfg.head_dim, cfg.hidden_size
    opt_len = min(1024, max_prefix_len)
    kv_axes = ["batch", "prefix_len", "num_kv_heads", "head_dim"]
    return {
        "component": "prefix",
        "onnx_filename": "model.onnx",
        "engine_filename": "prefix.engine",
        "batch_axis": {
            "inputs_embeds": 0,
            "rope_rotary_cos_sin": 0,
            "attention_pos_id": 0,
        },
        "optimization_profile": {
            "inputs_embeds":
            _shape([1, 2, hidden], [1, opt_len, hidden],
                   [max_batch_size, max_prefix_len, hidden]),
            "rope_rotary_cos_sin":
            _shape([1, 2, head_dim], [1, opt_len, head_dim],
                   [max_batch_size, max_prefix_len, head_dim]),
            "attention_pos_id":
            _shape([1, 2], [1, opt_len], [max_batch_size, max_prefix_len]),
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
                    "axes": kv_axes
                },
                "v_layerNN": {
                    "dtype": "fp16",
                    "rank": 4,
                    "axes": kv_axes
                },
            },
        },
        "builder_config": {
            "max_batch_size": max_batch_size,
            "max_prefix_len": max_prefix_len,
        },
        "num_hidden_layers": num_layers,
        "hidden_size": hidden,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads,
        "head_dim": head_dim,
        "intermediate_size": cfg.intermediate_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "vocab_size": cfg.vocab_size,
        "rope_theta": cfg.rope_theta,
    }


def make_action_config(
    cfg: Pi05ActionConfig,
    max_prefix_len: int,
    max_batch_size: int,
    num_denoise_steps: int,
    hoist_cond: bool = False,
) -> dict:
    n, d, hkv = cfg.num_hidden_layers, cfg.head_dim, cfg.num_key_value_heads
    horizon, action_dim = cfg.action_horizon, cfg.action_dim
    sites, mod = num_adarms_sites(cfg), cfg.hidden_size * 3
    profile = {
        "noise_trajectory":
        _shape([1, horizon, action_dim], [1, horizon, action_dim],
               [max_batch_size, horizon, action_dim]),
        "timestep":
        _shape([1], [1], [max_batch_size]),
        "rope_rotary_cos_sin":
        _shape([horizon, d], [horizon, d], [max_batch_size * horizon, d]),
        "attention_pos_id":
        _shape([horizon], [horizon], [max_batch_size * horizon]),
    }
    cond_contract = {
        "timestep": {
            "dtype": "fp32",
            "rank": 1,
            "axes": ["batch"]
        },
    }
    if hoist_cond:
        del profile["timestep"]
        profile["adarms_modulation"] = _shape([1, sites, mod], [1, sites, mod],
                                              [max_batch_size, sites, mod])
        cond_contract = {
            "adarms_modulation": {
                "dtype": "fp16",
                "rank": 3,
                "axes": ["batch", "adarms_sites", "scale_shift_gate"],
            },
        }
    batch_axis = {name: 0 for name in profile}
    # One request occupies a whole action horizon of the token-major axis.
    batch_stride: dict = {
        "rope_rotary_cos_sin": horizon,
        "attention_pos_id": horizon
    }
    batch_bias: dict = {}
    capacity = xqa_cache_capacity(max_prefix_len, horizon)
    pages = kv_pages_per_seq(max_prefix_len, horizon)
    words = packed_mask_words(horizon)
    profile["query_lengths"] = _shape([1], [1], [max_batch_size])
    batch_axis["query_lengths"] = 0
    profile["attention_sequence_lengths"] = _shape([1], [1], [max_batch_size])
    batch_axis["attention_sequence_lengths"] = 0
    profile["query_start_offsets"] = _shape([2], [2], [max_batch_size + 1])
    batch_axis["query_start_offsets"] = 0
    batch_bias["query_start_offsets"] = 1
    # Shape-only carriers: the extent IS the value, so all three bounds are the
    # one the runtime binds -- tree decoding, and no context sequence.
    profile["execution_phase_marker"] = _shape([DIFFUSION_DENOISE_PHASE],
                                               [DIFFUSION_DENOISE_PHASE],
                                               [DIFFUSION_DENOISE_PHASE])
    batch_axis["execution_phase_marker"] = None
    profile["context_sequence_count_carrier"] = _shape([0], [0], [0])
    batch_axis["context_sequence_count_carrier"] = None
    profile["kvcache_start_index"] = _shape([1], [1], [max_batch_size])
    batch_axis["kvcache_start_index"] = 0
    profile["kv_page_table"] = _shape([1, 2, pages], [1, 2, pages],
                                      [max_batch_size, 2, pages])
    batch_axis["kv_page_table"] = 0
    profile["attention_mask"] = _shape([horizon, words], [horizon, words],
                                       [max_batch_size * horizon, words])
    batch_axis["attention_mask"] = 0
    batch_stride["attention_mask"] = horizon
    pool = _shape([2, pages, KV_TOKENS_PER_PAGE, hkv, d],
                  [2, max_batch_size * pages, KV_TOKENS_PER_PAGE, hkv, d],
                  [2, max_batch_size * pages, KV_TOKENS_PER_PAGE, hkv, d])
    for i in range(n):
        profile[f"kv_cache_layer{i:02d}"] = pool
        # The pool grows by whole sequences, not by one page per request.
        batch_axis[f"kv_cache_layer{i:02d}"] = 1
        batch_stride[f"kv_cache_layer{i:02d}"] = pages
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
    return {
        "component": "action",
        "onnx_filename": "model.onnx",
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
                **cond_contract,
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
            # The runtime reads this to reject a bundle exported before the pool.
            "paged_kv_cache": True,
        },
        "num_hidden_layers": n,
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": hkv,
        "head_dim": d,
        "intermediate_size": cfg.intermediate_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "action_dim": action_dim,
        "action_horizon": horizon,
        # The graph emits the velocity only; the runtime owns the Euler step.
        "num_denoise_steps": num_denoise_steps,
        # False: the graph embeds the time embedder and the per-site AdaRMS
        # Denses and recomputes them every step. True: they live in the cond
        # component and the runtime binds one precomputed row per step.
        "hoisted_adarms_cond": hoist_cond,
        "num_adarms_sites": sites,
    }


def make_cond_config(cfg: Pi05ActionConfig, num_denoise_steps: int) -> dict:
    sites, mod = num_adarms_sites(cfg), cfg.hidden_size * 3
    return {
        "component": "cond",
        "onnx_filename": "model.onnx",
        "engine_filename": "cond.engine",
        "batch_axis": {
            "timestep": None
        },
        "optimization_profile": {
            "timestep": _shape([1], [num_denoise_steps], [MAX_DENOISE_STEPS]),
        },
        "tensor_contract": {
            "inputs": {
                "timestep": {
                    "dtype": "fp32",
                    "rank": 1,
                    "axes": ["num_steps"]
                },
            },
            "outputs": {
                "adarms_modulation": {
                    "dtype": "fp16",
                    "rank": 3,
                    "axes": ["num_steps", "adarms_sites", "scale_shift_gate"],
                },
            },
        },
        "builder_config": {
            "max_denoise_steps": MAX_DENOISE_STEPS
        },
        "hidden_size": cfg.hidden_size,
        "num_adarms_sites": sites,
        "modulation_dim": mod,
        # Site order: layer i -> rows 2i (input_layernorm) and 2i + 1
        # (post_attention_layernorm), final norm last.
        "num_denoise_steps": num_denoise_steps,
    }


# ---------------------------------------------------------------------------
# Sidecars
# ---------------------------------------------------------------------------


def _save_embedding_sidecar(out_dir: str, embedding: Dict[str, torch.Tensor],
                            dtype: torch.dtype) -> dict:
    """Write the token embedding table the host gathers on the prefix path."""
    import safetensors.torch as st

    if "embed_tokens.weight" not in embedding:
        raise KeyError("pi0.5 checkpoint has no language_model.embed_tokens")
    tensor = embedding["embed_tokens.weight"].to(dtype).contiguous()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "embed_tokens.safetensors")
    st.save_file({"embed_tokens.weight": tensor}, path)
    logger.info("Saved embed_tokens.safetensors %s", tuple(tensor.shape))
    return {
        "embed_tokens.safetensors": {
            "sha256": sha256_file(path),
            "dtype": str(dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
        }
    }


def _has_norm_stats(assets_dir: str) -> bool:
    return any("norm_stats.json" in files
               for _, _, files in os.walk(assets_dir))


def _stage_policy_assets(checkpoint: str, output_dir: str) -> None:
    """Copy the normalization statistics the policy layer unnormalizes with.

    The supported HF weight conversion does not include openpi's normalization
    assets, so their absence cannot be an export error; the documented workflow
    stages the matching ``norm_stats.json`` after export. An ``assets/`` carrying
    none is an error: it would pass a staging check while normalizing nothing.
    """
    src = os.path.join(checkpoint, "assets")
    dst = os.path.join(output_dir, "assets")
    if not os.path.isdir(src):
        logger.warning(
            "pi0.5 checkpoint has no assets/; stage norm_stats.json under %s "
            "before running the policy -- the runtime refuses to load without "
            "it", dst)
        return
    if not _has_norm_stats(src):
        raise FileNotFoundError(
            f"{src} carries no norm_stats.json; the policy layer cannot "
            "normalize the state or convert actions to robot units")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    logger.info("Staged policy assets: %s", dst)


# ---------------------------------------------------------------------------
# Per-component export
# ---------------------------------------------------------------------------


def export_visual(weights: Dict[str, torch.Tensor], out_dir: str,
                  dtype: torch.dtype, num_views: int, max_views: int,
                  export_id: str) -> None:
    cfg = Pi05VisualConfig()
    model: Pi05Visual = build_pi05_visual(cfg, weights, dtype).to("cpu")
    args, input_names, output_names, dynamic_shapes = (
        model.get_onnx_export_args("cpu"))
    os.makedirs(out_dir, exist_ok=True)
    _run_dynamo_export(model, args, os.path.join(out_dir, "model.onnx"),
                       input_names, output_names, dynamic_shapes)
    _write_component_config(out_dir,
                            make_visual_config(cfg, num_views, max_views),
                            export_id)
    logger.info("pi0.5 visual export complete: %s", out_dir)


def export_prefix(weights: Dict[str, torch.Tensor],
                  embedding: Dict[str, torch.Tensor], out_dir: str,
                  dtype: torch.dtype, variant: str, max_prefix_len: int,
                  max_batch_size: int, export_id: str) -> None:
    cfg = Pi05PrefixConfig(**_variant(variant))
    model: Pi05Prefix = build_pi05_prefix(cfg, weights, dtype).to("cpu")
    args, input_names, output_names, dynamic_shapes = (
        model.get_onnx_export_args("cpu"))
    os.makedirs(out_dir, exist_ok=True)
    _run_dynamo_export(model, args, os.path.join(out_dir, "model.onnx"),
                       input_names, output_names, dynamic_shapes)
    artifacts = _save_embedding_sidecar(out_dir, embedding, dtype)
    config = make_prefix_config(cfg, max_prefix_len, max_batch_size)
    config["artifacts"] = artifacts
    _write_component_config(out_dir, config, export_id)
    logger.info("pi0.5 prefix export complete: %s", out_dir)


def export_action(weights: Dict[str, torch.Tensor], out_dir: str,
                  dtype: torch.dtype, variant: str, action_dim: int,
                  action_horizon: int, max_prefix_len: int,
                  max_batch_size: int, num_denoise_steps: int,
                  hoist_cond: bool, cond_dir: str, export_id: str) -> None:
    cfg = Pi05ActionConfig(**_variant(variant),
                           action_dim=action_dim,
                           action_horizon=action_horizon)
    model: Pi05Action = build_pi05_action(cfg, weights, dtype,
                                          hoist_cond).to("cpu")
    args, input_names, output_names, dynamic_shapes = (
        model.get_onnx_export_args(max_prefix_len, "cpu"))
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "model.onnx")
    _run_dynamo_export(model, args, onnx_path, input_names, output_names,
                       dynamic_shapes)
    # The translation emits every optional AttentionPlugin input; the C++ contract
    # counts only the enabled groups.
    _strip_attention_plugin_optional_inputs(onnx_path)
    _write_component_config(
        out_dir,
        make_action_config(cfg, max_prefix_len, max_batch_size,
                           num_denoise_steps, hoist_cond), export_id)
    logger.info("pi0.5 action export complete: %s", out_dir)

    if hoist_cond:
        # Exported from the same loaded model: the cond graph's row order and
        # the action graph's slice indices come from one site enumeration.
        export_cond(model, cfg, cond_dir, num_denoise_steps, export_id)


def export_cond(action: Pi05Action, cfg: Pi05ActionConfig, out_dir: str,
                num_denoise_steps: int, export_id: str) -> None:
    model = build_pi05_cond(action)
    args, input_names, output_names, dynamic_shapes = (
        model.get_onnx_export_args(num_denoise_steps, "cpu"))
    os.makedirs(out_dir, exist_ok=True)
    _run_dynamo_export(model, args, os.path.join(out_dir, "model.onnx"),
                       input_names, output_names, dynamic_shapes)
    _write_component_config(out_dir, make_cond_config(cfg, num_denoise_steps),
                            export_id)
    logger.info("pi0.5 cond export complete: %s", out_dir)


def resolve_num_denoise_steps(config: dict, requested: "int | None") -> int:
    """Explicit argument beats the checkpoint, which beats the default.

    ``requested`` is None only when the caller supplied nothing; a value that
    happens to equal the default is still an explicit choice and is honoured.
    """
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


def export_pi05_components(
    checkpoint: str,
    output_dir: str,
    components: "list[str] | None" = None,
    dtype: torch.dtype = torch.float16,
    policy_config: "str | None" = None,
    num_views: "int | None" = None,
    max_views: int = MAX_CAMERA_SLOTS,
    max_token_len: int = DEFAULT_MAX_TOKEN_LEN,
    max_prefix_len: "int | None" = None,
    max_batch_size: int = 1,
    num_denoise_steps: "int | None" = None,
    hoist_adarms_cond: bool = True,
) -> None:
    """Export pi0.5 policy components from a converted openpi checkpoint.

    ``hoist_adarms_cond`` (default) moves the time embedder and the per-site
    AdaRMS Denses out of the per-step action graph into a one-shot ``cond``
    component. Turning it off exports no ``cond`` and keeps them per-step.
    """
    if dtype != torch.float16:
        # Rejected here rather than at load: the attention plugin and the component
        # contracts are fp16-only, so another dtype would export and build cleanly and
        # only then bind wrong shapes.
        raise ValueError(f"pi0.5 export supports float16 only, got {dtype}")

    components = list(components or PI05_COMPONENTS)
    unknown = [c for c in components if c not in PI05_COMPONENTS]
    if unknown:
        raise ValueError(f"Unsupported pi0.5 components: {unknown}; "
                         f"expected a subset of {list(PI05_COMPONENTS)}")

    config = load_pi05_config(checkpoint)
    action_dim = int(
        _config_field(config, ("action_dim", "max_action_dim"), "action dim"))
    # From the openpi configuration, not the checkpoint: a Hugging Face mirror ships
    # LeRobot's own horizon (chunk_size 50, n_action_steps 10) and the graph must be
    # built at the same H the manifest declares.
    config_name = policy_config_name(checkpoint, config, policy_config)
    policy = OPENPI_POLICY_CONTRACTS[config_name]
    action_horizon = int(policy["action_horizon"])
    if num_views is None:
        num_views = int(policy["opt_views"])
    num_denoise_steps = resolve_num_denoise_steps(config, num_denoise_steps)
    paligemma_variant = _config_field(config, ("paligemma_variant", ),
                                      "paligemma variant")
    expert_variant = _config_field(config, ("action_expert_variant", ),
                                   "action expert variant")

    if max_prefix_len is None:
        max_prefix_len = default_max_prefix_len(max_token_len)

    options = {
        "dtype": "float16",
        "num_views": num_views,
        "max_views": max_views,
        "max_token_len": max_token_len,
        "max_prefix_len": max_prefix_len,
        "max_batch_size": max_batch_size,
        "num_denoise_steps": num_denoise_steps,
        "hoist_adarms_cond": hoist_adarms_cond,
        "policy_config": config_name,
        # Shapes and variants the weights do not pin: the same checkpoint under a
        # changed config must not reuse a partial export built from the old one.
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "paligemma_variant": paligemma_variant,
        "expert_variant": expert_variant,
    }
    fingerprint = checkpoint_fingerprint(checkpoint)
    export_id = _export_id(fingerprint, options)
    _validate_existing_export(output_dir, fingerprint, export_id)
    logger.info(
        "pi0.5 export %s (%s): action_dim=%d action_horizon=%d "
        "max_prefix_len=%d num_denoise_steps=%d", export_id, config_name,
        action_dim, action_horizon, max_prefix_len, num_denoise_steps)

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, POLICY_CONTRACT_FILENAME)
    # Written last, and discarded on failure only when this run created it: it is the file
    # that claims the set is complete. A re-export into an existing directory keeps the
    # previous manifest and components, so a failure partway can still leave a directory
    # that looks complete and carries the same export id. Detecting that needs a
    # transactional export, which this is not.
    staged = [] if os.path.exists(manifest_path) else [manifest_path]
    try:
        # Only what this run creates is discardable. A partial re-export reuses whatever
        # is already here, and removing that on failure would destroy a valid export.
        def stage_if_new(path: str) -> None:
            if not os.path.exists(path):
                staged.append(path)

        assets_dir = os.path.join(output_dir, "assets")
        # Nothing in output_dir may be touched before the contract is accepted: a
        # rejected export must leave the bundle it refused exactly as it found it.
        with tempfile.TemporaryDirectory() as scratch:
            tokenizer = stage_text_tokenizer(scratch)
            contract = build_policy_contract(
                checkpoint, tokenizer, {
                    "contract_version": CONTRACT_VERSION,
                    "checkpoint_fingerprint": fingerprint,
                    "export_id": export_id,
                    "components": [],
                }, config_name)
            _validate_policy_semantics(manifest_path, contract)

            assets_existed = os.path.isdir(assets_dir)
            _stage_policy_assets(checkpoint, output_dir)
            if os.path.isdir(assets_dir) and not assets_existed:
                staged.append(assets_dir)
            tokenizer_dir = os.path.join(output_dir, TEXT_TOKENIZER_DIRNAME)
            stage_if_new(tokenizer_dir)
            shutil.rmtree(tokenizer_dir, ignore_errors=True)
            shutil.move(os.path.join(scratch, TEXT_TOKENIZER_DIRNAME),
                        tokenizer_dir)

        raw = load_checkpoint_weights(checkpoint)
        if not is_pi05_weights(raw):
            raise ValueError(
                f"{checkpoint} is not a pi0.5 checkpoint (no time_mlp/adaRMS "
                "signature); pi0 is not supported by this exporter")
        visual_w, prefix_w, embedding_w, action_w = split_pi05_weights(raw)

        for component in components:
            out_dir = os.path.join(output_dir, component)
            stage_if_new(out_dir)
            if component == "visual":
                export_visual(visual_w, out_dir, dtype, num_views, max_views,
                              export_id)
            elif component == "prefix":
                export_prefix(prefix_w, embedding_w, out_dir, dtype,
                              paligemma_variant, max_prefix_len,
                              max_batch_size, export_id)
            elif component == "action":
                if hoist_adarms_cond:
                    stage_if_new(os.path.join(output_dir, "cond"))
                export_action(action_w, out_dir, dtype, expert_variant,
                              action_dim, action_horizon, max_prefix_len,
                              max_batch_size, num_denoise_steps,
                              hoist_adarms_cond,
                              os.path.join(output_dir, "cond"), export_id)

        contract["components"] = _present_components(output_dir)
        write_policy_contract(output_dir, contract)
    except BaseException:
        _discard(staged)
        raise
