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
"""Cosmos3-Edge policy component export (library; no standalone CLI).

Invoked through the unified ``tensorrt-edgellm-export`` entry point, which
routes Cosmos3-Edge checkpoints here. Exports the three policy components
(``und_prefill``, ``gen``, ``vae_encoder``) plus tokenizer artifacts from a
HF/diffusers-format checkpoint.
"""

from __future__ import annotations

import json
import logging
import os
import shutil

import torch

from ..._version import __version__
from ...onnx.export_encoder import _run_dynamo_export
from .modeling_gen import (ACTION_CHUNK_SIZE, DEFAULT_FPS,
                           DEFAULT_MAX_VIDEO_SUBSAMPLE_FACTOR,
                           DEFAULT_NUM_FRAMES, build_cosmos3_gen,
                           gen_config_from_transformer, make_gen_config)
from .modeling_und_prefill import (build_cosmos3_und_prefill,
                                   make_und_prefill_config)
from .modeling_vae_encoder import (VAE_HEIGHT, VAE_WIDTH,
                                   build_cosmos3_vae_encoder,
                                   get_vae_onnx_export_args,
                                   make_vae_encoder_config)
from .weights import load_config_json, split_transformer_weights

logger = logging.getLogger(__name__)

COSMOS3_COMPONENTS = ("und_prefill", "gen", "vae_encoder")
CONTRACT_VERSION = 1

# Policy-path export default (droid context length).
DEFAULT_MAX_UND_LEN = 512


def _write_component_config(out_dir: str, config: dict) -> None:
    """Stamp contract/package versions on a component config and write it."""
    config = dict(config)
    config["contract_version"] = CONTRACT_VERSION
    config["edgellm_version"] = __version__
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def _transformer_dir(checkpoint: str) -> str:
    return os.path.join(checkpoint, "transformer")


def _save_embedding_artifacts(out_dir: str, und_weights: dict,
                              dtype: torch.dtype) -> None:
    import safetensors.torch as st

    artifacts = {
        "embed_tokens.weight": "embed_tokens.safetensors",
        "lm_head.weight": "lm_head.safetensors",
    }
    for key, filename in artifacts.items():
        if key in und_weights:
            st.save_file({key: und_weights[key].to(dtype).contiguous()},
                         os.path.join(out_dir, filename))
            logger.info("Saved %s", filename)


def _write_tokenizer_artifacts(checkpoint: str, output_dir: str) -> None:
    """Copy tokenizer files and materialize the provider Jinja template."""
    tokenizer_src = os.path.join(checkpoint, "text_tokenizer")
    if not os.path.isdir(tokenizer_src):
        raise FileNotFoundError(
            f"Cosmos3 checkpoint is missing text_tokenizer/: {tokenizer_src}")

    tokenizer_dst = os.path.join(output_dir, "text_tokenizer")
    shutil.copytree(tokenizer_src, tokenizer_dst, dirs_exist_ok=True)

    from ...chat_template import write_chat_template
    write_chat_template(checkpoint, tokenizer_dst)
    logger.info("Tokenizer artifacts complete: %s", tokenizer_dst)


def export_gen(
        checkpoint: str,
        out_dir: str,
        max_und_len: int,
        dtype: torch.dtype,
        action_chunk_size: "int | None" = None,
        num_frames: "int | None" = None,
        fps: float = DEFAULT_FPS,
        max_video_subsample_factor: int = DEFAULT_MAX_VIDEO_SUBSAMPLE_FACTOR,
        min_action_chunk: "int | None" = None,
        max_action_chunk: "int | None" = None) -> None:
    """Export the GEN diffusion expert."""
    tcfg = load_config_json(checkpoint, "transformer")
    _, gen_weights = split_transformer_weights(_transformer_dir(checkpoint))
    cfg = gen_config_from_transformer(tcfg,
                                      action_chunk_size=action_chunk_size,
                                      num_frames=num_frames)
    model = build_cosmos3_gen(cfg, gen_weights, dtype).to("cpu")
    args, input_names, output_names, dynamic_shapes = model.get_onnx_export_args(
        max_und_len, "cpu")
    os.makedirs(out_dir, exist_ok=True)
    _run_dynamo_export(model, args, os.path.join(out_dir, "model.onnx"),
                       input_names, output_names, dynamic_shapes)
    _write_component_config(
        out_dir,
        make_gen_config(cfg,
                        tcfg,
                        max_und_len,
                        fps=fps,
                        max_video_subsample_factor=max_video_subsample_factor,
                        min_action_chunk=min_action_chunk,
                        max_action_chunk=max_action_chunk))
    logger.info("GEN export complete: %s", out_dir)


def export_und_prefill(checkpoint: str,
                       out_dir: str,
                       dtype: torch.dtype,
                       max_und_len: int = DEFAULT_MAX_UND_LEN) -> None:
    """Export the policy UND-prefill tower."""
    tcfg = load_config_json(checkpoint, "transformer")
    und_weights, _ = split_transformer_weights(_transformer_dir(checkpoint))
    model, cfg = build_cosmos3_und_prefill(tcfg, und_weights, dtype)
    args, input_names, output_names, dynamic_shapes = model.get_onnx_export_args(
        "cpu")
    os.makedirs(out_dir, exist_ok=True)
    _run_dynamo_export(model, args, os.path.join(out_dir, "model.onnx"),
                       input_names, output_names, dynamic_shapes)
    _write_component_config(out_dir, make_und_prefill_config(cfg, max_und_len))
    _save_embedding_artifacts(out_dir, und_weights, dtype)
    logger.info("UND-prefill export complete: %s", out_dir)


def export_vae_encoder(checkpoint: str, out_dir: str, height: int, width: int,
                       dtype: torch.dtype, num_frames: int) -> None:
    """Export the Wan VAE encoder."""
    model = build_cosmos3_vae_encoder(os.path.join(checkpoint, "vae"),
                                      dtype).to("cpu")
    args, input_names, output_names, dynamic_shapes = get_vae_onnx_export_args(
        height, width, "cpu", dtype, num_frames)
    os.makedirs(out_dir, exist_ok=True)
    _run_dynamo_export(model, args, os.path.join(out_dir, "model.onnx"),
                       input_names, output_names, dynamic_shapes)
    _write_component_config(out_dir,
                            make_vae_encoder_config(height, width, num_frames))
    logger.info("VAE encoder export complete: %s", out_dir)


def export_cosmos3_components(checkpoint: str,
                              output_dir: str,
                              components: "list[str] | None" = None,
                              max_und_len: int = DEFAULT_MAX_UND_LEN,
                              height: int = VAE_HEIGHT,
                              width: int = VAE_WIDTH,
                              dtype: torch.dtype = torch.float16,
                              num_frames: int = DEFAULT_NUM_FRAMES,
                              action_chunk_size: int = ACTION_CHUNK_SIZE,
                              fps: float = DEFAULT_FPS,
                              max_video_subsample_factor: int = (
                                  DEFAULT_MAX_VIDEO_SUBSAMPLE_FACTOR),
                              min_action_chunk: "int | None" = None,
                              max_action_chunk: "int | None" = None) -> None:
    """Export Cosmos3 policy components from a HF/diffusers checkpoint.

    The defaults are the canonical policy request: one input image,
    ``num_frames=17``, ``fps=5``, ``action_chunk_size=16`` (action chunk
    [16, 8]). The generated frames are the rollout associated with the
    action chunk, not an input video.
    """
    components = list(components or COSMOS3_COMPONENTS)
    unknown = [c for c in components if c not in COSMOS3_COMPONENTS]
    if unknown:
        raise ValueError(f"Unsupported Cosmos3 components: {unknown}; "
                         f"expected a subset of {list(COSMOS3_COMPONENTS)}")
    os.makedirs(output_dir, exist_ok=True)
    _write_tokenizer_artifacts(checkpoint, output_dir)
    for component in components:
        out_dir = os.path.join(output_dir, component)
        if component == "gen":
            export_gen(checkpoint,
                       out_dir,
                       max_und_len,
                       dtype,
                       action_chunk_size=action_chunk_size,
                       num_frames=num_frames,
                       fps=fps,
                       max_video_subsample_factor=max_video_subsample_factor,
                       min_action_chunk=min_action_chunk,
                       max_action_chunk=max_action_chunk)
        elif component == "und_prefill":
            export_und_prefill(checkpoint, out_dir, dtype, max_und_len)
        elif component == "vae_encoder":
            export_vae_encoder(checkpoint, out_dir, height, width, dtype,
                               num_frames)
