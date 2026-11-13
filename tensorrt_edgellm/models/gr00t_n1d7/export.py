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
"""GR00T N1.7 component export, reached through ``tensorrt-edgellm-export``.
Writes ``visual``, ``vl_backbone`` and ``action`` plus the runtime sidecars in the bundle
layout the ONNX-less builder uses, so ``gr00t_policy_build`` yields the same bundle.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from typing import Dict

import numpy as np
import torch
from safetensors.numpy import save_file as save_numpy
from safetensors.torch import save_file as save_torch

from ..._version import __version__
from ...chat_template import write_chat_template
from ...config import ModelConfig
from ...onnx.export_encoder import _run_dynamo_export, export_visual_onnx
from . import configuration as facts
from . import contracts, identity, policy_contract, schedule
from .geometry import DEFAULT_MAX_IMAGE_LEN, DEFAULT_MAX_VL_LEN, Geometry
from .modeling_gr00t_action import build_gr00t_action
from .modeling_gr00t_vl_backbone import build_gr00t_vl_backbone
from .weights import (EMBED_TOKENS_KEY, load_checkpoint_weights,
                      split_gr00t_weights)

logger = logging.getLogger(__name__)

GR00T_COMPONENTS = ("visual", "vl_backbone", "action")
TOKENIZER_DIRNAME = facts.TOKENIZER_DIRNAME


def _write_component_config(out_dir: str, config: dict,
                            bundle_id: dict) -> None:
    """Stamp the contract/package versions and the bundle identity, and write it."""
    config = identity.stamp(config, bundle_id)
    config["onnx_filename"] = "model.onnx"
    config["contract_version"] = contracts.CONTRACT_VERSION
    config["edgellm_version"] = __version__
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as stream:
        json.dump(config, stream, indent=2)


def _batch_axis(profile: dict) -> dict:
    """Every profiled input carries the request batch on axis 0; the per-step
    constants are static and absent from the profile."""
    return {name: 0 for name in profile}


def _tower_model_config(text: dict) -> ModelConfig:
    """The FP16 ``ModelConfig`` the shared Qwen3-VL visual builder dispatches linears on."""
    return ModelConfig(model_type="qwen3_vl",
                       hidden_size=int(text["hidden_size"]),
                       num_hidden_layers=int(text["num_hidden_layers"]),
                       num_attention_heads=int(text["num_attention_heads"]),
                       num_key_value_heads=int(text["num_key_value_heads"]),
                       intermediate_size=int(text["intermediate_size"]),
                       head_dim=int(text["head_dim"]),
                       rms_norm_eps=float(text["rms_norm_eps"]),
                       vocab_size=int(text["vocab_size"]),
                       rope_theta=float(text["rope_theta"]),
                       max_position_embeddings=int(
                           text["max_position_embeddings"]),
                       default_attention_scale=float(text["head_dim"])**-0.5)


def export_visual(checkpoint: str, out_dir: str,
                  weights: Dict[str, torch.Tensor], text: dict,
                  geometry: Geometry, dtype: torch.dtype, bundle_id: dict, *,
                  min_image_tokens: int, max_image_tokens: int,
                  max_image_tokens_per_image: int) -> None:
    visual = dict(facts.VISION_CONFIG)
    visual["image_token_id"] = facts.IMAGE_TOKEN_ID
    os.makedirs(out_dir, exist_ok=True)
    export_visual_onnx(model_dir=checkpoint,
                       output_path=os.path.join(out_dir, "model.onnx"),
                       weights=weights,
                       config={"vision_config": visual},
                       model_type="qwen3_vl",
                       model_config=_tower_model_config(text),
                       dtype=dtype)
    _write_component_config(
        out_dir,
        contracts.visual_runtime_config(
            visual,
            text,
            geometry,
            min_image_tokens=min_image_tokens,
            max_image_tokens=max_image_tokens,
            max_image_tokens_per_image=max_image_tokens_per_image), bundle_id)
    # The shared Qwen ViT runner reads its normalization and patch geometry
    # from a preprocessor config beside the engine.
    for name in ("preprocessor_config.json", "processor_config.json"):
        for source in (os.path.join(checkpoint, TOKENIZER_DIRNAME,
                                    name), os.path.join(checkpoint, name)):
            if os.path.isfile(source):
                shutil.copy(source, os.path.join(out_dir, name))
                break
    logger.info("GR00T visual export complete: %s", out_dir)


def export_vl_backbone(out_dir: str, backbone: Dict[str, torch.Tensor],
                       embedding: Dict[str,
                                       torch.Tensor], head: Dict[str,
                                                                 torch.Tensor],
                       text: dict, geometry: Geometry, dtype: torch.dtype,
                       num_deepstack: int, bundle_id: dict) -> None:
    model = build_gr00t_vl_backbone(text, geometry, num_deepstack, backbone,
                                    head, dtype).to("cpu")
    args, input_names, output_names, dynamic_shapes = (
        model.get_onnx_export_args("cpu"))
    os.makedirs(out_dir, exist_ok=True)
    _run_dynamo_export(model, args, os.path.join(out_dir, "model.onnx"),
                       input_names, output_names, dynamic_shapes)
    # The backbone graph takes inputs_embeds, so the host gathers from this table.
    save_torch(
        {EMBED_TOKENS_KEY: embedding[EMBED_TOKENS_KEY].to(dtype).contiguous()},
        os.path.join(out_dir, "embed_tokens.safetensors"))
    config = contracts.vl_backbone_runtime_config(text,
                                                  geometry,
                                                  num_deepstack=num_deepstack,
                                                  max_batch=1)
    config["batch_axis"] = _batch_axis(config["optimization_profile"])
    _write_component_config(out_dir, config, bundle_id)
    logger.info("GR00T vl_backbone export complete: %s", out_dir)


def export_action(checkpoint: str, out_dir: str, head: Dict[str, torch.Tensor],
                  policy: dict, geometry: Geometry, dtype: torch.dtype,
                  bundle_id: dict) -> None:
    model = build_gr00t_action(geometry, head, dtype).to("cpu")
    args, input_names, output_names, dynamic_shapes = (
        model.get_onnx_export_args("cpu"))
    os.makedirs(out_dir, exist_ok=True)
    _run_dynamo_export(model, args, os.path.join(out_dir, "model.onnx"),
                       input_names, output_names, dynamic_shapes)

    def read(name: str) -> np.ndarray:
        return head[name].float().numpy()

    tables = schedule.action_constants(read, geometry)
    save_numpy({"adanorm_modulation": tables["adanorm_modulation"]},
               os.path.join(out_dir, schedule.MODULATION_FILENAME))
    save_numpy({"action_time_bias": tables["action_time_bias"]},
               os.path.join(out_dir, schedule.TIME_BIAS_FILENAME))
    visual = dict(facts.VISION_CONFIG)
    policy_contract.write_policy_contract(
        out_dir,
        identity.stamp_contract(
            policy_contract.build_policy_contract(checkpoint, policy, visual,
                                                  geometry), bundle_id))
    config = contracts.action_runtime_config(geometry, max_batch=1)
    config["batch_axis"] = _batch_axis(config["optimization_profile"])
    _write_component_config(out_dir, config, bundle_id)
    logger.info("GR00T action export complete: %s", out_dir)


def _stage_tokenizer(checkpoint: str, output_dir: str) -> None:
    """Copy the tokenizer whole, chat template included: the runtime renders the
    checkpoint's own Jinja rather than a derived form."""
    source = facts.require_tokenizer(checkpoint)
    staged = os.path.join(output_dir, TOKENIZER_DIRNAME)
    shutil.copytree(source, staged, dirs_exist_ok=True)
    write_chat_template(source, staged)


def export_gr00t_components(
    checkpoint: str,
    output_dir: str,
    components: "list[str] | None" = None,
    dtype: torch.dtype = torch.float16,
    embodiment_id: "int | None" = None,
    max_vl_len: int = DEFAULT_MAX_VL_LEN,
    max_image_len: int = DEFAULT_MAX_IMAGE_LEN,
    min_image_tokens: int = 4,
    max_image_tokens: int = 1024,
    max_image_tokens_per_image: int = 512,
) -> None:
    """Export the GR00T N1.7 policy components from the Hugging Face checkpoint.
    The embodiment defaults to the one the checkpoint directory name selects; the batch
    axis is widened by the C++ builder."""
    if dtype != torch.float16:
        # The component contracts are fp16-only; another dtype would export and
        # build cleanly and only then bind wrong shapes.
        raise ValueError(f"GR00T export supports float16 only, got {dtype}")
    components = list(components or GR00T_COMPONENTS)
    unknown = [c for c in components if c not in GR00T_COMPONENTS]
    if unknown:
        raise ValueError(f"Unsupported GR00T components: {unknown}; "
                         f"expected a subset of {list(GR00T_COMPONENTS)}")

    facts.require_tokenizer(checkpoint)
    policy = facts.load_policy_config(checkpoint)
    shapes = facts.checkpoint_shapes(checkpoint)
    facts.validate_backbone(shapes)
    text = facts.text_config(shapes, policy)
    if embodiment_id is None:
        embodiment_id = facts.resolve_embodiment(checkpoint)
    geometry = Geometry.from_policy(policy,
                                    dit_inner_dim=facts.dit_inner_dim(shapes),
                                    embodiment_id=embodiment_id,
                                    max_vl_len=max_vl_len,
                                    max_image_len=max_image_len)
    num_deepstack = len(facts.VISION_CONFIG["deepstack_visual_indexes"])
    bundle_id = identity.bundle_identity(checkpoint, geometry, "onnx")
    identity.validate_existing(output_dir, bundle_id, components)
    logger.info(
        "GR00T export %s: embodiment %d, %d backbone layers, max_vl_len=%d, "
        "max_image_len=%d", bundle_id["export_id"], geometry.embodiment_id,
        text["num_hidden_layers"], geometry.max_vl_len, geometry.max_image_len)

    os.makedirs(output_dir, exist_ok=True)
    _stage_tokenizer(checkpoint, output_dir)
    visual_w, backbone_w, embedding_w, head_w = split_gr00t_weights(
        load_checkpoint_weights(checkpoint))
    for component in components:
        out_dir = os.path.join(output_dir, component)
        if component == "visual":
            export_visual(
                checkpoint,
                out_dir,
                visual_w,
                text,
                geometry,
                dtype,
                bundle_id,
                min_image_tokens=min_image_tokens,
                max_image_tokens=max_image_tokens,
                max_image_tokens_per_image=max_image_tokens_per_image)
        elif component == "vl_backbone":
            export_vl_backbone(out_dir, backbone_w, embedding_w, head_w, text,
                               geometry, dtype, num_deepstack, bundle_id)
        elif component == "action":
            export_action(checkpoint, out_dir, head_w, policy, geometry, dtype,
                          bundle_id)
