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
"""GR00T N1.7 runtime artifact writing for the direct builder; the constant tables and the
policy contract come from :mod:`tensorrt_edgellm.models.gr00t_n1d7`, shared with the export.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import TYPE_CHECKING

from tensorrt_edgellm.chat_template import write_chat_template
from tensorrt_edgellm.models.gr00t_n1d7 import configuration as facts
from tensorrt_edgellm.models.gr00t_n1d7 import (identity, policy_contract,
                                                schedule)

from ...core import contracts
from ...core.artifacts.runtime_artifacts import write_component_artifacts
from ...core.artifacts.tensors import save_safetensors
from ...core.weights import Weights
from . import runtime_config
from . import weights as weight_conversion

if TYPE_CHECKING:
    from tensorrt_edgellm.models.gr00t_n1d7.geometry import Geometry

    from ...core.builder import BuildArgs
    from ...core.bundle import BundleConfig


def _component_name(component) -> str:
    return str(getattr(component, "value", component)).replace("-", "_")


def _write_policy_contract(bundle: BundleConfig, geometry: Geometry,
                           bundle_id: dict, output_dir: str) -> None:
    policy = bundle.root.get("_policy") or bundle.root
    visual = bundle.component_dict(contracts.Component.VISUAL)
    contract = policy_contract.build_policy_contract(bundle.model_dir, policy,
                                                     visual, geometry)
    policy_contract.write_policy_contract(
        output_dir, identity.stamp_contract(contract, bundle_id))


def _write_backbone_sidecars(bundle: BundleConfig, engine_dir: str) -> None:
    """Stage the token embedding table and the tokenizer beside the engines. The backbone
    takes ``inputs_embeds``, so the caller does the lookup; the tokenizer directory is copied
    whole because the runtime renders the checkpoint's own Jinja template."""
    checkpoint = Weights(bundle.model_dir,
                         component="vl-backbone",
                         conversion=weight_conversion)
    try:
        table = checkpoint.f16("model.embed_tokens.weight")
    finally:
        checkpoint.close()
    save_safetensors(os.path.join(engine_dir, "embed_tokens.safetensors"),
                     {"embed_tokens.weight": table})

    # validate_build already refused a checkpoint without a loadable tokenizer.
    source = facts.require_tokenizer(bundle.model_dir)
    staged = os.path.join(engine_dir, facts.TOKENIZER_DIRNAME)
    shutil.copytree(source, staged, dirs_exist_ok=True)
    # The runtime's Inja renderer loads chat_template.jinja; the checkpoint
    # carries the same template inside chat_template.json.
    write_chat_template(source, staged)


def _write_action_constants(bundle: BundleConfig, geometry: Geometry,
                            output_dir: str) -> None:
    checkpoint = Weights(bundle.model_dir,
                         component="action",
                         conversion=weight_conversion)
    try:
        tables = schedule.action_constants(checkpoint.f32, geometry)
    finally:
        checkpoint.close()
    save_safetensors(os.path.join(output_dir, schedule.MODULATION_FILENAME),
                     {"adanorm_modulation": tables["adanorm_modulation"]})
    save_safetensors(os.path.join(output_dir, schedule.TIME_BIAS_FILENAME),
                     {"action_time_bias": tables["action_time_bias"]})


def write_artifacts(bundle: BundleConfig, config: dict, args: BuildArgs,
                    engine_dir: str) -> None:
    """Write one GR00T component's runtime config and constant tables."""
    if args.resolved_component == contracts.Component.VISUAL:
        write_component_artifacts(bundle,
                                  args,
                                  engine_dir,
                                  runtime_config_module=runtime_config)
        # The shared Qwen ViT runner reads its normalization and patch geometry
        # from a preprocessor config beside the engine.
        source = os.path.join(bundle.model_dir, facts.TOKENIZER_DIRNAME,
                              "preprocessor_config.json")
        if os.path.isfile(source):
            shutil.copy(
                source,
                os.path.join(engine_dir, "visual", "preprocessor_config.json"))
        return

    name = _component_name(args.resolved_component)
    if name not in ("vl_backbone", "action"):
        raise ValueError(f"unsupported GR00T component {name!r}")
    output_dir = os.path.join(engine_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    payload = runtime_config.component_runtime_config(bundle,
                                                      args.resolved_component,
                                                      args)
    with open(os.path.join(output_dir, "config.json"), "w") as config_file:
        json.dump(payload, config_file, indent=2)
    if name == "action":
        geometry = runtime_config.geometry_from_bundle(bundle, args)
        _write_action_constants(bundle, geometry, output_dir)
        _write_policy_contract(bundle, geometry,
                               runtime_config.bundle_identity(bundle, args),
                               output_dir)
    if name == "vl_backbone":
        _write_backbone_sidecars(bundle, engine_dir)
