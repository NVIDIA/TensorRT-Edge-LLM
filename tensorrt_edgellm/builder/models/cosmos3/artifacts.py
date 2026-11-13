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
"""Cosmos3 policy runtime artifact writing."""

from __future__ import annotations

import json
import os
import shutil

from ...core import contracts
from ...core.artifacts.runtime_artifacts import (write_component_artifacts,
                                                 write_runtime_artifacts)
from ...core.artifacts.tensors import save_safetensors
from ...core.weights import Weights
from . import runtime_config, weights

_OUTPUT_DIRS = {
    "gen": "gen",
    "und_prefill": "und_prefill",
    "vae_encoder": "vae_encoder",
}


def _component_name(component) -> str:
    return str(getattr(component, "value", component)).replace("-", "_")


def _copy_tokenizer(bundle, engine_dir: str) -> None:
    source = os.path.join(bundle.model_dir, "text_tokenizer")
    if not os.path.isdir(source):
        raise FileNotFoundError(
            "Cosmos3 policy checkpoint is missing text_tokenizer/")
    shutil.copytree(source,
                    os.path.join(engine_dir, "text_tokenizer"),
                    dirs_exist_ok=True,
                    copy_function=shutil.copyfile)


def _write_und_embeddings(bundle, engine_dir: str) -> None:
    checkpoint = Weights(bundle.model_dir,
                         component="und_prefill",
                         conversion=weights)
    try:
        tensors = {
            "embed_tokens.weight": checkpoint.f16("embed_tokens.weight"),
        }
        save_safetensors(os.path.join(engine_dir, "embed_tokens.safetensors"),
                         tensors)
    finally:
        checkpoint.close()


def write_artifacts(bundle, config, args, engine_dir: str) -> None:
    """Write reasoning or policy artifacts from one Cosmos3 checkpoint."""
    if config is not None:
        write_runtime_artifacts(config,
                                args,
                                engine_dir,
                                weight_conversion=weights,
                                runtime_config_module=runtime_config)
        return
    if args.resolved_component == contracts.Component.VISUAL:
        write_component_artifacts(bundle,
                                  args,
                                  engine_dir,
                                  runtime_config_module=runtime_config)
        return

    name = _component_name(args.resolved_component)
    try:
        output_name = _OUTPUT_DIRS[name]
    except KeyError as error:
        raise ValueError(
            f"unsupported Cosmos3 policy component {name!r}") from error
    output_dir = os.path.join(engine_dir, output_name)
    os.makedirs(output_dir, exist_ok=True)
    payload = runtime_config.component_runtime_config(bundle,
                                                      args.resolved_component,
                                                      args)
    with open(os.path.join(output_dir, "config.json"), "w") as config_file:
        json.dump(payload, config_file, indent=2)
    _copy_tokenizer(bundle, engine_dir)
    if name == "und_prefill":
        _write_und_embeddings(bundle, engine_dir)
