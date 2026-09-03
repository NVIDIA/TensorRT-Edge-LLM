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
"""Reusable file emitters called by model-owned artifact modules."""

import json
import logging
import os
import shutil
from typing import Any, Dict

from .. import contracts
from ..config import DeviceConfig
from ..weights import Weights
from .chat_template import write_processed_chat_template
from .embeddings import (copy_vocab_artifacts, externalizes_embedding,
                         externalizes_ple, write_embedding,
                         write_ple_embedding)
from .runtime_config import (build_runtime_config,
                             default_component_runtime_config)
from .tensors import save_safetensors
from .tokenizer import (copy_tokenizer_artifacts,
                        write_tokenizer_json_if_missing)

logger = logging.getLogger(__name__)


def _require_artifact_hook(module, root_model_type: str, kind: str,
                           hook_name: str, component: str):
    if module is None or not hasattr(module, hook_name):
        raise ValueError(f"{root_model_type!r} {component!r} artifacts must "
                         f"define {kind}.{hook_name}")
    return getattr(module, hook_name)


def _runtime_artifact_model_dir(args) -> str:
    """Return the checkpoint that owns runtime tokenizer artifacts."""
    if (args.resolved_spec_role == contracts.SpecRole.DRAFT
            and args.target_model_dir):
        return args.target_model_dir
    return args.model_dir


def _load_root_config(model_dir: str) -> Dict[str, Any]:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as config_file:
        return json.load(config_file)


def _apply_generic_token_ids(config: Dict[str, Any], root: Dict[str, Any],
                             runtime_model_dir: str) -> None:
    thinker = root.get("thinker_config") or {}
    for key in ("audio_token_id", "image_token_id", "video_token_id",
                "vision_start_token_id", "vision_end_token_id",
                "audio_start_token_id", "audio_end_token_id"):
        if key in root:
            config[key] = root[key]
        elif key in thinker:
            config[key] = thinker[key]
    for runtime_key, checkpoint_key in (
        ("image_token_id", "img_context_token_id"),
        ("audio_token_id", "sound_context_token_id"),
        ("video_token_id", "video_context_token_id"),
    ):
        if runtime_key not in config:
            value = root.get(checkpoint_key, thinker.get(checkpoint_key))
            if isinstance(value, int):
                config[runtime_key] = value

    eos = root.get("eos_token_id")
    if eos is None:
        runtime_config_path = os.path.join(runtime_model_dir, "config.json")
        if os.path.isfile(runtime_config_path):
            with open(runtime_config_path) as runtime_config_file:
                eos = json.load(runtime_config_file).get("eos_token_id")
    if eos is None:
        generation_path = os.path.join(runtime_model_dir,
                                       "generation_config.json")
        if os.path.isfile(generation_path):
            with open(generation_path) as generation_file:
                eos = json.load(generation_file).get("eos_token_id")
    if eos is not None:
        config["eos_token_id"] = ([int(value) for value in eos] if isinstance(
            eos, list) else [int(eos)])


def write_runtime_artifacts(cfg: DeviceConfig,
                            args,
                            engine_dir: str,
                            *,
                            weight_conversion,
                            runtime_config_module=None,
                            tokenizer_module=None,
                            embedding_module=None) -> None:
    """Write runtime config, embeddings, tokenizer data for text-like engines."""
    output_dir = contracts.component_spec(
        args.resolved_component).output_dir(engine_dir)
    os.makedirs(output_dir, exist_ok=True)

    runtime_config = build_runtime_config(cfg, args)
    root = _load_root_config(args.model_dir)
    runtime_model_dir = _runtime_artifact_model_dir(args)
    runtime_model_artifacts = None
    if root:
        if tokenizer_module is not None and hasattr(tokenizer_module,
                                                    "prepare_runtime_model"):
            runtime_model_artifacts = tokenizer_module.prepare_runtime_model(
                root, args)
            runtime_model_dir = runtime_model_artifacts.name
        if root.get("vision_config"):
            runtime_config["vision_config"] = root["vision_config"]
        _apply_generic_token_ids(runtime_config, root, runtime_model_dir)
        if (runtime_config_module is not None
                and hasattr(runtime_config_module, "update_llm_config")):
            runtime_config_module.update_llm_config(runtime_config, root, cfg,
                                                    args)

    if cfg.component == contracts.Component.TALKER.value:
        if runtime_config_module is None or not hasattr(
                runtime_config_module, "update_talker_config"):
            raise ValueError(f"{cfg.root_model_type!r} does not define "
                             "talker runtime config artifacts")
        runtime_config_module.update_talker_config(runtime_config, root, cfg,
                                                   args)
    elif cfg.component == contracts.Component.CODE_PREDICTOR.value:
        if runtime_config_module is None or not hasattr(
                runtime_config_module, "update_code_predictor_config"):
            raise ValueError(f"{cfg.root_model_type!r} does not define "
                             "code-predictor runtime config artifacts")
        runtime_config_module.update_code_predictor_config(
            runtime_config, root, cfg, args)

    config_path = contracts.component_spec(
        args.resolved_component).config_path(engine_dir,
                                             args.resolved_spec_role,
                                             args.tp_size)
    with open(config_path, "w") as config_file:
        json.dump(runtime_config, config_file, indent=2)
    logger.info("Wrote %s", os.path.basename(config_path))

    if args.resolved_spec_role == contracts.SpecRole.DRAFT:
        from ...models import registry as model_registry
        weight_conversion = model_registry.weight_conversion_for(
            cfg.root_model_type, args.spec_type, args.resolved_spec_role)

    embedding_cfg = cfg
    weights_dir = args.model_dir
    embedding_source = getattr(weight_conversion,
                               "runtime_embedding_model_dir", None)
    if embedding_source is not None:
        weights_dir = embedding_source(args)
        embedding_cfg = DeviceConfig.from_pretrained(weights_dir,
                                                     contracts.Component.LLM,
                                                     tp_size=args.tp_size,
                                                     tp_rank=args.tp_rank)
        from ...models import registry as model_registry
        embedding_conversion = model_registry.weight_conversion_for(
            embedding_cfg.root_model_type)
    else:
        embedding_conversion = weight_conversion
    weights = Weights(weights_dir,
                      embedding_cfg.group_size,
                      embedding_cfg.quant,
                      component=cfg.component,
                      spec_type=args.spec_type,
                      spec_role=args.spec_role,
                      tie_word_embeddings=embedding_cfg.tie_word_embeddings,
                      conversion=embedding_conversion)
    try:
        if cfg.component == contracts.Component.CODE_PREDICTOR.value:
            writer = _require_artifact_hook(embedding_module,
                                            cfg.root_model_type, "embeddings",
                                            "write_code_predictor_embeddings",
                                            cfg.component)
            writer(weights, root, output_dir)
        else:
            writes_embedding = getattr(weight_conversion,
                                       "writes_runtime_embedding", None)
            if externalizes_embedding(args, weight_conversion):
                logger.info("Embedding stays in the checkpoint; the runtime "
                            "loads it through its checkpoint binding")
            elif writes_embedding is None or writes_embedding(args):
                write_embedding(weights, embedding_cfg, args, output_dir)
            if cfg.component == contracts.Component.TALKER.value:
                writer = _require_artifact_hook(embedding_module,
                                                cfg.root_model_type,
                                                "embeddings",
                                                "write_talker_embeddings",
                                                cfg.component)
                writer(weights, root, output_dir)
            if cfg.hidden_size_per_layer_input > 0:
                if externalizes_ple(args, cfg):
                    logger.info("PLE table stays in the checkpoint; the "
                                "runtime loads it through its checkpoint "
                                "binding")
                else:
                    write_ple_embedding(weights, cfg, output_dir)
            extra_artifacts = getattr(weight_conversion,
                                      "runtime_weight_artifacts", None)
            if extra_artifacts is not None:
                for filename, tensors in extra_artifacts(weights,
                                                         args).items():
                    save_safetensors(os.path.join(output_dir, filename),
                                     tensors)
    finally:
        weights.close()

    copy_vocab_artifacts(args, output_dir)

    try:
        copy_tokenizer_artifacts(runtime_model_dir, output_dir)
        write_tokenizer_json_if_missing(runtime_model_dir, output_dir)
        write_processed_chat_template(runtime_model_dir, output_dir,
                                      tokenizer_module)
        if tokenizer_module is not None and hasattr(tokenizer_module,
                                                    "patch_runtime_artifacts"):
            tokenizer_module.patch_runtime_artifacts(output_dir, args)
    finally:
        if runtime_model_artifacts is not None:
            runtime_model_artifacts.cleanup()


def write_component_artifacts(bundle,
                              args,
                              engine_dir: str,
                              *,
                              runtime_config_module=None,
                              embedding_module=None,
                              preprocessing_module=None) -> None:
    """Write config and preprocessing data for non-text components."""
    component = args.resolved_component
    output_dir = contracts.component_spec(component).output_dir(engine_dir)
    os.makedirs(output_dir, exist_ok=True)
    component_config = None
    if runtime_config_module is not None and hasattr(
            runtime_config_module, "component_runtime_config"):
        component_config = runtime_config_module.component_runtime_config(
            bundle, component, args)
    if component_config is None:
        component_config = default_component_runtime_config(
            bundle, component, args)
    if component_config is None:
        raise ValueError(f"no component artifact schema for {component.value}")

    with open(os.path.join(output_dir, "config.json"), "w") as config_file:
        json.dump(component_config, config_file, indent=2)

    for filename in ("preprocessor_config.json", "processor_config.json",
                     "feature_extractor_config.json", "audio_config.json"):
        source = os.path.join(args.model_dir, filename)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(output_dir, filename))

    if component == contracts.Component.VISUAL:
        preprocessor = os.path.join(output_dir, "preprocessor_config.json")
        processor_path = os.path.join(args.model_dir, "processor_config.json")
        if not os.path.isfile(preprocessor) and os.path.isfile(processor_path):
            with open(processor_path) as processor_file:
                image_processor = json.load(processor_file).get(
                    "image_processor")
            if isinstance(image_processor, dict):
                with open(preprocessor, "w") as preprocessor_file:
                    json.dump(image_processor, preprocessor_file, indent=2)

    if embedding_module is not None and hasattr(embedding_module,
                                                "write_component_embeddings"):
        embedding_module.write_component_embeddings(bundle, component, args,
                                                    output_dir)
    if preprocessing_module is not None and hasattr(preprocessing_module,
                                                    "write_component_assets"):
        preprocessing_module.write_component_assets(bundle, component, args,
                                                    output_dir)
