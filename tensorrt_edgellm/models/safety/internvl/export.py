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
"""Export the InternVL Safety decoder and visual encoder."""

import dataclasses
import json
import logging
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _image_token_id(model_dir: Path, config: dict):
    if config.get("image_token_id") is not None:
        return config["image_token_id"]
    tokenizer_path = model_dir / "tokenizer.json"
    if tokenizer_path.is_file():
        with tokenizer_path.open() as stream:
            tokenizer = json.load(stream)
        for entry in tokenizer.get("added_tokens", []):
            if entry.get("content") == "<IMG_CONTEXT>":
                return int(entry["id"])
        for token_id, entry in tokenizer.get("added_tokens_decoder",
                                             {}).items():
            if entry.get("content") == "<IMG_CONTEXT>":
                return int(token_id)
    for name in ("added_tokens.json", "vocab.json"):
        path = model_dir / name
        if path.is_file():
            with path.open() as stream:
                tokens = json.load(stream)
            if "<IMG_CONTEXT>" in tokens:
                return int(tokens["<IMG_CONTEXT>"])
    logger.warning("Could not resolve <IMG_CONTEXT> in %s", model_dir)
    return None


def _load_visual_weights(model_dir: Path) -> dict:
    import torch
    from safetensors import torch as safetensors_torch

    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        shards = sorted(model_dir.glob("pytorch_model*.bin")) or sorted(
            model_dir.glob("*.bin"))
    if not shards:
        raise ValueError(f"No checkpoint weights found in {model_dir}")
    weights = {}
    for shard in shards:
        tensors = (safetensors_torch.load_file(str(shard), device="cpu")
                   if shard.suffix == ".safetensors" else torch.load(
                       shard, map_location="cpu", weights_only=True))
        weights.update({
            key: value
            for key, value in tensors.items()
            if key.startswith(("vision_model.", "mlp1.", "vision_tower.",
                               "multi_modal_projector."))
        })
    if any(key.startswith("vision_tower.") for key in weights):
        weights = _map_hf_visual_weights(weights)
    return weights


def _map_hf_visual_weights(weights: dict) -> dict:
    import torch

    mapped = {}
    qkv_groups = {}
    projector = {"layer_norm": "0", "linear_1": "1", "linear_2": "3"}
    for key, tensor in weights.items():
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "Native HF Safety visual loading requires unquantized weights")
        if key.startswith("multi_modal_projector."):
            name, suffix = key[len("multi_modal_projector."):].split(".", 1)
            mapped[f"mlp1.{projector[name]}.{suffix}"] = tensor
            continue
        name = key[len("vision_tower."):]
        match = re.fullmatch(
            r"encoder\.layer\.(\d+)\.attention\.([qkv])_proj\.(weight|bias)",
            name)
        if match:
            layer, projection, suffix = match.groups()
            qkv_groups.setdefault((layer, suffix), set()).add(projection)
            mapped[
                f"vision_model.encoder.layers.{layer}.attn.{projection}_proj.{suffix}"] = tensor
            continue
        name = name.replace("embeddings.cls_token",
                            "embeddings.class_embedding")
        name = name.replace("embeddings.patch_embeddings.projection.",
                            "embeddings.patch_embedding.")
        name = name.replace("embeddings.position_embeddings",
                            "embeddings.position_embedding")
        name = name.replace("encoder.layer.", "encoder.layers.")
        name = name.replace(".attention.projection_layer.", ".attn.proj.")
        name = name.replace(".layernorm_before.", ".norm1.")
        name = name.replace(".layernorm_after.", ".norm2.")
        name = name.replace(".lambda_1", ".ls1").replace(".lambda_2", ".ls2")
        mapped[f"vision_model.{name}"] = tensor
    for (layer, suffix), projections in qkv_groups.items():
        if projections != {"q", "k", "v"}:
            raise ValueError(f"Incomplete HF visual QKV at layer {layer}")
    return mapped


def _normalize_hf_visual_config(config: dict) -> dict:
    config = dict(config, vision_config=dict(config["vision_config"]))
    vision = config["vision_config"]
    if (config.get("quantization_config") or vision.get("use_qk_norm", False)
            or config.get("projector_hidden_act", "gelu") != "gelu"
            or not vision.get("attention_bias", True)
            or not vision.get("use_mean_pooling", True)
            or config.get("vision_feature_layer", -1) != -1 or config.get(
                "vision_feature_select_strategy", "default") != "default"):
        raise ValueError(
            "Unsupported native HF InternVL Safety visual configuration")
    for name in ("image_size", "patch_size"):
        value = vision[name]
        if isinstance(value, list):
            if len(value) != 2 or value[0] != value[1]:
                raise ValueError(
                    f"Safety visual export requires square {name}")
            vision[name] = value[0]
    return config


def _export_visual(model_dir: Path, output_dir: Path, config: dict,
                   model_config, image_token_id) -> None:
    import torch

    from .... import config as config_module
    from .. import onnx_export
    from . import modeling_visual, onnx_translation

    weights = _load_visual_weights(model_dir)
    if not any(key.endswith(".weight_scale") for key in weights):
        model_config = dataclasses.replace(model_config,
                                           quant=config_module.QuantConfig())
    model = modeling_visual.build_internvl_visual(config,
                                                  weights,
                                                  model_config,
                                                  dtype=torch.float16)
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs, input_names, output_names, shapes = model.get_onnx_export_args(
        config, "cpu")
    onnx_export.export_visual(
        model,
        inputs,
        output_dir / "model.onnx",
        input_names,
        output_names,
        shapes,
        translation_table=onnx_translation.get_custom_translation_table())

    vision_config = dict(config["vision_config"], model_type="internvl")
    for name in ("image_size", "patch_size"):
        if isinstance(vision_config.get(name), int):
            vision_config[name] = [vision_config[name], vision_config[name]]
    runtime_config = {
        "model_type": "internvl",
        "vision_config": vision_config,
        "text_config": config.get("text_config") or config["llm_config"],
        "downsample_ratio": config["downsample_ratio"],
    }
    if image_token_id is not None:
        runtime_config["image_token_id"] = image_token_id
    with (output_dir / "config.json").open("w") as stream:
        json.dump(runtime_config, stream, indent=2)

    preprocessor = model_dir / "preprocessor_config.json"
    processor = model_dir / "processor_config.json"
    if preprocessor.is_file():
        shutil.copy2(preprocessor, output_dir)
    elif processor.is_file():
        with processor.open() as stream:
            image_processor = json.load(stream).get("image_processor")
        if image_processor:
            with (output_dir / "preprocessor_config.json").open("w") as stream:
                json.dump(image_processor, stream, indent=2)


def export_model(model_dir: Path,
                 output_dir: Path,
                 config: dict,
                 *,
                 skip_visual=False,
                 num_decoder_layers=None,
                 reduced_vocab_dir="") -> None:
    """Export InternVL components and their runtime configuration."""
    if config.get("model_type") not in ("internvl_chat", "internvl"):
        raise ValueError("Expected an InternVL checkpoint")
    if config.get("model_type") == "internvl":
        config = _normalize_hf_visual_config(config)
    config.setdefault("downsample_ratio", 0.5)

    from .. import onnx_export
    from . import modeling_internvl, onnx_translation

    model = modeling_internvl.load_safety_llm(
        str(model_dir),
        num_decoder_layers=num_decoder_layers,
        reduced_vocab_dir=reduced_vocab_dir or None)
    model_config = model.config
    llm_dir = output_dir / "llm"
    onnx_export.export_decoder(
        model,
        llm_dir / "model.onnx",
        model_dir=str(model_dir),
        reduced_vocab_dir=reduced_vocab_dir,
        translation_table=onnx_translation.get_custom_translation_table())
    del model
    image_token_id = _image_token_id(model_dir, config)
    if image_token_id is not None:
        config_path = llm_dir / "config.json"
        with config_path.open() as stream:
            llm_config = json.load(stream)
        llm_config["image_token_id"] = image_token_id
        with config_path.open("w") as stream:
            json.dump(llm_config, stream, indent=2)
    if not skip_visual:
        _export_visual(model_dir, output_dir / "visual", config, model_config,
                       image_token_id)
