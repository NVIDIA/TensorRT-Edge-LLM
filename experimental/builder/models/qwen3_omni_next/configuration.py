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
"""Qwen3-Omni-Next checkpoint and component configuration."""

import os
from dataclasses import replace
from typing import Any, Dict

from ...core import contracts, quantization, weight_policy

_TALKER_TYPES = frozenset((
    "qwen3_omni_next_talker",
    "qwen3_omni_next_talker_text",
))
_CODE_PREDICTOR_TYPES = frozenset((
    "qwen3_omni_next_code_predictor",
    "qwen3_omni_next_talker_code_predictor",
))


def _codec_loader():
    import yaml

    class CodecLoader(yaml.SafeLoader):
        pass

    def construct_unknown(loader, node):
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node, deep=True)
        return loader.construct_scalar(node)

    CodecLoader.add_constructor(None, construct_unknown)
    return yaml, CodecLoader


def _load_code2wav_config(codec_dir: str) -> Dict[str, Any]:
    config_path = os.path.join(codec_dir, "config.yaml")
    weights_path = os.path.join(codec_dir, "model_weights.pt")
    if not os.path.isfile(config_path) or not os.path.isfile(weights_path):
        raise FileNotFoundError("Qwen3-Omni-Next Code2Wav requires "
                                f"{config_path!r} and {weights_path!r}")

    yaml, loader = _codec_loader()
    with open(config_path) as config_file:
        raw = yaml.load(config_file, Loader=loader)
    model = raw["model"]
    dac = model["dac"]
    quantizer = model["quantizer"]
    post = model["post_module"]
    transformer = post["args"]
    rope = transformer.get("rope") or {}
    decoder_rates = list(dac["decoder_rates"])
    pre_upsample_rates = list(
        dac.get("upsample_rates", model.get("upsample_rates", (2, 2))))
    latent_dimension = dac.get("latent_dim", dac.get("latet_dim"))
    if latent_dimension is None:
        raise KeyError("Qwen3-Omni-Next codec config has no latent dimension")
    local_heads = transformer.get("n_local_heads")
    if local_heads in (None, -1):
        local_heads = transformer["n_head"]
    return {
        "model_type": "qwen3_omni_next_code2wav",
        "num_quantizers": int(quantizer["n_q"]),
        "num_semantic_quantizers": int(quantizer["n_q_semantic"]),
        "codebook_size": int(quantizer["bins"]),
        "quantizer_dimension": int(quantizer["dimension"]),
        "quantizer_input_dimension": int(quantizer["input_dimension"]),
        "quantizer_output_dimension": int(quantizer["output_dimension"]),
        "latent_dimension": int(latent_dimension),
        "decoder_dimension": int(dac["decoder_dim"]),
        "decoder_rates": decoder_rates,
        "pre_upsample_rates": pre_upsample_rates,
        "pre_conv_kernel": int(model.get("pre_conv_kernel", 3)),
        "pre_transformer_input_dimension": int(post["input_dim"]),
        "pre_transformer_window_size": int(post["window_size"]),
        "transformer": {
            "num_hidden_layers":
            int(transformer["n_layer"]),
            "num_attention_heads":
            int(transformer["n_head"]),
            "num_key_value_heads":
            int(local_heads),
            "hidden_size":
            int(transformer["dim"]),
            "head_dim":
            int(transformer["head_dim"]),
            "intermediate_size":
            int(transformer["intermediate_size"]),
            "rms_norm_eps":
            float(transformer.get("norm_eps", 1e-5)),
            "rope_theta":
            float(rope.get("rope_theta", 10000.0)),
            "max_position_embeddings":
            int(rope.get("max_position_embeddings", 16384)),
        },
        "sample_rate":
        int(model.get("sample_rate", raw.get("sample_rate", 24000))),
        "_checkpoint_dir": codec_dir,
    }


def prepare_root(model_dir: str, root: dict) -> dict:
    """Attach the separately shipped online-codec configuration."""
    root = dict(root)
    codec_dir = os.path.join(model_dir, "codec_decode_online")
    if os.path.isdir(codec_dir):
        root["_code2wav_config"] = _load_code2wav_config(codec_dir)
    return root


def available_components(root: dict, registered):
    """Narrow full-root registration for standalone component checkpoints."""
    model_type = str(root.get("model_type", ""))
    standalone = {
        "qwen3_omni_next_thinker": contracts.Component.LLM,
        "qwen3_omni_next_text": contracts.Component.LLM,
        "qwen3_omni_next_text_moe": contracts.Component.LLM,
        # ModelOpt uses this type for the CP-only replacement checkpoint.
        "qwen3_omni_next_talker": contracts.Component.CODE_PREDICTOR,
        "qwen3_omni_next_talker_text": contracts.Component.TALKER,
        "qwen3_omni_next_code_predictor": contracts.Component.CODE_PREDICTOR,
        "qwen3_omni_next_talker_code_predictor":
        contracts.Component.CODE_PREDICTOR,
        "qwen3_omni_next_vision_encoder": contracts.Component.VISUAL,
        "qwen3_omni_next_audio_encoder": contracts.Component.AUDIO,
        "qwen3_omni_next_code2wav": contracts.Component.CODE2WAV,
    }
    if model_type in standalone:
        return frozenset((standalone[model_type], ))
    return registered


def component_config(root: dict, component: contracts.Component) -> dict:
    """Select one model-owned component from a full or standalone root."""
    model_type = str(root.get("model_type", ""))
    thinker = root.get("thinker_config") or {}
    if component == contracts.Component.LLM:
        return thinker or root
    if component == contracts.Component.VISUAL:
        return (thinker.get("vision_config") or root.get("vision_config")
                or root)
    if component == contracts.Component.AUDIO:
        return (thinker.get("audio_config") or root.get("audio_config")
                or root)
    if component == contracts.Component.TALKER:
        talker = root.get("talker_config")
        if isinstance(talker, dict):
            return talker
        if model_type in _TALKER_TYPES:
            return root
        raise ValueError("Qwen3-Omni-Next checkpoint has no talker_config")
    if component == contracts.Component.CODE_PREDICTOR:
        talker = root.get("talker_config") or root
        predictor = talker.get("code_predictor_config")
        if isinstance(predictor, dict):
            return predictor
        if model_type in _CODE_PREDICTOR_TYPES:
            return root
        raise ValueError(
            "Qwen3-Omni-Next checkpoint has no code_predictor_config")
    if component == contracts.Component.CODE2WAV:
        codec = root.get("_code2wav_config")
        if isinstance(codec, dict):
            return codec
        if model_type == "qwen3_omni_next_code2wav":
            return root.get("code2wav_config") or root
        raise ValueError(
            "Qwen3-Omni-Next checkpoint has no codec_decode_online directory")
    raise ValueError(f"Qwen3-Omni-Next has no {component.value} configuration")


def component_weight_policy(args, policy):
    """Code2Wav weights are baked after loading the provider `.pt` archive."""
    if args.resolved_component != contracts.Component.CODE2WAV:
        return policy
    return policy.without((weight_policy.EXTERNAL_WEIGHT_FP16, ),
                          strict=bool(args.externalize_weights))


def _hybrid_layer_types(config: Dict[str, Any]) -> None:
    if config.get("layers_block_type") or config.get("layer_types"):
        return
    count = int(config["num_hidden_layers"])
    interval = int(config.get("full_attention_interval", 4) or 4)
    config["layer_types"] = [
        "full_attention" if (index + 1) % interval == 0 else "linear_attention"
        for index in range(count)
    ]


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    """Normalize the nested provider config without staging another model."""
    del model_dir
    config = dict(config)
    config.setdefault("partial_rotary_factor", 0.25)
    _hybrid_layer_types(config)

    thinker = root.get("thinker_config") or {}
    talker = root.get("talker_config") or {}
    if component == contracts.Component.LLM:
        vision = thinker.get("vision_config") or root.get(
            "vision_config") or {}
        indexes = vision.get("deepstack_visual_indexes")
        config.setdefault(
            "num_deepstack_features",
            len(indexes) if isinstance(indexes, (list, tuple)) else 0)
        if talker.get("accept_hidden_layer") is not None:
            config.setdefault("accept_hidden_layer",
                              talker["accept_hidden_layer"])
        for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers",
                    "mtp_use_dedicated_embeddings"):
            if key in thinker:
                config.setdefault(key, thinker[key])
            elif key in root:
                config.setdefault(key, root[key])
        config["model_type"] = ("qwen3_omni_next_text_moe" if int(
            config.get("num_experts", 0) or 0) > 0 else "qwen3_omni_next_text")
    elif component == contracts.Component.TALKER:
        for key in ("num_code_groups", "accept_hidden_layer"):
            if key in talker:
                config.setdefault(key, talker[key])
            elif key in root:
                config.setdefault(key, root[key])
        config["model_type"] = ("qwen3_omni_next_talker_text" if int(
            config.get("num_experts", 0) or 0) > 0 else
                                "qwen3_omni_next_talker")
        config["num_deepstack_features"] = 0
    elif component == contracts.Component.CODE_PREDICTOR:
        config.setdefault("num_code_groups",
                          int(talker.get("num_code_groups", 16)))
        config["model_type"] = "qwen3_omni_next_code_predictor"
        config["num_deepstack_features"] = 0
        config["layer_types"] = ["full_attention"] * int(
            config["num_hidden_layers"])
    return config


def configure_base(config, *, build_args=None, **kwargs) -> None:
    """Enable the native hybrid-state checkpointing contract."""
    del kwargs
    config.mtp_base = True
    config.mtp_tree_base = bool(build_args and build_args.tree_base)


def configure_draft(config, **kwargs) -> None:
    """Select the checkpoint-owned full-attention MoE MTP layers."""
    del kwargs
    if config.mtp_num_hidden_layers is None:
        raise ValueError("Qwen3-Omni-Next MTP requires mtp_num_hidden_layers")
    config.num_hidden_layers = config.mtp_num_hidden_layers
    config.layer_types = ["attention"] * config.num_hidden_layers
    config.attention_layer_types = ["full_attention"
                                    ] * config.num_hidden_layers
    config.gdn_cfg = None
    config.mtp_base = False
    config.tie_word_embeddings = False
    config.quant = replace(config.quant,
                           quant_type=quantization.QUANT_FP16,
                           excluded=(),
                           layer_overrides={},
                           is_mixed_precision=False)
