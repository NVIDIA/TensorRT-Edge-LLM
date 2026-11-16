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
"""EAGLE-4 target/draft pairing configuration.

The draft config is parsed and validated with the ONNX exporter's helpers.
The builder additionally rejects drafts that name no aux layers and drafts
with an FP8 KV cache.
"""

from typing import List

from tensorrt_edgellm.config import (_eagle_draft_feature_keys,
                                     _is_gemma4_model_type,
                                     _parse_eagle_draft_features,
                                     _parse_eagle_target_layer_ids)

from ...core import contracts, quantization
from ...core.bundle import BundleConfig
from ..eagle3.configuration import _clamp_unique, _validate_target_layers

# These base graphs feed back the final hidden state (allow_eagle3=False),
# which cannot bind the draft's aux-state input.
_BASES_WITHOUT_EAGLE_AUX_STATES = frozenset(("qwen3_5_moe", "muse_glimmer"))
# hidden_state_feedback emits aux states only from this many decoder layers on.
_MIN_FEEDBACK_DECODER_LAYERS = 5


def is_eagle4_draft(draft: dict) -> bool:
    """Whether an EAGLE draft config sets fc_norm, norm_output or a candidate head."""
    return bool(_eagle_draft_feature_keys(draft))


def _target_layers(draft: dict, num_target_layers: int,
                   build_args) -> List[int]:
    num_decoder_layers = getattr(build_args, "num_decoder_layers", 0) or 0
    # --num-decoder-layer truncates the base, so the draft's layers are
    # clamped into it instead of being range-checked against it.
    layers = _parse_eagle_target_layer_ids(
        draft, None if num_decoder_layers else num_target_layers)
    if not layers:
        raise ValueError(
            "EAGLE-4 drafts must name the base layers they read: set "
            "eagle_config.eagle_aux_hidden_state_layer_ids, "
            "eagle_aux_hidden_state_layer_ids or target_layer_ids in the "
            "draft config.json.")
    if num_decoder_layers:
        if num_decoder_layers < _MIN_FEEDBACK_DECODER_LAYERS:
            raise ValueError(
                "EAGLE-4 needs --num-decoder-layer of at least "
                f"{_MIN_FEEDBACK_DECODER_LAYERS}: fewer base layers feed back "
                "the final hidden state instead of the draft's aux states; "
                f"got {num_decoder_layers}")
        layers = _clamp_unique(layers, num_decoder_layers)
    _validate_target_layers(layers, num_target_layers)
    return layers


def _require_candidate_head_for_dense_head(features, build_args) -> None:
    if (getattr(build_args, "draft_dense_head", False)
            and features.candidate_count <= 0):
        raise ValueError("--draft-dense-head requires an EAGLE draft "
                         "checkpoint with a low-rank candidate head")


def configure_base(config,
                   *,
                   paired_draft_dir: str = "",
                   build_args=None,
                   **kwargs) -> None:
    """Enable EAGLE feedback of the HF hidden-state layers the draft names."""
    if not paired_draft_dir:
        raise ValueError("EAGLE-4 base requires a paired draft checkpoint")
    if _is_gemma4_model_type(config.model_type):
        raise ValueError(
            "A Gemma4 EAGLE base reads draft aux layers as decoder-layer "
            "outputs; EAGLE-4 drafts name HF hidden-state indices.")
    from .. import registry
    family = registry.family_for(config.root_model_type).name
    if family in _BASES_WITHOUT_EAGLE_AUX_STATES:
        raise ValueError(
            f"The checkpoint-direct {family} base does not output the EAGLE "
            "aux hidden states an EAGLE-4 draft reads; export the base with "
            "tensorrt-edgellm-export --eagle-base --eagle-draft-dir.")
    draft = BundleConfig.from_pretrained(paired_draft_dir).component_dict(
        contracts.Component.LLM)
    layers = _target_layers(draft, config.num_hidden_layers, build_args)
    # Validated with the base so an unsupported draft fails before the base
    # engine is built.
    _require_candidate_head_for_dense_head(
        _parse_eagle_draft_features(draft, layers), build_args)
    config.eagle_base = True
    config.eagle3_target_layer_ids = layers


def _require_supported_precision(config) -> None:
    if config.kv_cache_quant == "fp8":
        raise ValueError(
            "EAGLE-4 drafts with an FP8 KV cache (kv_cache_quant_algo FP8) "
            "are not supported. Quantize the draft again without "
            "--kv-cache-quantization.")
    lm_head_quant = config.module_quant_type("lm_head")
    if (config.draft_candidate_count > 0
            and lm_head_quant != quantization.QUANT_FP16):
        raise ValueError(
            "The low-rank candidate head gathers fp16 lm_head rows, but this "
            f"draft checkpoint stores lm_head as {lm_head_quant}. Quantize "
            "the draft with lm_head unquantized, or build it with "
            "--draft-dense-head.")


def configure_draft(config,
                    *,
                    paired_target=None,
                    build_args=None,
                    **kwargs) -> None:
    """Bind the draft features and fusion dimensions to the paired target."""
    if paired_target is None:
        raise ValueError("EAGLE-4 draft requires a target config")
    layers = _target_layers(config.raw_component,
                            paired_target.num_hidden_layers, build_args)
    features = _parse_eagle_draft_features(config.raw_component, layers)
    _require_candidate_head_for_dense_head(features, build_args)
    config.eagle3_target_layer_ids = layers
    config.target_hidden_size = paired_target.hidden_size
    config.eagle_fc_norm = features.fc_norm
    config.eagle_norm_output = features.norm_output
    if not getattr(build_args, "draft_dense_head", False):
        config.draft_candidate_rank = features.candidate_rank
        config.draft_candidate_count = features.candidate_count
    _require_supported_precision(config)
