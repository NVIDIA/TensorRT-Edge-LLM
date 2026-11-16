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
"""Qwen4-Exp (Qwen3.8-Flash-Next) checkpoint configuration."""

from typing import Any, Dict, List

from ...core import config as core_config
from ...core import contracts


def component_config(root: dict, component: contracts.Component) -> dict:
    if component == contracts.Component.LLM:
        return root
    if component == contracts.Component.VISUAL:
        vision = root.get("vision_config")
        if not vision:
            raise ValueError("Qwen4-Exp checkpoint has no vision_config")
        return vision
    raise ValueError(f"Qwen4-Exp has no {component.value} configuration")


def prepare_text_config(config: dict, root: dict,
                        component: contracts.Component,
                        model_dir: str) -> dict:
    config = dict(config)
    config.setdefault("partial_rotary_factor", 0.25)
    if not config.get("layer_types"):
        interval = int(config.get("full_attention_interval", 4) or 4)
        config["layer_types"] = [
            "full_attention" if
            (index + 1) % interval == 0 else "linear_attention"
            for index in range(int(config["num_hidden_layers"]))
        ]
    return config


def _ngram_layers(config: dict, num_hidden_layers: int) -> List[int]:
    """Decoder layers carrying an n-gram injection, as zero-based indices.

    ``ple_layer_ids`` counts layers from one. A truncated checkpoint keeps the
    original ids, so an id past the end means the injected layer was cut away
    and the remaining stack has none.
    """
    layers = sorted(
        int(layer_id) - 1 for layer_id in (config.get("ple_layer_ids") or []))
    for layer in layers:
        if layer < 0:
            raise ValueError(
                f"ple_layer_ids={config.get('ple_layer_ids')} is one-based and "
                f"names decoder layer {layer}")
    return [layer for layer in layers if layer < num_hidden_layers]


def update_device_config(config, root: dict,
                         component: contracts.Component) -> None:
    """Read the fields no other family has off the checkpoint config."""
    if component != contracts.Component.LLM:
        return
    text = config.raw_component
    if isinstance(text.get("text_config"), dict):
        text = text["text_config"]
    config.hc_count = int(text.get("hc_count", 0))
    config.hc_lowrank = int(text.get("hc_lowrank", 0))
    if config.hc_count <= 0 or config.hc_lowrank <= 0:
        raise ValueError("Qwen4-Exp requires hc_count and hc_lowrank; the "
                         "residual bus has no width without them")
    config.ngram_size = int(text.get("ngram_size", 0))
    config.ngram_heads_per_order = int(text.get("heads_per_ngram", 0))
    config.ple_embed_dim = int(text.get("ple_embed_dim", 0))
    config.ple_conv_kernel_size = int(text.get("ple_conv_kernel_size", 0))
    config.ngram_layers = _ngram_layers(text, config.num_hidden_layers)
    if config.ngram_layers:
        # A bigram is the shortest n-gram with a hash head, and a width-1
        # kernel would leave the dilated convolution with no state.
        geometry = {
            "ngram_size >= 2": config.ngram_size >= 2,
            "heads_per_ngram > 0": config.ngram_heads_per_order > 0,
            "ple_embed_dim > 0": config.ple_embed_dim > 0,
            "ple_conv_kernel_size > 1": config.ple_conv_kernel_size > 1,
        }
        unmet = [name for name, ok in geometry.items() if not ok]
        if unmet:
            raise ValueError("Qwen4-Exp n-gram injection needs " +
                             ", ".join(unmet))
    config.indexer_n_heads = int(text.get("indexer_n_heads") or 0)
    config.indexer_kv_heads = int(text.get("indexer_kv_heads") or 0)
    config.indexer_head_dim = int(text.get("indexer_head_dim") or 0)
    config.indexer_budget = int(text.get("indexer_budget") or 0)
    config.indexer_compress_ratio = int(
        text.get("indexer_compress_ratio") or 0)


def _mtp_section(config) -> Dict[str, Any]:
    text = config.raw_component
    if isinstance(text.get("text_config"), dict):
        text = text["text_config"]
    section = text.get("mtp")
    return section if isinstance(section, dict) else {}


def configure_base(config, *, build_args=None, **kwargs) -> None:
    """Enable the checkpoint's model-owned MTP feedback contract."""
    del kwargs, build_args
    config.mtp_base = True


def _reject_unconsumed_mtp_fields(config, section: Dict[str, Any]) -> None:
    """Fail on MTP config fields nothing implements.

    Both sit at their off value in the released checkpoint and nothing reads
    them, so a checkpoint that set either would silently get the behavior it
    did not ask for. This architecture's config is not frozen.
    """
    layer = section.get("mtp_use_hidden_state_from_layer")
    if layer is not None:
        raise ValueError(
            f"mtp.mtp_use_hidden_state_from_layer={layer!r} names a decoder "
            "layer, but the draft consumes the final multi-stream bus")
    text = config.raw_component
    if isinstance(text.get("text_config"), dict):
        text = text["text_config"]
    if text.get("mtp_use_dedicated_embeddings"):
        raise ValueError(
            "mtp_use_dedicated_embeddings is set, but the draft shares the "
            "decoder's embedding table")


def configure_draft(config, **kwargs) -> None:
    """Select the MTP predictor embedded in a Qwen4-Exp checkpoint."""
    del kwargs
    section = _mtp_section(config)
    _reject_unconsumed_mtp_fields(config, section)
    num_layers = section.get("num_hidden_layers", config.mtp_num_hidden_layers)
    if not num_layers:
        raise ValueError("Qwen4-Exp MTP requires mtp.num_hidden_layers in "
                         "config.json")
    if int(num_layers) != 1:
        raise ValueError("Qwen4-Exp MTP requires one predictor layer; the "
                         f"draft runs one layer per step, got {num_layers}")
    layer_types = section.get("layer_types") or ["full_attention"] * num_layers
    unsupported = sorted(set(layer_types) - {"full_attention"})
    if unsupported:
        raise ValueError("Qwen4-Exp MTP layers must be full_attention; got " +
                         ", ".join(unsupported))
    config.num_hidden_layers = int(num_layers)
    config.layer_types = [core_config.LAYER_ATTN] * config.num_hidden_layers
    config.attention_layer_types = list(layer_types)
    # The predictor is a plain attention stack: no recurrent layers, and the
    # checkpoint carries no mtp.*ple* tensors.
    config.gdn_cfg = None
    config.mamba_cfg = None
    config.ngram_layers = []
    config.mtp_base = False
    # The predictor's rotary embedding is built from the decoder's rope
    # parameters, so a value that disagrees with them has no defined effect.
    rope_theta = section.get("rope_theta")
    if rope_theta is not None and float(rope_theta) != float(
            config.rope_theta):
        raise ValueError(
            f"mtp.rope_theta={rope_theta} differs from the decoder's "
            f"{config.rope_theta}, which is the one the predictor's rotary "
            "embedding is built from")
