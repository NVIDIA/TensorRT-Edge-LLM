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
"""Qwen4-Exp runtime configuration artifacts."""

import logging
import os
from typing import Any, Dict

from ...core import contracts
from ...core.artifacts.runtime_config import normalize_rope_scaling

logger = logging.getLogger(__name__)


def component_runtime_config(bundle, component: contracts.Component, args):
    """Return the Qwen4-Exp visual component's runtime config.

    The tower is the Qwen3.5 one, so it is published under that ``model_type``: the runtime keys
    its preprocessing (patch layout, mRoPE sections, image/video token handling) off this string,
    and declaring the family name would ask it for a tower that does not exist.
    """
    if component != contracts.Component.VISUAL:
        return None
    root = bundle.root
    visual = dict(bundle.component_dict(component))
    visual["model_type"] = "qwen3_5"
    text = dict(root.get("text_config") or {})
    result: Dict[str, Any] = {
        "model_type": "qwen3_5",
        "vision_config": visual,
    }
    for key in ("vision_start_token_id", "vision_end_token_id",
                "image_token_id", "video_token_id", "vocab_size",
                "rope_theta"):
        for source in (root, text):
            if key in source:
                result[key] = source[key]
                break

    rope = (text.get("rope_scaling") or text.get("rope_parameters")
            or root.get("rope_scaling") or root.get("rope_parameters"))
    if rope:
        normalized = normalize_rope_scaling(rope)
        result["rope_scaling"] = normalized
        text["rope_scaling"] = normalized
        if "rope_theta" in normalized:
            text.setdefault("rope_theta", normalized["rope_theta"])
    for key in ("vocab_size", "rope_theta"):
        if key in result:
            text.setdefault(key, result[key])
    if text:
        result["text_config"] = text
    result["builder_config"] = {
        "min_image_tokens": args.min_image_tokens,
        "max_image_tokens": args.max_image_tokens,
        "max_image_tokens_per_image": args.max_image_tokens_per_image,
    }
    return result


def _describe_ngram_pool(config: dict, cfg, model_dir: str) -> None:
    """Point the runtime at the checkpoint-resident n-gram table.

    The table is never copied: the runtime opens it through ``--checkpointDir`` using this source
    metadata, which is what keeps a 47.7 GiB tensor out of the engine bundle. The row-ID hash
    parameters are named here and read from the same checkpoint, so they cannot drift out of step
    with the rows they index.
    """
    from tensorrt_edgellm.checkpoint.ple_source import make_ngram_pool_metadata

    pool = make_ngram_pool_metadata(model_dir, cfg.ple_embed_dim,
                                    cfg.ngram_heads)
    config["ple_pool"] = pool
    config["checkpoint_dir"] = os.path.abspath(model_dir)
    logger.info("Described the n-gram pool: %d rows x %d, %d shards",
                pool["row_count"], pool["row_width"],
                len(pool["source"]["shards"]))


def _update_mtp_draft_config(config: dict, cfg) -> None:
    """Describe the MTP predictor's own stack and its feedback width.

    The generic writer publishes ``base_model_hidden_size = hidden_size``,
    which holds for every other MTP model. Here the base hands over the whole
    multi-stream residual, so the draft's two feedback bindings are hc_count
    times wider than its token embedding.
    """
    config["base_model_hidden_size"] = cfg.hc_hidden_size
    config["num_attention_layers"] = cfg.num_hidden_layers
    config["layer_types"] = ["attention"] * cfg.num_hidden_layers
    config["kv_layer_configs"] = [{
        "num_kv_heads": cfg.num_key_value_heads,
        "head_dim": cfg.head_dim,
    } for _ in range(cfg.num_hidden_layers)]


def update_llm_config(config: dict, root: dict, cfg, args) -> None:
    """Add the runtime contracts only this family declares."""
    if args.resolved_spec_role == contracts.SpecRole.DRAFT:
        _update_mtp_draft_config(config, cfg)
    if cfg.ngram_layers:
        config["ngram_layers"] = list(cfg.ngram_layers)
        config["ngram_embed_dim"] = cfg.ple_embed_dim
        config["ngram_conv_kernel"] = cfg.ple_conv_kernel_size
        # The injection writes into every residual stream at once, so its
        # convolution spans the full multi-stream width.
        config["ngram_conv_dim"] = cfg.hc_hidden_size
        config["ngram_size"] = cfg.ngram_size
        config["ngram_heads_per_order"] = cfg.ngram_heads_per_order
        # Must match the ``ngram_conv_state_*`` binding dtype in
        # modeling_qwen4_exp.py::input_tensors; the runtime rejects a mismatch at load.
        config["ngram_conv_state_dtype"] = "fp16"
        _describe_ngram_pool(config, cfg, args.model_dir)
    # The sparse-attention pool carries the indexer state as a tail on each
    # row, so its rows are wider than head_dim and the runtime has to allocate
    # them that way.
    for layer in config.get("kv_layer_configs") or []:
        if layer is not None:
            layer["head_dim"] = cfg.qsa_pool_head_dim
    # Every attention layer is Qwen Sparse Attention, which the runtime's
    # compact capability check refuses.
    if cfg.indexer_head_dim > 0:
        config["indexer_head_dim"] = cfg.indexer_head_dim
