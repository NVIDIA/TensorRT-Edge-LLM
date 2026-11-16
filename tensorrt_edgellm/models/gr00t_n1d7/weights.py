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
"""GR00T N1.7 checkpoint loading and per-component weight splitting."""

from __future__ import annotations

import glob
import logging
import os
from typing import Dict, Tuple

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

VISUAL_PREFIX = "backbone.model.model.visual."
BACKBONE_LM_PREFIX = "backbone.model.model.language_model."
ACTION_HEAD_PREFIX = "action_head."
EMBED_TOKENS_KEY = "embed_tokens.weight"
# Tied to embed_tokens; the policy never decodes a token.
_UNUSED_HEAD_KEY = "backbone.model.lm_head.weight"


def load_checkpoint_weights(model_dir: str) -> Dict[str, torch.Tensor]:
    """Load and merge every safetensors shard in the checkpoint directory."""
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")
    merged: Dict[str, torch.Tensor] = {}
    for shard in shards:
        merged.update(load_file(shard))
    logger.info("Loaded %d tensors from %d shard(s) in %s", len(merged),
                len(shards), model_dir)
    return merged


def split_gr00t_weights(
    weights: Dict[str, torch.Tensor]
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[
        str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split the checkpoint into ``(visual, backbone, embedding, action_head)``: the tower
    keeps ``model.visual.``, the backbone is rekeyed to ``model.layers.N``/``model.norm``, the
    token embedding is separated for the host gather, the head loses ``action_head.``."""
    visual: Dict[str, torch.Tensor] = {}
    backbone: Dict[str, torch.Tensor] = {}
    embedding: Dict[str, torch.Tensor] = {}
    head: Dict[str, torch.Tensor] = {}
    for key, tensor in weights.items():
        if key.startswith(VISUAL_PREFIX):
            visual["model.visual." + key[len(VISUAL_PREFIX):]] = tensor
        elif key == BACKBONE_LM_PREFIX + EMBED_TOKENS_KEY:
            embedding[EMBED_TOKENS_KEY] = tensor
        elif key.startswith(BACKBONE_LM_PREFIX):
            backbone["model." + key[len(BACKBONE_LM_PREFIX):]] = tensor
        elif key.startswith(ACTION_HEAD_PREFIX):
            head[key[len(ACTION_HEAD_PREFIX):]] = tensor
        elif key == _UNUSED_HEAD_KEY:
            continue
        else:
            logger.debug("Unassigned checkpoint key: %s", key)
    if EMBED_TOKENS_KEY not in embedding:
        raise KeyError("GR00T checkpoint has no language_model.embed_tokens; "
                       "the backbone cannot embed the VL context without it")
    logger.info(
        "Split GR00T checkpoint: %d visual, %d backbone, %d action-head "
        "tensors", len(visual), len(backbone), len(head))
    return visual, backbone, embedding, head
