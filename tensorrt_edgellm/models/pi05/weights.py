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
"""pi0.5 checkpoint detection and weight splitting.

The supported input is a PyTorch pi0.5 checkpoint: ``model.safetensors`` plus a
``config.json``. Two schemas ship the same weights -- the LeRobot releases, which
are the validated ones and write ``max_action_dim`` / ``n_action_steps`` and no
normalization assets, and openpi's ``convert_jax_model_to_pytorch.py`` output,
which writes ``action_dim`` / ``action_horizon`` and an ``assets/`` directory.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Dict, Tuple

import safetensors
import torch

logger = logging.getLogger(__name__)

try:
    from safetensors.torch import load_file as _load_safetensors
except ImportError:  # pragma: no cover
    _load_safetensors = None

# Checkpoint prefixes (openpi ``PI0Pytorch`` module paths).
_VISION_PREFIX = "paligemma_with_expert.paligemma.model."
_PREFIX_LM_PREFIX = "paligemma_with_expert.paligemma.model.language_model."
_EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model."

_VISION_SUBMODULES = ("vision_tower.", "multi_modal_projector.")

# Flow-matching projections and the pi05 time MLP live at the checkpoint root.
_ACTION_TOPLEVEL = (
    "action_in_proj.",
    "action_out_proj.",
    "time_mlp_in.",
    "time_mlp_out.",
)

_EMBED_TOKENS_KEY = _PREFIX_LM_PREFIX + "embed_tokens.weight"
# Gemma ties lm_head to the token embedding and ``safetensors.save_model`` deduplicates
# tied tensors, so the table usually survives only under the lm_head name; the file's
# ``__metadata__`` records the alias. Both spellings are accepted.
_TIED_EMBED_KEY = "paligemma_with_expert.paligemma.lm_head.weight"
_UNUSED_HEAD_KEY = "paligemma_with_expert.gemma_expert.lm_head.weight"

# The variant fields cannot identify the architecture -- pi0 carries them too. These
# weights exist only in pi0.5: the time MLP feeding adaRMS, and expert norms carrying a
# modulation Dense instead of a plain scale.
_PI05_TIME_MLP_KEY = "time_mlp_in.weight"
_PI05_EXPERT_NORM_MARKER = ".input_layernorm.dense."
# pi0 puts the continuous state into the suffix through this projection; pi0.5
# never has it (state is either discretized into the prompt or unused).
_PI0_STATE_PROJ_KEY = "state_proj.weight"

# Uniform prefix of a checkpoint saved from the LeRobot policy wrapper.
_WRAPPER_PREFIX = "model."


def load_checkpoint_weights(model_dir: str) -> Dict[str, torch.Tensor]:
    """Load and merge every safetensors shard in a pi0.5 checkpoint directory."""
    if _load_safetensors is None:
        raise RuntimeError("safetensors is required to load pi0.5 checkpoints")
    shards = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")
    merged: Dict[str, torch.Tensor] = {}
    for shard in shards:
        merged.update(_load_safetensors(shard))
    # Some LeRobot releases save the policy wrapper rather than the model, which puts
    # every tensor one level down. Stripping it here keeps one naming below.
    if merged and all(k.startswith(_WRAPPER_PREFIX) for k in merged):
        merged = {k[len(_WRAPPER_PREFIX):]: v for k, v in merged.items()}
    logger.info("Loaded %d tensors from %d shard(s) in %s", len(merged),
                len(shards), model_dir)
    return merged


def is_pi05_weights(weights: Dict[str, torch.Tensor]) -> bool:
    """Identify pi0.5 from its weight signature.

    Raises:
        ValueError: If the checkpoint carries both pi0 and pi0.5 markers. A
            silent guess would export a structurally wrong graph, so an
            ambiguous checkpoint is rejected rather than assumed.
    """
    has_time_mlp = _PI05_TIME_MLP_KEY in weights
    has_adarms = any(_PI05_EXPERT_NORM_MARKER in k for k in weights)
    has_state_proj = _PI0_STATE_PROJ_KEY in weights

    if (has_time_mlp or has_adarms) and has_state_proj:
        raise ValueError(
            "Checkpoint carries both pi0.5 markers (time_mlp/adaRMS) and the "
            "pi0 state_proj; cannot determine the architecture.")
    if has_time_mlp != has_adarms:
        raise ValueError(
            f"Inconsistent pi0.5 markers: time_mlp={has_time_mlp}, "
            f"adaRMS={has_adarms}. Expected both or neither.")
    return has_time_mlp


def is_pi05_checkpoint(model_dir: str) -> bool:
    """Whether ``model_dir`` holds a pi0.5 checkpoint.

    Prefers the name config.json gives itself -- the LeRobot releases write
    ``type``, and ``model_type`` is accepted too -- and falls back to the weight
    signature, read from the safetensors headers, for checkpoints converted
    before either field existed.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return False
    with open(config_path) as f:
        config = json.load(f)
    declared = str(config.get("model_type", config.get("type", ""))).lower()
    if declared:
        return declared == "pi05"
    if "paligemma_variant" not in config:
        return False
    # Key names alone decide this, so read the safetensors headers rather than
    # materializing every tensor: the caller loads them again straight after.
    return is_pi05_weights(_checkpoint_key_set(model_dir))


def _checkpoint_key_set(model_dir: str) -> Dict[str, None]:
    """Every tensor name in the checkpoint, without reading tensor data.

    Strips the LeRobot policy-wrapper prefix exactly as the full loader does, so
    the two paths see the same names and cannot disagree about the architecture.
    """
    keys: Dict[str, None] = {}
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".safetensors"):
            continue
        with safetensors.safe_open(os.path.join(model_dir, name),
                                   framework="pt") as handle:
            for key in handle.keys():
                keys[key] = None
    if keys and all(k.startswith(_WRAPPER_PREFIX) for k in keys):
        keys = {k[len(_WRAPPER_PREFIX):]: None for k in keys}
    return keys


def split_pi05_weights(
    weights: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[
        str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Split a pi0.5 checkpoint into per-component tensor dicts.

    Returns:
        ``(visual, prefix, embedding, action)``. Keys are rewritten to the
        module-local names each component's ``build_*`` expects; the token
        embedding table is separated out because the host gathers it instead of
        the prefix graph.
    """
    visual: Dict[str, torch.Tensor] = {}
    prefix: Dict[str, torch.Tensor] = {}
    embedding: Dict[str, torch.Tensor] = {}
    action: Dict[str, torch.Tensor] = {}

    for key, tensor in weights.items():
        if key in (_EMBED_TOKENS_KEY, _TIED_EMBED_KEY):
            embedding["embed_tokens.weight"] = tensor
        elif key == _UNUSED_HEAD_KEY:
            continue  # expert head; pi0.5 never decodes tokens
        elif key.startswith(_PREFIX_LM_PREFIX):
            prefix[key[len(_PREFIX_LM_PREFIX):]] = tensor
        elif key.startswith(_VISION_PREFIX) and any(
                key[len(_VISION_PREFIX):].startswith(sub)
                for sub in _VISION_SUBMODULES):
            visual[key[len(_VISION_PREFIX):]] = tensor
        elif key.startswith(_EXPERT_PREFIX):
            action[key[len(_EXPERT_PREFIX):]] = tensor
        elif key.startswith(_ACTION_TOPLEVEL):
            action[key] = tensor
        elif key.startswith("lm_head."):
            continue  # tied to embed_tokens; pi0.5 never runs it
        else:
            logger.debug("Unassigned checkpoint key: %s", key)

    if not embedding:
        raise KeyError(
            "pi0.5 checkpoint has no token embedding table (looked for "
            f"{_EMBED_TOKENS_KEY!r} and the tied {_TIED_EMBED_KEY!r}); the "
            "prefix graph cannot embed language tokens without it")

    logger.info(
        "Split pi0.5 checkpoint: %d visual, %d prefix, %d embedding, "
        "%d action tensors", len(visual), len(prefix), len(embedding),
        len(action))
    return visual, prefix, embedding, action


def load_pi05_config(model_dir: str) -> dict:
    """Load ``config.json`` from a converted pi0.5 checkpoint."""
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)
