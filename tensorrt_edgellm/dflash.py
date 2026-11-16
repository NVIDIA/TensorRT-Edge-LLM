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
"""Shared checkpoint contract for DFlash frontends."""

import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Tuple

# Last-layer projections of a Qwen3-style draft behind a Gemma4 target that the
# overflow guard computes in FP32.
LAST_LAYER_PROJECTIONS = ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                          "mlp.down_proj")


def dense_last_layer_modules(num_hidden_layers: int) -> list:
    """Return the last layer's projections to run as dense FP16.

    Empty unless ``EDGELLM_DFLASH_DENSE_LAST_LAYER=1``.
    """
    if os.environ.get("EDGELLM_DFLASH_DENSE_LAST_LAYER") != "1":
        return []
    return [
        f"layers.{num_hidden_layers - 1}.{module}"
        for module in LAST_LAYER_PROJECTIONS
    ]


class DFlashVersion(IntEnum):
    """DFlash engine ABI selected by the draft checkpoint architecture."""

    V1 = 1
    V2 = 2


@dataclass(frozen=True)
class DFlashContract:
    """Normalized DFlash checkpoint fields used by export and engine build."""

    version: DFlashVersion
    target_layer_ids: Tuple[int, ...]
    block_size: int
    mask_token_id: int
    is_causal: bool
    conv_kernel_size: int = 0
    conv_group_size: int = 0
    selector_rank: int = 0
    selector_top_k: int = 0
    # DFlash V2 selector unary values are the target's final logits:
    # softcap(lm_head(h) * output_multiplier); 0 disables the softcap.
    output_multiplier: float = 1.0
    final_logit_softcapping: float = 0.0
    # Domino: DFlash V1 draft ABI plus a runtime causal correction head
    # (GRU over realized-token embeddings + rank-`proj_dim` vocab codebook).
    projector_type: str = ""
    gru_hidden_dim: int = 0
    proj_dim: int = 0
    prefix_len: int = 0
    pool_size: int = 0
    # shift_label=False (DFlash convention): block position i predicts the
    # masked token i, position 0 only carries the anchor. shift_label=True:
    # position i predicts token i + 1 and the last position is dropped.
    # DeepSpec DFlash checkpoints (``*DSparkModel``) use shift_label=True.
    shift_label: bool = False
    # LiLiCorr: DFlash V2 backbone whose selector is replaced by a lattice
    # correction head over the per-slot top-`lilicorr_candidate_topk` tokens.
    lilicorr_candidate_topk: int = 0
    lilicorr_hidden_size: int = 0
    lilicorr_num_layers: int = 0
    lilicorr_num_heads: int = 0
    lilicorr_mlp_ratio: float = 0.0
    lilicorr_factor_dim: int = 0
    lilicorr_vector_eps: float = 0.0
    lilicorr_logit_scale: float = 0.0

    @property
    def is_domino(self) -> bool:
        return self.projector_type == "domino"

    @property
    def is_lilicorr(self) -> bool:
        return self.projector_type == LILICORR_PROJECTOR_TYPE

    @property
    def supports_probabilistic_sampling(self) -> bool:
        return self.version == DFlashVersion.V2


LILICORR_PROJECTOR_TYPE = "lilicorr"
LILICORR_EMBEDDING_TABLE_INPUT = "lilicorr_embedding_table"

# The released LiLiCorr backbone is trained in BF16; its o_proj output,
# silu(gate) * up and down_proj output exceed the FP16 range (65504). o_proj
# and up_proj run on inputs divided by this power of two (both are bias-free,
# so their outputs scale exactly up to FP16 rounding and subnormal underflow),
# which leaves silu(gate) * up and down_proj at 1/scale; down_proj must be
# bias-free too. The conv finish restores the scale in its FP32 residual.
LILICORR_SUBLAYER_OUTPUT_SCALE = 1024.0

# The packed fc emits FP16. On Bolt target hidden its output reaches 2.3e4
# (2.8x below the FP16 limit); the bias-free fc runs on inputs divided by this
# power of two and the FP32 projection is scaled back before hidden_norm.
LILICORR_FC_INPUT_SCALE = 64.0


def _architectures(config: Mapping[str, Any]) -> set[str]:
    value = config.get("architectures") or []
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


_LILICORR_HEAD_FIELDS = (
    "lilicorr_candidate_topk",
    "lilicorr_hidden_size",
    "lilicorr_num_layers",
    "lilicorr_num_heads",
    "lilicorr_mlp_ratio",
    "lilicorr_factor_dim",
    "lilicorr_vector_eps",
    "lilicorr_logit_scale",
)


def _find_is_causal(*scopes: Mapping[str, Any]) -> Any:
    """Return ``is_causal`` from the first scope that has it, else ``causal``."""
    for name in ("is_causal", "causal"):
        for scope in scopes:
            if name in scope:
                return scope[name]
    return None


def _target_logit_transform(fields: Mapping[str, Any]) -> Tuple[float, float]:
    """``(output_multiplier, final_logit_softcapping)`` of a DFlash V2 checkpoint; a missing or null field is unset.

    The draft applies them after its top-k, which keeps the support only for a positive multiplier.
    """
    multiplier = fields.get("output_multiplier")
    softcap = fields.get("final_logit_softcapping")
    multiplier = 1.0 if multiplier is None else float(multiplier)
    softcap = 0.0 if softcap is None else float(softcap)
    if multiplier <= 0.0:
        raise ValueError(
            f"DFlash V2 output_multiplier must be positive, got {multiplier}")
    if softcap < 0.0:
        raise ValueError(
            f"DFlash V2 final_logit_softcapping must be non-negative, got {softcap}"
        )
    return multiplier, softcap


def resolve_dflash_contract(root: Mapping[str, Any],
                            llm: Mapping[str, Any]) -> DFlashContract:
    """Resolve and validate a DFlash contract without framework imports."""
    architectures = _architectures(root) | _architectures(llm)

    root_fields = root.get("dflash_config") or {}
    llm_fields = llm.get("dflash_config") or {}
    fields = {**root_fields, **llm_fields}
    projector_type = str(fields.get("projector_type", "") or "")
    if not projector_type and "LiLiCorrDraftModel" in architectures:
        projector_type = LILICORR_PROJECTOR_TYPE
    is_lilicorr = projector_type == LILICORR_PROJECTOR_TYPE
    version = (DFlashVersion.V2 if "DFlash2DraftModel" in architectures
               or is_lilicorr else DFlashVersion.V1)
    target_layer_ids = tuple(
        int(value) for value in (fields.get("target_layer_ids") or llm.get(
            "target_layer_ids") or root.get("target_layer_ids") or []))
    block_size = int(
        fields.get("block_size",
                   llm.get("block_size", root.get("block_size", 16))))
    mask_token_id = int(
        fields.get("mask_token_id",
                   llm.get("mask_token_id", root.get("mask_token_id",
                                                     248070))))
    causal_value = _find_is_causal(fields, llm, root)
    is_causal = True if causal_value is None else bool(causal_value)

    gru_hidden_dim = int(fields.get("gru_hidden_dim", 0) or 0)
    proj_dim = int(fields.get("emb_dim", 0) or 0)
    prefix_len = int(fields.get("pure_draft_prefix_len", 0) or 0)
    pool_size = int(fields.get("pool_size", 2048) or 0)
    is_deepspec = any(arch.endswith("DSparkModel") for arch in architectures)
    shift_label = bool(fields.get("shift_label", is_deepspec))

    if projector_type and projector_type not in ("domino",
                                                 LILICORR_PROJECTOR_TYPE):
        raise ValueError(
            f"Unsupported dflash_config.projector_type {projector_type!r}; "
            "supported: 'domino', 'lilicorr'")
    if projector_type == "domino":
        if version != DFlashVersion.V1:
            raise ValueError(
                "Domino requires the DFlash V1 draft architecture")
        for name in ("block_size", "mask_token_id"):
            if name not in fields and name not in llm and name not in root:
                raise ValueError(f"Domino requires {name}")
        if gru_hidden_dim <= 0 or proj_dim <= 0:
            raise ValueError(
                "Domino requires positive dflash_config.gru_hidden_dim and "
                "dflash_config.emb_dim")
        if prefix_len != 1:
            raise ValueError(
                "Domino runtime supports pure_draft_prefix_len=1 only, "
                f"got {prefix_len}")
        if not 3 <= block_size <= 16:
            raise ValueError(
                f"Domino block_size must be in [3, 16], got {block_size}")
        if mask_token_id < 0:
            raise ValueError("Domino mask_token_id must be non-negative")
        if not target_layer_ids:
            raise ValueError("Domino requires dflash_config.target_layer_ids")
        if pool_size <= 0:
            raise ValueError(
                f"Domino pool_size must be positive, got {pool_size}")

    lilicorr_fields = {}
    if is_lilicorr:
        if not fields.get("lilicorr_enabled", False):
            raise ValueError(
                "LiLiCorr requires dflash_config.lilicorr_enabled=true")
        missing = [
            name for name in _LILICORR_HEAD_FIELDS if name not in fields
        ]
        if missing:
            raise ValueError(
                f"LiLiCorr requires dflash_config.{', '.join(missing)}")
        lilicorr_fields = {
            "lilicorr_candidate_topk": int(fields["lilicorr_candidate_topk"]),
            "lilicorr_hidden_size": int(fields["lilicorr_hidden_size"]),
            "lilicorr_num_layers": int(fields["lilicorr_num_layers"]),
            "lilicorr_num_heads": int(fields["lilicorr_num_heads"]),
            "lilicorr_mlp_ratio": float(fields["lilicorr_mlp_ratio"]),
            "lilicorr_factor_dim": int(fields["lilicorr_factor_dim"]),
            "lilicorr_vector_eps": float(fields["lilicorr_vector_eps"]),
            "lilicorr_logit_scale": float(fields["lilicorr_logit_scale"]),
        }
        topk = lilicorr_fields["lilicorr_candidate_topk"]
        if topk <= 0 or topk > 16 or topk & (topk - 1):
            raise ValueError(
                "LiLiCorr lilicorr_candidate_topk must be a power of two "
                f"<= 16, got {topk}")
        for name in ("lilicorr_hidden_size", "lilicorr_num_layers",
                     "lilicorr_num_heads", "lilicorr_mlp_ratio",
                     "lilicorr_factor_dim", "lilicorr_vector_eps",
                     "lilicorr_logit_scale"):
            if not lilicorr_fields[name] > 0:
                raise ValueError(f"LiLiCorr {name} must be positive, got "
                                 f"{lilicorr_fields[name]}")

    if version == DFlashVersion.V2:
        for name in ("block_size", "mask_token_id"):
            if name not in fields and name not in llm and name not in root:
                raise ValueError(f"DFlash V2 requires {name}")
        if causal_value is None:
            raise ValueError("DFlash V2 requires is_causal")
        required = {
            "conv_kernel_size": 2,
            "conv_group_size": 16,
        }
        if not is_lilicorr:
            required.update({
                "selector_rank": 256,
                "selector_top_k": 16,
            })
        for name, expected in required.items():
            if name not in fields:
                raise ValueError(f"DFlash V2 requires dflash_config.{name}")
            actual = int(fields[name])
            if actual != expected:
                raise ValueError(
                    f"DFlash V2 {name} must be {expected}, got {actual}")
        if not 2 <= block_size <= 16:
            raise ValueError(
                f"DFlash V2 block_size must be in [2, 16], got {block_size}")
        if mask_token_id < 0:
            raise ValueError("DFlash V2 mask_token_id must be non-negative")
        if is_lilicorr:
            if not target_layer_ids:
                raise ValueError(
                    "LiLiCorr requires at least one target-layer ID")
        elif len(target_layer_ids) != 5:
            raise ValueError(
                "DFlash V2 requires exactly five target-layer IDs")
        if len(set(target_layer_ids)) != len(target_layer_ids):
            raise ValueError("DFlash V2 target-layer IDs must be unique")
        if is_causal:
            raise ValueError(
                "DFlash V2 draft checkpoint requires is_causal=false")

    logit_transform = (_target_logit_transform(fields)
                       if version == DFlashVersion.V2 else (1.0, 0.0))
    return DFlashContract(
        version=version,
        target_layer_ids=target_layer_ids,
        block_size=block_size,
        mask_token_id=mask_token_id,
        is_causal=is_causal,
        conv_kernel_size=int(fields.get("conv_kernel_size", 0)),
        conv_group_size=int(fields.get("conv_group_size", 0)),
        selector_rank=int(fields.get("selector_rank", 0)),
        selector_top_k=int(fields.get("selector_top_k", 0)),
        output_multiplier=logit_transform[0],
        final_logit_softcapping=logit_transform[1],
        projector_type=projector_type,
        gru_hidden_dim=gru_hidden_dim,
        proj_dim=proj_dim,
        prefix_len=prefix_len,
        pool_size=pool_size if projector_type == "domino" else 0,
        shift_label=shift_label if version == DFlashVersion.V1 else False,
        **lilicorr_fields,
    )
