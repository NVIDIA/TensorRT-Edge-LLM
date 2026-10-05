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

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, Tuple


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

    @property
    def supports_probabilistic_sampling(self) -> bool:
        return self.version == DFlashVersion.V2


def _architectures(config: Mapping[str, Any]) -> set[str]:
    value = config.get("architectures") or []
    if isinstance(value, str):
        return {value}
    return {str(item) for item in value}


def resolve_dflash_contract(root: Mapping[str, Any],
                            llm: Mapping[str, Any]) -> DFlashContract:
    """Resolve and validate a DFlash contract without framework imports."""
    architectures = _architectures(root) | _architectures(llm)
    version = (DFlashVersion.V2
               if "DFlash2DraftModel" in architectures else DFlashVersion.V1)

    root_fields = root.get("dflash_config") or {}
    llm_fields = llm.get("dflash_config") or {}
    fields = {**root_fields, **llm_fields}
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
    is_causal = bool(
        fields.get("is_causal",
                   llm.get("is_causal", root.get("is_causal", True))))

    if version == DFlashVersion.V2:
        for name in ("block_size", "mask_token_id"):
            if name not in fields and name not in llm and name not in root:
                raise ValueError(f"DFlash V2 requires {name}")
        if ("is_causal" not in fields and "is_causal" not in llm
                and "is_causal" not in root):
            raise ValueError("DFlash V2 requires is_causal")
        required = {
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "selector_rank": 256,
            "selector_top_k": 16,
        }
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
        if len(target_layer_ids) != 5:
            raise ValueError(
                "DFlash V2 requires exactly five target-layer IDs")
        if len(set(target_layer_ids)) != len(target_layer_ids):
            raise ValueError("DFlash V2 target-layer IDs must be unique")
        if is_causal:
            raise ValueError(
                "DFlash V2 draft checkpoint requires is_causal=false")

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
    )
