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
"""Experimental-builder policy for checkpoint-backed runtime weights.

Supported INT4 FFN, INT4 MoE, NVFP4 MoE, TP NVFP4 plugin, and FP16 layouts
become runtime-owned engine inputs or plugin resources when the model family
provides a reproducible checkpoint transform recipe. FP8 and MXFP8 weights
remain folded into the engine by design.
"""

from dataclasses import dataclass, replace
from typing import Iterable, Optional, Sequence, Union

EXTERNAL_WEIGHT_INT4_FFN = "int4_ffn"
EXTERNAL_WEIGHT_INT4_MOE = "int4_moe"
EXTERNAL_WEIGHT_NVFP4_MOE = "nvfp4_moe"
EXTERNAL_WEIGHT_NVFP4_TP = "nvfp4_tp"
EXTERNAL_WEIGHT_LM_HEAD = "lm_head"
EXTERNAL_WEIGHT_FP16 = "fp16"
EXTERNAL_WEIGHT_EMBEDDING = "embedding"
EXTERNAL_WEIGHT_ALL = "all"

EXTERNAL_WEIGHT_KINDS = (
    EXTERNAL_WEIGHT_INT4_FFN,
    EXTERNAL_WEIGHT_INT4_MOE,
    EXTERNAL_WEIGHT_NVFP4_MOE,
    EXTERNAL_WEIGHT_NVFP4_TP,
    EXTERNAL_WEIGHT_LM_HEAD,
    EXTERNAL_WEIGHT_FP16,
    EXTERNAL_WEIGHT_EMBEDDING,
)
EXTERNAL_WEIGHT_CHOICES = (*EXTERNAL_WEIGHT_KINDS, EXTERNAL_WEIGHT_ALL)

# Engine name / role reserved for the checkpoint-direct embedding table, which
# the runtime loads outside the TensorRT input set.
CHECKPOINT_BINDING_ROLE_EMBEDDING = "embedding"
CHECKPOINT_BINDING_ENGINE_EMBEDDING = "__embedding__"
CHECKPOINT_BINDING_ROLE_PLE = "ple_embedding"
CHECKPOINT_BINDING_ENGINE_PLE = "__ple_embedding__"

__all__ = [
    "CHECKPOINT_BINDING_ENGINE_EMBEDDING",
    "CHECKPOINT_BINDING_ENGINE_PLE",
    "CHECKPOINT_BINDING_ROLE_EMBEDDING",
    "CHECKPOINT_BINDING_ROLE_PLE",
    "EXTERNAL_WEIGHT_CHOICES",
    "EXTERNAL_WEIGHT_KINDS",
    "WeightPolicy",
    "resolve_externalize_weights",
]


def resolve_externalize_weights(
        values: Optional[Union[str, Iterable[str]]]) -> "tuple[str, ...]":
    """Normalize and validate requested external weight kinds."""
    if values is None:
        return ()
    if isinstance(values, str):
        values = (values, )
    requested: list[str] = []
    for value in values:
        if value == EXTERNAL_WEIGHT_ALL:
            return EXTERNAL_WEIGHT_KINDS
        if value not in EXTERNAL_WEIGHT_KINDS:
            raise ValueError(
                f"unsupported external weight kind {value!r}; supported kinds: "
                + ", ".join(EXTERNAL_WEIGHT_CHOICES))
        if value not in requested:
            requested.append(value)
    return tuple(requested)


def _is_lm_head(name: str) -> bool:
    return (name == "lm_head" or name.endswith(".lm_head")
            or name == "model.embed_tokens" or name.endswith(".embed_tokens"))


@dataclass(frozen=True)
class WeightPolicy:
    """Externalization decisions shared by lowering and artifact writing."""

    kinds: Sequence[str] = ()
    bake_lm_head: bool = False

    @classmethod
    def from_request(cls, externalize_weights=None) -> "WeightPolicy":
        """Build a policy from CLI-shaped arguments."""
        kinds = resolve_externalize_weights(externalize_weights)
        if not kinds:
            kinds = EXTERNAL_WEIGHT_KINDS
        return cls(kinds=tuple(kinds))

    def without(self,
                kinds: Iterable[str],
                *,
                strict: bool = False) -> "WeightPolicy":
        """Drop kinds this build cannot reproduce from a raw checkpoint."""
        dropped = tuple(kind for kind in kinds if kind in self.kinds)
        if not dropped:
            return self
        if strict:
            raise ValueError("cannot externalize " + ", ".join(dropped) +
                             " for this build")
        return replace(self,
                       kinds=tuple(kind for kind in self.kinds
                                   if kind not in dropped))

    def baking_lm_head(self, *, strict: bool = False) -> "WeightPolicy":
        """Keep the LM head in the engine even when FP16 weights leave it."""
        if strict and self.wants(EXTERNAL_WEIGHT_LM_HEAD):
            raise ValueError("cannot externalize lm_head for this build")
        return replace(self, bake_lm_head=True)

    def wants(self, kind: str) -> bool:
        return kind in self.kinds

    def externalizes_parameter(self, kind: str, name: str = "") -> bool:
        """Return whether one static TensorRT parameter leaves the engine."""
        if kind == EXTERNAL_WEIGHT_FP16:
            module_name = name.removesuffix(".weight")
            return self.externalizes_fp16(module_name)
        return self.wants(kind)

    def externalizes_fp16(self, name: str) -> bool:
        """Whether one dense FP16 projection becomes an engine input."""
        if _is_lm_head(name):
            return (not self.bake_lm_head
                    and (self.wants(EXTERNAL_WEIGHT_FP16)
                         or self.wants(EXTERNAL_WEIGHT_LM_HEAD)))
        return self.wants(EXTERNAL_WEIGHT_FP16)

    @property
    def externalizes_embedding(self) -> bool:
        """Whether the embedding table is a runtime-owned artifact."""
        return self.wants(EXTERNAL_WEIGHT_EMBEDDING)
