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
"""Qwen3 checkpoint weight mapping."""

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from ...core import quantization
from ...core.weights import LinearWeights, ParameterSpec


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend tensor names to Qwen3 checkpoint aliases."""
    del component, spec_type, spec_role
    if name == "lm_head.weight" and quant_type == "fp16":
        return ("model.embed_tokens.weight", )
    return ()


def fuse_nvfp4_qkv(
        projections: Sequence[LinearWeights]) -> Optional[LinearWeights]:
    """Fuse provider Q/K/V NVFP4 tensors when their scalar scales match."""
    if len(projections) != 3:
        raise ValueError("Qwen3 QKV fusion requires three projections")
    query = projections[0]
    for name, projection in zip(("query", "key", "value"), projections):
        if projection.quant_type != quantization.QUANT_NVFP4:
            raise ValueError(f"{name} projection is not NVFP4")
        if projection.in_features != query.in_features:
            raise ValueError("Qwen3 Q/K/V projections must share input size")
        if projection.group_size != query.group_size:
            raise ValueError("Qwen3 Q/K/V projections must share group size")
        if isinstance(projection.weight, ParameterSpec) or isinstance(
                projection.weight_scale, ParameterSpec):
            # Separate projections retain their checkpoint refit recipes.
            return None
    if any(projection.input_scale != query.input_scale
           or projection.weight_scale_2 != query.weight_scale_2
           for projection in projections[1:]):
        return None

    biases = [projection.bias for projection in projections]
    if any(bias is None for bias in biases):
        if not all(bias is None for bias in biases):
            raise ValueError("Qwen3 Q/K/V projections must use uniform bias")
        bias = None
    else:
        bias = np.ascontiguousarray(np.concatenate(biases, axis=0))

    return replace(
        query,
        weight=np.ascontiguousarray(
            np.concatenate([projection.weight for projection in projections],
                           axis=0)),
        weight_scale=np.ascontiguousarray(
            np.concatenate(
                [projection.weight_scale for projection in projections],
                axis=0)),
        bias=bias,
        weight_recipe=None,
        scale_recipe=None,
        logical_out_features=sum(projection.out_features
                                 for projection in projections),
        logical_in_features=query.in_features,
    )
