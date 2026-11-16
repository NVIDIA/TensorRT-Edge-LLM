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
"""DSpark draft checkpoint weight mapping."""

import numpy as np

from tensorrt_edgellm.vocab_reduction.constants import DRAFT_VOCAB_MAP_NAME


def writes_runtime_embedding(args) -> bool:
    """DSpark reuses the base model's runtime embedding table."""
    del args
    return False


def runtime_weight_artifacts(weights, args):
    del args
    if not weights.has("d2t"):
        return {}
    if not weights.has("t2d"):
        raise ValueError(
            "DSpark reduced vocabulary requires both d2t and t2d tensors")
    offset_map = weights.array("d2t").astype(np.int64)
    vocab_map = offset_map + np.arange(offset_map.size, dtype=np.int64)
    target_vocab_size = weights.array("t2d").size
    if np.any(vocab_map < 0) or np.any(vocab_map >= target_vocab_size):
        raise ValueError(
            "DSpark draft-to-target vocabulary map contains invalid IDs")
    return {DRAFT_VOCAB_MAP_NAME: {"vocab_map": vocab_map.astype(np.int32)}}


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """DSpark backbone tensors already use frontend names."""
    del name, component, spec_type, spec_role, quant_type
    return ()
