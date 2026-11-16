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
"""Low-rank candidate head of EAGLE drafters (``LowRankCandidateHeadPlugin``)."""

from typing import Tuple

import torch
import torch.nn as nn

from ..config import ModelConfig
from .linear import FP16Linear
from .ops import low_rank_candidate_head

__all__ = ["LowRankCandidateHead", "fp16_head_weight"]


class LowRankCandidateHead(nn.Module):
    """Selects ``draft_candidate_count`` full-vocabulary candidates per row.

    Module tree (checkpoint factors are remapped by ``_eagle4_key_remap``):
        down_proj   ``[rank, hidden]``  factor B
        up_proj     ``[vocab, rank]``   factor A

    The draft model's ``lm_head`` weight is passed to :meth:`forward`, so the
    exported graph holds it once, as an input of the plugin node.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_candidates = config.draft_candidate_count
        self.down_proj = FP16Linear(config.hidden_size,
                                    config.draft_candidate_rank)
        self.up_proj = FP16Linear(config.draft_candidate_rank,
                                  config.vocab_size)

    def forward(
            self, nrm: torch.Tensor,
            lm_head_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(support_ids [R, Kc] int32, support_log_probs [R, Kc] fp32)``."""
        return low_rank_candidate_head(nrm, self.down_proj.weight,
                                       self.up_proj.weight, lm_head_weight,
                                       self.num_candidates)


def fp16_head_weight(name: str, weight: torch.Tensor) -> torch.Tensor:
    """Return ``weight.to(torch.float16)``; raise if a cast value is not finite."""
    cast = weight.to(torch.float16)
    if not bool(torch.isfinite(cast).all()):
        raise ValueError(
            f"{name} has values that are not finite in fp16; the candidate "
            "head plugin takes fp16 weights.")
    return cast
