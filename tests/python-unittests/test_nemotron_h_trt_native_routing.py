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
"""The in-graph routing of the TRT-native Nemotron-H MoE arm matches the router.

``TRT_MoE`` consumes precomputed routing, so ``NemotronHMoEMLP`` materialises
the sigmoid group-top-k routing of ``NemotronHTopkRouter`` as graph ops. The
two must agree on the chosen experts and their weights (order is not part of
the contract), for the Nemotron-3-Nano shape (``n_group == 1``) and for a
grouped configuration.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_edgellm.models.nemotron_h.modeling_nemotron_h import (
    NemotronHTopkRouter, trt_native_sigmoid_group_topk_routing)


def _dense(indices: torch.Tensor, weights: torch.Tensor,
           num_experts: int) -> torch.Tensor:
    """Scatter per-token (expert, weight) pairs into a dense [T, E] map."""
    out = torch.zeros(indices.shape[0], num_experts, dtype=torch.float32)
    return out.scatter(1, indices.long(), weights.float())


@pytest.mark.parametrize("n_group,topk_group", [(1, 1), (8, 4)])
@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_routing_matches_router_module(n_group: int, topk_group: int,
                                       norm_topk_prob: bool) -> None:
    torch.manual_seed(0)
    num_experts, hidden, top_k, num_tokens = 128, 64, 6, 37
    config = SimpleNamespace(num_experts_per_tok=top_k,
                             n_routed_experts=num_experts,
                             routed_scaling_factor=2.5,
                             n_group=n_group,
                             topk_group=topk_group,
                             norm_topk_prob=norm_topk_prob,
                             hidden_size=hidden)
    router = NemotronHTopkRouter(config)
    router.weight.data = torch.randn(num_experts, hidden).to(torch.float16)
    router.e_score_correction_bias.copy_(
        torch.randn(num_experts).to(torch.float16))
    hidden_states = torch.randn(num_tokens, hidden).to(torch.float16)

    ref_indices, ref_weights = router(hidden_states)

    router_logits = torch.nn.functional.linear(hidden_states,
                                               router.weight).float()
    indices, weights = trt_native_sigmoid_group_topk_routing(
        router_logits, router.e_score_correction_bias.float(), top_k, n_group,
        topk_group, norm_topk_prob, 2.5)

    assert indices.dtype == torch.int32
    assert weights.dtype == torch.float32
    assert indices.shape == (num_tokens, top_k)
    assert weights.shape == (num_tokens, top_k)
    torch.testing.assert_close(_dense(indices, weights, num_experts),
                               _dense(ref_indices, ref_weights, num_experts),
                               rtol=1e-5,
                               atol=1e-6)
