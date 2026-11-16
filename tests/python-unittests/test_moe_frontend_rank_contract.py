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
"""Token-major rank contracts for the official MoE custom-op frontend."""

import pytest
import torch

from tensorrt_edgellm.models import ops


def _common_tensors():
    return torch.zeros(2, 4), torch.zeros(1, 2, 8, dtype=torch.float16)


@pytest.mark.parametrize(
    "invoke",
    [
        lambda r, h, w: ops.int4_moe_plugin(r, h, w, w, w, w, 4, 2, 8, 16, 0, 8
                                            ),
        lambda r, h, w: ops.
        nvfp4_moe_plugin(r, h, w, w, w, w, w, w, w, w, w, 4, 2, 8, 16, 4, 1, 1,
                         1, 1.0, 1, 0, 1, 0),
        lambda r, h, w: ops.nvfp4_moe_plugin_geforce(
            r, h, w, w, w, w, w, w, w, w, w, 4, 2, 8, 16, 4, 1, 1, 1, 1.0, 1,
            0, 1, 0),
        lambda r, h, w: ops.nvfp4_a16_moe_plugin(
            r, h, w, w, w, w, w, w, w, 4, 2, 8, 16, 4, 1, 1, 1, 1.0, 1, 0),
        lambda r, h, w: ops.nvfp4_a16_blackwell_moe_plugin(
            r, h, w, w, w, w, w, w, w, 4, 2, 8, 16, 4, 1, 1, 1, 1.0, 1, 0, 0),
        lambda r, h, w: ops.fp16_moe_plugin(r, h, w, w, 4, 2, 8, 16, 2, 1, 0),
        lambda r, h, w: ops.fp16_moe_plugin_sigmoid(r, h, w, w, w, 4, 2, 8, 16,
                                                    4, 1, 1, 1, 1.0, 0),
    ],
    ids=[
        "int4", "nvfp4", "nvfp4_geforce", "nvfp4_a16", "nvfp4_a16_blackwell",
        "fp16", "fp16_sigmoid"
    ],
)
def test_moe_custom_ops_reject_rank3_hidden_states(invoke):
    router_logits, hidden_states = _common_tensors()
    weight = torch.zeros(1, dtype=torch.int8)

    with pytest.raises(ValueError,
                       match=r"hidden_states must have shape \[T, H\]"):
        invoke(router_logits, hidden_states, weight)


def test_moe_custom_op_rejects_rank3_router_logits():
    hidden_states = torch.zeros(2, 8, dtype=torch.float16)
    router_logits = torch.zeros(1, 2, 4)
    weight = torch.zeros(1, dtype=torch.int8)

    with pytest.raises(ValueError,
                       match=r"router_logits must have shape \[T, E\]"):
        ops.int4_moe_plugin(router_logits, hidden_states, weight, weight,
                            weight, weight, 4, 2, 8, 16, 0, 8)
