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
"""Recurrent and stateful sequence operations."""

from typing import Optional, Tuple, Union

from ..ragged import RaggedDecoderInputs
from ..tensor import Tensor
from ._operation import operation


def causal_conv1d(
    hidden_states: Tensor,
    weight: Tensor,
    bias: Tensor,
    conv_state: Tensor,
    ragged: RaggedDecoderInputs,
    groups: int,
    padding: int,
    tree_parent_ids: Optional[Tensor] = None,
    tree_depths: Optional[Tensor] = None,
    use_intermediate: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Run stateful depthwise causal convolution."""
    if (tree_parent_ids is None) != (tree_depths is None):
        raise ValueError(
            "tree parent IDs and depths must be supplied together")
    use_ddtree = tree_parent_ids is not None
    if use_ddtree and not use_intermediate:
        raise ValueError("DDTree recurrent state requires intermediate output")
    attributes = {
        "stride": 1,
        "padding": padding,
        "dilation": 1,
        "groups": groups,
        "use_mtp": int(use_intermediate and not use_ddtree),
        "use_ddtree": int(use_ddtree),
    }
    inputs = [
        hidden_states, weight, bias, conv_state, ragged.query_lengths,
        ragged.query_start_offsets, ragged.state_indices,
        ragged.execution_phase_marker, ragged.context_sequence_count_carrier
    ]
    if use_ddtree:
        inputs.extend((tree_parent_ids, tree_depths))
    result = operation("causal_conv1d",
                       inputs,
                       output_count=3 if use_intermediate else 2,
                       **attributes)
    hidden_states, conv_state = result[:2]
    intermediate = result[2] if use_intermediate else None
    return hidden_states, conv_state, intermediate


def gated_delta_net(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    a: Tensor,
    b: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    state: Tensor,
    ragged: RaggedDecoderInputs,
    key_head_dim: int,
    value_head_dim: int,
    tree_parent_ids: Optional[Tensor] = None,
    tree_depths: Optional[Tensor] = None,
    use_intermediate: bool = False,
    use_diffusion_state: bool = False,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """Run recurrent Gated DeltaNet attention."""
    if (tree_parent_ids is None) != (tree_depths is None):
        raise ValueError(
            "tree parent IDs and depths must be supplied together")
    use_ddtree = tree_parent_ids is not None
    if use_ddtree and not use_intermediate:
        raise ValueError("DDTree recurrent state requires intermediate output")
    attributes = {
        "k_dim": key_head_dim,
        "v_dim": value_head_dim,
        "use_mtp": int(use_intermediate and not use_ddtree),
        "use_ddtree": int(use_ddtree),
        "use_diffusion_state": int(use_diffusion_state),
    }
    inputs = [
        q, k, v, a, b, a_log, dt_bias, state, ragged.query_lengths,
        ragged.query_start_offsets, ragged.state_indices,
        ragged.execution_phase_marker, ragged.context_sequence_count_carrier
    ]
    if use_ddtree:
        inputs.extend((tree_parent_ids, tree_depths))
    result = operation("gated_delta_net",
                       inputs,
                       output_count=3 if use_intermediate else 2,
                       **attributes)
    hidden_states, state = result[:2]
    intermediate = result[2] if use_intermediate else None
    return hidden_states, state, intermediate


def update_ssm_state(
    x: Tensor,
    a: Tensor,
    b: Tensor,
    c: Tensor,
    d: Tensor,
    dt: Tensor,
    dt_bias: Tensor,
    state: Tensor,
    ragged: RaggedDecoderInputs,
    dim: int,
    dstate: int,
    nheads: int,
    ngroups: int,
    tree_parent_ids: Optional[Tensor] = None,
    tree_depths: Optional[Tensor] = None,
    use_intermediate: bool = False,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor, Tensor,
                                        Tensor]]:
    """Run selective state update."""
    if (tree_parent_ids is None) != (tree_depths is None):
        raise ValueError(
            "tree parent IDs and depths must be supplied together")
    use_ddtree = tree_parent_ids is not None
    if use_ddtree and not use_intermediate:
        raise ValueError("DDTree recurrent state requires intermediate output")
    inputs = [
        x, a, b, c, d, dt, dt_bias, state, ragged.query_lengths,
        ragged.query_start_offsets, ragged.state_indices,
        ragged.execution_phase_marker, ragged.context_sequence_count_carrier
    ]
    if use_ddtree:
        inputs.extend((tree_parent_ids, tree_depths))
    return operation("update_ssm_state",
                     inputs,
                     output_count=6 if use_intermediate else 2,
                     dim=dim,
                     dstate=dstate,
                     nheads=nheads,
                     ngroups=ngroups,
                     dt_softplus=1,
                     chunk_size=1,
                     use_spec_verify_state=int(use_intermediate),
                     use_ddtree=int(use_ddtree),
                     time_step_limit=[0.0, float("inf")])
