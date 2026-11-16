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
"""Qwen4-Exp multi-stream residual: grouped normalization and its gate pair.

Qwen4-Exp replaces the single residual bus with ``hc_count`` parallel streams
("Gated Residual", also published as HyperConnections). These gates carry the
only normalization on the residual path: the architecture has no per-layer
input/post norms and no final ``model.norm``.
"""

from typing import Tuple, Union

import numpy as np
import tensorrt as trt

from ...ops import BuildContext, Linear, Module, Tensor
from ...ops import functional as F

__all__ = ["Qwen4ExpGroupedRMSNorm", "Qwen4ExpGatedResidual"]


class Qwen4ExpGroupedRMSNorm(Module):
    """Zero-centered RMSNorm over independently normalized groups.

    Splitting the last dimension into ``group_size``-wide groups keeps one
    residual stream's magnitude from rescaling another's. Both the statistic
    and the ``1 + weight`` scale are evaluated in float32 and rounded once on
    the way out, matching the reference implementation.
    """

    def __init__(self, ctx: BuildContext, prefix: str,
                 group_size: int) -> None:
        super().__init__(ctx, prefix)
        self.group_size = int(group_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        weight = self.weights.f16(self.key("weight"))
        width = int(weight.shape[0])
        if width % self.group_size:
            raise ValueError(
                f"{self.key('weight')} width {width} is not a multiple of "
                f"group size {self.group_size}")
        groups = width // self.group_size
        grouped = hidden_states.cast(trt.float32).reshape(
            (0, groups, self.group_size))
        variance = (grouped * grouped).mean(-1, keepdim=True)
        normalized = grouped / (variance +
                                np.float32(self.cfg.rms_norm_eps)).sqrt()
        scale = F.constant((1.0 + weight.astype(np.float32)).reshape(1, width),
                           "hc_norm_gain")
        return (normalized.reshape((0, width)) * scale).cast(trt.float16)


class Qwen4ExpGatedResidual(Module):
    """Read/write gate pair wrapping one block of the multi-stream residual.

    :meth:`forward` is the read half: it normalizes the streams, builds an
    element-wise gate through a ``hc_lowrank`` bottleneck and averages the
    gated streams into one block input. It also returns the per-stream
    injection weights that :meth:`combine` applies once the block has run.

    ``use_combine=False`` builds the stack-exit mixer, which only folds the
    streams down and has no ``block_inject_weight``.
    """

    def __init__(self,
                 ctx: BuildContext,
                 prefix: str,
                 use_combine: bool = True) -> None:
        super().__init__(ctx, prefix)
        self.hc_norm = Qwen4ExpGroupedRMSNorm(ctx, self.key("hc_norm"),
                                              ctx.cfg.hidden_size)
        self.input_mix_weight_down = Linear(ctx,
                                            self.key("input_mix_weight_down"),
                                            tensor_parallel=False)
        self.input_mix_weight_up = Linear(ctx,
                                          self.key("input_mix_weight_up"),
                                          tensor_parallel=False)
        self.block_inject_weight = (Linear(
            ctx, self.key("block_inject_weight"), tensor_parallel=False)
                                    if use_combine else None)

    def forward(self,
                hyper_states: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        cfg = self.cfg
        normed = self.hc_norm(hyper_states)
        # The /hc_count temperature keeps the gates off saturation at init. It
        # stays in the graph because the projections it would fold into are
        # checkpoint-resident inputs with no build-time payload.
        mix = (self.input_mix_weight_down(normed) / cfg.hc_count).silu()
        mix = self.input_mix_weight_up(mix).sigmoid()
        streams = (0, cfg.hc_count, cfg.hidden_size)
        mixed = (mix.reshape(streams) * normed.reshape(streams)).mean(1)
        if self.block_inject_weight is None:
            return mixed
        injection_weights = (self.block_inject_weight(normed) /
                             cfg.hc_count).sigmoid() * 2.0
        return mixed, injection_weights

    def combine(self, hyper_states: Tensor, block_output: Tensor,
                injection_weights: Tensor) -> Tensor:
        """Write ``block_output`` into every stream, per-stream weighted.

        The un-normalized ``hyper_states`` is what carries forward: the
        normalized view exists only to decide the gates, and the residual trunk
        has to stay an identity path.
        """
        cfg = self.cfg
        injection = (block_output.unsqueeze(1, 2) *
                     injection_weights.unsqueeze(2, 2))
        return hyper_states + injection.reshape((0, cfg.hc_hidden_size))
