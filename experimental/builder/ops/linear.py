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
"""Checkpoint-backed dense projection shared by model families."""

from dataclasses import replace
from typing import Optional

import numpy as np
import tensorrt as trt

from ..core import weight_policy
from . import functional as F
from .module import BuildContext, Module


class Linear(Module):
    """Apply one checkpoint linear with optional decoder TP and LoRA.

    Decoder modules use tensor parallelism by default. Encoder and action
    modules set ``tensor_parallel=False`` and use the same projection without
    decoder-specific weight sharding or collectives.
    """

    _ROW_PARALLEL = frozenset(("o_proj", "down_proj", "out_proj"))
    _COLUMN_PARALLEL = frozenset((
        "q_proj",
        "k_proj",
        "v_proj",
        "qkv_proj",
        "gate_proj",
        "up_proj",
        "gate_up_proj",
        "in_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
    ))

    def __init__(self,
                 ctx: BuildContext,
                 prefix: str,
                 rank: int = 3,
                 *,
                 tensor_parallel: bool = True,
                 tp_mode: Optional[str] = None) -> None:
        super().__init__(ctx, prefix)
        self.rank = rank
        self.tensor_parallel = tensor_parallel
        self.explicit_tp_mode = tp_mode

    def forward(self, hidden_states, rank: Optional[int] = None):
        rank = self.rank if rank is None else rank
        if not self.tensor_parallel:
            output = F.linear_from_weights(hidden_states,
                                           self.weight_descriptor(),
                                           rank,
                                           name=self.prefix)
            return self._apply_static_adapter(hidden_states, output,
                                              "replicated", rank)

        tp_mode = self._tp_mode()
        quant_type = self.quant_type()
        # The fused CuTeDSL kernels currently target SM100/101/103/110. SM12x
        # uses TensorRT's NVFP4 Q/DQ matmul followed by the generic all-reduce.
        use_fused_nvfp4_tp = (tp_mode == "row" and self.cfg.tp_size > 1
                              and quant_type == "nvfp4"
                              and not self.has_adapter()
                              and not self.ctx.options.sm12x)
        full_descriptor = self.weights.linear_descriptor(
            self.prefix,
            quant_type,
            external_kind=(weight_policy.EXTERNAL_WEIGHT_NVFP4_TP
                           if use_fused_nvfp4_tp else ""),
        )
        descriptor = self.weights.shard_linear(full_descriptor, tp_mode,
                                               self.cfg.tp_size,
                                               self.cfg.tp_rank)
        if use_fused_nvfp4_tp:
            sharded_weights = replace(descriptor, bias=None, bias_recipe=None)
            return F.fused_nvfp4_gemm_all_reduce(
                hidden_states,
                sharded_weights,
                full_descriptor.bias,
                self.cfg.tp_size,
                rank,
                name=self.prefix,
                bias_recipe=full_descriptor.bias_recipe)
        output = F.linear_from_weights(hidden_states,
                                       descriptor,
                                       rank,
                                       name=self.prefix)
        output = self._apply_static_adapter(hidden_states, output, tp_mode,
                                            rank)
        if self._uses_lora():
            output = F.dynamic_lora(hidden_states, output, self.prefix,
                                    descriptor.in_features,
                                    descriptor.out_features)
        if tp_mode == "row" and self.cfg.tp_size > 1:
            output = F.all_reduce(output, self.cfg.tp_size)
        return output

    def forward_f32(self, hidden_states, rank: Optional[int] = None):
        """Apply an FP16 checkpoint projection with FP32 accumulation."""
        if self.has_adapter():
            raise ValueError(
                f"{self.prefix}: FP32 projection does not support adapters")
        rank = self.rank if rank is None else rank
        output = F.linear_f32_from_weights(hidden_states,
                                           self.weight_descriptor(),
                                           self.prefix, rank)
        if self._tp_mode() == "row" and self.cfg.tp_size > 1:
            output = F.all_reduce(output, self.cfg.tp_size)
        return output

    def quant_type(self) -> str:
        """Return the compiled precision selected for this projection."""
        mode = self.ctx.options.dense_quant
        if mode == "fp16":
            return "fp16"
        if mode == "nvfp4-qdq" and self.weights.is_nvfp4(self.prefix):
            return "nvfp4"
        return self.weights.module_quant_type(
            self.prefix,
            tie_word_embeddings=bool(
                getattr(self.cfg, "tie_word_embeddings", False)))

    def weight_descriptor(self):
        """Load and tensor-parallel shard the base projection weights."""
        descriptor = self.weights.linear_descriptor(self.prefix,
                                                    self.quant_type())
        if not self.tensor_parallel:
            return descriptor
        tp_mode = self._tp_mode()
        return self.weights.shard_linear(descriptor, tp_mode, self.cfg.tp_size,
                                         self.cfg.tp_rank)

    def has_adapter(self) -> bool:
        """Whether this projection needs model or runtime LoRA handling."""
        return (self.weights.linear_adapter(self.prefix) is not None
                or self._uses_lora())

    def _apply_static_adapter(self, hidden_states, output, tp_mode: str,
                              rank: int):
        adapter = self.weights.linear_adapter(self.prefix)
        if adapter is None:
            return output
        adapter_a, adapter_b, scale = adapter
        if self.tensor_parallel and self.cfg.tp_size > 1:
            if tp_mode == "column":
                adapter_b = np.split(adapter_b, self.cfg.tp_size,
                                     axis=0)[self.cfg.tp_rank]
            elif tp_mode == "row":
                adapter_a = np.split(adapter_a, self.cfg.tp_size,
                                     axis=1)[self.cfg.tp_rank]
        low_rank = F.linear_f32(hidden_states,
                                np.ascontiguousarray(adapter_a),
                                rank=rank)
        residual = F.linear_f32(low_rank,
                                np.ascontiguousarray(adapter_b),
                                rank=rank)
        promoted = output.cast(trt.float32) + residual * np.float32(scale)
        return promoted.cast(output.dtype)

    def _tp_mode(self) -> str:
        if not self.tensor_parallel:
            return "replicated"
        if self.explicit_tp_mode is not None:
            return self.explicit_tp_mode
        if self.cfg.tp_size == 1:
            return "replicated"
        suffix = self.prefix.rsplit(".", 1)[-1]
        if suffix in self._ROW_PARALLEL:
            return "row"
        if suffix in self._COLUMN_PARALLEL:
            return "column"
        return "replicated"

    def _uses_lora(self) -> bool:
        return (self.ctx.options.max_lora_rank > 0
                and self.prefix not in ("lm_head", "model.embed_tokens")
                and not self.prefix.endswith("embed_tokens"))


class DynamicLinear(Module):
    """Apply a runtime-provided output matrix instead of checkpoint weights."""

    def __init__(self, ctx: BuildContext, in_features: int) -> None:
        super().__init__(ctx)
        self.in_features = in_features

    def forward(self, hidden_states, weight):
        weight = weight.reshape((1, -1, self.in_features))
        return hidden_states.matmul(weight,
                                    rhs_op=trt.MatrixOperation.TRANSPOSE)
