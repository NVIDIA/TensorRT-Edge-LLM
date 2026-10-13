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
"""Input declarations for the unified token-major decoder ABI."""

from dataclasses import dataclass
from typing import Any, Callable, Dict

import tensorrt as trt


@dataclass(frozen=True)
class RaggedDecoderInputs:
    """Runtime-owned decoder metadata with an optional logits selector."""

    positions: Any
    query_start_offsets: Any
    query_lengths: Any
    past_lengths: Any
    attention_sequence_lengths: Any
    state_indices: Any
    logits_indices: Any
    execution_phase_marker: Any
    context_sequence_count_carrier: Any
    kv_page_table: Any

    def as_dict(self) -> Dict[str, Any]:
        return {
            name: value
            for name, value in vars(self).items() if value is not None
        }

    @classmethod
    def from_dict(cls, inputs: Dict[str, Any]) -> "RaggedDecoderInputs":
        """Reconstruct the bundle from flattened network inputs."""
        values = {
            name: inputs[name]
            for name in cls.__dataclass_fields__ if name != "logits_indices"
        }
        values["logits_indices"] = inputs.get("logits_indices")
        return cls(**values)


def add_ragged_decoder_inputs(
    add_input: Callable[[str, object, tuple], Any],
    *,
    include_logits_indices: bool = True,
) -> RaggedDecoderInputs:
    """Declare decoder bindings through ``NetworkModule.add_input``."""

    def int32_vector(name):
        return add_input(name, trt.int32, (-1, ))

    return RaggedDecoderInputs(
        positions=int32_vector("positions"),
        query_start_offsets=int32_vector("query_start_offsets"),
        query_lengths=int32_vector("query_lengths"),
        past_lengths=int32_vector("past_lengths"),
        attention_sequence_lengths=int32_vector("attention_sequence_lengths"),
        state_indices=int32_vector("state_indices"),
        logits_indices=(add_input("logits_indices", trt.int64,
                                  (-1, )) if include_logits_indices else None),
        execution_phase_marker=int32_vector("execution_phase_marker"),
        context_sequence_count_carrier=add_input(
            "context_sequence_count_carrier", trt.int32, (-1, )),
        kv_page_table=add_input("kv_page_table", trt.int32, (-1, 2, -1)),
    )
