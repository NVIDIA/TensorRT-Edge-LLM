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
"""Private bridge from semantic operations to the active graph backend.

Functional operation definitions pass tensors and ordinary Python attributes.
TensorRT-specific creator names and attribute encoding stay behind this
boundary, just like native TensorRT layer construction.
"""

from typing import Sequence

from ..scope import current_net
from ..tensor import Tensor


def operation_attributes(name: str) -> frozenset[str]:
    """Return attributes supported by one operation implementation."""
    return current_net().operation_attributes(name)


def parameter(name: str, value, consumer: str, *, recipe=None) -> Tensor:
    """Create an externalized parameter input for one operation.

    Without a recipe the runtime has no way to rebuild the
    tensor, so a checkpoint-backed build keeps it in the engine as a constant instead
    of declaring an input nothing can fill.
    """
    return Tensor(current_net().parameter(name, value, consumer, recipe))


def network_input(name: str, dtype, shape: Sequence[int]) -> Tensor:
    """Return the network input ``name``, declaring it on first use."""
    return Tensor(current_net().add_input(name, dtype, shape))


def operation(name: str,
              inputs: Sequence[Tensor],
              *,
              output_count: int = 1,
              **attributes):
    """Apply a semantic operation and wrap its symbolic outputs."""
    if output_count < 1:
        raise ValueError("operation output_count must be positive")
    layer = current_net().operation(name, attributes, inputs)
    result = tuple(
        Tensor(layer.get_output(index)) for index in range(output_count))
    return result[0] if output_count == 1 else result
