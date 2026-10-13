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
"""PyTorch-style modules for checkpoint-direct TensorRT graphs.

There are two module kinds:

* ``Module`` models a submodule such as attention, MLP, norm, or projector.
  These modules belong to a network context but do not declare engine inputs or
  mark engine outputs.
* ``NetworkModule`` models an engine-level component such as an LLM, ViT,
  audio encoder, action head, or vocoder. Only these modules own the I/O
  contract for a TensorRT ``INetwork``.

Modules contain other modules and ordinary configuration. TensorRT ownership
is confined to ``BuildContext`` and the active build scope; model ``forward``
methods use symbolic tensors and free operations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterator, Mapping, Sequence, Tuple

from ..core.config import DeviceConfig
from ..core.weights import Weights
from .backend import Net
from .scope import build_scope
from .tensor import Tensor


@dataclass
class BuildOptions:
    """Build-time implementation choices, matching module constructor args."""

    backend: str = "edgellm"  # edgellm | eager
    dense_quant: str = "auto"  # auto | fp16 | nvfp4-qdq
    int4_gemm_plugin_version: int = 2
    sm12x: bool = False
    sm110: bool = False
    max_lora_rank: int = 0


@dataclass
class BuildContext:
    """Backend and checkpoint state bound while a module is executing."""

    net: Net
    cfg: DeviceConfig
    weights: Weights
    options: BuildOptions
    bundle: object
    args: object

    @property
    def backend(self) -> str:
        return self.options.backend

    def with_checkpoint(self, cfg: DeviceConfig,
                        weights: Weights) -> "BuildContext":
        """Bind alternate checkpoint state without exposing graph ownership."""
        return replace(self, cfg=cfg, weights=weights)

    def open_weights(self, model_dir: str, **kwargs) -> Weights:
        """Open related checkpoint weights under the active storage policy."""
        return Weights(model_dir, policy=self.net.policy, **kwargs)


class Module:
    """Base class for graph submodules.

    Subclasses follow the PyTorch convention: construct child modules in
    ``__init__`` and implement ``forward``. ``__call__`` delegates to
    ``forward`` in the module's build scope. Submodules do not declare engine
    inputs or mark outputs.
    """

    def __init__(self, ctx: BuildContext, prefix: str = "") -> None:
        self.ctx = ctx
        self.prefix = prefix

    @property
    def cfg(self) -> DeviceConfig:
        return self.ctx.cfg

    @property
    def weights(self) -> Weights:
        return self.ctx.weights

    def key(self, suffix: str) -> str:
        return f"{self.prefix}.{suffix}" if self.prefix else suffix

    def __call__(self, *args, **kwargs):
        with build_scope(self.ctx):
            return self.forward(*args, **kwargs)

    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        """Yield immediate child modules, including flat lists and mappings."""
        for name, value in vars(self).items():
            if isinstance(value, Module):
                yield name, value
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    if isinstance(child, Module):
                        yield f"{name}.{index}", child
            elif isinstance(value, dict):
                for key, child in value.items():
                    if isinstance(child, Module):
                        yield f"{name}.{key}", child

    def __repr__(self) -> str:
        children = list(self.named_children())
        if not children:
            return f"{type(self).__name__}()"
        body = "\n".join(f"  ({name}): {child!r}" for name, child in children)
        return f"{type(self).__name__}(\n{body}\n)"

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()")


class NetworkModule(Module):
    """Engine-level module that owns a component's I/O contract."""

    @classmethod
    def from_config(cls, ctx: BuildContext):
        """Construct a component from its checkpoint and build settings.

        Text models normally need only ``ctx``. Vision, audio, action, and
        generation components can read component metadata from ``ctx.bundle``
        and profile limits from ``ctx.args``.
        """
        return cls(ctx)

    def build(self) -> None:
        """Declare this component's TensorRT inputs, graph, and outputs."""
        try:
            with build_scope(self.ctx):
                outputs = self.forward(**self.input_tensors())
                self.mark_outputs(outputs)
        finally:
            self.close()

    def input_tensors(self) -> Mapping[str, object]:
        """Declare tensors passed by name to ``forward``."""
        return {}

    def close(self) -> None:
        """Release model-owned resources after graph construction."""

    def add_input(self, name: str, dtype, shape: Sequence[int]) -> Tensor:
        return Tensor(self.ctx.net.add_input(name, dtype, shape))

    def mark_output(self, value, name: str) -> None:
        self.ctx.net.mark_output(Tensor.unwrap(value), name)

    def mark_outputs(self, outputs) -> None:
        """Mark a mapping or a sequence of explicitly named tensors."""
        if outputs is None:
            return
        if isinstance(outputs, Mapping):
            for name, value in outputs.items():
                self.mark_output(value, name)
            return
        if isinstance(outputs, Tensor):
            if not outputs.name:
                raise ValueError("unnamed network output tensor")
            self.mark_output(outputs, outputs.name)
            return
        if isinstance(outputs, (list, tuple)):
            for value in outputs:
                self.mark_outputs(value)
            return
        raise TypeError(f"{type(self).__name__}.forward must return named "
                        "tensors, a mapping, or None")
