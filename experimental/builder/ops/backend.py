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
"""Internal TensorRT lowering backend for symbolic operations.

A thin, stateful wrapper over ``trt.INetworkDefinition`` providing the
lowerings used by the public ``Tensor`` and ``functional`` APIs. Public model
definitions do not know whether a lowering is a TensorRT-native layer or an
Edge-LLM extension operation, and never receive this backend object.
"""

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import tensorrt as trt

from ..core import quantization
from ..core.weight_policy import WeightPolicy
from ..core.weights import CheckpointParameter, ParameterSpec

__all__ = ["Net"]

_NP_TO_TRT = {
    np.dtype(np.float16): trt.float16,
    np.dtype(np.float32): trt.float32,
    np.dtype(np.int8): trt.int8,
    np.dtype(np.int32): trt.int32,
    np.dtype(np.int64): trt.int64,
    np.dtype(np.uint8): trt.uint8,
    np.dtype(np.bool_): trt.bool,
}

# Tags the runtime's safetensors reader understands, for binding metadata.
_SAFETENSORS_DTYPE = {
    np.dtype(np.float16): "F16",
    np.dtype(np.float32): "F32",
    np.dtype(np.int8): "I8",
    np.dtype(np.int32): "I32",
    np.dtype(np.int64): "I64",
    np.dtype(np.uint8): "U8",
    np.dtype(np.bool_): "BOOL",
}

_PLUGIN_RESOURCE_DTYPES = frozenset(("U8", "F8_E4M3"))

_OPERATION_CREATORS = {
    "all_reduce": "AllReducePlugin",
    "fused_nvfp4_gemm_all_reduce": "FusedNvfp4GemmAllReducePlugin",
    "attention": "AttentionPlugin",
    "dflash_target_cache_update": "DFlashTargetKVCacheUpdate",
    "gemma4_attention": "Gemma4AudioAttentionPlugin",
    "fp16_moe": "Fp16MoePlugin",
    "int4_groupwise_gemm": "Int4GroupwiseGemmPlugin",
    "int4_groupwise_gemm_v2": "Int4GroupwiseGemmPluginV2",
    "int4_moe": "Int4MoePlugin",
    "nvfp4_moe": "Nvfp4MoePlugin",
    "nvfp4_moe_sm12x": "NvFP4MoEPluginGeforce",
    "vit_attention": "ViTAttentionPlugin",
}


def _e8m0_dtype():
    """Return TensorRT's E8M0 scale type across Python API spellings."""
    for name in ("E8M0", "UE8M0"):
        if hasattr(trt.DataType, name):
            return getattr(trt.DataType, name)
    raise RuntimeError("MXFP8 requires TensorRT E8M0 support")


class Net:
    """Lower symbolic operations into one ``INetworkDefinition``."""

    def __init__(self,
                 builder: "trt.Builder",
                 network: "trt.INetworkDefinition",
                 policy: Optional[WeightPolicy] = None,
                 int4_gemm_plugin_version: int = 2) -> None:
        self.builder = builder
        self.network = network
        self.policy = policy or WeightPolicy()
        if int4_gemm_plugin_version not in (1, 2):
            raise ValueError("int4_gemm_plugin_version must be 1 or 2")
        self.int4_gemm_plugin_version = int4_gemm_plugin_version
        self._weight_refs: List[np.ndarray] = []  # keep numpy alive for TRT
        self._inputs = {}
        self._weight_bindings: Dict[str, Dict[str, object]] = {}
        self._next_plugin_resource_id = 0
        self._n = 0

    # -- low level ----------------------------------------------------------

    def _name(self, base: str) -> str:
        self._n += 1
        return f"{base}_{self._n}"

    @staticmethod
    def _unwrap(value):
        unwrap = getattr(value, "_as_trt", None)
        return unwrap() if unwrap is not None else value

    def const(self,
              arr: np.ndarray,
              name: Optional[str] = None) -> "trt.ITensor":
        arr = np.ascontiguousarray(arr)
        if arr.dtype not in _NP_TO_TRT:
            raise TypeError(f"unsupported constant dtype {arr.dtype}")
        self._weight_refs.append(arr)  # TRT references the buffer; keep alive
        layer = self.network.add_constant(tuple(int(d) for d in arr.shape),
                                          trt.Weights(arr))
        if name:
            layer.name = self._name(name)
        return layer.get_output(0)

    def add_input(self, name: str, dtype: "trt.DataType",
                  shape: Sequence[int]) -> "trt.ITensor":
        if name in self._inputs:
            return self._inputs[name]
        tensor = self.network.add_input(name, dtype, tuple(shape))
        self._inputs[name] = tensor
        return tensor

    def weight_input(
            self,
            name: str,
            value,
            kind: str,
            *,
            recipe: Optional[Mapping[str, object]] = None) -> "trt.ITensor":
        """Declare one static engine input the runtime fills at load time.

        The payload is represented by final TensorRT metadata and a checkpoint
        transform recipe; it is never read while the engine is built.
        """
        metadata_only = isinstance(value, ParameterSpec)
        if self.policy.externalizes_parameter(kind,
                                              name) and not metadata_only:
            raise ValueError(
                f"external parameter {name!r} was materialized during engine "
                "build; model weight loading must return ParameterSpec")
        if metadata_only:
            shape = value.shape
            numpy_dtype = value.dtype
        else:
            value = np.ascontiguousarray(value)
            shape = value.shape
            numpy_dtype = value.dtype
        dtype = _NP_TO_TRT.get(numpy_dtype)
        if dtype is None:
            raise TypeError(f"unsupported external weight dtype {numpy_dtype}")
        if recipe is None:
            raise ValueError(
                f"checkpoint-backed input {name!r} has no load recipe")
        self._record_binding(name, shape, numpy_dtype, recipe)
        return self.add_input(name, dtype, shape)

    def parameter(
            self,
            name: str,
            value,
            kind: str,
            recipe: Optional[Mapping[str, object]] = None) -> "trt.ITensor":
        """Create a runtime weight input, or bake unreproducible layouts."""
        if not self.policy.externalizes_parameter(kind, name):
            if isinstance(value, ParameterSpec):
                raise ValueError(
                    f"constant parameter {name!r} has metadata but no payload")
            return self.const(value, name.rsplit(".", 1)[-1])
        if recipe is None:
            if isinstance(value, ParameterSpec):
                raise ValueError(
                    f"external parameter {name!r} has no transform recipe")
            return self.const(value, name.rsplit(".", 1)[-1])
        return self.weight_input(name, value, kind, recipe=recipe)

    def plugin_resource(self, name: str, value: ParameterSpec, kind: str,
                        recipe: Mapping[str, object], *, resource_id: int,
                        resource_kind: str, storage_dtype: str) -> None:
        """Record a persistent plugin resource without an engine binding."""
        if not self.policy.externalizes_parameter(kind, name):
            raise ValueError(f"plugin resource {name!r} is not externalized")
        if not isinstance(value, ParameterSpec):
            raise ValueError(
                f"plugin resource {name!r} was materialized during engine build"
            )
        if storage_dtype not in _PLUGIN_RESOURCE_DTYPES:
            raise TypeError(f"unsupported plugin resource storage dtype "
                            f"{storage_dtype}")
        resource_recipe = dict(recipe)
        extra = dict(resource_recipe.get("extra", {}))
        extra.update({
            "plugin_resource_id": int(resource_id),
            "plugin_resource_kind": resource_kind,
        })
        resource_recipe["extra"] = extra
        self._record_binding(name, value.shape, storage_dtype, resource_recipe)

    def _record_binding(self, name: str, shape: Sequence[int], dtype,
                        recipe: Mapping[str, object]) -> None:
        checkpoint_keys = recipe.get("checkpoint_keys")
        assemble = recipe.get("assemble")
        if not checkpoint_keys and not assemble:
            raise ValueError(
                f"engine input {name!r} has no checkpoint keys, so the runtime "
                "could not fill it; give it a recipe or emit it as a constant")
        unknown = set(recipe).difference(
            ("checkpoint_keys", "source_layout", "assemble", "extra"))
        if unknown:
            raise ValueError(f"unknown checkpoint recipe fields: {unknown}")
        storage_dtype = (dtype if isinstance(dtype, str) else
                         _SAFETENSORS_DTYPE[np.dtype(dtype)])
        binding: Dict[str, object] = {
            "engine_name": name,
            "checkpoint_keys": list(checkpoint_keys or ()),
            "source_layout": recipe.get("source_layout", "plugin"),
            "dtype": storage_dtype,
            "shape": [int(dim) for dim in shape],
        }
        if assemble:
            binding["assemble"] = assemble
        extra = recipe.get("extra")
        if extra:
            if not isinstance(extra, Mapping):
                raise TypeError("checkpoint recipe extra must be a mapping")
            binding.update(extra)
        previous = self._weight_bindings.get(name)
        if previous is not None and previous != binding:
            raise ValueError(f"conflicting checkpoint binding {name!r}")
        self._weight_bindings.setdefault(name, binding)

    def take_weight_bindings(self) -> List[dict]:
        """Transfer checkpoint weight bindings to the artifact writer."""
        bindings = list(self._weight_bindings.values())
        self._weight_bindings = {}
        return bindings

    def mark_output(self,
                    tensor: "trt.ITensor",
                    name: str,
                    dtype: Optional["trt.DataType"] = None) -> None:
        tensor = self._unwrap(tensor)
        tensor.name = name
        self.network.mark_output(tensor)
        if dtype is not None:
            tensor.dtype = dtype

    # -- shape ops ----------------------------------------------------------

    def reshape(self, x: "trt.ITensor", shape: Sequence[int]) -> "trt.ITensor":
        x = self._unwrap(x)
        layer = self.network.add_shuffle(x)
        layer.reshape_dims = tuple(int(d) for d in shape)
        return layer.get_output(0)

    def dynamic_reshape(self, x: "trt.ITensor",
                        shape: "trt.ITensor") -> "trt.ITensor":
        """Reshape ``x`` using a runtime shape tensor."""
        layer = self.network.add_shuffle(self._unwrap(x))
        layer.set_input(1, self._unwrap(shape))
        return layer.get_output(0)

    def unsqueeze(self, x: "trt.ITensor", axis: int,
                  rank: int) -> "trt.ITensor":
        """Insert one unit dimension into a tensor with dynamic extents."""
        x = self._unwrap(x)
        normalized_axis = axis if axis >= 0 else axis + rank + 1
        if not 0 <= normalized_axis <= rank:
            raise ValueError(
                f"unsqueeze axis {axis} is invalid for rank {rank}")
        shape = self.cast(self.network.add_shape(x).get_output(0), trt.int32)
        pieces = []
        if normalized_axis:
            pieces.append(
                self.network.add_slice(shape, (0, ), (normalized_axis, ),
                                       (1, )).get_output(0))
        pieces.append(self.const(np.array([1], dtype=np.int32), "unit_dim"))
        if normalized_axis < rank:
            pieces.append(
                self.network.add_slice(shape, (normalized_axis, ),
                                       (rank - normalized_axis, ),
                                       (1, )).get_output(0))
        layer = self.network.add_shuffle(x)
        layer.set_input(1, self.concat(pieces, 0))
        return layer.get_output(0)

    def squeeze(self, x: "trt.ITensor", axis: int, rank: int) -> "trt.ITensor":
        """Remove one static unit dimension while preserving dynamic extents."""
        x = self._unwrap(x)
        normalized_axis = axis if axis >= 0 else axis + rank
        if not 0 <= normalized_axis < rank:
            raise ValueError(f"squeeze axis {axis} is invalid for rank {rank}")
        if int(x.shape[normalized_axis]) != 1:
            raise ValueError("squeeze requires a static unit dimension")
        if rank == 1:
            layer = self.network.add_shuffle(x)
            layer.reshape_dims = ()
            return layer.get_output(0)
        shape = self.cast(self.network.add_shape(x).get_output(0), trt.int32)
        pieces = []
        if normalized_axis:
            pieces.append(
                self.network.add_slice(shape, (0, ), (normalized_axis, ),
                                       (1, )).get_output(0))
        trailing = rank - normalized_axis - 1
        if trailing:
            pieces.append(
                self.network.add_slice(shape, (normalized_axis + 1, ),
                                       (trailing, ), (1, )).get_output(0))
        layer = self.network.add_shuffle(x)
        layer.set_input(1, self.concat(pieces, 0))
        return layer.get_output(0)

    def transpose(self, x: "trt.ITensor",
                  permutation: Sequence[int]) -> "trt.ITensor":
        """Permute tensor axes without changing element type."""
        x = self._unwrap(x)
        layer = self.network.add_shuffle(x)
        layer.second_transpose = tuple(int(axis) for axis in permutation)
        return layer.get_output(0)

    def cast(self, x: "trt.ITensor", dtype: "trt.DataType") -> "trt.ITensor":
        x = self._unwrap(x)
        layer = self.network.add_cast(x, dtype)
        return layer.get_output(0)

    def elementwise(self, a: "trt.ITensor", b: "trt.ITensor",
                    op: "trt.ElementWiseOperation") -> "trt.ITensor":
        a, b = self._unwrap(a), self._unwrap(b)
        return self.network.add_elementwise(a, b, op).get_output(0)

    def matmul(
        self,
        lhs: "trt.ITensor",
        rhs: "trt.ITensor",
        lhs_op: "trt.MatrixOperation" = trt.MatrixOperation.NONE,
        rhs_op: "trt.MatrixOperation" = trt.MatrixOperation.NONE
    ) -> "trt.ITensor":
        """Lower a symbolic matrix multiplication."""
        return self.network.add_matrix_multiply(self._unwrap(lhs), lhs_op,
                                                self._unwrap(rhs),
                                                rhs_op).get_output(0)

    def reduce(self,
               x: "trt.ITensor",
               op: "trt.ReduceOperation",
               axes: int,
               keep_dims: bool = False) -> "trt.ITensor":
        x = self._unwrap(x)
        return self.network.add_reduce(x, op, axes, keep_dims).get_output(0)

    def unary(self, x: "trt.ITensor",
              op: "trt.UnaryOperation") -> "trt.ITensor":
        x = self._unwrap(x)
        return self.network.add_unary(x, op).get_output(0)

    def activation(self, x: "trt.ITensor",
                   act: "trt.ActivationType") -> "trt.ITensor":
        x = self._unwrap(x)
        return self.network.add_activation(x, act).get_output(0)

    def softmax(self, x: "trt.ITensor", axis: int) -> "trt.ITensor":
        x = self._unwrap(x)
        layer = self.network.add_softmax(x)
        layer.axes = 1 << axis
        return layer.get_output(0)

    def log_softmax(self, x: "trt.ITensor", axis: int) -> "trt.ITensor":
        """Numerically stable softmax followed by logarithm."""
        return self.unary(self.softmax(x, axis), trt.UnaryOperation.LOG)

    def concat(self, tensors: Sequence["trt.ITensor"],
               axis: int) -> "trt.ITensor":
        layer = self.network.add_concatenation(
            [self._unwrap(tensor) for tensor in tensors])
        layer.axis = axis
        return layer.get_output(0)

    def silu(self, x: "trt.ITensor") -> "trt.ITensor":
        s = self.activation(x, trt.ActivationType.SIGMOID)
        return self.elementwise(x, s, trt.ElementWiseOperation.PROD)

    def relu(self, x: "trt.ITensor") -> "trt.ITensor":
        return self.activation(x, trt.ActivationType.RELU)

    def tanh(self, x: "trt.ITensor") -> "trt.ITensor":
        return self.activation(x, trt.ActivationType.TANH)

    def gelu(self, x: "trt.ITensor") -> "trt.ITensor":
        """Exact GELU represented with elementwise operations."""
        x = self._unwrap(x)
        output_dtype = x.dtype
        x32 = self.cast(x, trt.float32)
        scalar_shape = (1, ) * len(x.shape)
        sqrt2 = self.const(
            np.array(np.sqrt(2.0), dtype=np.float32).reshape(scalar_shape),
            "sqrt2")
        half = self.const(
            np.array(0.5, dtype=np.float32).reshape(scalar_shape), "half")
        one = self.const(
            np.array(1.0, dtype=np.float32).reshape(scalar_shape), "one")
        scaled = self.elementwise(x32, sqrt2, trt.ElementWiseOperation.DIV)
        erf = self.unary(scaled, trt.UnaryOperation.ERF)
        factor = self.elementwise(one, erf, trt.ElementWiseOperation.SUM)
        factor = self.elementwise(factor, half, trt.ElementWiseOperation.PROD)
        return self.cast(
            self.elementwise(x32, factor, trt.ElementWiseOperation.PROD),
            output_dtype)

    def gelu_tanh(self, x: "trt.ITensor") -> "trt.ITensor":
        """Tanh-approximate GELU with rank-matched scalar constants."""
        x = self._unwrap(x)
        output_dtype = x.dtype
        x32 = self.cast(x, trt.float32)
        scalar_shape = (1, ) * len(x.shape)

        def scalar(value: float, name: str) -> "trt.ITensor":
            data = np.array(value, dtype=np.float32).reshape(scalar_shape)
            return self.const(data, name)

        half = scalar(0.5, "gelu_half")
        one = scalar(1.0, "gelu_one")
        cubic = scalar(0.044715, "gelu_cubic")
        scale = scalar(np.sqrt(2.0 / np.pi), "gelu_scale")
        squared = self.elementwise(x32, x32, trt.ElementWiseOperation.PROD)
        cubed = self.elementwise(squared, x32, trt.ElementWiseOperation.PROD)
        inner = self.elementwise(
            x32, self.elementwise(cubic, cubed, trt.ElementWiseOperation.PROD),
            trt.ElementWiseOperation.SUM)
        tanh = self.tanh(
            self.elementwise(scale, inner, trt.ElementWiseOperation.PROD))
        factor = self.elementwise(one, tanh, trt.ElementWiseOperation.SUM)
        result = self.elementwise(
            self.elementwise(half, x32, trt.ElementWiseOperation.PROD), factor,
            trt.ElementWiseOperation.PROD)
        return self.cast(result, output_dtype)

    def gather(self, x: "trt.ITensor", indices: np.ndarray,
               axis: int) -> "trt.ITensor":
        """Gather a constant index vector along one axis."""
        x = self._unwrap(x)
        index_tensor = self.const(np.asarray(indices, dtype=np.int64),
                                  "indices")
        return self.network.add_gather(x, index_tensor, axis).get_output(0)

    def gather_tensor(self, x: "trt.ITensor", indices: "trt.ITensor",
                      axis: int) -> "trt.ITensor":
        """Gather runtime indices along one axis."""
        x, indices = self._unwrap(x), self._unwrap(indices)
        return self.network.add_gather(x, indices, axis).get_output(0)

    def gather_nd(self,
                  x: "trt.ITensor",
                  indices: "trt.ITensor",
                  num_elementwise_dims: int = 0) -> "trt.ITensor":
        """Gather runtime indices with TensorRT's ND semantics."""
        layer = self.network.add_gather_v2(self._unwrap(x),
                                           self._unwrap(indices),
                                           trt.GatherMode.ND)
        layer.num_elementwise_dims = num_elementwise_dims
        return layer.get_output(0)

    def shape_of(self, x: "trt.ITensor") -> "trt.ITensor":
        """Return the runtime shape vector as INT32."""
        shape = self.network.add_shape(self._unwrap(x)).get_output(0)
        return self.cast(shape, trt.int32)

    def dynamic_slice(self, x: "trt.ITensor", start: "trt.ITensor",
                      size: "trt.ITensor",
                      stride: Sequence[int]) -> "trt.ITensor":
        """Slice ``x`` using runtime start and size vectors."""
        rank = len(stride)
        layer = self.network.add_slice(self._unwrap(x), (0, ) * rank,
                                       (0, ) * rank, tuple(stride))
        layer.set_input(1, self._unwrap(start))
        layer.set_input(2, self._unwrap(size))
        return layer.get_output(0)

    def slice_last_dim(self, x: "trt.ITensor", offset: int, size: int,
                       rank: int) -> "trt.ITensor":
        """Slice ``x[..., offset:offset+size]`` keeping dynamic leading dims."""
        x = self._unwrap(x)
        start = tuple([0] * (rank - 1) + [offset])
        stride = tuple([1] * rank)
        layer = self.network.add_slice(x, start, tuple([0] * rank), stride)
        shp = self.network.add_shape(x).get_output(0)
        shp32 = self.cast(shp, trt.int32)
        lead = self.network.add_slice(shp32, (0, ), (rank - 1, ),
                                      (1, )).get_output(0)
        size_c = self.const(np.array([size], dtype=np.int32))
        full = self.concat([lead, size_c], 0)
        layer.set_input(2, full)
        return layer.get_output(0)

    def slice_axis(self, x: "trt.ITensor", axis: int, offset: int, size: int,
                   rank: int) -> "trt.ITensor":
        """Slice one axis while copying all dynamic extents from the input."""
        x = self._unwrap(x)
        start = [0] * rank
        start[axis] = offset
        layer = self.network.add_slice(x, tuple(start), tuple([0] * rank),
                                       tuple([1] * rank))
        shape = self.cast(self.network.add_shape(x).get_output(0), trt.int32)
        before = (self.network.add_slice(
            shape, (0, ), (axis, ), (1, )).get_output(0) if axis else None)
        selected = self.const(np.array([size], dtype=np.int32), "slice_size")
        after_count = rank - axis - 1
        after = (self.network.add_slice(shape, (axis + 1, ), (after_count, ),
                                        (1, )).get_output(0)
                 if after_count else None)
        pieces = [
            piece for piece in (before, selected, after) if piece is not None
        ]
        layer.set_input(2, self.concat(pieces, 0))
        return layer.get_output(0)

    def slice_lead_dims(self, x: "trt.ITensor", size: int,
                        rank: int) -> "trt.ITensor":
        """Slice ``x[..., :size]`` along the last axis (alias of slice_last_dim)."""
        return self.slice_last_dim(x, 0, size, rank)

    def pad_last_dim(self, x: "trt.ITensor", padding: int,
                     rank: int) -> "trt.ITensor":
        """Append dynamic-shape zeros to the last dimension."""
        x = self._unwrap(x)
        if padding <= 0:
            return x
        source = self.slice_last_dim(x, 0, padding, rank)
        zero = self.const(np.zeros((1, ) * rank, dtype=np.float16), "zero")
        zeros = self.elementwise(source, zero, trt.ElementWiseOperation.PROD)
        return self.concat((x, zeros), rank - 1)

    def empty_sequence(self, x: "trt.ITensor", last_dim: int) -> "trt.ITensor":
        """Create a ``[B, 0, last_dim]`` view carrying x's dynamic batch."""
        x = self._unwrap(x)
        layer = self.network.add_slice(x, (0, 0, 0), (0, 0, last_dim),
                                       (1, 1, 1))
        source_shape = self.cast(
            self.network.add_shape(x).get_output(0), trt.int32)
        batch = self.network.add_slice(source_shape, (0, ), (1, ),
                                       (1, )).get_output(0)
        empty_shape = self.concat(
            (batch, self.const(np.array([0, last_dim], dtype=np.int32))), 0)
        layer.set_input(2, empty_shape)
        return layer.get_output(0)

    def topk(self, x: "trt.ITensor", k: int,
             axis: int) -> tuple["trt.ITensor", "trt.ITensor"]:
        x = self._unwrap(x)
        layer = self.network.add_topk(x, trt.TopKOperation.MAX, k, 1 << axis)
        return layer.get_output(0), layer.get_output(1)

    def select(self, condition: "trt.ITensor", when_true: "trt.ITensor",
               when_false: "trt.ITensor") -> "trt.ITensor":
        condition = self._unwrap(condition)
        when_true = self._unwrap(when_true)
        when_false = self._unwrap(when_false)
        return self.network.add_select(condition, when_true,
                                       when_false).get_output(0)

    # -- TensorRT attention -------------------------------------------------

    def rotary_embedding(self,
                         x: "trt.ITensor",
                         cos_cache: "trt.ITensor",
                         sin_cache: "trt.ITensor",
                         position_ids: "trt.ITensor",
                         rotary_dim: int,
                         interleaved: bool = False) -> "trt.ITensor":
        """Lower rotary embedding."""
        layer = self.network.add_rotary_embedding(self._unwrap(x),
                                                  self._unwrap(cos_cache),
                                                  self._unwrap(sin_cache),
                                                  interleaved, rotary_dim)
        layer.set_input(3, self._unwrap(position_ids))
        return layer.get_output(0)

    def kv_cache_update(self, cache: "trt.ITensor", update: "trt.ITensor",
                        write_indices: "trt.ITensor") -> "trt.ITensor":
        """Update a linear TensorRT KV cache at per-request positions."""
        layer = self.network.add_kv_cache_update(self._unwrap(cache),
                                                 self._unwrap(update),
                                                 self._unwrap(write_indices),
                                                 trt.KVCacheMode.LINEAR)
        return layer.get_output(0)

    def scaled_dot_product_attention(
        self,
        query: "trt.ITensor",
        key: "trt.ITensor",
        value: "trt.ITensor",
        mask: Optional["trt.ITensor"] = None,
        key_value_lengths: Optional["trt.ITensor"] = None,
        scale: Optional[float] = None,
        is_causal: bool = False,
    ) -> "trt.ITensor":
        """Apply TensorRT scaled dot-product attention."""
        query = self._unwrap(query)
        if scale is None:
            head_size = int(query.shape[-1])
            if head_size <= 0:
                raise ValueError(
                    "attention scale must be explicit for a dynamic head size")
            scale = head_size**-0.5
        if scale != 1.0:
            if query.dtype != trt.float16:
                raise TypeError(
                    "scaled attention currently requires FP16 inputs")
            scalar = self.const(
                np.array(scale, dtype=np.float16).reshape(
                    (1, ) * len(query.shape)), "attention_scale")
            query = self.elementwise(query, scalar,
                                     trt.ElementWiseOperation.PROD)
        key = self._unwrap(key)
        value = self._unwrap(value)
        if hasattr(self.network, "add_attention_v2"):
            causal_mask = (trt.CausalMaskKind.UPPER_LEFT
                           if is_causal else trt.CausalMaskKind.NONE)
            layer = self.network.add_attention_v2(
                query, key, value, trt.AttentionNormalizationOp.SOFTMAX,
                causal_mask)
        else:
            layer = self.network.add_attention(
                query, key, value, trt.AttentionNormalizationOp.SOFTMAX,
                is_causal)
        layer.decomposable = True
        if mask is not None:
            layer.mask = self._unwrap(mask)
        if key_value_lengths is not None:
            key_value_lengths = self._unwrap(key_value_lengths)
            if hasattr(layer, "key_value_lengths"):
                layer.key_value_lengths = key_value_lengths
            else:
                if mask is not None:
                    raise ValueError(
                        "this variable-length attention lowering cannot also "
                        "accept an explicit mask")
                capacity = int(key.shape[-2])
                if capacity <= 0:
                    raise ValueError(
                        "this variable-length attention lowering requires a "
                        "static key/value capacity")
                positions = self.const(
                    np.arange(capacity,
                              dtype=np.int32).reshape(1, 1, 1, capacity),
                    "attention_positions")
                lengths = self.reshape(key_value_lengths, (-1, 1, 1, 1))
                valid = self.elementwise(positions, lengths,
                                         trt.ElementWiseOperation.LESS)
                allowed = self.const(np.zeros((1, 1, 1, 1), dtype=np.float16),
                                     "attention_allowed")
                blocked = self.const(
                    np.full((1, 1, 1, 1), -65504.0, dtype=np.float16),
                    "attention_blocked")
                layer.mask = self.select(valid, allowed, blocked)
        return layer.get_output(0)

    # -- linear / matmul ----------------------------------------------------

    def linear(self,
               x: "trt.ITensor",
               weight: np.ndarray,
               bias: Optional[np.ndarray] = None,
               rank: int = 3) -> "trt.ITensor":
        """FP16 linear: ``y = x @ W^T (+ bias)`` with ``W`` of shape ``[out, in]``.

        TRT matmul requires both operands to have the same rank, so the weight
        is reshaped with leading 1s (``[1,...,out,in]``) to match ``x``.
        """
        x = self._unwrap(x)
        w16 = weight.astype(np.float16)
        wshape = [1] * (rank - 2) + list(w16.shape)
        w = self.const(np.ascontiguousarray(w16.reshape(wshape)), "w")
        out = self.network.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, w,
            trt.MatrixOperation.TRANSPOSE).get_output(0)
        if bias is not None:
            bshape = [1] * (rank - 1) + [int(bias.shape[0])]
            b = self.const(bias.astype(np.float16).reshape(bshape), "b")
            out = self.elementwise(out, b, trt.ElementWiseOperation.SUM)
        return out

    def external_fp16_linear(self,
                             x: "trt.ITensor",
                             linear_weights,
                             name: str,
                             rank: int = 3) -> "trt.ITensor":
        """FP16 linear whose ``[out, in]`` weight is a runtime engine input."""
        tensor = self.weight_input(name + ".weight",
                                   linear_weights.weight,
                                   "fp16",
                                   recipe=linear_weights.weight_recipe)
        if rank > 2:
            tensor = self.reshape(tensor, [1] * (rank - 2) +
                                  list(linear_weights.weight.shape))
        output = self.matmul(x, tensor, trt.MatrixOperation.NONE,
                             trt.MatrixOperation.TRANSPOSE)
        return self._add_bias(output,
                              linear_weights.bias,
                              rank,
                              name=name,
                              recipe=linear_weights.bias_recipe)

    def linear_f32(self,
                   x: "trt.ITensor",
                   weight: np.ndarray,
                   bias: Optional[np.ndarray] = None,
                   rank: int = 3) -> "trt.ITensor":
        """FP32 linear used where the pre-normalization accumulation is wide."""
        x = self._unwrap(x)
        x32 = self.cast(x, trt.float32)
        weight = np.ascontiguousarray(weight, dtype=np.float32)
        shape = [1] * (rank - 2) + list(weight.shape)
        constant = self.const(weight.reshape(shape), "w32")
        output = self.network.add_matrix_multiply(
            x32, trt.MatrixOperation.NONE, constant,
            trt.MatrixOperation.TRANSPOSE).get_output(0)
        if bias is not None:
            bias_shape = [1] * (rank - 1) + [int(bias.shape[0])]
            bias_tensor = self.const(
                np.asarray(bias, dtype=np.float32).reshape(bias_shape), "b32")
            output = self.elementwise(output, bias_tensor,
                                      trt.ElementWiseOperation.SUM)
        return output

    def linear_f32_from_weights(self,
                                x: "trt.ITensor",
                                linear_weights,
                                name: str,
                                rank: int = 3) -> "trt.ITensor":
        """FP32 accumulation with an FP16 checkpoint-backed parameter."""
        if linear_weights.quant_type != quantization.QUANT_FP16:
            raise ValueError("FP32 accumulation requires an FP16 projection")
        x32 = self.cast(self._unwrap(x), trt.float32)
        if linear_weights.weight_recipe and self.policy.externalizes_fp16(
                name):
            weight = self.weight_input(name + ".weight",
                                       linear_weights.weight,
                                       "fp16",
                                       recipe=linear_weights.weight_recipe)
            weight = self.cast(weight, trt.float32)
            if rank > 2:
                weight = self.reshape(weight, [1] * (rank - 2) +
                                      list(linear_weights.weight.shape))
        else:
            value = np.ascontiguousarray(linear_weights.weight,
                                         dtype=np.float32)
            weight = self.const(
                value.reshape([1] * (rank - 2) + list(value.shape)), "w32")
        output = self.matmul(x32, weight, trt.MatrixOperation.NONE,
                             trt.MatrixOperation.TRANSPOSE)
        if linear_weights.bias is not None:
            bias_shape = [1] * (rank - 1) + [int(linear_weights.bias.shape[0])]
            if isinstance(linear_weights.bias, ParameterSpec):
                bias = self.weight_input(name + ".bias",
                                         linear_weights.bias,
                                         "fp16",
                                         recipe=linear_weights.bias_recipe)
                bias = self.cast(self.reshape(bias, bias_shape), trt.float32)
            else:
                bias = self.const(
                    np.asarray(linear_weights.bias,
                               dtype=np.float32).reshape(bias_shape), "b32")
            output = self.elementwise(output, bias,
                                      trt.ElementWiseOperation.SUM)
        return output

    def dynamic_lora(self, x: "trt.ITensor", base: "trt.ITensor", prefix: str,
                     in_features: int, out_features: int) -> "trt.ITensor":
        """Add runtime LoRA A/B matrices to a linear's base output."""
        x, base = self._unwrap(x), self._unwrap(base)
        lora_a = self.add_input(f"{prefix}.lora_A.weight", trt.float16,
                                (in_features, -1))
        lora_b = self.add_input(f"{prefix}.lora_B.weight", trt.float16,
                                (-1, out_features))
        intermediate = self.network.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, lora_a,
            trt.MatrixOperation.NONE).get_output(0)
        update = self.network.add_matrix_multiply(
            intermediate, trt.MatrixOperation.NONE, lora_b,
            trt.MatrixOperation.NONE).get_output(0)
        return self.elementwise(base, update, trt.ElementWiseOperation.SUM)

    def convolution(
            self,
            x: "trt.ITensor",
            weight,
            bias=None,
            stride: Sequence[int] = (1, ),
            padding: Sequence[int] = (0, ),
            dilation: Sequence[int] = (1, ),
            groups: int = 1,
            pre_padding: Optional[Sequence[int]] = None,
            post_padding: Optional[Sequence[int]] = None) -> "trt.ITensor":
        """Add an N-D convolution with baked or checkpoint-backed weights."""
        x = self._unwrap(x)
        weight_shape = tuple(int(dimension) for dimension in weight.shape)
        promoted_1d = len(weight_shape) == 3
        if promoted_1d:
            x = self.unsqueeze(x, -1, 3)
            weight_shape += (1, )
        spatial_rank = len(weight_shape) - 2

        def spatial(values: Sequence[int], name: str,
                    promoted_value: int) -> tuple:
            result = tuple(int(value) for value in values)
            if promoted_1d and len(result) == 1:
                return result + (promoted_value, )
            if len(result) == 1 and spatial_rank > 1:
                result *= spatial_rank
            if len(result) != spatial_rank:
                raise ValueError(
                    f"{name} rank {len(result)} does not match convolution "
                    f"spatial rank {spatial_rank}")
            return result

        kernel_tensor = None
        if isinstance(weight, CheckpointParameter):
            kernel_tensor = self.parameter(weight.name, weight.value, "fp16",
                                           weight.recipe)
            if promoted_1d:
                kernel_tensor = self.reshape(kernel_tensor, weight_shape)
            kernel_weights = trt.Weights()
        else:
            weight = np.ascontiguousarray(np.asarray(weight, dtype=np.float16))
            if promoted_1d:
                weight = np.expand_dims(weight, -1)
            self._weight_refs.append(weight)
            kernel_weights = trt.Weights(weight)

        bias_tensor = None
        bias_weights = trt.Weights()
        if isinstance(bias, CheckpointParameter):
            bias_tensor = self.parameter(bias.name, bias.value, "fp16",
                                         bias.recipe)
        elif bias is not None and np.asarray(bias).size:
            bias_array = np.ascontiguousarray(
                np.asarray(bias, dtype=np.float16))
            self._weight_refs.append(bias_array)
            bias_weights = trt.Weights(bias_array)
        layer = self.network.add_convolution_nd(
            x,
            weight_shape[0],
            weight_shape[2:],
            kernel_weights,
            bias_weights,
        )
        if kernel_tensor is not None:
            layer.set_input(1, kernel_tensor)
        if bias_tensor is not None:
            layer.set_input(2, bias_tensor)
        layer.stride_nd = spatial(stride, "stride", 1)
        layer.dilation_nd = spatial(dilation, "dilation", 1)
        if pre_padding is None and post_padding is None:
            layer.padding_nd = spatial(padding, "padding", 0)
        else:
            layer.pre_padding = spatial(pre_padding or padding, "pre_padding",
                                        0)
            layer.post_padding = spatial(post_padding or padding,
                                         "post_padding", 0)
        layer.num_groups = groups
        output = layer.get_output(0)
        return self.reshape(output, (0, 0, 0)) if promoted_1d else output

    def deconvolution(
            self,
            x: "trt.ITensor",
            weight: np.ndarray,
            bias: Optional[np.ndarray] = None,
            stride: Sequence[int] = (1, ),
            padding: Sequence[int] = (0, ),
            groups: int = 1,
            pre_padding: Optional[Sequence[int]] = None,
            post_padding: Optional[Sequence[int]] = None) -> "trt.ITensor":
        """Add an N-D transposed convolution."""
        x = self._unwrap(x)
        weight = np.ascontiguousarray(weight.astype(np.float16))
        bias_array = (None if bias is None or bias.size == 0 else
                      np.ascontiguousarray(bias.astype(np.float16)))
        promoted_1d = weight.ndim == 3
        if promoted_1d:
            x = self.unsqueeze(x, -1, 3)
            weight = np.expand_dims(weight, -1)
        spatial_rank = weight.ndim - 2

        def spatial(values: Sequence[int], name: str,
                    promoted_value: int) -> tuple:
            result = tuple(int(value) for value in values)
            if promoted_1d and len(result) == 1:
                return result + (promoted_value, )
            if len(result) == 1 and spatial_rank > 1:
                result *= spatial_rank
            if len(result) != spatial_rank:
                raise ValueError(
                    f"{name} rank {len(result)} does not match deconvolution "
                    f"spatial rank {spatial_rank}")
            return result

        self._weight_refs.append(weight)
        bias_weights = trt.Weights()
        if bias_array is not None:
            self._weight_refs.append(bias_array)
            bias_weights = trt.Weights(bias_array)
        layer = self.network.add_deconvolution_nd(
            x, int(weight.shape[1] * groups),
            tuple(int(dim) for dim in weight.shape[2:]), trt.Weights(weight),
            bias_weights)
        layer.stride_nd = spatial(stride, "stride", 1)
        if pre_padding is None and post_padding is None:
            layer.padding_nd = spatial(padding, "padding", 0)
        else:
            layer.pre_padding = spatial(pre_padding or padding, "pre_padding",
                                        0)
            layer.post_padding = spatial(post_padding or padding,
                                         "post_padding", 0)
        layer.num_groups = groups
        output = layer.get_output(0)
        return self.reshape(output, (0, 0, 0)) if promoted_1d else output

    def linear_from_weights(self,
                            x: "trt.ITensor",
                            linear_weights,
                            rank: int = 3,
                            name: str = "") -> "trt.ITensor":
        """Emit a linear using its checkpoint quantization.

        FP8 and MXFP8 weights are always folded into the engine: their Q/DQ
        constants have no checkpoint-equivalent layout for the runtime to
        rebuild.
        """
        quant_type = linear_weights.quant_type
        if quant_type == quantization.QUANT_FP16:
            if (name and linear_weights.weight_recipe
                    and self.policy.externalizes_fp16(name)):
                return self.external_fp16_linear(x, linear_weights, name, rank)
            return self.linear(x, linear_weights.weight, linear_weights.bias,
                               rank)
        if quant_type == quantization.QUANT_NVFP4:
            raw = {
                "packed": linear_weights.weight,
                "weight_scale": linear_weights.weight_scale,
                "weight_scale_2": linear_weights.weight_scale_2,
                "input_scale": linear_weights.input_scale,
                "bias": linear_weights.bias,
            }
            return self.nvfp4_linear(x, raw, rank)
        if quant_type == quantization.QUANT_FP8:
            return self.fp8_linear(x, linear_weights, rank)
        if quant_type == quantization.QUANT_FP8_BLOCK:
            return self.fp8_block_linear(x, linear_weights, rank)
        if quant_type == quantization.QUANT_MXFP8:
            return self.mxfp8_linear(x, linear_weights, rank)
        if quant_type in (quantization.QUANT_INT4_AWQ,
                          quantization.QUANT_INT4_AWQ_MODELOPT,
                          quantization.QUANT_INT4_GPTQ):
            return self.int4_linear(x, linear_weights, rank, name)
        if quant_type == quantization.QUANT_INT8_SQ:
            return self.int8_sq_linear(x, linear_weights, rank)
        raise ValueError(f"unsupported linear quantization {quant_type!r}")

    # -- normalization ------------------------------------------------------

    def rmsnorm(self,
                x: "trt.ITensor",
                weight,
                eps: float,
                rank: int = 3,
                weight_before_cast: bool = False) -> "trt.ITensor":
        """Apply decomposed RMSNorm over the last axis."""
        last_axis = rank - 1
        x32 = self.cast(x, trt.float32)
        sq = self.elementwise(x32, x32, trt.ElementWiseOperation.PROD)
        var = self.network.add_reduce(sq, trt.ReduceOperation.AVG,
                                      1 << last_axis, True).get_output(0)
        eps_c = self.const(
            np.array(eps, dtype=np.float32).reshape([1] * rank), "eps")
        var = self.elementwise(var, eps_c, trt.ElementWiseOperation.SUM)
        std = self.unary(var, trt.UnaryOperation.SQRT)
        normed = self.elementwise(x32, std, trt.ElementWiseOperation.DIV)
        wshape = [1] * (rank - 1) + [int(weight.shape[0])]
        if isinstance(weight, CheckpointParameter):
            w = self.parameter(weight.name, weight.value, "fp16",
                               weight.recipe)
            w = self.reshape(w, wshape)
        else:
            w = None
        if weight_before_cast:
            if w is not None:
                w = self.cast(w, trt.float32)
            else:
                w = self.const(
                    weight.astype(np.float32).reshape(wshape), "rmsw")
            weighted = self.elementwise(normed, w,
                                        trt.ElementWiseOperation.PROD)
            return self.cast(weighted, trt.float16)
        normed16 = self.cast(normed, trt.float16)
        if w is None:
            w = self.const(weight.astype(np.float16).reshape(wshape), "rmsw")
        return self.elementwise(normed16, w, trt.ElementWiseOperation.PROD)

    def layernorm(self, x: "trt.ITensor", weight, bias, eps: float,
                  rank: int) -> "trt.ITensor":
        """Apply LayerNorm over the last tensor axis."""
        axis = rank - 1
        x32 = self.cast(x, trt.float32)
        mean = self.reduce(x32, trt.ReduceOperation.AVG, 1 << axis, True)
        centered = self.elementwise(x32, mean, trt.ElementWiseOperation.SUB)
        squared = self.elementwise(centered, centered,
                                   trt.ElementWiseOperation.PROD)
        variance = self.reduce(squared, trt.ReduceOperation.AVG, 1 << axis,
                               True)
        epsilon = self.const(
            np.array(eps, dtype=np.float32).reshape([1] * rank), "eps")
        deviation = self.unary(
            self.elementwise(variance, epsilon, trt.ElementWiseOperation.SUM),
            trt.UnaryOperation.SQRT)
        normalized = self.cast(
            self.elementwise(centered, deviation,
                             trt.ElementWiseOperation.DIV), trt.float16)
        shape = [1] * (rank - 1) + [int(weight.shape[0])]
        if isinstance(weight, CheckpointParameter):
            scale = self.parameter(weight.name, weight.value, "fp16",
                                   weight.recipe)
            scale = self.reshape(scale, shape)
        else:
            scale = self.const(
                weight.astype(np.float16).reshape(shape), "ln_w")
        if isinstance(bias, CheckpointParameter):
            shift = self.parameter(bias.name, bias.value, "fp16", bias.recipe)
            shift = self.reshape(shift, shape)
        else:
            shift = self.const(bias.astype(np.float16).reshape(shape), "ln_b")
        normalized = self.elementwise(normalized, scale,
                                      trt.ElementWiseOperation.PROD)
        return self.elementwise(normalized, shift,
                                trt.ElementWiseOperation.SUM)

    # -- token selection ----------------------------------------------------

    def gather_last_tokens(self, hidden: "trt.ITensor",
                           last_token_ids: "trt.ITensor") -> "trt.ITensor":
        """Select token states with GatherND and batch dimensions enabled.

        ``hidden`` has shape ``[B,S,H]`` and ``last_token_ids`` has shape
        ``[B,T]``. The indices are expanded to ``[B,T,1]`` and the result has
        shape ``[B,T,H]``.
        """
        hidden = self._unwrap(hidden)
        idx32 = self.cast(last_token_ids, trt.int32)
        # [B,T] -> [B,T,1] via shuffle (0 copies the corresponding input dim).
        sh = self.network.add_shuffle(idx32)
        sh.reshape_dims = (0, 0, 1)
        gather = self.network.add_gather_v2(hidden, sh.get_output(0),
                                            trt.GatherMode.ND)
        gather.num_elementwise_dims = 1
        return gather.get_output(0)

    # -- operation implementations -----------------------------------------

    @staticmethod
    def _creator(name: str):
        creator_name = _OPERATION_CREATORS.get(name, name)
        registry = trt.get_plugin_registry()
        creator = registry.get_creator(creator_name, "1", "")
        if creator is None:
            raise RuntimeError(
                f"implementation for operation {name!r} ({creator_name!r}, v1) "
                "was not found; is libNvInfer_edgellm_plugin.so loaded?")
        return creator

    def _operation_field(self, name: str, value) -> "trt.PluginField":
        data = np.asarray(value)
        if data.dtype.kind in "biu":
            data = np.ascontiguousarray(data, dtype=np.int32).reshape(-1)
            field_type = trt.PluginFieldType.INT32
        elif data.dtype.kind == "f":
            data = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
            field_type = trt.PluginFieldType.FLOAT32
        else:
            raise TypeError(
                f"operation attribute {name!r} has unsupported dtype "
                f"{data.dtype}")
        self._weight_refs.append(data)
        return trt.PluginField(name, data, field_type)

    def operation(self, name: str, attributes: Mapping[str, object],
                  inputs: Sequence["trt.ITensor"]):
        """Lower one semantic operation into the network."""
        fields = [
            self._operation_field(key, value)
            for key, value in attributes.items()
        ]
        instance_name = self._name(name)
        implementation = self._creator(name).create_plugin(
            instance_name, trt.PluginFieldCollection(fields),
            trt.TensorRTPhase.BUILD)
        if implementation is None:
            raise RuntimeError(
                f"operation {name!r} rejected the requested configuration; "
                "check the TensorRT log and rebuild "
                "libNvInfer_edgellm_plugin.so with the required optional "
                "kernels")
        layer = self.network.add_plugin_v3(
            [self._unwrap(value) for value in inputs], [], implementation)
        layer.name = instance_name
        return layer

    def operation_attributes(self, name: str) -> frozenset[str]:
        """Return attributes accepted by one operation implementation."""
        return frozenset(field.name
                         for field in self._creator(name).field_names)

    # -- NVFP4 dense Q/DQ (activation dynamic quantization) -----------------

    def const_fp4(self,
                  packed: np.ndarray,
                  shape,
                  name: str = "w_fp4") -> "trt.ITensor":
        """FP4 constant from packed nibbles (low nibble = even element)."""
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
        self._weight_refs.append(packed)
        count = int(np.prod(shape))
        weights = trt.Weights(trt.DataType.FP4, packed.ctypes.data, count)
        layer = self.network.add_constant(tuple(int(d) for d in shape),
                                          weights)
        layer.name = self._name(name)
        return layer.get_output(0)

    def const_fp8(self,
                  raw_bytes: np.ndarray,
                  shape,
                  name: str = "w_fp8") -> "trt.ITensor":
        """FP8 (E4M3) constant from raw bytes."""
        raw_bytes = np.ascontiguousarray(raw_bytes, dtype=np.uint8)
        self._weight_refs.append(raw_bytes)
        count = int(np.prod(shape))
        weights = trt.Weights(trt.DataType.FP8, raw_bytes.ctypes.data, count)
        layer = self.network.add_constant(tuple(int(d) for d in shape),
                                          weights)
        layer.name = self._name(name)
        return layer.get_output(0)

    def const_ue8m0(self,
                    raw_bytes: np.ndarray,
                    shape,
                    name: str = "scale_ue8m0") -> "trt.ITensor":
        """E8M0 constant used as an MXFP8 block scale."""
        raw_bytes = np.ascontiguousarray(raw_bytes, dtype=np.uint8)
        self._weight_refs.append(raw_bytes)
        count = int(np.prod(shape))
        weights = trt.Weights(_e8m0_dtype(), raw_bytes.ctypes.data, count)
        layer = self.network.add_constant(tuple(int(dim) for dim in shape),
                                          weights)
        layer.name = self._name(name)
        return layer.get_output(0)

    def _add_bias(self,
                  x: "trt.ITensor",
                  bias,
                  rank: int,
                  *,
                  name: str = "",
                  recipe=None) -> "trt.ITensor":
        x = self._unwrap(x)
        if bias is None:
            return x
        shape = [1] * (rank - 1) + [int(bias.shape[0])]
        if isinstance(bias, ParameterSpec):
            if not name or recipe is None:
                raise ValueError("external FP16 bias has no checkpoint recipe")
            tensor = self.weight_input(name + ".bias",
                                       bias,
                                       "fp16",
                                       recipe=recipe)
            tensor = self.reshape(tensor, shape)
        else:
            tensor = self.const(bias.astype(np.float16).reshape(shape), "b")
        return self.elementwise(x, tensor, trt.ElementWiseOperation.SUM)

    def fp8_linear(self,
                   x: "trt.ITensor",
                   linear_weights,
                   rank: int = 3) -> "trt.ITensor":
        """FP8 Q/DQ MatMul with per-tensor activation and weight scales."""
        x = self._unwrap(x)
        input_scale = self.const(
            np.array(linear_weights.input_scale, dtype=np.float16), "x_scale")
        quantized_x = self.network.add_quantize(x, input_scale,
                                                trt.DataType.FP8)
        dequantized_x = self.network.add_dequantize(quantized_x.get_output(0),
                                                    input_scale, trt.float16)
        out_features, in_features = linear_weights.weight.shape
        weight = self.const_fp8(linear_weights.weight,
                                (out_features, in_features), "w_fp8")
        weight_scale = self.const(
            np.asarray(linear_weights.weight_scale, dtype=np.float16),
            "w_scale")
        dequantized_weight = self.network.add_dequantize(
            weight, weight_scale, trt.float16)
        dequantized_weight.axis = 0
        w_tensor = dequantized_weight.get_output(0)
        if rank > 2:
            w_tensor = self.reshape(w_tensor, [1] * (rank - 2) +
                                    [out_features, in_features])
        output = self.network.add_matrix_multiply(
            dequantized_x.get_output(0), trt.MatrixOperation.NONE, w_tensor,
            trt.MatrixOperation.TRANSPOSE).get_output(0)
        return self._add_bias(output, linear_weights.bias, rank)

    def mxfp8_linear(self,
                     x: "trt.ITensor",
                     linear_weights,
                     rank: int = 3) -> "trt.ITensor":
        """MXFP8 dynamic-activation Q/DQ and E8M0 block-weight DQ."""
        x = self._unwrap(x)
        axis = rank - 1
        dynamic = self.network.add_dynamic_quantize(x, axis,
                                                    linear_weights.group_size,
                                                    trt.DataType.FP8,
                                                    _e8m0_dtype())
        activation_scale = dynamic.get_output(1)
        dequantized_x = self.network.add_dequantize(dynamic.get_output(0),
                                                    activation_scale,
                                                    trt.float16)
        out_features, in_features = linear_weights.weight.shape
        weight = self.const_fp8(linear_weights.weight,
                                (out_features, in_features), "w_mxfp8")
        weight_scale = self.const_ue8m0(
            linear_weights.weight_scale,
            (out_features, in_features // linear_weights.group_size), "w_e8m0")
        dequantized_weight = self.network.add_dequantize(
            weight, weight_scale, trt.float16)
        dequantized_weight.axis = 1
        w_tensor = dequantized_weight.get_output(0)
        if rank > 2:
            w_tensor = self.reshape(w_tensor, [1] * (rank - 2) +
                                    [out_features, in_features])
        output = self.network.add_matrix_multiply(
            dequantized_x.get_output(0), trt.MatrixOperation.NONE, w_tensor,
            trt.MatrixOperation.TRANSPOSE).get_output(0)
        return self._add_bias(output, linear_weights.bias, rank)

    def fp8_block_linear(self,
                         x: "trt.ITensor",
                         linear_weights,
                         rank: int = 3) -> "trt.ITensor":
        """FP8 weight-only MatMul with FP32 two-dimensional block scales."""
        x = self._unwrap(x)
        out_features, in_features = linear_weights.weight.shape
        scale_shape = tuple(
            int(dim) for dim in linear_weights.weight_scale.shape)
        if (len(scale_shape) != 4 or scale_shape[1] != 1
                or scale_shape[3] != 1):
            raise ValueError("FP8 block scale must have shape [out_blocks, 1, "
                             f"in_blocks, 1], got {scale_shape}")
        out_blocks, _, in_blocks, _ = scale_shape
        if out_features % out_blocks or in_features % in_blocks:
            raise ValueError(
                f"FP8 weight shape {(out_features, in_features)} is not "
                f"divisible by block scale shape {scale_shape}")
        out_block = out_features // out_blocks
        weight = self.const_fp8(linear_weights.weight,
                                (out_features, in_features), "w_fp8_block")
        scale = self.const(
            np.ascontiguousarray(linear_weights.weight_scale,
                                 dtype=np.float32).reshape(
                                     (out_blocks, in_blocks)),
            "w_block_scale",
        )
        dequantized_weight = self.network.add_dequantize(
            weight, scale, trt.float16)
        dequantized_weight.block_shape = (out_block, in_features // in_blocks)
        weight = dequantized_weight.get_output(0)
        input_shape = None
        if rank > 2:
            input_shape = self.shape_of(x)
            x = self.reshape(x, (-1, in_features))
        output = self.network.add_matrix_multiply(
            x, trt.MatrixOperation.NONE, weight,
            trt.MatrixOperation.TRANSPOSE).get_output(0)
        if input_shape is not None:
            leading = self.network.add_slice(input_shape, (0, ), (rank - 1, ),
                                             (1, )).get_output(0)
            output_shape = self.concat(
                (leading, self.const(np.array([out_features],
                                              dtype=np.int32))), 0)
            output = self.dynamic_reshape(output, output_shape)
        return self._add_bias(output, linear_weights.bias, rank)

    def int4_linear(self,
                    x: "trt.ITensor",
                    linear_weights,
                    rank: int = 3,
                    name: str = "") -> "trt.ITensor":
        """Lower groupwise INT4 GEMM."""
        if not name:
            raise ValueError("INT4 linear requires a stable module name")
        if linear_weights.pre_quant_scale is not None:
            shape = [1] * (rank - 1) + [linear_weights.in_features]
            pre_quant = linear_weights.pre_quant_scale
            if not isinstance(pre_quant, ParameterSpec):
                pre_quant = np.ascontiguousarray(pre_quant, dtype=np.float16)
            if (linear_weights.pre_quant_recipe and
                    self.policy.externalizes_fp16(name + ".pre_quant_scale")):
                smoother = self.weight_input(
                    name + ".pre_quant_scale",
                    pre_quant,
                    "fp16",
                    recipe=linear_weights.pre_quant_recipe,
                )
                smoother = self.reshape(smoother, shape)
            else:
                smoother = self.const(pre_quant.reshape(shape),
                                      "pre_quant_scale")
            x = self.elementwise(x, smoother, trt.ElementWiseOperation.PROD)
        weight_recipe = linear_weights.weight_recipe
        permutation = linear_weights.activation_permutation
        if isinstance(permutation, ParameterSpec):
            permutation_name = name + ".activation_permutation"
            permutation_tensor = self.weight_input(
                permutation_name,
                permutation,
                "int4_ffn",
                recipe=linear_weights.activation_permutation_recipe,
            )
            x = self.gather_tensor(x, permutation_tensor, rank - 1)
            if weight_recipe is not None:
                weight_recipe = dict(weight_recipe)
                extra = dict(weight_recipe.get("extra") or {})
                extra["activation_permutation_engine_name"] = permutation_name
                weight_recipe["extra"] = extra
        elif permutation is not None:
            x = self.gather(x, permutation, rank - 1)
        weight_value = linear_weights.weight
        if not isinstance(weight_value, ParameterSpec):
            weight_value = weight_value.astype(np.int8, copy=False)
        weight = self.parameter(name + ".qweight", weight_value, "int4_ffn",
                                weight_recipe)
        scale_value = linear_weights.weight_scale
        if not isinstance(scale_value, ParameterSpec):
            scale_value = np.asarray(scale_value, dtype=np.float16)
        scales = self.parameter(name + ".scales", scale_value, "int4_ffn",
                                linear_weights.scale_recipe)
        operation_name = ("int4_groupwise_gemm_v2"
                          if self.int4_gemm_plugin_version == 2 else
                          "int4_groupwise_gemm")
        plugin_input = x
        if rank == 2:
            plugin_input = self.unsqueeze(x, 0, rank)
        elif rank != 3:
            raise ValueError(
                f"INT4 linear supports rank 2 or 3, received rank {rank}")
        output = self.operation(
            operation_name, {
                "gemm_n": linear_weights.out_features,
                "gemm_k": linear_weights.in_features,
                "group_size": linear_weights.group_size,
            }, [plugin_input, weight, scales]).get_output(0)
        if rank == 2:
            output = self.squeeze(output, 0, 3)
        return self._add_bias(output,
                              linear_weights.bias,
                              rank,
                              name=name,
                              recipe=linear_weights.bias_recipe)

    def int8_sq_linear(self,
                       x: "trt.ITensor",
                       linear_weights,
                       rank: int = 3) -> "trt.ITensor":
        """SmoothQuant W8A8 Q/DQ MatMul."""
        smooth_shape = [1] * (rank - 1) + [linear_weights.in_features]
        smoother = self.const(
            linear_weights.pre_quant_scale.astype(
                np.float16).reshape(smooth_shape), "pre_quant_scale")
        smoothed = self.elementwise(x, smoother, trt.ElementWiseOperation.PROD)
        input_scale = self.const(
            np.array(linear_weights.input_scale, dtype=np.float32), "x_scale")
        quantized_x = self.network.add_quantize(smoothed, input_scale,
                                                trt.int8)
        dequantized_x = self.network.add_dequantize(quantized_x.get_output(0),
                                                    input_scale, trt.float16)
        weight = self.const(linear_weights.weight.astype(np.int8), "w_int8")
        weight_scale = self.const(
            linear_weights.weight_scale.astype(np.float32), "w_scale")
        dequantized_weight = self.network.add_dequantize(
            weight, weight_scale, trt.float16)
        dequantized_weight.axis = 0
        w_tensor = dequantized_weight.get_output(0)
        if rank > 2:
            w_tensor = self.reshape(
                w_tensor, [1] * (rank - 2) +
                [linear_weights.out_features, linear_weights.in_features])
        output = self.network.add_matrix_multiply(
            dequantized_x.get_output(0), trt.MatrixOperation.NONE, w_tensor,
            trt.MatrixOperation.TRANSPOSE).get_output(0)
        return self._add_bias(output, linear_weights.bias, rank)

    def nvfp4_act_qdq(self,
                      x: "trt.ITensor",
                      input_scale: float,
                      rank: int = 3) -> "trt.ITensor":
        """Quantize activations to FP4 with FP8 block scales, then
        dequantize the scales and activations to FP16."""
        x = self._unwrap(x)
        axis = rank - 1
        scale32 = self.const(np.array(input_scale, dtype=np.float32),
                             "act_scale32")
        dynq = self.network.add_dynamic_quantize(x, axis, 16, trt.DataType.FP4,
                                                 trt.DataType.FP8)
        dynq.set_input(1, scale32)
        x_f4 = dynq.get_output(0)
        block_scales_f8 = dynq.get_output(1)

        scale16 = self.const(np.array(input_scale, dtype=np.float16),
                             "act_scale16")
        dq_scales = self.network.add_dequantize(block_scales_f8, scale16,
                                                trt.float16)
        dq_x = self.network.add_dequantize(x_f4, dq_scales.get_output(0),
                                           trt.float16)
        dq_x.axis = axis
        return dq_x.get_output(0)

    def nvfp4_linear(self,
                     x: "trt.ITensor",
                     raw: dict,
                     rank: int = 3,
                     weight_dq_in_graph: bool = True) -> "trt.ITensor":
        """NVFP4 dense linear with activation and weight Q/DQ.

        ``raw`` comes from ``Weights.linear_nvfp4_raw``. With
        ``weight_dq_in_graph`` the FP4 weight and FP8 block scales become
        typed constants dequantized in-graph; otherwise the weight is
        pre-decoded to an FP16 constant (numerically equivalent).
        """
        x_dq = self.nvfp4_act_qdq(x, float(raw["input_scale"]), rank=rank)

        packed = raw["packed"]
        out_features = int(packed.shape[0])
        in_features = int(packed.shape[1]) * 2
        if weight_dq_in_graph:
            w_f4 = self.const_fp4(packed, (out_features, in_features), "w_fp4")
            ws_f8 = self.const_fp8(raw["weight_scale"],
                                   (out_features, in_features // 16),
                                   "w_scales")
            ws2 = self.const(np.array(raw["weight_scale_2"], dtype=np.float32),
                             "ws2")
            dq_ws = self.network.add_dequantize(ws_f8, ws2, trt.float32)
            dq_w = self.network.add_dequantize(w_f4, dq_ws.get_output(0),
                                               trt.float32)
            dq_w.axis = 1
            w16 = self.cast(dq_w.get_output(0), trt.float16)
            if rank != 2:
                w16 = self.reshape(w16, [1] * (rank - 2) +
                                   [out_features, in_features])
        else:
            from ..weight_packing import nvfp4 as nvfp4_pack
            dense = nvfp4_pack.decode_modelopt_nvfp4(packed,
                                                     raw["weight_scale"],
                                                     raw["weight_scale_2"], 16)
            wshape = [1] * (rank - 2) + [out_features, in_features]
            w16 = self.const(
                np.ascontiguousarray(dense.astype(np.float16).reshape(wshape)),
                "w")

        out = self.network.add_matrix_multiply(
            x_dq, trt.MatrixOperation.NONE, w16,
            trt.MatrixOperation.TRANSPOSE).get_output(0)
        bias = raw.get("bias")
        if bias is not None:
            bshape = [1] * (rank - 1) + [int(bias.shape[0])]
            b = self.const(bias.astype(np.float16).reshape(bshape), "b")
            out = self.elementwise(out, b, trt.ElementWiseOperation.SUM)
        return out

    def fused_nvfp4_gemm_all_reduce(self,
                                    x: "trt.ITensor",
                                    linear_weights,
                                    bias,
                                    tp_size: int,
                                    rank: int = 3,
                                    name: str = "",
                                    bias_recipe=None) -> "trt.ITensor":
        """Row-parallel NVFP4 GEMM fused with the TP all-reduce."""
        x = self._unwrap(x)
        input_scale = self.const(
            np.array(linear_weights.input_scale, dtype=np.float32),
            "act_scale")
        dynamic_quantize = self.network.add_dynamic_quantize(
            x, rank - 1, 16, trt.DataType.FP4, trt.DataType.FP8)
        dynamic_quantize.set_input(1, input_scale)
        activation = dynamic_quantize.get_output(0)
        activation_scale = self.network.add_dequantize(
            dynamic_quantize.get_output(1), input_scale,
            trt.float32).get_output(0)

        packed = linear_weights.weight
        out_features = int(linear_weights.out_features)
        in_features = int(linear_weights.in_features)
        attributes = {"tp_size": tp_size}
        if isinstance(packed, ParameterSpec):
            resource_id = self._next_plugin_resource_id
            self._next_plugin_resource_id += 1
            self.plugin_resource(
                name + ".weight",
                packed,
                "nvfp4_tp",
                linear_weights.weight_recipe,
                resource_id=resource_id,
                resource_kind="weight",
                storage_dtype="U8",
            )
            self.plugin_resource(
                name + ".weight_scale",
                linear_weights.weight_scale,
                "nvfp4_tp",
                linear_weights.scale_recipe,
                resource_id=resource_id,
                resource_kind="scale",
                storage_dtype="F8_E4M3",
            )
            scale_shape = linear_weights.weight_scale.shape
            attributes.update({
                "external_weight_resource_id": resource_id,
                "weight_out_features": out_features,
                "weight_in_features": in_features,
                "weight_scale_cols": int(scale_shape[1]),
            })
        else:
            weight = self.const_fp4(packed, (out_features, in_features),
                                    "w_fp4")
            weight_scale = self.const_fp8(
                linear_weights.weight_scale,
                (out_features, in_features // linear_weights.group_size),
                "w_scales")
        weight_scale_2 = self.const(
            np.array(linear_weights.weight_scale_2, dtype=np.float32),
            "w_scale_2")
        inputs = ([activation, activation_scale, weight_scale_2] if isinstance(
            packed, ParameterSpec) else [
                activation, activation_scale, weight, weight_scale,
                weight_scale_2
            ])
        layer = self.operation("fused_nvfp4_gemm_all_reduce", attributes,
                               inputs)
        return self._add_bias(layer.get_output(0),
                              bias,
                              rank,
                              name=name,
                              recipe=bias_recipe)
