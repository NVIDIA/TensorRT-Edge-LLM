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
"""High-level weight accessors over a safetensors checkpoint.

Linears retain their checkpoint precision. The returned descriptors feed
TensorRT FP8/NVFP4/MXFP8 Q/DQ patterns, INT4 plugins, INT8 Q/DQ, or native
FP16 MatMul. MoE expert weights remain packed for their dedicated plugins.
"""

import re
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..weight_packing import int4 as int4_pack
from ..weight_packing import nvfp4 as nvfp4_pack
from . import quantization
from .safetensors_np import SafetensorsStore
from .weight_policy import WeightPolicy

__all__ = ["CheckpointParameter", "LinearWeights", "ParameterSpec", "Weights"]


@dataclass(frozen=True)
class ParameterSpec:
    """Final TensorRT parameter metadata without a materialized payload."""

    shape: Tuple[int, ...]
    dtype: np.dtype

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape",
                           tuple(int(dimension) for dimension in self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))


ParameterValue = Union[np.ndarray, ParameterSpec]


@dataclass(frozen=True)
class CheckpointParameter:
    """One ordinary parameter and its runtime checkpoint load recipe."""

    name: str
    value: ParameterValue
    recipe: dict

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(int(dimension) for dimension in self.value.shape)


@dataclass(frozen=True)
class LinearWeights:
    """Tensor buffers and metadata for one checkpoint linear."""

    quant_type: str
    weight: ParameterValue
    bias: Optional[ParameterValue] = None
    weight_scale: Optional[ParameterValue] = None
    weight_scale_2: Optional[float] = None
    input_scale: Optional[float] = None
    pre_quant_scale: Optional[ParameterValue] = None
    activation_permutation: Optional[ParameterValue] = None
    group_size: int = 1
    # How the runtime rebuilds parameters from a raw checkpoint, or None when
    # only this builder can produce the layout.
    # Sharding and vocabulary reduction clear both: they rewrite the tensors
    # in ways no single checkpoint key describes.
    weight_recipe: Optional[dict] = None
    bias_recipe: Optional[dict] = None
    scale_recipe: Optional[dict] = None
    pre_quant_recipe: Optional[dict] = None
    activation_permutation_recipe: Optional[dict] = None
    logical_out_features: Optional[int] = None
    logical_in_features: Optional[int] = None

    @property
    def out_features(self) -> int:
        """Infer the output dimension for supported checkpoint layouts."""
        if self.logical_out_features is not None:
            return self.logical_out_features
        if self.quant_type in (quantization.QUANT_INT4_AWQ,
                               quantization.QUANT_INT4_AWQ_MODELOPT,
                               quantization.QUANT_INT4_GPTQ):
            return int(self.weight.shape[0]) * 2
        return int(self.weight.shape[0])

    @property
    def in_features(self) -> int:
        """Infer the input dimension for supported checkpoint layouts."""
        if self.logical_in_features is not None:
            return self.logical_in_features
        if self.quant_type == quantization.QUANT_NVFP4:
            return int(self.weight.shape[1]) * 2
        return int(self.weight.shape[1])


class Weights:

    def __init__(self,
                 model_dir: str,
                 group_size: int = 16,
                 quant: Optional[quantization.QuantConfig] = None,
                 component: str = "llm",
                 spec_type: str = "none",
                 spec_role: str = "none",
                 vocab_map: Optional[np.ndarray] = None,
                 conversion: Optional[ModuleType] = None,
                 int4_gemm_plugin_version: int = 2,
                 checkpoint_source: str = "component",
                 tie_word_embeddings: bool = False,
                 policy: Optional[WeightPolicy] = None) -> None:
        self.conversion = conversion
        checkpoint_dir = getattr(conversion, "checkpoint_dir", None)
        source_dir = (checkpoint_dir(model_dir, component)
                      if checkpoint_dir is not None else model_dir)
        self.source_dir = source_dir
        self.store = SafetensorsStore(source_dir)
        self.group_size = group_size
        self.quant = quant or quantization.QuantConfig(group_size=group_size)
        self.component = component
        self.spec_type = spec_type
        self.spec_role = spec_role
        self.vocab_map = (None if vocab_map is None else np.ascontiguousarray(
            vocab_map, dtype=np.int64))
        if int4_gemm_plugin_version not in (1, 2):
            raise ValueError("int4_gemm_plugin_version must be 1 or 2")
        self.int4_gemm_plugin_version = int4_gemm_plugin_version
        if checkpoint_source not in ("component", "target"):
            raise ValueError("checkpoint_source must be component or target")
        self.checkpoint_source = checkpoint_source
        self.tie_word_embeddings = bool(tie_word_embeddings)
        self._policy = policy or WeightPolicy()

    def close(self) -> None:
        self.store.close()

    # -- presence -----------------------------------------------------------

    def has(self, name: str) -> bool:
        return self._resolve(name, required=False) is not None

    def _resolve(self, name: str, required: bool = True) -> Optional[str]:
        candidates = [name]
        resolve_candidates = getattr(self.conversion, "resolve_candidates",
                                     None)
        if resolve_candidates is not None:
            candidates.extend(
                resolve_candidates(name,
                                   component=self.component,
                                   spec_type=self.spec_type,
                                   spec_role=self.spec_role,
                                   quant_type=self.quant.quant_type))
        if name == "lm_head.weight":
            can_tie = (self.tie_word_embeddings and self.module_quant_type(
                "lm_head", tie_word_embeddings=True)
                       == quantization.QUANT_FP16)
            if can_tie:
                embedding_candidates = ["model.embed_tokens.weight"]
                if resolve_candidates is not None:
                    embedding_candidates.extend(
                        resolve_candidates(
                            "model.embed_tokens.weight",
                            component=self.component,
                            spec_type=self.spec_type,
                            spec_role=self.spec_role,
                            quant_type=self.quant.quant_type,
                        ))
                candidates = embedding_candidates + candidates
            else:
                candidates = [
                    candidate for candidate in candidates
                    if not candidate.endswith("embed_tokens.weight")
                ]
        candidates = list(dict.fromkeys(candidates))
        for candidate in candidates:
            if self.store.has(candidate):
                return candidate
        if required:
            raise KeyError(f"checkpoint tensor not found: {name!r}")
        return None

    def checkpoint_key(self, name: str) -> str:
        """Return the concrete checkpoint key backing a model tensor name."""
        return self._resolve(name, required=False) or name

    def causal_lm_head_prefix(self, *preferred: str) -> str:
        """Resolve a model-preferred head, HF LM head, or tied embedding."""
        fallback = (("model.embed_tokens", ) if self.tie_word_embeddings else
                    ())
        for prefix in (*preferred, "lm_head", *fallback):
            if (self.has(f"{prefix}.weight") or self.has(f"{prefix}.qweight")):
                return prefix
        raise KeyError(
            "checkpoint has neither lm_head nor tied token embeddings")

    def checkpoint_binding(self,
                           names: Sequence[str],
                           source_layout: str = "plugin",
                           assemble: Optional[str] = None,
                           **extra) -> dict:
        """Describe how the runtime rebuilds one tensor from the checkpoint."""
        checkpoint_keys = [self.checkpoint_key(name) for name in names]
        recipe = {
            "checkpoint_keys": checkpoint_keys,
            "source_layout": source_layout,
        }
        if assemble:
            recipe["assemble"] = assemble
        binding_extra = dict(extra)
        locations = self.checkpoint_locations(checkpoint_keys)
        if locations:
            binding_extra["checkpoint_locations"] = locations
        if self.checkpoint_source != "component":
            binding_extra["checkpoint_source"] = self.checkpoint_source
        if binding_extra:
            recipe["extra"] = binding_extra
        return recipe

    def checkpoint_locations(self, names: Sequence[str]) -> dict:
        """Return explicit archive ranges for keys stored in PyTorch ``.bin``."""
        locations = {}
        for name in names:
            location = self.store.checkpoint_location(name)
            if location is not None:
                locations[name] = location
        return locations

    def module_quant_type(self,
                          name: str,
                          *,
                          tie_word_embeddings: bool = False) -> str:
        """Return the checkpoint precision owned by one model projection."""
        normalize = getattr(self.conversion, "normalize_checkpoint_name", None)
        if normalize is not None:
            name = normalize(name)
        return self.quant.module_type(name, tie_word_embeddings)

    def parameter_spec(self,
                       name: str,
                       dtype=np.float16,
                       *,
                       transpose: bool = False) -> ParameterSpec:
        """Describe a checkpoint tensor without reading its payload."""
        key = self._resolve(name)
        shape = self.store.shape(key)
        if transpose:
            if len(shape) != 2:
                raise ValueError(
                    f"cannot transpose non-matrix checkpoint tensor {key!r}")
            shape = (shape[1], shape[0])
        return ParameterSpec(shape, np.dtype(dtype))

    def parameter_value(self, kind: str, name: str,
                        metadata: Callable[[], ParameterValue],
                        materialize: Callable[[], ParameterValue]):
        """Resolve a parameter without exposing storage policy to a model.

        Externalized parameters are represented only by final TensorRT
        metadata while the network is built. Parameters retained by the engine
        are materialized from the checkpoint as usual.
        """
        if self._policy.externalizes_parameter(kind, name):
            return metadata()
        return materialize()

    def fp16_parameter(self, name: str) -> CheckpointParameter:
        """Describe one checkpoint tensor consumed directly by an operation."""
        value = self.parameter_value("fp16", name,
                                     lambda: self.parameter_spec(name),
                                     lambda: self.f16(name))
        return CheckpointParameter(
            name,
            value,
            self.checkpoint_binding([name], source_layout="fp16"),
        )

    def opt_fp16_parameter(self, name: str) -> Optional[CheckpointParameter]:
        """Return an FP16 parameter descriptor when the checkpoint has it."""
        return self.fp16_parameter(name) if self.has(name) else None

    def is_nvfp4(self, prefix: str) -> bool:
        return self.has(prefix + ".weight_scale") and (
            self.has(prefix + ".weight_scale_2")
            or self.has(prefix + ".weight_global_scale"))

    def _nvfp4_names(self, prefix: str) -> Tuple[str, str, str, str, bool]:
        """Resolve ModelOpt or compressed-tensors NVFP4 tensor names."""
        if self.has(prefix + ".weight_global_scale"):
            return (prefix + ".weight_packed", prefix + ".weight_scale",
                    prefix + ".weight_global_scale",
                    prefix + ".input_global_scale", True)
        return (prefix + ".weight", prefix + ".weight_scale",
                prefix + ".weight_scale_2", prefix + ".input_scale", False)

    def nvfp4_checkpoint_names(self,
                               prefix: str) -> Tuple[str, str, str, bool]:
        """Return packed-weight, block-scale, and alpha checkpoint keys."""
        weight, scale, alpha, _, reciprocal = self._nvfp4_names(prefix)
        return (self.checkpoint_key(weight), self.checkpoint_key(scale),
                self.checkpoint_key(alpha), reciprocal)

    @staticmethod
    def _nvfp4_global_scale(value: float, reciprocal: bool) -> float:
        """Return a scale in the multiplier convention used by the builder."""
        if not reciprocal:
            return value
        if value == 0.0:
            raise ValueError("compressed-tensors NVFP4 global scale is zero")
        return 1.0 / value

    # -- plain tensors ------------------------------------------------------

    def f16(self, name: str) -> np.ndarray:
        return self.store.get_f16(self._resolve(name))

    def f32(self, name: str) -> np.ndarray:
        return self.store.get_f32(self._resolve(name))

    def array(self, name: str) -> np.ndarray:
        """Return one tensor while retaining its stored integer/float dtype."""
        return np.ascontiguousarray(self.store.get_numpy(self._resolve(name)))

    def opt_f16(self, name: str) -> Optional[np.ndarray]:
        return self.f16(name) if self.has(name) else None

    def find(self, *names: str) -> str:
        """Return the first checkpoint key that exists."""
        for name in names:
            resolved = self._resolve(name, required=False)
            if resolved is not None:
                return resolved
        raise KeyError("none of the checkpoint keys exist: " +
                       ", ".join(names))

    def keys(self) -> Tuple[str, ...]:
        """Return checkpoint tensor names for component-specific discovery."""
        return tuple(self.store.keys())

    def layer_prefixes(self, markers: Sequence[str]) -> List[str]:
        """Discover and numerically order model-specific layer prefixes."""
        found = set()
        patterns = [re.compile(marker) for marker in markers]
        for key in self.keys():
            for pattern in patterns:
                match = pattern.search(key)
                if match:
                    found.add(match.group(1))
                    break

        def order(prefix: str) -> Tuple[int, str]:
            numbers = re.findall(r"\.(\d+)(?:\.|$)", prefix)
            return (int(numbers[-1]) if numbers else -1, prefix)

        return sorted(found, key=order)

    def find_suffix(self, suffix: str, contains: str = "") -> str:
        """Find a unique component tensor by suffix and optional substring."""
        matches = [
            name for name in self.store.keys()
            if name.endswith(suffix) and (not contains or contains in name)
        ]
        if len(matches) != 1:
            raise KeyError(
                f"expected one tensor ending {suffix!r} containing {contains!r}; "
                f"found {len(matches)}")
        return matches[0]

    def _linear_unreduced(self, prefix: str, quant_type: str) -> LinearWeights:
        """Load one linear in the representation consumed by TensorRT."""
        bias = self.opt_f16(prefix + ".bias")
        if quant_type == quantization.QUANT_FP16:
            weight, bias = self.linear_fp16(prefix)
            verbatim = (not self.is_nvfp4(prefix)
                        and self.has(prefix + ".weight"))
            return LinearWeights(
                quant_type,
                weight,
                bias,
                weight_recipe=(self.checkpoint_binding(
                    [prefix + ".weight"], "fp16") if verbatim else None))
        if quant_type == quantization.QUANT_NVFP4:
            raw = self.linear_nvfp4_raw(prefix)
            return LinearWeights(
                quant_type,
                raw["packed"],
                raw["bias"],
                weight_scale=raw["weight_scale"],
                weight_scale_2=raw["weight_scale_2"],
                input_scale=raw["input_scale"],
                group_size=self.group_size,
            )
        if quant_type == quantization.QUANT_FP8:
            return LinearWeights(
                quant_type,
                self.store.get_fp8_bytes(self._resolve(prefix + ".weight")),
                bias,
                weight_scale=self.f16(prefix + ".weight_scale"),
                input_scale=self.store.get_scalar_f32(
                    self._resolve(prefix + ".input_scale")),
            )
        if quant_type == quantization.QUANT_FP8_BLOCK:
            return LinearWeights(
                quant_type,
                self.store.get_fp8_bytes(self._resolve(prefix + ".weight")),
                bias,
                weight_scale=self.f32(prefix + ".weight_scale"),
            )
        if quant_type == quantization.QUANT_MXFP8:
            return LinearWeights(
                quant_type,
                self.store.get_fp8_bytes(self._resolve(prefix + ".weight")),
                bias,
                weight_scale=self.array(prefix + ".weight_scale").astype(
                    np.uint8, copy=False),
                group_size=self.group_size,
            )
        if quant_type == quantization.QUANT_INT4_AWQ:
            qweight = self.array(prefix + ".qweight")
            qzeros = self.array(prefix + ".qzeros")
            scales = self.f16(prefix + ".scales")
            in_features = int(qweight.shape[0])
            out_features = int(qweight.shape[1]) * 8
            if self._reduce_lm_head(prefix):
                order = (0, 2, 4, 6, 1, 3, 5, 7)
                qweight = int4_pack.select_column_packed(
                    qweight, self.vocab_map, order)
                qzeros = int4_pack.select_column_packed(
                    qzeros, self.vocab_map, order)
                scales = self._select_output_axis(scales, self.vocab_map)
            reduced = self._reduce_lm_head(prefix)
            logical_out_features = (int(self.vocab_map.size)
                                    if reduced else out_features)
            packed = int4_pack.repack_awq(qweight, qzeros,
                                          self.int4_gemm_plugin_version)
            return LinearWeights(
                quant_type,
                packed,
                bias,
                weight_scale=scales,
                group_size=self.group_size,
                weight_recipe=None if reduced else self.checkpoint_binding(
                    [prefix + ".qweight", prefix + ".qzeros"],
                    assemble="awq_ffn_qweight",
                    int4_gemm_plugin_version=self.int4_gemm_plugin_version),
                scale_recipe=None
                if reduced else self.checkpoint_binding([prefix + ".scales"]),
                logical_out_features=logical_out_features,
                logical_in_features=in_features,
            )
        if quant_type == quantization.QUANT_INT4_AWQ_MODELOPT:
            weight = self.array(prefix + ".weight")
            scales = self.f16(prefix + ".weight_scale")
            in_features = int(weight.shape[1])
            out_features = int(weight.shape[0]) * 2
            if self._reduce_lm_head(prefix):
                weight = int4_pack.select_pair_packed_rows(
                    weight, self.vocab_map)
                scales = self._select_output_axis(scales, self.vocab_map)
            reduced = self._reduce_lm_head(prefix)
            logical_out_features = (int(self.vocab_map.size)
                                    if reduced else out_features)
            packed = int4_pack.repack_modelopt_awq(
                weight, self.int4_gemm_plugin_version)
            scales = scales.T
            pre_quant_scale = (self.f16(prefix + ".pre_quant_scale")
                               if self.has(prefix + ".pre_quant_scale") else
                               np.ones(in_features, dtype=np.float16))
            return LinearWeights(
                quant_type,
                packed,
                bias,
                weight_scale=np.ascontiguousarray(scales),
                pre_quant_scale=np.ascontiguousarray(pre_quant_scale),
                group_size=self.group_size,
                weight_recipe=None if reduced else self.checkpoint_binding(
                    [prefix + ".weight"],
                    "int4_modelopt_uint8",
                    assemble="modelopt_awq_ffn_qweight",
                    int4_gemm_plugin_version=self.int4_gemm_plugin_version),
                scale_recipe=None if reduced else self.checkpoint_binding(
                    [prefix + ".weight_scale"]),
                pre_quant_recipe=(self.checkpoint_binding(
                    [prefix + ".pre_quant_scale"], "fp16")
                                  if self.has(prefix +
                                              ".pre_quant_scale") else None),
                logical_out_features=logical_out_features,
                logical_in_features=in_features,
            )
        if quant_type == quantization.QUANT_INT4_GPTQ:
            qweight = self.array(prefix + ".qweight")
            qzeros = (self.array(prefix + ".qzeros")
                      if self.has(prefix + ".qzeros") else np.empty(
                          (1, 0), dtype=np.int32))
            scales = self.f16(prefix + ".scales")
            in_features = int(qweight.shape[0]) * 8
            out_features = int(qweight.shape[1])
            if self._reduce_lm_head(prefix):
                qweight = np.ascontiguousarray(qweight[:, self.vocab_map])
                if qzeros.size:
                    qzeros = int4_pack.select_column_packed(
                        qzeros, self.vocab_map, tuple(range(8)))
                scales = self._select_output_axis(scales, self.vocab_map)
            g_idx = (self.array(prefix +
                                ".g_idx") if self.has(prefix +
                                                      ".g_idx") else None)
            reduced = self._reduce_lm_head(prefix)
            logical_out_features = (int(self.vocab_map.size)
                                    if reduced else out_features)
            packed, permutation = int4_pack.repack_gptq(
                qweight,
                qzeros,
                g_idx,
                self.quant.gptq_zero_point_offset,
                self.int4_gemm_plugin_version,
            )
            return LinearWeights(
                quant_type,
                packed,
                bias,
                weight_scale=scales,
                activation_permutation=permutation,
                group_size=self.group_size,
                weight_recipe=None if reduced else self.checkpoint_binding(
                    [
                        prefix + ".qweight", prefix + ".qzeros", prefix +
                        ".g_idx"
                    ],
                    assemble="gptq_ffn_qweight",
                    zero_point_offset=int(self.quant.gptq_zero_point_offset),
                    int4_gemm_plugin_version=self.int4_gemm_plugin_version),
                scale_recipe=None
                if reduced else self.checkpoint_binding([prefix + ".scales"]),
                logical_out_features=logical_out_features,
                logical_in_features=in_features,
            )
        if quant_type == quantization.QUANT_INT8_SQ:
            pre_quant_scale = (self.f16(prefix + ".pre_quant_scale")
                               if self.has(prefix + ".pre_quant_scale") else
                               np.ones(self.store.shape(
                                   self._resolve(prefix + ".weight"))[1],
                                       dtype=np.float16))
            return LinearWeights(
                quant_type,
                self.array(prefix + ".weight").astype(np.int8, copy=False),
                bias,
                weight_scale=self.f32(prefix + ".weight_scale"),
                input_scale=self.store.get_scalar_f32(
                    self._resolve(prefix + ".input_scale")),
                pre_quant_scale=np.ascontiguousarray(pre_quant_scale),
            )
        raise ValueError(f"unsupported linear quantization {quant_type!r}")

    def linear(self, prefix: str, quant_type: str) -> LinearWeights:
        """Load a linear and apply an optional output-vocabulary map."""
        if (quant_type == quantization.QUANT_NVFP4
                and not self.is_nvfp4(prefix)):
            raise ValueError(
                f"{prefix}: quantization config selects NVFP4, but the "
                "checkpoint has no NVFP4 weight and scale tensors")
        linear = self._linear_unreduced(prefix, quant_type)
        if not self._reduce_lm_head(prefix) or quant_type in (
                quantization.QUANT_INT4_AWQ,
                quantization.QUANT_INT4_AWQ_MODELOPT,
                quantization.QUANT_INT4_GPTQ,
        ):
            return linear
        indices = self.vocab_map
        weight = np.ascontiguousarray(linear.weight[indices])
        bias = (None if linear.bias is None else np.ascontiguousarray(
            linear.bias[indices]))
        weight_scale = linear.weight_scale
        if (weight_scale is not None
                and (isinstance(weight_scale, ParameterSpec)
                     or np.ndim(weight_scale) > 0)):
            weight_scale = self._select_output_axis(weight_scale, indices)
        return replace(linear,
                       weight=weight,
                       bias=bias,
                       weight_scale=weight_scale,
                       weight_recipe=None,
                       bias_recipe=None,
                       scale_recipe=None)

    def linear_metadata(
            self,
            prefix: str,
            quant_type: str,
            *,
            externalize_pre_quant: bool = False) -> Optional[LinearWeights]:
        """Describe a reproducible external linear without loading its data."""
        if self._reduce_lm_head(prefix):
            return None
        bias_name = prefix + ".bias"
        bias = None
        bias_recipe = None
        if self.has(bias_name):
            bias = self.parameter_value("fp16", bias_name,
                                        lambda: self.parameter_spec(bias_name),
                                        lambda: self.f16(bias_name))
            bias_recipe = self.checkpoint_binding([bias_name], "fp16")
        if quant_type == quantization.QUANT_FP16:
            if self.is_nvfp4(prefix) or not self.has(prefix + ".weight"):
                return None
            weight = self.parameter_spec(prefix + ".weight", np.float16)
            return LinearWeights(
                quant_type,
                weight,
                bias,
                weight_recipe=self.checkpoint_binding([prefix + ".weight"],
                                                      "fp16"),
                bias_recipe=bias_recipe,
            )
        if quant_type not in (
                quantization.QUANT_INT4_AWQ,
                quantization.QUANT_INT4_AWQ_MODELOPT,
                quantization.QUANT_INT4_GPTQ,
        ):
            return None

        def packed_shape(out_features: int,
                         in_features: int) -> Tuple[int, int]:
            if self.int4_gemm_plugin_version == 1:
                if out_features % 4 or in_features % 64:
                    raise ValueError(
                        "INT4 V1 packing requires output features divisible "
                        f"by 4 and input features divisible by 64, got "
                        f"{(out_features, in_features)}")
                return (out_features // 2, in_features)
            if in_features % 64:
                raise ValueError(
                    "INT4 V2 packing requires input features divisible by 64, "
                    f"got {(out_features, in_features)}")
            rows = ((out_features + 127) // 128) * (in_features // 64) * 8
            return (rows, 512)

        pre_quant_scale = None
        pre_quant_recipe = None
        activation_permutation = None
        activation_permutation_recipe = None
        if quant_type == quantization.QUANT_INT4_AWQ:
            qweight_shape = self.store.shape(self._resolve(prefix +
                                                           ".qweight"))
            in_features = int(qweight_shape[0])
            out_features = int(qweight_shape[1]) * 8
            weight_recipe = self.checkpoint_binding(
                [prefix + ".qweight", prefix + ".qzeros"],
                assemble="awq_ffn_qweight",
                int4_gemm_plugin_version=self.int4_gemm_plugin_version)
            scale = self.parameter_spec(prefix + ".scales", np.float16)
            scale_recipe = self.checkpoint_binding([prefix + ".scales"])
        elif quant_type == quantization.QUANT_INT4_AWQ_MODELOPT:
            source_shape = self.store.shape(self._resolve(prefix + ".weight"))
            in_features = int(source_shape[1])
            out_features = int(source_shape[0]) * 2
            weight_recipe = self.checkpoint_binding(
                [prefix + ".weight"],
                "int4_modelopt_uint8",
                assemble="modelopt_awq_ffn_qweight",
                int4_gemm_plugin_version=self.int4_gemm_plugin_version)
            scale = self.parameter_spec(prefix + ".weight_scale",
                                        np.float16,
                                        transpose=True)
            scale_recipe = self.checkpoint_binding([prefix + ".weight_scale"])
            if self.has(prefix + ".pre_quant_scale"):
                pre_quant_recipe = self.checkpoint_binding(
                    [prefix + ".pre_quant_scale"], "fp16")
                if externalize_pre_quant:
                    pre_quant_scale = self.parameter_spec(
                        prefix + ".pre_quant_scale", np.float16)
                else:
                    pre_quant_scale = self.f16(prefix + ".pre_quant_scale")
            else:
                pre_quant_scale = np.ones(in_features, dtype=np.float16)
        else:
            qweight_shape = self.store.shape(self._resolve(prefix +
                                                           ".qweight"))
            in_features = int(qweight_shape[0]) * 8
            out_features = int(qweight_shape[1])
            if self.has(prefix + ".g_idx"):
                activation_permutation = ParameterSpec((in_features, ),
                                                       np.int32)
                activation_permutation_recipe = self.checkpoint_binding(
                    [prefix + ".g_idx"],
                    assemble="gptq_activation_permutation",
                    group_size=self.group_size)
            else:
                activation_permutation = np.arange(in_features, dtype=np.int64)
                activation_permutation_recipe = None
            weight_recipe = self.checkpoint_binding(
                [prefix + ".qweight", prefix + ".qzeros", prefix + ".g_idx"],
                assemble="gptq_ffn_qweight",
                zero_point_offset=int(self.quant.gptq_zero_point_offset),
                int4_gemm_plugin_version=self.int4_gemm_plugin_version)
            scale = self.parameter_spec(prefix + ".scales", np.float16)
            scale_recipe = self.checkpoint_binding([prefix + ".scales"])

        return LinearWeights(
            quant_type,
            ParameterSpec(packed_shape(out_features, in_features), np.int8),
            bias,
            weight_scale=scale,
            pre_quant_scale=pre_quant_scale,
            activation_permutation=activation_permutation,
            group_size=self.group_size,
            weight_recipe=weight_recipe,
            scale_recipe=scale_recipe,
            pre_quant_recipe=pre_quant_recipe,
            activation_permutation_recipe=activation_permutation_recipe,
            logical_out_features=out_features,
            logical_in_features=in_features,
        )

    def linear_nvfp4_tp_metadata(self, prefix: str) -> Optional[LinearWeights]:
        """Describe raw NVFP4 tensors consumed by the fused TP plugin.

        TensorRT-native NVFP4 GEMMs retain constant weights so Myelin can own
        their lowering. The fused TP plugin instead consumes the provider's
        packed bytes directly, which lets the runtime materialize one rank's
        row-parallel shard without reading the payload during engine build.
        """
        if self._reduce_lm_head(prefix):
            return None
        (weight_name, scale_name, global_scale_name, input_scale_name,
         reciprocal) = self._nvfp4_names(prefix)
        weight_key = self._resolve(weight_name)
        scale_key = self._resolve(scale_name)
        weight_dtype = self.store.dtype(weight_key)
        scale_dtype = self.store.dtype(scale_key)
        # The plugin resource ABI owns packed NVFP4 bytes as UINT8. Signed I8
        # and sub-byte F4 provider layouts stay on the existing build-time path
        # until their storage contracts can be represented without changing
        # the payload type.
        if weight_dtype != "U8" or scale_dtype != "F8_E4M3":
            return None
        weight_shape = self.store.shape(weight_key)
        scale_shape = self.store.shape(scale_key)
        if len(weight_shape) != 2 or len(scale_shape) != 2:
            raise ValueError(
                f"{prefix}: NVFP4 weight and scale must be rank-2")
        out_features = int(weight_shape[0])
        in_features = int(weight_shape[1]) * 2
        expected_scale_shape = (out_features, in_features // self.group_size)
        if tuple(scale_shape) != expected_scale_shape:
            raise ValueError(
                f"{prefix}: NVFP4 scale shape {scale_shape} does not match "
                f"{expected_scale_shape}")
        weight_scale_2 = self._nvfp4_global_scale(
            self.store.get_scalar_f32(self._resolve(global_scale_name)),
            reciprocal)
        input_scale = (self._nvfp4_global_scale(
            self.store.get_scalar_f32(self._resolve(input_scale_name)),
            reciprocal) if self.has(input_scale_name) else 1.0)
        bias_name = prefix + ".bias"
        bias = self.f16(bias_name) if self.has(bias_name) else None
        # Keep the checkpoint's byte-addressable packed shape. Logical FP4
        # dimensions are carried separately for plugin shape validation.
        return LinearWeights(
            quantization.QUANT_NVFP4,
            ParameterSpec(weight_shape, np.uint8),
            bias,
            weight_scale=ParameterSpec(scale_shape, np.uint8),
            weight_scale_2=weight_scale_2,
            input_scale=input_scale,
            group_size=self.group_size,
            weight_recipe=self.checkpoint_binding([weight_name],
                                                  "nvfp4_packed"),
            scale_recipe=self.checkpoint_binding([scale_name], "nvfp4_scale"),
            logical_out_features=out_features,
            logical_in_features=in_features,
        )

    def linear_descriptor(self,
                          prefix: str,
                          quant_type: str,
                          *,
                          external_kind: str = "") -> LinearWeights:
        """Return a dense projection payload or metadata, as required."""
        if (quant_type == quantization.QUANT_NVFP4
                and not self.is_nvfp4(prefix)):
            raise ValueError(
                f"{prefix}: quantization config selects NVFP4, but the "
                "checkpoint has no NVFP4 weight and scale tensors")
        if (quant_type == quantization.QUANT_NVFP4
                and external_kind and self._policy.externalizes_parameter(
                    external_kind, prefix)):
            descriptor = self.linear_nvfp4_tp_metadata(prefix)
            if descriptor is not None:
                return descriptor
        kind = ("fp16" if quant_type == quantization.QUANT_FP16 else
                "int4_ffn" if quant_type in (
                    quantization.QUANT_INT4_AWQ,
                    quantization.QUANT_INT4_AWQ_MODELOPT,
                    quantization.QUANT_INT4_GPTQ,
                ) else "")
        if kind and self._policy.externalizes_parameter(kind, prefix):
            descriptor = self.linear_metadata(
                prefix,
                quant_type,
                externalize_pre_quant=self._policy.externalizes_fp16(
                    prefix + ".pre_quant_scale"))
            if descriptor is not None:
                return descriptor
        return self.linear(prefix, quant_type)

    def linear_adapter(self, prefix: str):
        """Return a model-owned static low-rank adapter, when present."""
        adapter = getattr(self.conversion, "linear_adapter", None)
        return adapter(self, prefix) if adapter is not None else None

    def _reduce_lm_head(self, prefix: str) -> bool:
        return (self.vocab_map is not None
                and (prefix == "lm_head" or prefix.endswith(".lm_head")
                     or prefix.endswith("embed_tokens")))

    @staticmethod
    def _select_output_axis(array: np.ndarray,
                            indices: np.ndarray) -> np.ndarray:
        max_index = int(indices.max())
        axes = [
            axis for axis, extent in enumerate(array.shape)
            if extent > max_index
        ]
        if not axes:
            return array
        return np.ascontiguousarray(np.take(array, indices, axis=axes[-1]))

    @staticmethod
    def shard_linear(linear: LinearWeights, mode: str, tp_size: int,
                     tp_rank: int) -> LinearWeights:
        """Return one contiguous tensor-parallel shard of a linear."""
        if tp_size == 1 or mode == "replicated":
            return linear
        full_out = linear.out_features
        full_in = linear.in_features
        split_size = full_out if mode == "column" else full_in
        if split_size % tp_size:
            raise ValueError(
                f"cannot {mode}-shard linear dimension {split_size} over {tp_size} ranks"
            )

        def rank_neutral_recipe(recipe, **fields):
            if recipe is None:
                return None
            result = dict(recipe)
            extra = dict(result.get("extra", {}))
            extra.update(fields)
            result["extra"] = extra
            return result

        def split(value, recipe, axis):
            if value is None:
                return value, recipe
            if isinstance(value, ParameterSpec):
                shape = list(value.shape)
                if shape[axis] % tp_size:
                    raise ValueError(
                        f"cannot shard parameter shape {value.shape} on axis {axis} over {tp_size} ranks"
                    )
                shape[axis] //= tp_size
                return (replace(value, shape=tuple(shape)),
                        rank_neutral_recipe(recipe,
                                            tp_shard={
                                                "axis": axis,
                                                "size": tp_size
                                            }))
            if np.ndim(value) == 0:
                return value, recipe
            return (np.ascontiguousarray(
                np.split(value, tp_size, axis=axis)[tp_rank]), None)

        weight, weight_recipe = split(linear.weight, linear.weight_recipe,
                                      0 if mode == "column" else 1)
        if mode == "column":
            bias, bias_recipe = split(linear.bias, linear.bias_recipe, 0)
        else:
            bias, bias_recipe = linear.bias, linear.bias_recipe
            if isinstance(bias, ParameterSpec):
                bias_recipe = rank_neutral_recipe(bias_recipe,
                                                  tp_rank0_only=True)
            elif bias is not None and tp_rank != 0:
                bias = np.zeros_like(bias)
                bias_recipe = None

        weight_scale = linear.weight_scale
        if (weight_scale is not None
                and (isinstance(weight_scale, ParameterSpec)
                     or np.ndim(weight_scale) > 0)):
            if linear.quant_type == quantization.QUANT_NVFP4:
                matching = [0 if mode == "column" else 1]
            elif mode == "column":
                matching = [
                    axis for axis, extent in enumerate(weight_scale.shape)
                    if extent in (full_out, full_out // 2)
                ]
            else:
                groups = full_in // max(1, linear.group_size)
                matching = [
                    axis for axis, extent in enumerate(weight_scale.shape)
                    if extent in (full_in, groups)
                ]
            if matching:
                weight_scale, scale_recipe = split(weight_scale,
                                                   linear.scale_recipe,
                                                   matching[0])
            else:
                scale_recipe = linear.scale_recipe
        else:
            scale_recipe = linear.scale_recipe

        pre_quant_scale = linear.pre_quant_scale
        if (mode == "row" and pre_quant_scale is not None
                and (isinstance(pre_quant_scale, ParameterSpec)
                     or np.ndim(pre_quant_scale) > 0)):
            matching = [
                axis for axis, extent in enumerate(pre_quant_scale.shape)
                if extent == full_in
            ]
            if matching:
                pre_quant_scale, pre_quant_recipe = split(
                    pre_quant_scale, linear.pre_quant_recipe, matching[0])
            else:
                pre_quant_recipe = linear.pre_quant_recipe
        else:
            pre_quant_recipe = linear.pre_quant_recipe

        permutation = linear.activation_permutation
        if mode == "row" and permutation is not None:
            start = tp_rank * (full_in // tp_size)
            stop = start + full_in // tp_size
            local = permutation[(permutation >= start) & (permutation < stop)]
            permutation = np.ascontiguousarray(local - start, dtype=np.int64)

        return replace(
            linear,
            weight=weight,
            bias=bias,
            weight_scale=weight_scale,
            pre_quant_scale=pre_quant_scale,
            activation_permutation=permutation,
            weight_recipe=weight_recipe,
            bias_recipe=bias_recipe,
            scale_recipe=scale_recipe,
            pre_quant_recipe=pre_quant_recipe,
            activation_permutation_recipe=None,
            logical_out_features=(full_out //
                                  tp_size if mode == "column" else full_out),
            logical_in_features=(full_in //
                                 tp_size if mode == "row" else full_in))

    # -- dense linear (NVFP4 decoded to FP16, or plain) ---------------------

    def linear_fp16(self,
                    prefix: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return ``(W [out, in] fp16, bias [out] fp16 | None)`` for a linear.

        NVFP4 weights are dequantized to FP16; plain weights are cast to FP16.
        """
        if self.is_nvfp4(prefix):
            weight_name, scale_name, global_scale_name, _, reciprocal = \
                self._nvfp4_names(prefix)
            packed = self.store.get_packed_fp4(self._resolve(weight_name))
            sf = self.store.get_fp8_bytes(self._resolve(scale_name))
            ws2 = self._nvfp4_global_scale(
                self.store.get_scalar_f32(self._resolve(global_scale_name)),
                reciprocal)
            dense = nvfp4_pack.decode_modelopt_nvfp4(packed, sf, ws2,
                                                     self.group_size)
            w = dense.astype(np.float16)
        elif self.has(prefix + ".weight"):
            w = self.f16(prefix + ".weight")
        else:
            convert = getattr(self.conversion, "convert_linear_fp16", None)
            converted = convert(self, prefix) if convert is not None else None
            if converted is None:
                raise KeyError(f"no weight for linear {prefix!r}")
            return converted
        bias = self.opt_f16(prefix + ".bias")
        return np.ascontiguousarray(w), (np.ascontiguousarray(bias)
                                         if bias is not None else None)

    def qkv_scales(self, attn_prefix: str) -> Tuple[float, float, float]:
        """Per-layer ``[q, k, v]`` scales for the FP8 KV cache.

        The k/v scales come from checkpoint metadata when present and default
        to 1.0 otherwise.
        """

        def scalar(name: str) -> float:
            return (float(self.store.get_scalar_f32(self._resolve(name)))
                    if self.has(name) else 1.0)

        return (1.0, scalar(f"{attn_prefix}.k_proj.k_scale"),
                scalar(f"{attn_prefix}.v_proj.v_scale"))

    def linear_nvfp4_raw(self, prefix: str) -> dict:
        """Raw NVFP4 pieces of a dense linear for the in-graph Q/DQ path."""
        (weight_name, scale_name, global_scale_name, input_scale_name,
         reciprocal) = \
            self._nvfp4_names(prefix)
        weight_scale_2 = self._nvfp4_global_scale(
            self.store.get_scalar_f32(self._resolve(global_scale_name)),
            reciprocal)
        input_scale = (self._nvfp4_global_scale(
            self.store.get_scalar_f32(self._resolve(input_scale_name)),
            reciprocal) if self.has(input_scale_name) else 1.0)
        return {
            "packed": self.store.get_packed_fp4(self._resolve(weight_name)),
            "weight_scale":
            self.store.get_fp8_bytes(self._resolve(scale_name)),
            "weight_scale_2": weight_scale_2,
            "input_scale": input_scale,
            "bias": self.opt_f16(prefix + ".bias"),
        }

    # -- NVFP4 MoE expert accessors -----------------------------------------

    def expert_dense_f32(self, prefix: str) -> np.ndarray:
        """Load one NVFP4 or plain expert projection as dense fp32."""
        if not self.is_nvfp4(prefix):
            return np.ascontiguousarray(self.f32(prefix + ".weight"))
        (weight_name, scale_name, global_scale_name, _,
         reciprocal) = self._nvfp4_names(prefix)
        packed = self.store.get_packed_fp4(self._resolve(weight_name))
        sf = self.store.get_fp8_bytes(self._resolve(scale_name))
        ws2 = self._nvfp4_global_scale(
            self.store.get_scalar_f32(self._resolve(global_scale_name)),
            reciprocal)
        return nvfp4_pack.decode_modelopt_nvfp4(packed, sf, ws2,
                                                self.group_size)

    def expert_raw_nvfp4(self, prefix: str) -> dict:
        """Return raw NVFP4 bytes for one expert projection (byte-reuse path)."""
        (weight_name, scale_name, global_scale_name, _,
         reciprocal) = self._nvfp4_names(prefix)
        alpha = self._nvfp4_global_scale(
            self.store.get_scalar_f32(self._resolve(global_scale_name)),
            reciprocal)
        return {
            "packed": self.store.get_packed_fp4(self._resolve(weight_name)),
            "sf": self.store.get_fp8_bytes(self._resolve(scale_name)),
            "alpha": alpha,
        }
