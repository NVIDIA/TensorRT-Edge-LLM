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

import logging
import re
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ...checkpoint import iq2_xxs
from ..weight_packing import int4 as int4_pack
from ..weight_packing import int8 as int8_pack
from ..weight_packing import nvfp4 as nvfp4_pack
from . import contracts, quantization
from .numpy_dtypes import f32_to_fp8_e4m3_bytes
from .safetensors_np import SafetensorsStore
from .weight_policy import WeightPolicy

__all__ = ["CheckpointParameter", "LinearWeights", "ParameterSpec", "Weights"]

logger = logging.getLogger(__name__)


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
    zero_correction: Optional[ParameterValue] = None
    weight_scale_2: Optional[ParameterValue] = None
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
    weight_scale_2_recipe: Optional[dict] = None

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
        if self.quant_type in (quantization.QUANT_NVFP4,
                               quantization.QUANT_NVFP4_A16):
            return int(self.weight.shape[1]) * 2
        return int(self.weight.shape[1])


class Weights:

    def __init__(
        self,
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
        policy: Optional[WeightPolicy] = None,
        w4a16_fallback_types: Tuple[str, ...] = (),
        w8a16_fallback_types: Tuple[str, ...] = (),
        mxfp8_conversion_types: Tuple[str, ...] = (),
        weightless_engine: bool = False,
    ) -> None:
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
        self._weightless_engine = weightless_engine
        self._w4a16_fallback_types = frozenset(w4a16_fallback_types)
        self._w8a16_fallback_types = frozenset(w8a16_fallback_types)
        self._mxfp8_conversion_types = frozenset(mxfp8_conversion_types)

    def close(self) -> None:
        self.store.close()

    # -- presence -----------------------------------------------------------

    def has(self, name: str) -> bool:
        return self._resolve(name, required=False) is not None

    def dtype(self, name: str) -> str:
        """Return a resolved checkpoint tensor's storage dtype without loading it."""
        return self.store.dtype(self._resolve(name))

    def _resolve(self,
                 name: str,
                 required: bool = True,
                 pin: bool = True) -> Optional[str]:
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
        if pin:
            owned = self._module_tensor_key(name)
            if owned is not None:
                candidates = [owned]
        for candidate in candidates:
            if self.store.has(candidate):
                return candidate
        if required:
            raise KeyError(f"checkpoint tensor not found: {name!r}")
        return None

    # Every tensor one module can own, and the subset that can be its primary
    # weight. A draft resolves its modules in its own namespace (Qwen3.5 MTP
    # stores them under ``mtp.``) and then falls back to the base model's
    # names, so resolving a module's tensors one name at a time lets the base
    # model's identically named module answer for whatever the draft lacks.
    _MODULE_TENSORS = (".weight", ".qweight", ".weight_packed", ".bias",
                       ".weight_scale", ".weight_scale_inv", ".weight_scale_2",
                       ".weight_global_scale", ".input_scale",
                       ".input_global_scale", ".pre_quant_scale", ".scales",
                       ".qzeros", ".g_idx")
    _PRIMARY_WEIGHTS = (".weight", ".qweight", ".weight_packed")
    # Carrying anything besides a plain weight and a bias means a module's
    # weights are stored quantized.
    _QUANT_MARKERS = tuple(suffix for suffix in _MODULE_TENSORS
                           if suffix not in (".weight", ".bias"))

    def _module_namespace(self, prefix: str) -> Optional[str]:
        """Concrete checkpoint prefix owning one module's tensors.

        The first primary weight that resolves decides the namespace, so a
        module is read as a whole rather than one tensor name at a time.
        Returns None when no primary weight resolves.
        """
        for suffix in self._PRIMARY_WEIGHTS:
            key = self._resolve(prefix + suffix, required=False, pin=False)
            if key is not None:
                return key[:-len(suffix)]
        return None

    def _module_tensor_key(self, name: str) -> Optional[str]:
        """Key for a module tensor, pinned to the namespace owning its module.

        Returns None when *name* is not a module tensor or its module has no
        primary weight, leaving ordinary candidate resolution to run.
        """
        for suffix in self._MODULE_TENSORS:
            if name.endswith(suffix) and len(name) > len(suffix):
                namespace = self._module_namespace(name[:-len(suffix)])
                if namespace is None:
                    return None
                owned = namespace + suffix
                if (suffix == ".weight_scale" and not self.store.has(owned)
                        and self.store.has(owned + "_inv")):
                    return owned + "_inv"
                return owned
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
        lookup = name
        normalize = getattr(self.conversion, "normalize_checkpoint_name", None)
        if normalize is not None:
            lookup = normalize(lookup)
        quant_type = self.quant.module_type(lookup, tie_word_embeddings)
        if (quant_type == quantization.QUANT_NVFP4 and self.is_nvfp4(name)
                and not self.nvfp4_quantizes_activations(name)):
            # Without a calibrated activation scale the stored weights are
            # weight-only NVFP4, whatever the metadata declares.
            quant_type = quantization.QUANT_NVFP4_A16
        # Overrides are keyed by module name, which a draft shares with the
        # base layer it falls back to, so a quantized base layer would claim
        # an unquantized draft layer. The weights stored for this module
        # decide. ``lm_head`` is exempt because resolving it consults this
        # method for embedding tying.
        if (quant_type in (quantization.QUANT_FP16, quantization.QUANT_Q8_0,
                           quantization.QUANT_IQ2_XXS) or name == "lm_head"):
            return quant_type
        namespace = self._module_namespace(name)
        if namespace is None or any(
                self.store.has(namespace + marker)
                for marker in self._QUANT_MARKERS):
            return quant_type
        return quantization.QUANT_FP16

    def module_quant_group_size(self, name: str) -> int:
        """Return the checkpoint group size owned by one model projection."""
        normalize = getattr(self.conversion, "normalize_checkpoint_name", None)
        if normalize is not None:
            name = normalize(name)
        return self.quant.module_group_size(name)

    def uses_w4a16_fallback(self, quant_type: str) -> bool:
        """Return whether W4A4 checkpoint weights must execute as W4A16."""
        return quant_type in getattr(self, "_w4a16_fallback_types", ())

    def uses_w8a16_fallback(self, quant_type: str) -> bool:
        """Return whether floating checkpoint weights must execute as W8A16."""
        return quant_type in getattr(self, "_w8a16_fallback_types", ())

    def converts_to_mxfp8(self, quant_type: str) -> bool:
        """Return whether this build converts the source layout to MXFP8."""
        return quant_type in getattr(self, "_mxfp8_conversion_types", ())

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

    def nvfp4_quantizes_activations(self, prefix: str) -> bool:
        """Return whether NVFP4 weights carry a calibrated input scale."""
        return self.has(self._nvfp4_names(prefix)[3])

    def _nvfp4_names(self, prefix: str) -> Tuple[str, str, str, str, bool]:
        """Resolve ModelOpt or compressed-tensors NVFP4 tensor names."""
        if self.has(prefix + ".weight_global_scale"):
            return (prefix + ".weight_packed", prefix + ".weight_scale",
                    prefix + ".weight_global_scale",
                    prefix + ".input_global_scale", True)
        return (prefix + ".weight", prefix + ".weight_scale",
                prefix + ".weight_scale_2", prefix + ".input_scale", False)

    def nvfp4_checkpoint_names(
            self, prefix: str) -> Tuple[str, str, str, bool, Optional[str]]:
        """Return NVFP4 weight, scale, alpha, convention, and input-scale keys."""
        weight, scale, alpha, input_scale, reciprocal = self._nvfp4_names(
            prefix)
        return (self.checkpoint_key(weight), self.checkpoint_key(scale),
                self.checkpoint_key(alpha), reciprocal,
                self.checkpoint_key(input_scale)
                if self.has(input_scale) else None)

    def nvfp4_moe_input_scale_binding(self, experts_prefix: str,
                                      num_experts: int,
                                      projections: Sequence[str]):
        """Bind calibrated per-expert activation scales for one MoE stage."""
        keys = []
        conventions = set()
        missing = 0
        for expert in range(num_experts):
            for projection in projections:
                record = self.nvfp4_checkpoint_names(
                    f"{experts_prefix}.{expert}.{projection}")
                conventions.add(record[3])
                if record[4] is None:
                    missing += 1
                else:
                    keys.append(record[4])
        expected = num_experts * len(projections)
        if missing == expected:
            return self.checkpoint_binding([],
                                           "generated",
                                           "fill",
                                           fill_value=1.0)
        if missing or len(conventions) != 1:
            raise ValueError(
                f"{experts_prefix}: inconsistent NVFP4 input-scale metadata")
        return self.checkpoint_binding(keys,
                                       "plugin",
                                       "nvfp4_moe_input_scale",
                                       num_experts=num_experts,
                                       keys_per_expert=len(projections),
                                       reciprocal_alpha=conventions.pop(),
                                       require_equal=len(projections) > 1)

    def nvfp4_input_scale(self, prefix: str) -> float:
        """Return the effective calibrated activation scale for one linear."""
        _, _, _, input_scale, reciprocal = self._nvfp4_names(prefix)
        if not self.has(input_scale):
            return 1.0
        return self._nvfp4_global_scale(
            self.store.get_scalar_f32(self._resolve(input_scale)), reciprocal)

    def is_compressed_gptq(self, prefix: str) -> bool:
        """Return whether a GPTQ projection uses compressed-tensors names."""
        return (self.quant.checkpoint_format
                == quantization.CHECKPOINT_FORMAT_COMPRESSED_TENSORS
                and self.has(prefix + ".weight_packed"))

    def is_compressed_int8(self, prefix: str) -> bool:
        """Return whether a projection stores dense compressed INT8 weights."""
        return (self.quant.checkpoint_format
                == quantization.CHECKPOINT_FORMAT_COMPRESSED_TENSORS
                and self.has(prefix + ".weight")
                and self.has(prefix + ".weight_scale") and self.store.dtype(
                    self._resolve(prefix + ".weight")) == "I8")

    def gptq_tensors(self, prefix: str):
        """Load GPTQ tensors in canonical ``[K/pack,N]`` provider layout."""
        if self.is_compressed_gptq(prefix):
            qweight = np.ascontiguousarray(
                self.array(prefix + ".weight_packed").T)
        else:
            qweight = self.array(prefix + ".qweight")
        qzeros, scales, g_idx = self.gptq_sidecars(prefix)
        return qweight, qzeros, scales, g_idx

    def gptq_sidecars(self, prefix: str):
        """Load GPTQ zero points, scales, and group indices canonically."""
        if self.is_compressed_gptq(prefix):
            qzeros = (np.ascontiguousarray(
                self.array(prefix + ".weight_zero_point").T)
                      if self.has(prefix + ".weight_zero_point") else np.empty(
                          (0, 0), dtype=np.int32))
            scales = np.ascontiguousarray(self.f16(prefix + ".weight_scale").T)
            g_idx = (self.array(prefix + ".weight_g_idx")
                     if self.has(prefix + ".weight_g_idx") else None)
            return qzeros, scales, g_idx
        qzeros = (self.array(prefix + ".qzeros")
                  if self.has(prefix + ".qzeros") else np.empty(
                      (0, 0), dtype=np.int32))
        scales = self.f16(prefix + ".scales")
        g_idx = (self.array(prefix + ".g_idx") if self.has(prefix +
                                                           ".g_idx") else None)
        return qzeros, scales, g_idx

    def gptq_checkpoint_names(self, prefix: str) -> Tuple[str, str, str, str]:
        """Return source names for weight, zero point, scales, and groups."""
        if self.is_compressed_gptq(prefix):
            return (prefix + ".weight_packed", prefix + ".weight_zero_point",
                    prefix + ".weight_scale", prefix + ".weight_g_idx")
        return (prefix + ".qweight", prefix + ".qzeros", prefix + ".scales",
                prefix + ".g_idx")

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

    def f32(self, name: str, *, index: Optional[int] = None) -> np.ndarray:
        return self.store.get_f32(self._resolve(name), index=index)

    def array(self, name: str) -> np.ndarray:
        """Return one tensor while retaining its stored integer/float dtype."""
        return np.ascontiguousarray(self.store.get_numpy(self._resolve(name)))

    def gptq_expert_projection(self, experts_prefix: str, expert_index: int,
                               projection: str):
        """Load one GPTQ expert projection in its provider layout."""
        prefix = f"{experts_prefix}.{expert_index}.{projection}"
        qweight, qzeros, scales, group_index = self.gptq_tensors(prefix)
        if group_index is not None:
            group_index = group_index.reshape(-1)
            group_size = self.module_quant_group_size(prefix)
            expected = np.arange(group_index.size) // group_size
            if not np.array_equal(group_index, expected):
                raise ValueError(
                    f"act-order GPTQ experts are not supported: {prefix}")
        return qweight, qzeros, scales

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
        if self.uses_w4a16_fallback(quant_type):
            return self._linear_w4a16_fallback(prefix, quant_type)
        if self.uses_w8a16_fallback(quant_type):
            return self._linear_w8a16_fallback(prefix, quant_type, bias)
        if self.converts_to_mxfp8(quant_type):
            return self._linear_mxfp8_conversion(prefix, quant_type, bias)
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
        if quant_type in (quantization.QUANT_Q8_0, quantization.QUANT_IQ2_XXS):
            return self._linear_gguf(prefix, quant_type, bias, False)
        if quant_type in (quantization.QUANT_NVFP4,
                          quantization.QUANT_NVFP4_A16):
            raw = self.linear_nvfp4_raw(prefix)
            return LinearWeights(
                quant_type,
                raw["packed"],
                raw["bias"],
                weight_scale=raw["weight_scale"],
                weight_scale_2=raw["weight_scale_2"],
                input_scale=raw["input_scale"],
                group_size=self.module_quant_group_size(prefix),
            )
        if quant_type in (quantization.QUANT_FP8, quantization.QUANT_FP8_A16):
            return LinearWeights(
                quant_type,
                self.store.get_fp8_bytes(self._resolve(prefix + ".weight")),
                bias,
                weight_scale=np.ascontiguousarray(
                    self.f16(prefix + ".weight_scale").reshape(-1)),
                input_scale=(self.store.get_scalar_f32(
                    self._resolve(prefix + ".input_scale")) if quant_type
                             == quantization.QUANT_FP8 else None),
            )
        if quant_type == quantization.QUANT_FP8_BLOCK:
            scale = self.f32(prefix + ".weight_scale")
            if scale.ndim == 2:
                scale = scale[:, None, :, None]
            return LinearWeights(
                quant_type,
                self.store.get_fp8_bytes(self._resolve(prefix + ".weight")),
                bias,
                weight_scale=np.ascontiguousarray(scale),
            )
        if quant_type == quantization.QUANT_MXFP8:
            return LinearWeights(
                quant_type,
                self.store.get_fp8_bytes(self._resolve(prefix + ".weight")),
                bias,
                weight_scale=self.array(prefix + ".weight_scale").astype(
                    np.uint8, copy=False),
                group_size=self.module_quant_group_size(prefix),
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
                zero_correction=int4_pack.awq_zero_correction(qzeros, scales),
                group_size=self.module_quant_group_size(prefix),
                weight_recipe=None if reduced else self.checkpoint_binding(
                    [prefix + ".qweight"],
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
                group_size=self.module_quant_group_size(prefix),
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
            qweight, qzeros, scales, g_idx = self.gptq_tensors(prefix)
            in_features = int(qweight.shape[0]) * 8
            out_features = int(qweight.shape[1])
            if self._reduce_lm_head(prefix):
                qweight = np.ascontiguousarray(qweight[:, self.vocab_map])
                if qzeros.size:
                    qzeros = int4_pack.select_column_packed(
                        qzeros, self.vocab_map, tuple(range(8)))
                scales = self._select_output_axis(scales, self.vocab_map)
            reduced = self._reduce_lm_head(prefix)
            logical_out_features = (int(self.vocab_map.size)
                                    if reduced else out_features)
            packed, permutation = int4_pack.repack_gptq(
                qweight,
                qzeros,
                g_idx,
                self.quant.gptq_zero_point_offset,
                self.int4_gemm_plugin_version,
                num_groups=int(scales.shape[0]),
            )
            zero_correction = int4_pack.gptq_zero_correction(
                qzeros, scales, self.quant.gptq_zero_point_offset)
            weight_name, zero_name, scale_name, group_name = (
                self.gptq_checkpoint_names(prefix))
            compressed = self.is_compressed_gptq(prefix)
            return LinearWeights(
                quant_type,
                packed,
                bias,
                weight_scale=scales,
                zero_correction=zero_correction,
                activation_permutation=permutation,
                group_size=self.module_quant_group_size(prefix),
                weight_recipe=None if reduced else self.checkpoint_binding(
                    [weight_name, zero_name, group_name],
                    "compressed-tensors-gptq" if compressed else "plugin",
                    assemble="gptq_ffn_qweight",
                    zero_point_offset=int(self.quant.gptq_zero_point_offset),
                    preserve_zero_points=True,
                    checkpoint_layout=(
                        "output-major" if compressed else "gptq"),
                    int4_gemm_plugin_version=self.int4_gemm_plugin_version),
                scale_recipe=None if reduced else self.checkpoint_binding(
                    [scale_name],
                    "compressed-tensors-gptq" if compressed else "plugin"),
                logical_out_features=logical_out_features,
                logical_in_features=in_features,
            )
        if quant_type == quantization.QUANT_INT8_GPTQ:
            if self.is_compressed_int8(prefix):
                weight = self.array(prefix + ".weight").astype(np.int8,
                                                               copy=False)
                scales = np.ascontiguousarray(
                    self.f32(prefix + ".weight_scale").reshape(
                        weight.shape[0], -1))
                if scales.shape[1] != 1:
                    raise ValueError(
                        f"{prefix}: dense compressed INT8 scales must be "
                        f"per-channel, got {scales.shape}")
                zero_correction = None
                if self.has(prefix + ".weight_zero_point"):
                    zero_point = self.f32(prefix +
                                          ".weight_zero_point").reshape(
                                              weight.shape[0], -1)
                    if zero_point.shape != scales.shape:
                        raise ValueError(
                            f"{prefix}: dense compressed INT8 zero points "
                            f"must have shape {scales.shape}, got "
                            f"{zero_point.shape}")
                    zero_correction = np.ascontiguousarray(-zero_point *
                                                           scales,
                                                           dtype=np.float16)
                return LinearWeights(
                    quant_type,
                    np.ascontiguousarray(weight),
                    bias,
                    weight_scale=scales,
                    zero_correction=zero_correction,
                    group_size=int(weight.shape[1]),
                    logical_out_features=int(weight.shape[0]),
                    logical_in_features=int(weight.shape[1]),
                )
            qweight, qzeros, scales, g_idx = self.gptq_tensors(prefix)
            in_features = int(qweight.shape[0]) * 4
            out_features = int(qweight.shape[1])
            weight, permutation, group_size = int8_pack.unpack_gptq(
                qweight,
                qzeros,
                num_groups=int(scales.shape[0]),
                g_idx=g_idx,
                zero_point_offset=self.quant.gptq_zero_point_offset,
            )
            zero_correction = int8_pack.gptq_zero_correction(
                qzeros, scales, self.quant.gptq_zero_point_offset)
            if np.array_equal(permutation, np.arange(in_features)):
                permutation = None
            return LinearWeights(
                quant_type,
                weight,
                bias,
                weight_scale=np.ascontiguousarray(scales.T, dtype=np.float16),
                zero_correction=zero_correction,
                activation_permutation=permutation,
                group_size=group_size,
                logical_out_features=out_features,
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

    def _linear_w4a16_fallback(self, prefix: str,
                               source_quant_type: str) -> LinearWeights:
        """Preserve provider NVFP4 weights and disable activation FP4."""
        if source_quant_type != quantization.QUANT_NVFP4:
            raise ValueError(
                f"{prefix}: cannot convert {source_quant_type!r} to W4A16")
        raw = self.linear_nvfp4_raw(prefix)
        return LinearWeights(
            quantization.QUANT_NVFP4_A16,
            raw["packed"],
            raw["bias"],
            weight_scale=raw["weight_scale"],
            weight_scale_2=raw["weight_scale_2"],
            group_size=self.module_quant_group_size(prefix),
        )

    def _linear_w8a16_fallback(self, prefix: str, source_quant_type: str,
                               bias: Optional[np.ndarray]) -> LinearWeights:
        """Requantize an unsupported floating-point weight as W8A16."""
        if source_quant_type in (
                quantization.QUANT_FP8,
                quantization.QUANT_FP8_A16,
                quantization.QUANT_FP8_BLOCK,
        ):
            weight = self._fp8_dense_f32(prefix, source_quant_type)
        elif source_quant_type == quantization.QUANT_MXFP8:
            weight = self._mxfp8_dense_f32(prefix)
        else:
            raise ValueError(
                f"{prefix}: cannot convert {source_quant_type!r} to W8A16")
        return self._quantize_w8a16(weight, bias)

    @staticmethod
    def _quantize_w8a16(weight: np.ndarray,
                        bias: Optional[np.ndarray] = None) -> LinearWeights:
        """Symmetrically quantize a dense output-major matrix per channel."""
        weight = np.asarray(weight, dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError(
                f"W8A16 fallback expects a rank-2 weight, got {weight.shape}")
        absmax = np.max(np.abs(weight), axis=1, keepdims=True)
        scale = np.where(absmax > 0.0, absmax / 127.0, 1.0)
        quantized = np.clip(np.rint(weight / scale), -127, 127).astype(np.int8)
        return LinearWeights(
            quantization.QUANT_INT8_GPTQ,
            np.ascontiguousarray(quantized),
            bias,
            weight_scale=np.ascontiguousarray(scale, dtype=np.float32),
            group_size=int(weight.shape[1]),
            logical_out_features=int(weight.shape[0]),
            logical_in_features=int(weight.shape[1]),
        )

    def _linear_mxfp8_conversion(self, prefix: str, source_quant_type: str,
                                 bias: Optional[np.ndarray]) -> LinearWeights:
        """Requantize a provider FP8 block matrix to native MXFP8."""
        if source_quant_type != quantization.QUANT_FP8_BLOCK:
            raise ValueError(
                f"{prefix}: cannot convert {source_quant_type!r} to MXFP8")
        return self._quantize_mxfp8(
            self._fp8_dense_f32(prefix, source_quant_type), bias)

    @staticmethod
    def _quantize_mxfp8(weight: np.ndarray,
                        bias: Optional[np.ndarray] = None) -> LinearWeights:
        """Quantize an output-major matrix to FP8 E4M3 with E8M0 blocks."""
        weight = np.asarray(weight, dtype=np.float32)
        block_size = 32
        if weight.ndim != 2 or weight.shape[1] % block_size:
            raise ValueError(
                "MXFP8 conversion expects a rank-2 weight with input features "
                f"divisible by {block_size}, got {weight.shape}")
        out_features, in_features = weight.shape
        blocks = weight.reshape(out_features, in_features // block_size,
                                block_size)
        amax = np.max(np.abs(blocks), axis=-1)
        safe = np.where(amax > 0.0, amax, 1.0)
        shared_exp = np.clip(np.floor(np.log2(safe)) - 8.0, -127.0, 127.0)
        shared_exp = np.where(amax > 0.0, shared_exp, 0.0)
        scale = np.where(amax > 0.0, np.exp2(shared_exp), 1.0)
        quantized = f32_to_fp8_e4m3_bytes(blocks / scale[..., None])
        encoded_scale = (shared_exp.astype(np.int16) + 127).astype(np.uint8)
        return LinearWeights(
            quantization.QUANT_MXFP8,
            np.ascontiguousarray(quantized.reshape(weight.shape)),
            bias,
            weight_scale=np.ascontiguousarray(encoded_scale),
            group_size=block_size,
            logical_out_features=out_features,
            logical_in_features=in_features,
        )

    def _fp8_dense_f32(self, prefix: str, quant_type: str) -> np.ndarray:
        """Decode one provider FP8 matrix and apply its weight scales."""
        weight = self.f32(prefix + ".weight")
        scale = self.f32(prefix + ".weight_scale")
        if quant_type == quantization.QUANT_FP8_BLOCK:
            group_size = self.module_quant_group_size(prefix)
            scale = np.asarray(scale, dtype=np.float32)
            if scale.ndim == 4 and scale.shape[1::2] == (1, 1):
                scale = scale[:, 0, :, 0]
            if scale.ndim != 2:
                raise ValueError(
                    f"{prefix}: FP8 block scale must be rank-2, got "
                    f"{scale.shape}")
            out_features, in_features = weight.shape
            expected = (
                (out_features + group_size - 1) // group_size,
                (in_features + group_size - 1) // group_size,
            )
            if scale.shape != expected:
                raise ValueError(
                    f"{prefix}: FP8 block scale shape {scale.shape} does not "
                    f"match {expected} for weight {weight.shape}")
            scale = np.repeat(
                np.repeat(scale, group_size, axis=0),
                group_size,
                axis=1,
            )[:out_features, :in_features]
        elif scale.size == 1:
            scale = float(scale.reshape(-1)[0])
        elif scale.shape in ((weight.shape[0], ), (weight.shape[0], 1)):
            scale = scale.reshape(-1, 1)
        else:
            raise ValueError(
                f"{prefix}: unsupported FP8 weight scale shape {scale.shape}")
        return np.ascontiguousarray(weight * scale, dtype=np.float32)

    def _mxfp8_dense_f32(self, prefix: str) -> np.ndarray:
        """Decode one native MXFP8 matrix for a lower-precision fallback."""
        weight = self.f32(prefix + ".weight")
        scale = self.array(prefix + ".weight_scale").astype(np.int16)
        group_size = self.module_quant_group_size(prefix)
        if scale.ndim != 2 or scale.shape != (weight.shape[0],
                                              weight.shape[1] // group_size):
            raise ValueError(
                f"{prefix}: MXFP8 scale shape {scale.shape} does not match "
                f"weight {weight.shape} and block size {group_size}")
        multiplier = np.exp2(scale.astype(np.float32) - 127.0)
        multiplier = np.repeat(multiplier, group_size, axis=1)
        return np.ascontiguousarray(weight * multiplier, dtype=np.float32)

    def linear(self, prefix: str, quant_type: str) -> LinearWeights:
        """Load a linear and apply an optional output-vocabulary map."""
        if (quant_type
                in (quantization.QUANT_NVFP4, quantization.QUANT_NVFP4_A16)
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
        zero_correction = None
        activation_permutation = None
        activation_permutation_recipe = None
        if quant_type == quantization.QUANT_INT4_AWQ:
            qweight_shape = self.store.shape(self._resolve(prefix +
                                                           ".qweight"))
            in_features = int(qweight_shape[0])
            out_features = int(qweight_shape[1]) * 8
            weight_recipe = self.checkpoint_binding(
                [prefix + ".qweight"],
                assemble="awq_ffn_qweight",
                int4_gemm_plugin_version=self.int4_gemm_plugin_version)
            scale = self.parameter_spec(prefix + ".scales", np.float16)
            scale_recipe = self.checkpoint_binding([prefix + ".scales"])
            zero_correction = int4_pack.awq_zero_correction(
                self.array(prefix + ".qzeros"), self.f16(prefix + ".scales"))
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
            weight_name, zero_name, scale_name, group_name = (
                self.gptq_checkpoint_names(prefix))
            compressed = self.is_compressed_gptq(prefix)
            qweight_shape = self.store.shape(self._resolve(weight_name))
            if compressed:
                out_features = int(qweight_shape[0])
                in_features = int(qweight_shape[1]) * 8
            else:
                in_features = int(qweight_shape[0]) * 8
                out_features = int(qweight_shape[1])
            if self.has(group_name):
                activation_permutation = ParameterSpec((in_features, ),
                                                       np.int32)
                activation_permutation_recipe = self.checkpoint_binding(
                    [group_name],
                    assemble="gptq_activation_permutation",
                    group_size=self.module_quant_group_size(prefix))
            weight_recipe = self.checkpoint_binding(
                [weight_name, zero_name, group_name],
                "compressed-tensors-gptq" if compressed else "plugin",
                assemble="gptq_ffn_qweight",
                zero_point_offset=int(self.quant.gptq_zero_point_offset),
                preserve_zero_points=True,
                checkpoint_layout=("output-major" if compressed else "gptq"),
                int4_gemm_plugin_version=self.int4_gemm_plugin_version)
            source_scale_shape = self.store.shape(self._resolve(scale_name))
            scale_shape = ((int(source_scale_shape[1]),
                            int(source_scale_shape[0])) if compressed else
                           tuple(int(size) for size in source_scale_shape))
            scale = ParameterSpec(scale_shape, np.float16)
            scale_recipe = self.checkpoint_binding(
                [scale_name],
                "compressed-tensors-gptq" if compressed else "plugin")
            if self.has(zero_name):
                qzeros, scales, _ = self.gptq_sidecars(prefix)
                zero_correction = int4_pack.gptq_zero_correction(
                    qzeros, scales, self.quant.gptq_zero_point_offset)

        return LinearWeights(
            quant_type,
            ParameterSpec(packed_shape(out_features, in_features), np.int8),
            bias,
            weight_scale=scale,
            zero_correction=zero_correction,
            pre_quant_scale=pre_quant_scale,
            activation_permutation=activation_permutation,
            group_size=self.module_quant_group_size(prefix),
            weight_recipe=weight_recipe,
            bias_recipe=bias_recipe,
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
        (weight_name, scale_name, global_scale_name, _,
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
        expected_scale_shape = (out_features, in_features //
                                self.module_quant_group_size(prefix))
        if tuple(scale_shape) != expected_scale_shape:
            raise ValueError(
                f"{prefix}: NVFP4 scale shape {scale_shape} does not match "
                f"{expected_scale_shape}")
        weight_scale_2 = self._nvfp4_global_scale(
            self.store.get_scalar_f32(self._resolve(global_scale_name)),
            reciprocal)
        input_scale = self.nvfp4_input_scale(prefix)
        bias_name = prefix + ".bias"
        bias = None
        if self.has(bias_name):
            bias = (self.parameter_spec(bias_name)
                    if self._weightless_engine else self.f16(bias_name))
        # Keep the checkpoint's byte-addressable packed shape. Logical FP4
        # dimensions are carried separately for plugin shape validation.
        return LinearWeights(
            quantization.QUANT_NVFP4,
            ParameterSpec(weight_shape, np.uint8),
            bias,
            weight_scale=ParameterSpec(scale_shape, np.uint8),
            weight_scale_2=weight_scale_2,
            input_scale=input_scale,
            group_size=self.module_quant_group_size(prefix),
            weight_recipe=self.checkpoint_binding([weight_name],
                                                  "nvfp4_packed"),
            scale_recipe=self.checkpoint_binding([scale_name], "nvfp4_scale"),
            logical_out_features=out_features,
            logical_in_features=in_features,
        )

    def linear_nvfp4_a16_metadata(self,
                                  prefix: str) -> Optional[LinearWeights]:
        """Describe the Marlin W4A16 plugin tensors from checkpoint headers."""
        if self._reduce_lm_head(prefix):
            return None
        (weight_name, scale_name, global_scale_name, _, reciprocal) = \
            self._nvfp4_names(prefix)
        weight_key = self._resolve(weight_name)
        scale_key = self._resolve(scale_name)
        weight_shape = self.store.shape(weight_key)
        scale_shape = self.store.shape(scale_key)
        if len(weight_shape) != 2 or len(scale_shape) != 2:
            raise ValueError(
                f"{prefix}: NVFP4 weight and scale must be rank-2")
        out_features = int(weight_shape[0])
        in_features = int(
            scale_shape[1]) * self.module_quant_group_size(prefix)
        if int(scale_shape[0]) != out_features:
            raise ValueError(
                f"{prefix}: NVFP4 weight and scale rows do not match")
        packed_bytes = int(self.store.tensor_identity(weight_key, 0)["bytes"])
        if packed_bytes != out_features * in_features // 2:
            raise ValueError(
                f"{prefix}: NVFP4 packed weight byte count is inconsistent")
        padded_out = (out_features + 127) // 128 * 128
        common = {
            "num_experts": 1,
            "keys_per_expert": 1,
            "source_n": out_features,
            "source_k": in_features,
            "output_n": padded_out,
            "output_k": in_features,
        }
        bias_name = prefix + ".bias"
        bias = None
        bias_recipe = None
        if self.has(bias_name):
            bias = self.parameter_value("fp16", bias_name,
                                        lambda: self.parameter_spec(bias_name),
                                        lambda: self.f16(bias_name))
            bias_recipe = self.checkpoint_binding([bias_name], "fp16")
        return LinearWeights(
            quantization.QUANT_NVFP4_A16,
            ParameterSpec((1, in_features // 16, 8 * padded_out), np.int8),
            bias,
            weight_scale=ParameterSpec((1, in_features // 16, padded_out),
                                       np.int8),
            weight_scale_2=ParameterSpec((1, ), np.float16),
            weight_recipe=self.checkpoint_binding([weight_name],
                                                  "nvfp4_packed",
                                                  "nvfp4_a16_qweight",
                                                  **common),
            bias_recipe=bias_recipe,
            scale_recipe=self.checkpoint_binding([scale_name], "nvfp4_scale",
                                                 "nvfp4_a16_scale", **common),
            weight_scale_2_recipe=self.checkpoint_binding(
                [global_scale_name],
                "plugin",
                "nvfp4_a16_alpha",
                num_experts=1,
                keys_per_expert=1,
                reciprocal_alpha=reciprocal,
                alpha_multiplier=128.0),
            logical_out_features=out_features,
            logical_in_features=in_features,
        )

    def linear_refit_metadata(self, prefix: str,
                              quant_type: str) -> Optional[LinearWeights]:
        """Describe native quantized constants without loading weight payloads.

        Global quantization scales remain build-time constants; matrix weights
        and NVFP4 block scales are supplied through checkpoint refitting.
        """
        bias_name = prefix + ".bias"
        bias = self.parameter_spec(bias_name) if self.has(bias_name) else None
        bias_recipe = (self.checkpoint_binding([bias_name], "fp16")
                       if bias is not None else None)
        if quant_type in (quantization.QUANT_FP8, quantization.QUANT_FP8_A16):
            weight_name = prefix + ".weight"
            if self.store.dtype(self._resolve(weight_name)) != "F8_E4M3":
                return None
            return LinearWeights(
                quant_type,
                self.parameter_spec(weight_name, np.uint8),
                bias,
                weight_scale=np.ascontiguousarray(
                    self.f16(prefix + ".weight_scale").reshape(-1)),
                input_scale=(self.store.get_scalar_f32(
                    self._resolve(prefix + ".input_scale")) if quant_type
                             == quantization.QUANT_FP8 else None),
                weight_recipe=self.checkpoint_binding([weight_name], "fp8"),
                bias_recipe=bias_recipe,
            )
        descriptor = self.linear_nvfp4_tp_metadata(prefix)
        if descriptor is None:
            return None
        return replace(descriptor, bias=bias, bias_recipe=bias_recipe)

    def linear_descriptor(self,
                          prefix: str,
                          quant_type: str,
                          *,
                          external_kind: str = "") -> LinearWeights:
        """Return a dense projection payload or metadata, as required."""
        if self._weightless_engine and quant_type == quantization.QUANT_FP16:
            descriptor = self.linear_metadata(prefix, quant_type)
            if descriptor is not None:
                return descriptor
        if self._weightless_engine and quant_type in (
                quantization.QUANT_FP8, quantization.QUANT_FP8_A16,
                quantization.QUANT_NVFP4):
            if not (self.uses_w4a16_fallback(quant_type)
                    or self.uses_w8a16_fallback(quant_type)
                    or self.converts_to_mxfp8(quant_type)):
                descriptor = self.linear_refit_metadata(prefix, quant_type)
                if descriptor is not None:
                    return descriptor
        if (quant_type
                in (quantization.QUANT_NVFP4, quantization.QUANT_NVFP4_A16)
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
        if (quant_type == quantization.QUANT_NVFP4_A16
                and external_kind and self._policy.externalizes_parameter(
                    external_kind, prefix)):
            descriptor = self.linear_nvfp4_a16_metadata(prefix)
            if descriptor is not None:
                return descriptor
        if (quant_type in (quantization.QUANT_Q8_0, quantization.QUANT_IQ2_XXS)
                and not self._reduce_lm_head(prefix)
                and self._policy.externalizes_parameter(quant_type, prefix)):
            return self._linear_gguf(prefix, quant_type,
                                     self.opt_f16(prefix + ".bias"), True)
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

    def _linear_gguf(self, prefix: str, quant_type: str,
                     bias: Optional[ParameterValue],
                     metadata: bool) -> LinearWeights:
        """Describe a packed Q8_0 or IQ2_XXS projection.

        Q8_0 blocks are consumed exactly as stored. IQ2_XXS blocks are split
        into payload and FP16 bases for W2A16GemmPlugin, by the runtime when
        ``metadata`` is set and here otherwise. Nothing is dequantized.
        """
        name = prefix + ".weight"
        shape = tuple(
            int(dim) for dim in self.store.shape(self._resolve(name)))
        block_bytes = 34 if quant_type == quantization.QUANT_Q8_0 else 66
        if len(shape) != 3 or shape[2] != block_bytes:
            raise ValueError(f"{prefix}: expected {quant_type} uint8 "
                             f"[N, K/block, {block_bytes}], got {shape}")
        rows, blocks, _ = shape
        if quant_type == quantization.QUANT_Q8_0:
            weight = (ParameterSpec(shape, np.dtype(np.int8)) if metadata else
                      np.ascontiguousarray(self.array(name)).view(np.int8))
            return LinearWeights(quant_type,
                                 weight,
                                 bias,
                                 weight_recipe=self.checkpoint_binding([name]),
                                 logical_out_features=rows,
                                 logical_in_features=blocks * 32)
        packing = 128 if rows % 128 == 0 else 8
        if metadata:
            values = ParameterSpec((rows // packing, blocks, packing, 64),
                                   np.dtype(np.int8))
            bases = ParameterSpec((blocks, rows), np.dtype(np.float16))
        else:
            values, bases = iq2_xxs.pack_dense_weights(self.array(name))
            values = values.view(np.int8)
        return LinearWeights(quant_type,
                             values,
                             bias,
                             weight_scale=bases,
                             weight_recipe=self.checkpoint_binding(
                                 [name], assemble="iq2_xxs_gemm_values"),
                             scale_recipe=self.checkpoint_binding(
                                 [name], assemble="iq2_xxs_gemm_bases"),
                             logical_out_features=rows,
                             logical_in_features=blocks * 256)

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
    def shard_segments(value: np.ndarray,
                       segments: Sequence[int],
                       tp_size: int,
                       tp_rank: int,
                       axis: int = 0) -> np.ndarray:
        """Shard each concatenated segment independently along one axis."""
        if tp_size < 1:
            raise ValueError("tp_size must be positive")
        if tp_rank < 0 or tp_rank >= tp_size:
            raise ValueError("tp_rank must be in [0, tp_size)")
        if axis < 0 or axis >= value.ndim:
            raise ValueError(f"invalid shard axis {axis} for {value.shape}")
        segment_sizes = tuple(int(size) for size in segments)
        if not segment_sizes or any(size <= 0 for size in segment_sizes):
            raise ValueError("shard segments must be positive")
        if sum(segment_sizes) != value.shape[axis]:
            raise ValueError(
                f"shard segments {segment_sizes} do not cover axis {axis} "
                f"of shape {value.shape}")
        if any(size % tp_size for size in segment_sizes):
            raise ValueError(
                f"shard segments {segment_sizes} are not divisible by "
                f"tp_size {tp_size}")
        if tp_size == 1:
            return np.ascontiguousarray(value)

        pieces = []
        offset = 0
        for size in segment_sizes:
            local_size = size // tp_size
            selected = [slice(None)] * value.ndim
            start = offset + tp_rank * local_size
            selected[axis] = slice(start, start + local_size)
            pieces.append(value[tuple(selected)])
            offset += size
        return np.ascontiguousarray(np.concatenate(pieces, axis=axis))

    @staticmethod
    def shard_linear(
        linear: LinearWeights,
        mode: str,
        tp_size: int,
        tp_rank: int,
        output_segments: Sequence[int] = ()) -> LinearWeights:
        """Return one tensor-parallel shard of a linear."""
        if tp_size == 1 or mode == "replicated":
            return linear
        full_out = linear.out_features
        full_in = linear.in_features
        split_size = full_out if mode == "column" else full_in
        if split_size % tp_size:
            raise ValueError(
                f"cannot {mode}-shard linear dimension {split_size} over {tp_size} ranks"
            )
        output_segments = tuple(int(size) for size in output_segments)
        if output_segments:
            if mode != "column":
                raise ValueError(
                    "output segments require column-parallel sharding")
            if sum(output_segments) != full_out:
                raise ValueError(
                    f"output segments {output_segments} do not match "
                    f"linear output size {full_out}")

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
            if output_segments and axis == 0:
                if isinstance(value, ParameterSpec):
                    raise ValueError(
                        "segmented TP projections must remain engine "
                        "constants")
                return (Weights.shard_segments(value, output_segments, tp_size,
                                               tp_rank, axis), None)
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
            if linear.quant_type in (quantization.QUANT_NVFP4,
                                     quantization.QUANT_NVFP4_A16):
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

        zero_correction = linear.zero_correction
        if zero_correction is not None:
            zero_correction, _ = split(zero_correction, None,
                                       0 if mode == "column" else 1)

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
            zero_correction=zero_correction,
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
            dense = nvfp4_pack.decode_modelopt_nvfp4(
                packed, sf, ws2, self.module_quant_group_size(prefix))
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
        """Load one quantized or plain expert projection as dense FP32."""
        if not self.is_nvfp4(prefix):
            weight_name = prefix + ".weight"
            weight = self.f32(weight_name)
            stored_dtype = self.store.dtype(self._resolve(weight_name))
            if stored_dtype in ("F8_E4M3", "F8_E4M3FN"):
                scale = self.f32(prefix + ".weight_scale")
                quant_type = (quantization.QUANT_FP8_BLOCK
                              if scale.ndim >= 2 and scale.shape[-1] > 1 else
                              quantization.QUANT_FP8_A16)
                weight = self._fp8_dense_f32(prefix, quant_type)
            return np.ascontiguousarray(weight, dtype=np.float32)
        (weight_name, scale_name, global_scale_name, _,
         reciprocal) = self._nvfp4_names(prefix)
        packed = self.store.get_packed_fp4(self._resolve(weight_name))
        sf = self.store.get_fp8_bytes(self._resolve(scale_name))
        ws2 = self._nvfp4_global_scale(
            self.store.get_scalar_f32(self._resolve(global_scale_name)),
            reciprocal)
        return nvfp4_pack.decode_modelopt_nvfp4(
            packed, sf, ws2, self.module_quant_group_size(prefix))

    def expert_raw_nvfp4(self, prefix: str) -> dict:
        """Return raw NVFP4 bytes for one expert projection (byte-reuse path)."""
        (weight_name, scale_name, global_scale_name, _,
         reciprocal) = self._nvfp4_names(prefix)
        alpha = self._nvfp4_global_scale(
            self.store.get_scalar_f32(self._resolve(global_scale_name)),
            reciprocal)
        input_scale = self.nvfp4_input_scale(prefix)
        return {
            "packed": self.store.get_packed_fp4(self._resolve(weight_name)),
            "sf": self.store.get_fp8_bytes(self._resolve(scale_name)),
            "alpha": alpha,
            "input_scale": input_scale,
        }


def component_quantization(model_dir: str, conversion: Optional[ModuleType],
                           component: str, quant: quantization.QuantConfig,
                           spec_type: str,
                           spec_role: str) -> quantization.QuantConfig:
    """Return *quant*, or fp16 when this component ships no quantized tensor.

    Only conversions that set ``QUANTIZATION_FOLLOWS_COMPONENT_WEIGHTS`` take
    part: their checkpoints state one root ``quantization_config`` that may
    cover only some components.
    """
    if (not getattr(conversion, "QUANTIZATION_FOLLOWS_COMPONENT_WEIGHTS",
                    False) or quant.quant_type == quantization.QUANT_FP16
            or _carries_quantized_tensors(model_dir, conversion, component,
                                          quant.quant_type, spec_type,
                                          spec_role)):
        return quant
    logger.info(
        "Building %s in fp16: the checkpoint declares %s at its root but "
        "ships this component unquantized", component, quant.quant_type)
    return quantization.QuantConfig(group_size=quant.group_size,
                                    kv_cache_quant=quant.kv_cache_quant)


def _carries_quantized_tensors(model_dir: str, conversion: ModuleType,
                               component: str, quant_type: str, spec_type: str,
                               spec_role: str) -> bool:
    """Whether a quantized tensor in the checkpoint belongs to *component*.

    A tensor belongs to the component with the longest matching checkpoint
    prefix, so ``talker.code_predictor.*`` is not the talker's. A name no
    component prefixes counts for every component, because ``Weights._resolve``
    tries the bare name first.
    """
    resolve_candidates = getattr(conversion, "resolve_candidates", None)

    def prefixes(name: str) -> Tuple[str, ...]:
        if resolve_candidates is None:
            return ()
        return tuple(prefix
                     for prefix in resolve_candidates("",
                                                      component=name,
                                                      spec_type=spec_type,
                                                      spec_role=spec_role,
                                                      quant_type=quant_type)
                     if prefix)

    own = prefixes(component)
    known = {
        prefix
        for other in contracts.Component
        for prefix in prefixes(other.value)
    }
    checkpoint_dir = getattr(conversion, "checkpoint_dir", None)
    source_dir = (checkpoint_dir(model_dir, component)
                  if checkpoint_dir is not None else model_dir)
    try:
        with SafetensorsStore(source_dir) as store:
            names = store.keys()
    except (OSError, ValueError):
        return True
    for name in names:
        if not name.endswith(Weights._QUANT_MARKERS):
            continue
        longest = max(
            (len(prefix) for prefix in known if name.startswith(prefix)),
            default=0)
        if longest == 0 or any(
                len(prefix) == longest and name.startswith(prefix)
                for prefix in own):
            return True
    return False
