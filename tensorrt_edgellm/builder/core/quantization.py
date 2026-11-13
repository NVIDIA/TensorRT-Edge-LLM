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
"""Checkpoint quantization contracts used by direct TensorRT graph builders."""

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from types import ModuleType
from typing import Dict, List, Optional, Set, Tuple

from . import safetensors_np

QUANT_FP16 = "fp16"
QUANT_FP8 = "fp8"
QUANT_FP8_A16 = "fp8_a16"
QUANT_FP8_BLOCK = "fp8_block"
QUANT_MXFP8 = "mxfp8"
QUANT_NVFP4 = "nvfp4"
QUANT_NVFP4_A16 = "nvfp4_a16"
QUANT_Q8_0 = "q8_0"
QUANT_IQ2_XXS = "iq2_xxs"
QUANT_INT4_AWQ = "int4_awq"
QUANT_INT4_AWQ_MODELOPT = "int4_awq_modelopt"
QUANT_INT4_GPTQ = "int4_gptq"
QUANT_INT8_GPTQ = "int8_gptq"
QUANT_INT8_SQ = "int8_sq"
QUANT_MIXED = "mixed_precision"

CHECKPOINT_FORMAT_NATIVE = "native"
CHECKPOINT_FORMAT_COMPRESSED_TENSORS = "compressed-tensors"
CHECKPOINT_FORMAT_FP8_SCALE_INV = "fp8-scale-inv"

QUANT_TYPES = frozenset((
    QUANT_FP16,
    QUANT_FP8,
    QUANT_FP8_A16,
    QUANT_FP8_BLOCK,
    QUANT_MXFP8,
    QUANT_NVFP4,
    QUANT_NVFP4_A16,
    QUANT_Q8_0,
    QUANT_IQ2_XXS,
    QUANT_INT4_AWQ,
    QUANT_INT4_AWQ_MODELOPT,
    QUANT_INT4_GPTQ,
    QUANT_INT8_GPTQ,
    QUANT_INT8_SQ,
))


def _module_setting(values: dict, module_name: str):
    """Return the nearest setting inherited from a module or its parents."""
    candidate = module_name
    while candidate:
        if candidate in values:
            return values[candidate]
        candidate = candidate.rpartition(".")[0]
    return None


@dataclass(frozen=True)
class QuantConfig:
    """Effective quantization for checkpoint-backed linear layers."""

    quant_type: str = QUANT_FP16
    group_size: int = 1
    gptq_zero_point_offset: int = 1
    kv_cache_quant: Optional[str] = None
    excluded: Tuple[str, ...] = ()
    layer_overrides: Dict[str, str] = field(default_factory=dict)
    layer_group_sizes: Dict[str, int] = field(default_factory=dict)
    is_mixed_precision: bool = False
    checkpoint_format: str = CHECKPOINT_FORMAT_NATIVE

    def module_type(self,
                    module_name: str,
                    tie_word_embeddings: bool = False) -> str:
        """Return the concrete precision used by one linear module."""
        normalized = normalize_module_name(module_name)
        if normalized in self.excluded:
            return QUANT_FP16
        if (normalized == "lm_head" and tie_word_embeddings
                and normalized not in self.layer_overrides
                and self.quant_type == QUANT_FP16):
            return QUANT_FP16
        if self.layer_overrides:
            fallback = QUANT_FP16 if self.is_mixed_precision else self.quant_type
            override = _module_setting(self.layer_overrides, normalized)
            return fallback if override is None else override
        return self.quant_type

    def module_group_size(self, module_name: str) -> int:
        """Return the checkpoint group size used by one linear module."""
        normalized = normalize_module_name(module_name)
        group_size = _module_setting(self.layer_group_sizes, normalized)
        return int(self.group_size if group_size is None else group_size)

    @property
    def is_quantized(self) -> bool:
        """Return whether at least one module uses quantized weights."""
        return self.quant_type != QUANT_FP16 or bool(self.layer_overrides)


def normalize_module_name(name: str) -> str:
    """Normalize the frontend's ordinary ``model.`` graph namespace."""
    if name.startswith("model."):
        name = name[len("model."):]
    return name


def parse_quantization(model_dir: str,
                       root: dict,
                       component: dict,
                       conversion: Optional[ModuleType] = None,
                       quantize_activations: bool = True) -> QuantConfig:
    """Parse ModelOpt, AWQ, GPTQ, SmoothQuant, and mixed checkpoints."""
    embedded = (component.get("quantization_config")
                or root.get("quantization_config"))
    hf_path = os.path.join(model_dir, "hf_quant_config.json")
    if os.path.isfile(hf_path):
        with open(hf_path) as quant_file:
            quantization = json.load(quant_file).get("quantization", {})
        algorithm = str(quantization.get("quant_algo") or "").upper()
        if algorithm == "MIXED_PRECISION":
            if (isinstance(embedded, dict) and str(
                    embedded.get("quant_method", "")).lower() == "modelopt"
                    and _has_modelopt_weight_groups(embedded)):
                return _parse_modelopt_mixed_precision(model_dir, embedded,
                                                       conversion,
                                                       quantize_activations,
                                                       quantization)
            dominant, group_size, overrides, layer_group_sizes = _parse_mixed_precision(
                quantization.get("quantized_layers", {}), conversion,
                quantize_activations)
            return QuantConfig(
                quant_type=dominant,
                group_size=group_size,
                kv_cache_quant=_resolve_kv_cache_quant(
                    model_dir, quantization.get("kv_cache_quant_algo")),
                excluded=tuple(
                    _effective_exclusions(
                        model_dir,
                        list(quantization.get("exclude_modules",
                                              [])), conversion)),
                layer_overrides=overrides,
                layer_group_sizes=layer_group_sizes,
                is_mixed_precision=True,
            )

        quant_type = _activation_quant_type(
            _algorithm_to_type(algorithm, conversion), quantize_activations)
        group_size = int(quantization.get("group_size", 1))
        if quant_type == QUANT_MXFP8 and group_size == 1:
            group_size = 32
        if quant_type in (QUANT_NVFP4, QUANT_NVFP4_A16) and group_size == 1:
            group_size = 16
        if quant_type in (QUANT_INT4_AWQ,
                          QUANT_INT4_AWQ_MODELOPT) and group_size == 1:
            group_size = 128
        excluded = _effective_exclusions(
            model_dir, list(quantization.get("exclude_modules", [])),
            conversion)
        excluded.extend(
            name for name in _detect_plain_weights(model_dir, conversion)
            if name not in excluded)
        return QuantConfig(
            quant_type=quant_type,
            group_size=group_size,
            kv_cache_quant=_resolve_kv_cache_quant(
                model_dir, quantization.get("kv_cache_quant_algo")),
            excluded=tuple(sorted(set(excluded))),
        )

    if not isinstance(embedded, dict):
        return QuantConfig()

    algorithm = str(embedded.get("quant_algo") or "").upper()
    if algorithm == "MIXED_PRECISION":
        if embedded.get("config_groups"):
            return _parse_modelopt_mixed_precision(model_dir, embedded,
                                                   conversion,
                                                   quantize_activations)
        dominant, group_size, overrides, layer_group_sizes = _parse_mixed_precision(
            embedded.get("quantized_layers", {}), conversion,
            quantize_activations)
        return QuantConfig(
            quant_type=dominant,
            group_size=group_size,
            kv_cache_quant=_normalize_kv(embedded.get("kv_cache_quant_algo")),
            excluded=tuple(
                _effective_exclusions(
                    model_dir,
                    list(
                        embedded.get("exclude_modules",
                                     embedded.get("ignore", []))),
                    conversion)),
            layer_overrides=overrides,
            layer_group_sizes=layer_group_sizes,
            is_mixed_precision=True,
        )

    method = str(embedded.get("quant_method", "")).lower()
    if method in ("awq", "autoawq"):
        return QuantConfig(
            quant_type=QUANT_INT4_AWQ,
            group_size=int(embedded.get("group_size", 128)),
            excluded=tuple(
                _detect_unquantized_packed_modules(model_dir, conversion)),
        )
    if method in ("gptq", "auto-round", "autoround"):
        bits = int(embedded.get("bits", 4))
        if bits not in (4, 8):
            raise ValueError(
                f"unsupported GPTQ bit width {bits}; expected 4 or 8")
        return QuantConfig(
            quant_type=(QUANT_INT4_GPTQ if bits == 4 else QUANT_INT8_GPTQ),
            group_size=int(embedded.get("group_size", 128)),
            gptq_zero_point_offset=_detect_gptq_zero_point_offset(
                model_dir, embedded, bits),
            excluded=tuple(
                _detect_unquantized_packed_modules(model_dir, conversion)),
        )
    if method in ("compressed-tensors", "compressed_tensors"):
        return _parse_compressed_tensors(model_dir, embedded, conversion,
                                         quantize_activations)
    if method == "fp8":
        block_size = embedded.get("weight_block_size")
        static_activations = str(embedded.get("activation_scheme",
                                              "")).lower() == "static"
        return QuantConfig(
            quant_type=(QUANT_FP8_BLOCK if block_size else
                        QUANT_FP8 if static_activations else QUANT_FP8_A16),
            group_size=(int(block_size[-1]) if block_size else 1),
            excluded=tuple(_detect_plain_weights(model_dir, conversion)),
            checkpoint_format=(CHECKPOINT_FORMAT_FP8_SCALE_INV
                               if block_size else CHECKPOINT_FORMAT_NATIVE),
        )

    if not algorithm:
        if method:
            raise ValueError(
                f"unsupported checkpoint quantization method {method!r}")
        return QuantConfig()
    quant_type = _activation_quant_type(
        _algorithm_to_type(algorithm, conversion), quantize_activations)
    group_size = int(embedded.get("group_size", 1))
    config_groups = embedded.get("config_groups") or {}
    if config_groups:
        first_group = next(iter(config_groups.values()), {})
        group_size = int(
            first_group.get("weights", {}).get("group_size", group_size))
    if quant_type in (QUANT_NVFP4, QUANT_NVFP4_A16) and group_size == 1:
        group_size = 16
    if quant_type == QUANT_MXFP8 and group_size == 1:
        group_size = 32
    if quant_type in (QUANT_INT4_AWQ,
                      QUANT_INT4_AWQ_MODELOPT) and group_size == 1:
        group_size = 128
    return QuantConfig(
        quant_type=quant_type,
        group_size=group_size,
        kv_cache_quant=_resolve_kv_cache_quant(
            model_dir, "fp8" if embedded.get("kv_cache_scheme") else None),
        excluded=tuple(
            _effective_exclusions(model_dir, list(embedded.get("ignore", [])),
                                  conversion)),
    )


def algorithm_to_type(algorithm: str) -> str:
    """Map checkpoint algorithm names to direct-builder precision names."""
    algorithm = str(algorithm).upper()
    if "IQ2_XXS" in algorithm:
        return QUANT_IQ2_XXS
    if "Q8_0" in algorithm:
        return QUANT_Q8_0
    # ModelOpt writes the same block-scaled FP8 recipe as FP8_PB_WO in
    # config.json and FP8_BLOCK_SCALES in hf_quant_config.json; the sidecar
    # wins, so both names have to reach the block decoder.
    if "FP8_PB" in algorithm or "FP8_BLOCK" in algorithm:
        return QUANT_FP8_BLOCK
    if "MXFP4" in algorithm:
        raise ValueError(
            f"unsupported checkpoint quantization algorithm {algorithm!r}: "
            "MXFP4 weight layouts are not implemented")
    if "NVFP4" in algorithm:
        # Weight-only NVFP4 (ModelOpt ``W4A16_NVFP4``) keeps FP16 activations.
        return QUANT_NVFP4_A16 if "W4A16" in algorithm else QUANT_NVFP4
    if "MXFP8" in algorithm:
        return QUANT_MXFP8
    if "FP8" in algorithm:
        return QUANT_FP8
    if "FP4" in algorithm:
        raise ValueError(
            f"unsupported checkpoint quantization algorithm {algorithm!r}: "
            "only NVFP4 weight layouts are implemented")
    if "W4A16" in algorithm and "AWQ" in algorithm:
        return QUANT_INT4_AWQ_MODELOPT
    if "AWQ" in algorithm or "INT4_AWQ" in algorithm:
        return QUANT_INT4_AWQ
    if "GPTQ" in algorithm and ("INT8" in algorithm or "W8A16" in algorithm):
        return QUANT_INT8_GPTQ
    if "GPTQ" in algorithm:
        return QUANT_INT4_GPTQ
    if "W8A8" in algorithm or "INT8" in algorithm:
        return QUANT_INT8_SQ
    if not algorithm:
        return QUANT_FP16
    raise ValueError(
        f"unsupported checkpoint quantization algorithm {algorithm!r}")


def _algorithm_to_type(algorithm: str,
                       conversion: Optional[ModuleType]) -> str:
    quant_type = algorithm_to_type(algorithm)
    hook = getattr(conversion, "quant_type_for_algorithm", None)
    return hook(str(algorithm).upper(), quant_type) if hook else quant_type


def _activation_quant_type(quant_type: str, quantize_activations: bool) -> str:
    """Select W4A16 when packed NVFP4 weights keep FP16 activations."""
    if (not quantize_activations
            and quant_type in (QUANT_NVFP4, QUANT_NVFP4_A16)):
        return QUANT_NVFP4_A16
    return quant_type


def _parse_compressed_tensors(model_dir: str, embedded: dict,
                              conversion: Optional[ModuleType],
                              quantize_activations: bool) -> QuantConfig:
    """Translate compressed-tensors metadata into executable contracts."""
    config_groups = embedded.get("config_groups") or {}
    if not config_groups:
        raise ValueError(
            "unsupported compressed-tensors checkpoint: config_groups is empty"
        )

    default_format = str(embedded.get("format", "")).lower()
    contracts = [
        _compressed_group_contract(group, default_format)
        for group in config_groups.values()
    ]
    contracts = [(_activation_quant_type(quant_type,
                                         quantize_activations), group_size)
                 for quant_type, group_size in contracts]
    quant_types = {quant_type for quant_type, _ in contracts}

    layer_overrides: Dict[str, str] = {}
    is_mixed_precision = len(quant_types) > 1
    if is_mixed_precision:
        layer_overrides = _infer_serialized_module_types(
            model_dir, conversion, quant_types, quantize_activations)
        represented = set(layer_overrides.values())
        missing = quant_types.difference(represented)
        if missing:
            raise ValueError(
                "compressed-tensors mixed-precision metadata does not match "
                "checkpoint tensor layouts; missing module payloads for "
                f"{', '.join(sorted(missing))}")
        counts = Counter(layer_overrides.values())
        quant_type = max(sorted(quant_types), key=counts.__getitem__)
    else:
        quant_type = next(iter(quant_types))

    grouped_sizes = {
        size
        for contract_type, size in contracts
        if contract_type in (QUANT_NVFP4, QUANT_NVFP4_A16, QUANT_INT4_GPTQ,
                             QUANT_INT8_GPTQ) and size > 0
    }
    if len(grouped_sizes) > 1:
        raise ValueError(
            "compressed-tensors checkpoint uses incompatible grouped weight "
            f"sizes: {sorted(grouped_sizes)}")
    group_size = (next(iter(grouped_sizes)) if grouped_sizes else next(
        size for contract_type, size in contracts
        if contract_type == quant_type))

    excluded = _effective_exclusions(model_dir,
                                     list(embedded.get("ignore",
                                                       [])), conversion)
    excluded.extend(name
                    for name in _detect_plain_weights(model_dir, conversion)
                    if name not in excluded)
    return QuantConfig(
        quant_type=quant_type,
        group_size=group_size,
        gptq_zero_point_offset=_compressed_zero_point_offset(
            model_dir, config_groups),
        kv_cache_quant=_resolve_kv_cache_quant(
            model_dir, _compressed_kv_cache_quant(embedded)),
        excluded=tuple(sorted(set(excluded))),
        layer_overrides=layer_overrides,
        is_mixed_precision=is_mixed_precision,
        checkpoint_format=CHECKPOINT_FORMAT_COMPRESSED_TENSORS,
    )


def _compressed_zero_point_offset(model_dir: str, config_groups: dict) -> int:
    """Return the integer zero-point convention declared by the checkpoint."""
    integer_weights = [
        group.get("weights") or {} for group in config_groups.values()
        if str((group.get("weights") or {}).get("type", "")).lower() == "int"
    ]
    if not integer_weights:
        return 1
    if any(weights.get("symmetric") is False for weights in integer_weights):
        keys = _checkpoint_keys(model_dir)
        if not any(key.endswith(".weight_zero_point") for key in keys):
            raise ValueError(
                "asymmetric compressed-tensors weights require explicit "
                "weight_zero_point tensors")
    return 0


def _compressed_kv_cache_quant(embedded: dict) -> Optional[str]:
    """Parse the compressed-tensors KV-cache quantization contract."""
    scheme = embedded.get("kv_cache_scheme")
    if not scheme:
        return None
    if not isinstance(scheme, dict):
        raise ValueError(
            "compressed-tensors kv_cache_scheme must be an object")
    value_type = str(scheme.get("type", "")).lower()
    bits = int(scheme.get("num_bits") or 0)
    if value_type == "float" and bits == 8:
        return "fp8"
    raise ValueError("unsupported compressed-tensors KV-cache quantization: "
                     f"type={value_type!r}, bits={bits}")


def _compressed_group_contract(group: dict,
                               default_format: str) -> Tuple[str, int]:
    """Return the Edge-LLM lowering represented by one compression group."""
    weights = group.get("weights") or {}
    format_name = str(group.get("format") or default_format).lower()
    weight_type = str(weights.get("type", "")).lower()
    bits = int(weights.get("num_bits") or 0)

    if "nvfp4" in format_name:
        return (QUANT_NVFP4 if group.get("input_activations") else
                QUANT_NVFP4_A16), int(weights.get("group_size") or 16)
    if weight_type == "int" and bits == 4:
        return QUANT_INT4_GPTQ, int(weights.get("group_size") or 128)
    if weight_type == "int" and bits == 8:
        return QUANT_INT8_GPTQ, int(weights.get("group_size") or 0)
    if weight_type == "float" and bits == 8:
        block_structure = weights.get("block_structure")
        if isinstance(block_structure, (list, tuple)) and block_structure:
            return QUANT_FP8_BLOCK, int(block_structure[-1])
        return QUANT_FP8_A16, 1

    raise ValueError("unsupported compressed-tensors checkpoint group: "
                     f"format={format_name or '<missing>'}, "
                     f"weight_type={weight_type!r}, bits={bits}")


def _has_modelopt_weight_groups(embedded: dict) -> bool:
    groups = embedded.get("config_groups") or {}
    for group in groups.values():
        weights = group.get("weights") or {}
        if (group.get("quant_algo") or group.get("format")
                or weights.get("type") != "float"
                or weights.get("num_bits") not in (4, 8)
                or weights.get("block_structure")
                or (weights.get("num_bits") == 8
                    and weights.get("group_size", 1) != 1)):
            return False
    return bool(groups)


def _parse_modelopt_mixed_precision(
        model_dir: str,
        embedded: dict,
        conversion: Optional[ModuleType],
        quantize_activations: bool,
        sidecar: Optional[dict] = None) -> QuantConfig:
    """Parse ModelOpt mixed precision from serialized tensor layouts."""
    config_groups = embedded.get("config_groups") or {}
    if not config_groups:
        raise ValueError(
            "unsupported ModelOpt mixed-precision checkpoint: config_groups "
            "is empty")

    contracts = [_modelopt_group_contract(g) for g in config_groups.values()]
    contracts = [(_activation_quant_type(quant_type,
                                         quantize_activations), group_size)
                 for quant_type, group_size in contracts]
    quant_types = {quant_type for quant_type, _ in contracts}
    nvfp4_types = {QUANT_NVFP4, QUANT_NVFP4_A16}
    allowed_types = quant_types | (nvfp4_types if quant_types
                                   & nvfp4_types else set())
    # A statically-quantized FP8 group also covers FP8 tensors with no
    # activation scale, such as an embedding table: it is a lookup, not a
    # GEMM, so it can only execute weight-only.
    if QUANT_FP8 in quant_types:
        allowed_types.add(QUANT_FP8_A16)
    layer_overrides = _infer_serialized_module_types(model_dir, conversion,
                                                     allowed_types,
                                                     quantize_activations)
    represented = set(layer_overrides.values())
    missing = quant_types.difference(represented)
    if represented & nvfp4_types:
        missing.difference_update(nvfp4_types)
    if QUANT_FP8_A16 in represented and represented - {QUANT_FP8_A16}:
        missing.discard(QUANT_FP8)
    if missing:
        raise ValueError(
            "ModelOpt mixed-precision metadata does not match checkpoint "
            "tensor layouts; missing module payloads for "
            f"{', '.join(sorted(missing))}")

    counts = Counter(layer_overrides.values())
    quant_type = max(sorted(represented), key=counts.__getitem__)
    grouped_sizes = {
        size
        for contract_type, size in contracts
        if contract_type in (QUANT_NVFP4, QUANT_NVFP4_A16) and size > 0
    }
    if len(grouped_sizes) > 1:
        raise ValueError("ModelOpt mixed-precision checkpoint uses "
                         "incompatible NVFP4 group sizes: "
                         f"{sorted(grouped_sizes)}")
    group_size = next(iter(grouped_sizes), 1)

    sidecar = sidecar or {}
    excluded = _effective_exclusions(
        model_dir,
        list(embedded.get("ignore", [])) +
        list(sidecar.get("exclude_modules", [])), conversion)
    excluded.extend(name
                    for name in _detect_plain_weights(model_dir, conversion)
                    if name not in excluded)
    return QuantConfig(
        quant_type=quant_type,
        group_size=group_size,
        kv_cache_quant=_resolve_kv_cache_quant(
            model_dir,
            sidecar.get("kv_cache_quant_algo",
                        "fp8" if embedded.get("kv_cache_scheme") else None)),
        excluded=tuple(sorted(set(excluded))),
        layer_overrides=layer_overrides,
        layer_group_sizes={
            name: group_size if module_type in nvfp4_types else 1
            for name, module_type in layer_overrides.items()
        },
        is_mixed_precision=True,
    )


def _modelopt_group_contract(group: dict) -> Tuple[str, int]:
    weights = group.get("weights") or {}
    activations = group.get("input_activations")
    weight_type = str(weights.get("type", "")).lower()
    bits = int(weights.get("num_bits") or 0)
    if weight_type == "float" and bits == 4:
        group_size = int(weights.get("group_size") or 16)
        if group_size != 16:
            raise ValueError(
                f"unsupported ModelOpt FP4 group size {group_size}; expected "
                "NVFP4 group size 16")
        return (QUANT_NVFP4 if activations else QUANT_NVFP4_A16), group_size
    if weight_type == "float" and bits == 8:
        if (isinstance(activations, dict)
                and activations.get("dynamic") is False):
            return QUANT_FP8, 1
        return QUANT_FP8_A16, 1
    raise ValueError("unsupported ModelOpt mixed-precision checkpoint group: "
                     f"weight_type={weight_type!r}, bits={bits}")


def _infer_serialized_module_types(
        model_dir: str, conversion: Optional[ModuleType],
        allowed_types: Set[str], quantize_activations: bool) -> Dict[str, str]:
    """Infer effective per-module precision from checkpoint tensor layouts."""
    try:
        store = safetensors_np.SafetensorsStore(model_dir)
    except (FileNotFoundError, OSError, ValueError):
        return {}

    overrides: Dict[str, str] = {}
    try:
        keys = set(store.keys())

        def record(raw_prefix: str, quant_type: str) -> None:
            quant_type = _activation_quant_type(quant_type,
                                                quantize_activations)
            if quant_type not in allowed_types:
                return
            normalized = _normalize_checkpoint_name(raw_prefix, conversion)
            for module_name in _expand_quantized_module(
                    normalized, conversion):
                previous = overrides.get(module_name)
                if previous is not None and previous != quant_type:
                    raise ValueError(
                        f"{module_name}: conflicting serialized layouts "
                        f"{previous!r} and {quant_type!r}")
                overrides[module_name] = quant_type

        # NVFP4 weights quantize activations only with a calibrated input
        # scale; without one they execute as W4A16.
        for suffix, input_suffix in ((".weight_global_scale",
                                      ".input_global_scale"),
                                     (".weight_scale_2", ".input_scale")):
            for key in keys:
                if key.endswith(suffix):
                    prefix = key[:-len(suffix)]
                    record(
                        prefix, QUANT_NVFP4 if prefix +
                        input_suffix in keys else QUANT_NVFP4_A16)

        for scale_suffix in (".weight_scale", ".weight_scale_inv"):
            for key in keys:
                if not key.endswith(scale_suffix):
                    continue
                prefix = key[:-len(scale_suffix)]
                if (prefix + ".weight_global_scale" in keys
                        or prefix + ".weight_scale_2" in keys):
                    continue
                weight_name = prefix + ".weight"
                if weight_name not in keys or store.dtype(weight_name) not in (
                        "F8_E4M3", "F8_E4M3FN"):
                    continue
                scale_shape = store.shape(key)
                if (scale_suffix == ".weight_scale_inv"
                        or (len(scale_shape) >= 2 and scale_shape[-1] > 1)):
                    module_type = QUANT_FP8_BLOCK
                elif prefix + ".input_scale" in keys:
                    module_type = QUANT_FP8
                else:
                    module_type = QUANT_FP8_A16
                record(prefix, module_type)
    finally:
        store.close()
    return overrides


def _parse_mixed_precision(
    quantized_layers: dict,
    conversion: Optional[ModuleType],
    quantize_activations: bool = True,
) -> Tuple[str, int, Dict[str, str], Dict[str, int]]:

    def group_size(layer_config: dict, quant_type: str) -> int:
        configured = int(layer_config.get("group_size", 1))
        if configured != 1:
            return configured
        if quant_type == QUANT_MXFP8:
            return 32
        if quant_type in (QUANT_NVFP4, QUANT_NVFP4_A16):
            return 16
        return configured

    algorithm_counts: Counter = Counter()
    group_sizes: Dict[str, int] = {}
    for layer_config in quantized_layers.values():
        algorithm = str(layer_config.get("quant_algo", "")).upper()
        algorithm_counts[algorithm] += 1
        quant_type = _activation_quant_type(
            _algorithm_to_type(algorithm, conversion), quantize_activations)
        group_sizes.setdefault(algorithm, group_size(layer_config, quant_type))
    if not algorithm_counts:
        return QUANT_FP16, 1, {}, {}

    dominant_algorithm = algorithm_counts.most_common(1)[0][0]
    overrides: Dict[str, str] = {}
    layer_group_sizes: Dict[str, int] = {}
    for name, layer_config in quantized_layers.items():
        quant_type = _activation_quant_type(
            _algorithm_to_type(layer_config.get("quant_algo", ""), conversion),
            quantize_activations)
        module_group_size = group_size(layer_config, quant_type)
        short_name = _normalize_checkpoint_name(name, conversion)
        for module_name in _expand_quantized_module(short_name, conversion):
            overrides[module_name] = quant_type
            layer_group_sizes[module_name] = module_group_size
    return (_activation_quant_type(
        _algorithm_to_type(dominant_algorithm, conversion),
        quantize_activations), group_sizes.get(dominant_algorithm, 1),
            overrides, layer_group_sizes)


def _checkpoint_keys(model_dir: str) -> List[str]:
    try:
        with safetensors_np.SafetensorsStore(model_dir) as store:
            return store.keys()
    except (FileNotFoundError, OSError, ValueError):
        return []


def _normalize_checkpoint_name(name: str,
                               conversion: Optional[ModuleType]) -> str:
    normalize = getattr(conversion, "normalize_checkpoint_name", None)
    if normalize is not None:
        name = normalize(name)
    return normalize_module_name(name)


def _expand_quantized_module(
        name: str, conversion: Optional[ModuleType]) -> Tuple[str, ...]:
    expand = getattr(conversion, "expand_quantized_module", None)
    return tuple(expand(name)) if expand is not None else (name, )


def _finalize_exclusions(modules: List[str],
                         conversion: Optional[ModuleType]) -> List[str]:
    finalize = getattr(conversion, "finalize_exclusions", None)
    if finalize is not None:
        modules = list(finalize(modules))
    return sorted(set(modules))


def _detect_plain_weights(model_dir: str,
                          conversion: Optional[ModuleType]) -> List[str]:
    keys = _checkpoint_keys(model_dir)
    weights = {
        key.rsplit(".", 1)[0]
        for key in keys if key.endswith(".weight")
    }
    scales = {
        key.rsplit(".", 1)[0]
        for key in keys if key.endswith((".weight_scale", ".weight_scale_inv"))
    }
    quantized_names = {
        _normalize_checkpoint_name(name, conversion)
        for name in scales
    }
    excluded = set()
    for name in weights - scales:
        normalized = _normalize_checkpoint_name(name, conversion)
        # Multi-component checkpoints can contain a plain Talker projection
        # and a quantized Thinker projection with the same frontend name.
        # Keep the shared name quantized here; module_quant_type() verifies
        # the concrete checkpoint namespace and restores FP16 for the plain
        # component.
        if normalized in quantized_names:
            continue
        excluded.update(_expand_quantized_module(normalized, conversion))
    return _finalize_exclusions(list(excluded), conversion)


def _detect_unquantized_packed_modules(
        model_dir: str, conversion: Optional[ModuleType]) -> List[str]:
    keys = _checkpoint_keys(model_dir)
    quantized = {
        key.rsplit(".", 1)[0]
        for key in keys if key.endswith((".qweight", ".weight_packed"))
    }
    weights = {
        key.rsplit(".", 1)[0]
        for key in keys if key.endswith(".weight")
    }
    excluded = [
        module_name for name in weights - quantized
        for module_name in _expand_quantized_module(
            _normalize_checkpoint_name(name, conversion), conversion)
    ]
    known = {
        _normalize_checkpoint_name(name, conversion)
        for name in weights | quantized
    }
    if "lm_head" not in known:
        excluded.append("lm_head")
    return _finalize_exclusions(excluded, conversion)


def _effective_exclusions(model_dir: str, configured: List[str],
                          conversion: Optional[ModuleType]) -> List[str]:
    keys = _checkpoint_keys(model_dir)
    sidecar_suffixes = (".qweight", ".weight_packed", ".weight_scale",
                        ".weight_scale_inv", ".weight_scale_2", ".input_scale",
                        ".scales")
    quantized = {
        _normalize_checkpoint_name(key.rsplit(".", 1)[0], conversion)
        for key in keys if key.endswith(sidecar_suffixes)
    }
    normalized = []
    for name in configured:
        short_name = _normalize_checkpoint_name(name, conversion)
        if short_name not in quantized:
            normalized.extend(_expand_quantized_module(short_name, conversion))
    return _finalize_exclusions(normalized, conversion)


def _detect_gptq_zero_point_offset(model_dir: str,
                                   config: dict,
                                   bits: int = 4) -> int:
    if not bool(config.get("sym", False)):
        return 1
    try:
        with safetensors_np.SafetensorsStore(model_dir) as store:
            key = next(name for name in store.keys()
                       if name.endswith(".qzeros"))
            packed = store.get_numpy(key).reshape(-1)[:1024]
    except (FileNotFoundError, KeyError, OSError, StopIteration, ValueError):
        return 1
    if packed.size == 0:
        return 1
    values = []
    values_per_word = 32 // bits
    mask = (1 << bits) - 1
    for value in packed:
        unsigned = int(value) & 0xFFFFFFFF
        values.extend((unsigned >> (bits * index)) & mask
                      for index in range(values_per_word))
    midpoint = 1 << (bits - 1)
    return 0 if values and all(value == midpoint for value in values) else 1


_VISUAL_PATH_HINTS = ("visual", "vision_tower", "vision_model",
                      "multi_modal_projector", "mlp1", "image_embed",
                      "embed_vision")


def _has_llm_kv_scales(model_dir: str) -> bool:
    """Whether the checkpoint carries LLM (non-visual) KV-cache K scales.

    ModelOpt writes ``kv_cache_quant_algo``/``kv_cache_scheme`` whenever any
    ``k_bmm_quantizer`` is enabled, including a visual tower's, so the metadata
    alone reports FP8 for checkpoints whose LLM attention has no K/V scales.
    Mirrors ``tensorrt_edgellm.config._detect_llm_kv_cache_fp8``.
    """
    return any(
        key.endswith(".k_proj.k_scale") and not any(
            hint in key for hint in _VISUAL_PATH_HINTS)
        for key in _checkpoint_keys(model_dir))


def _resolve_kv_cache_quant(model_dir: str,
                            claimed: Optional[str]) -> Optional[str]:
    """Honor a checkpoint's KV-cache claim only when its scales are present."""
    normalized = _normalize_kv(claimed)
    if normalized is None:
        return None
    return normalized if _has_llm_kv_scales(model_dir) else None


def _normalize_kv(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    normalized = str(value).strip().lower()
    return normalized or None
