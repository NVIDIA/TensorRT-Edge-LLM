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
"""Qwen4-Exp checkpoint weight mapping."""

import numpy as np

from ...core import quantization
from ...core.weights import ParameterSpec

_PREFIXES = {
    "llm": ("model.language_model.", "language_model."),
    "visual": ("model.visual.", "visual."),
}
_WRAPPERS = ("model.language_model.", "language_model.")

NGRAM_EMBEDDING_PREFIX = "ple.ple_embedding"
NGRAM_CONSTANT_NAMES = ("layer_multipliers", "ngram_heads_vocab_sizes",
                        "ngram_heads_offsets")


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    """Map frontend tensor names to Qwen4-Exp component checkpoints."""
    prefixes = _PREFIXES.get(component, ())
    candidates = [prefix + name for prefix in prefixes]
    if component == "llm" and name.startswith("model."):
        nested_name = name[len("model."):]
        candidates.extend(prefix + nested_name for prefix in prefixes)
    return tuple(candidates)


def normalize_checkpoint_name(name: str) -> str:
    """Remove only wrappers used by this model family's checkpoints."""
    for prefix in _WRAPPERS:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def expert_quant_type(weights, experts_prefix: str) -> str:
    """Recognize unquantized stacked experts independently of base precision."""
    keys = tuple(f"{experts_prefix}.{name}"
                 for name in ("gate_up_proj", "down_proj"))
    if any(weights.has(key) for key in keys):
        for key in keys:
            if not weights.has(key) or weights.dtype(key) not in ("F16",
                                                                  "BF16",
                                                                  "F32"):
                raise ValueError(f"{experts_prefix}: stacked experts require "
                                 "floating gate_up_proj and down_proj tensors")
        return quantization.QUANT_FP16
    return weights.module_quant_type(f"{experts_prefix}.0.up_proj")


def nvfp4_expert_bindings(weights, experts_prefix: str, num_experts: int,
                          sm12x: bool) -> dict:
    """Map Qwen4-Exp provider-packed NVFP4 experts to plugin inputs."""
    records = {}
    for expert in range(num_experts):
        for projection in ("up_proj", "gate_proj", "down_proj"):
            records[expert, projection] = weights.nvfp4_checkpoint_names(
                f"{experts_prefix}.{expert}.{projection}")

    def fields(projections, order):
        return [
            records[expert, projection][field] for expert in range(num_experts)
            for field in order for projection in projections
        ]

    fc1_reciprocal = {
        records[expert, projection][3]
        for expert in range(num_experts)
        for projection in ("up_proj", "gate_proj")
    }
    fc2_reciprocal = {
        records[expert, "down_proj"][3]
        for expert in range(num_experts)
    }
    if len(fc1_reciprocal) != 1 or len(fc2_reciprocal) != 1:
        raise ValueError(
            "Qwen4-Exp NVFP4 experts use inconsistent alpha formats")
    fc1_reciprocal = fc1_reciprocal.pop()
    fc2_reciprocal = fc2_reciprocal.pop()

    common = {
        "num_experts": num_experts,
        "fc1_layout": "concat" if sm12x else "interleave",
    }
    gate_up = ("up_proj", "gate_proj")
    down = ("down_proj", )
    return {
        "fc1_qweights":
        weights.checkpoint_binding(fields(gate_up, (0, 1, 2)),
                                   "nvfp4_qweight",
                                   "nvfp4_gated_fc1_qweight",
                                   reciprocal_alpha=fc1_reciprocal,
                                   **common),
        "fc1_blocks_scale":
        weights.checkpoint_binding(fields(gate_up, (0, 1, 2)),
                                   "nvfp4_scale_linear",
                                   "nvfp4_gated_fc1_scale",
                                   reciprocal_alpha=fc1_reciprocal,
                                   **common),
        "fc1_alpha":
        weights.checkpoint_binding([], "generated", "fill", fill_value=1.0),
        "fc2_qweights":
        weights.checkpoint_binding(fields(down, (0, 1, 2)),
                                   "nvfp4_qweight",
                                   "nvfp4_gated_fc2_qweight",
                                   reciprocal_alpha=fc2_reciprocal,
                                   **common),
        "fc2_blocks_scale":
        weights.checkpoint_binding(fields(down, (0, 1, 2)),
                                   "nvfp4_scale_linear",
                                   "nvfp4_gated_fc2_scale",
                                   reciprocal_alpha=fc2_reciprocal,
                                   **common),
        "fc2_alpha":
        weights.checkpoint_binding([], "generated", "fill", fill_value=1.0),
        "input_global_scale":
        weights.checkpoint_binding([], "generated", "fill", fill_value=1.0),
        "down_input_scale":
        weights.checkpoint_binding([], "generated", "fill", fill_value=1.0),
        "e_score_correction_bias":
        weights.checkpoint_binding([], "generated", "fill", fill_value=0.0),
    }


def nvfp4_expert_specs(weights, experts_prefix: str, num_experts: int) -> dict:
    """Describe architecture-specific NVFP4 MoE buffers from headers only."""
    up_names = weights.nvfp4_checkpoint_names(f"{experts_prefix}.0.up_proj")
    down_names = weights.nvfp4_checkpoint_names(
        f"{experts_prefix}.0.down_proj")
    up_weight = weights.store.shape(up_names[0])
    up_scale = weights.store.shape(up_names[1])
    down_weight = weights.store.shape(down_names[0])
    down_scale = weights.store.shape(down_names[1])
    return {
        "fc1_qweights":
        ParameterSpec((num_experts, 2 * up_weight[0], up_weight[1]), np.int8),
        "fc1_blocks_scale":
        ParameterSpec((num_experts, (2 * up_scale[0] + 127) // 128,
                       (up_scale[1] + 3) // 4, 32, 4, 4), np.int8),
        "fc1_alpha":
        ParameterSpec((num_experts, ), np.float32),
        "fc2_qweights":
        ParameterSpec((num_experts, *down_weight), np.int8),
        "fc2_blocks_scale":
        ParameterSpec((num_experts, (down_scale[0] + 127) // 128,
                       (down_scale[1] + 3) // 4, 32, 4, 4), np.int8),
        "fc2_alpha":
        ParameterSpec((num_experts, ), np.float32),
        "input_global_scale":
        ParameterSpec((num_experts, ), np.float32),
        "down_input_scale":
        ParameterSpec((num_experts, ), np.float32),
        "e_score_correction_bias":
        ParameterSpec((num_experts, ), np.float32),
    }


# Fp16MoePlugin packs FC1 as 64-row chunks alternating up and gate.
_FP16_MOE_CHUNK_ROWS = 64


def fp16_moe_supports(num_experts: int, top_k: int) -> bool:
    """Whether ``Fp16MoePlugin`` can route this many experts this wide."""
    return num_experts in (128, 256) and 0 < top_k <= 8


def dequantized_fp16_experts(experts, hidden_size: int,
                             intermediate_size: int) -> dict:
    """Pack individual or stacked floating experts for ``Fp16MoePlugin``."""
    if intermediate_size % _FP16_MOE_CHUNK_ROWS:
        raise ValueError("FP16 MoE intermediate size must be a multiple of "
                         f"{_FP16_MOE_CHUNK_ROWS}, got {intermediate_size}")
    chunks = intermediate_size // _FP16_MOE_CHUNK_ROWS
    fc1_weights = []
    fc2_weights = []
    for expert in range(len(experts)):
        prefix = experts.key(str(expert))
        dense = experts.load_expert_dense(expert)
        gate, up, down = dense["gate"], dense["up"], dense["down"]
        expected_fc1 = (intermediate_size, hidden_size)
        if gate.shape != expected_fc1 or up.shape != expected_fc1:
            raise ValueError(f"{prefix} gate/up shape must be {expected_fc1}, "
                             f"got {gate.shape} and {up.shape}")
        if down.shape != (hidden_size, intermediate_size):
            raise ValueError(f"{prefix}.down_proj shape must be "
                             f"{(hidden_size, intermediate_size)}, got "
                             f"{down.shape}")
        interleaved = np.stack(
            (up.reshape(chunks, _FP16_MOE_CHUNK_ROWS, hidden_size),
             gate.reshape(chunks, _FP16_MOE_CHUNK_ROWS, hidden_size)),
            axis=1,
        ).reshape(2 * intermediate_size, hidden_size)
        fc1_weights.append(np.ascontiguousarray(interleaved, dtype=np.float16))
        fc2_weights.append(np.ascontiguousarray(down, dtype=np.float16))
    return {
        "fc1_weights": np.stack(fc1_weights),
        "fc2_weights": np.stack(fc2_weights),
    }
