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
"""Phi-4-MM visual embedding artifacts."""

import os

import numpy as np

from ...core import contracts
from ...core.artifacts.tensors import save_safetensors
from ...core.numpy_dtypes import f32_to_fp8_e4m3_bytes, fp8_e4m3_bytes_to_f32
from ...core.weights import Weights
from . import weights as weight_conversion


def _fp8_qdq(value: np.ndarray, scale: float) -> np.ndarray:
    encoded = f32_to_fp8_e4m3_bytes(value.astype(np.float32) / scale)
    return fp8_e4m3_bytes_to_f32(encoded) * scale


def write_component_embeddings(bundle, component: contracts.Component, args,
                               output_dir: str) -> None:
    """Write Phi-4-MM projected image separator embeddings."""
    if component != contracts.Component.VISUAL:
        return
    weights = Weights(args.model_dir,
                      component=contracts.Component.VISUAL.value,
                      conversion=weight_conversion)
    try:
        projection_key = weights.find_suffix("img_projection.0.weight")
        projection_root = projection_key[:-len("0.weight")]

        def linear(prefix: str):
            weight_key = weights.checkpoint_key(prefix + ".weight")
            if weights.store.dtype(weight_key) not in ("F8_E4M3", "F8_E4M3FN"):
                weight, bias = weights.linear_fp16(prefix)
                return weight.astype(np.float32), bias, None

            scale_key = weights.checkpoint_key(prefix + ".weight_scale")
            if weights.store.dtype(scale_key) == "U8":
                raise ValueError(
                    "Phi-4-MM separator projection does not support "
                    "block-scaled FP8 weights")
            weight = weights.f32(prefix + ".weight")
            weight *= weights.f32(prefix + ".weight_scale")
            input_scale = float(
                weights.f32(prefix + ".input_scale").reshape(-1)[0])
            return (weight.astype(np.float16).astype(np.float32),
                    weights.opt_f16(prefix + ".bias"), input_scale)

        first_weight, first_bias, first_input_scale = linear(projection_root +
                                                             "0")
        second_weight, second_bias, second_input_scale = linear(
            projection_root + "2")

        def apply_linear(value, weight, bias, input_scale):
            if input_scale is not None:
                value = _fp8_qdq(value, input_scale)
            value = value @ weight.T
            if bias is not None:
                value += bias.astype(np.float32)
            return value.astype(np.float16).astype(np.float32)

        def project(key: str) -> np.ndarray:
            value = weights.f16(weights.find_suffix(key)).reshape(1, -1)
            value = apply_linear(value.astype(np.float32), first_weight,
                                 first_bias, first_input_scale)
            value = (0.5 * value * (1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) *
                (value + 0.044715 * value * value * value)))).astype(
                    np.float16)
            value = apply_linear(value.astype(np.float32), second_weight,
                                 second_bias, second_input_scale)
            return np.ascontiguousarray(value.reshape(-1), dtype=np.float16)

        save_safetensors(
            os.path.join(output_dir, "phi4mm_gn_proj.safetensors"), {
                "glb_GN": project("image_embed.glb_GN"),
                "sub_GN": project("image_embed.sub_GN"),
            })
    finally:
        weights.close()
    _ = bundle
