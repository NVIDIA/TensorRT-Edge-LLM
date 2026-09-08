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
"""Draft-namespace quantization lookup in the direct builder."""

import numpy as np
import pytest

safetensors_numpy = pytest.importorskip("safetensors.numpy")

from experimental.builder.core import quantization
from experimental.builder.core.weights import Weights
from experimental.builder.models.qwen3_5 import weights as qwen3_5_weights

_BASE = "model.language_model.layers.0.mlp."


def _nvfp4(prefix):
    return {
        prefix + "weight": np.zeros((8, 4), dtype=np.uint8),
        prefix + "weight_scale": np.ones((8, 1), dtype=np.float16),
        prefix + "weight_scale_2": np.ones((), dtype=np.float32),
        prefix + "input_scale": np.ones((), dtype=np.float32),
    }


@pytest.fixture
def checkpoint(tmp_path):
    tensors = {}
    tensors.update(_nvfp4(_BASE + "gate_proj."))
    tensors.update(_nvfp4(_BASE + "up_proj."))
    tensors.update(_nvfp4("lm_head."))
    # Unquantized MTP draft MLP: plain weight, no quantization sidecar.
    tensors["mtp.layers.0.mlp.gate_proj.weight"] = np.zeros((8, 8),
                                                            dtype=np.float16)
    # Quantized MTP draft projection keeps its declared precision.
    tensors.update(_nvfp4("mtp.layers.0.mlp.up_proj."))
    safetensors_numpy.save_file(tensors, str(tmp_path / "model.safetensors"))
    return str(tmp_path)


_QUANT = quantization.QuantConfig(
    quant_type=quantization.QUANT_NVFP4,
    group_size=16,
    layer_overrides={
        "layers.0.mlp.gate_proj": quantization.QUANT_NVFP4,
        "layers.0.mlp.up_proj": quantization.QUANT_NVFP4,
        "lm_head": quantization.QUANT_NVFP4,
    },
    is_mixed_precision=True,
)


def test_unquantized_draft_projection_is_fp16_despite_base_override(
        checkpoint):
    weights = Weights(checkpoint,
                      quant=_QUANT,
                      conversion=qwen3_5_weights,
                      spec_type="mtp",
                      spec_role="draft")
    try:
        assert weights.checkpoint_key("layers.0.mlp.gate_proj.weight") == (
            "mtp.layers.0.mlp."
            "gate_proj.weight")
        assert weights.module_quant_type(
            "layers.0.mlp.gate_proj") == quantization.QUANT_FP16
        # Sidecars stay pinned to the draft namespace: the base layer's
        # scales must not be borrowed, so the module loads as a plain weight.
        assert not weights.has("layers.0.mlp.gate_proj.weight_scale")
        assert not weights.is_nvfp4("layers.0.mlp.gate_proj")
        weight, bias = weights.linear_fp16("layers.0.mlp.gate_proj")
        assert weight.shape == (8, 8) and bias is None
        assert weights.module_quant_type(
            "layers.0.mlp.up_proj") == quantization.QUANT_NVFP4
        assert weights.is_nvfp4("layers.0.mlp.up_proj")
        # The draft shares the base lm_head, which stays quantized.
        assert weights.module_quant_type("lm_head") == quantization.QUANT_NVFP4
        assert weights.is_nvfp4("lm_head")
    finally:
        weights.close()


def test_base_projection_keeps_its_override(checkpoint):
    weights = Weights(checkpoint, quant=_QUANT, conversion=qwen3_5_weights)
    try:
        assert weights.module_quant_type(
            "layers.0.mlp.gate_proj") == quantization.QUANT_NVFP4
        assert weights.module_quant_type(
            "layers.0.mlp.up_proj") == quantization.QUANT_NVFP4
    finally:
        weights.close()
