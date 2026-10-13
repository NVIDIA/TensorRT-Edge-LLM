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
"""Draft-namespace precision and sidecar resolution in the direct builder."""

import numpy as np
import pytest

safetensors_numpy = pytest.importorskip("safetensors.numpy")

from experimental.builder.core import quantization
from experimental.builder.core.weights import Weights
from experimental.builder.models.qwen3_5 import weights as qwen3_5_weights

_BASE = "model.language_model.layers.0.mlp."

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


def _nvfp4(prefix):
    return {
        prefix + "weight": np.zeros((8, 4), dtype=np.uint8),
        prefix + "weight_scale": np.ones((8, 1), dtype=np.float16),
        prefix + "weight_scale_2": np.ones((), dtype=np.float32),
        prefix + "input_scale": np.ones((), dtype=np.float32),
    }


@pytest.fixture
def checkpoint(tmp_path):
    """NVFP4 base model whose MTP draft leaves one projection unquantized."""
    tensors = {}
    tensors.update(_nvfp4(_BASE + "gate_proj."))
    tensors.update(_nvfp4(_BASE + "up_proj."))
    tensors.update(_nvfp4("lm_head."))
    tensors["mtp.layers.0.mlp.gate_proj.weight"] = np.zeros((8, 8),
                                                            dtype=np.float16)
    tensors.update(_nvfp4("mtp.layers.0.mlp.up_proj."))
    safetensors_numpy.save_file(tensors, str(tmp_path / "model.safetensors"))
    return str(tmp_path)


def test_unquantized_draft_projection_is_fp16_despite_base_override(
        checkpoint):
    weights = Weights(checkpoint,
                      quant=_QUANT,
                      conversion=qwen3_5_weights,
                      spec_type="mtp",
                      spec_role="draft")
    try:
        assert weights.checkpoint_key("layers.0.mlp.gate_proj.weight") == (
            "mtp.layers.0.mlp.gate_proj.weight")
        assert weights.module_quant_type(
            "layers.0.mlp.gate_proj") == quantization.QUANT_FP16
        # The base layer's scales must not answer for the draft module, or
        # its plain weight is decoded as packed FP4.
        assert not weights.has("layers.0.mlp.gate_proj.weight_scale")
        assert not weights.is_nvfp4("layers.0.mlp.gate_proj")
        weight, bias = weights.linear_fp16("layers.0.mlp.gate_proj")
        assert weight.shape == (8, 8)
        assert bias is None
        # A draft projection that is quantized keeps its declared precision.
        assert weights.module_quant_type(
            "layers.0.mlp.up_proj") == quantization.QUANT_NVFP4
        assert weights.is_nvfp4("layers.0.mlp.up_proj")
        # The draft shares the base lm_head, which stays quantized.
        assert weights.module_quant_type("lm_head") == quantization.QUANT_NVFP4
        assert weights.is_nvfp4("lm_head")
    finally:
        weights.close()


def _awq(prefix):
    return {
        prefix + "qweight": np.zeros((8, 1), dtype=np.int32),
        prefix + "scales": np.ones((8, 1), dtype=np.float16),
        prefix + "qzeros": np.zeros((8, 1), dtype=np.int32),
    }


@pytest.fixture
def awq_checkpoint(tmp_path):
    """INT4 base model whose MTP draft leaves a projection unquantized.

    The base module has no plain weight, so its packed primary is what an
    unpinned lookup for the draft's module would fall back to.
    """
    tensors = _awq(_BASE + "gate_proj.")
    tensors["mtp.layers.0.mlp.gate_proj.weight"] = np.zeros((8, 8),
                                                            dtype=np.float16)
    safetensors_numpy.save_file(tensors, str(tmp_path / "model.safetensors"))
    return str(tmp_path)


def test_unquantized_draft_projection_is_fp16_beside_packed_base(
        awq_checkpoint):
    quant = quantization.QuantConfig(
        quant_type=quantization.QUANT_INT4_AWQ,
        group_size=128,
        layer_overrides={
            "layers.0.mlp.gate_proj": quantization.QUANT_INT4_AWQ,
        },
        is_mixed_precision=True,
    )
    weights = Weights(awq_checkpoint,
                      quant=quant,
                      conversion=qwen3_5_weights,
                      spec_type="mtp",
                      spec_role="draft")
    try:
        assert weights.checkpoint_key("layers.0.mlp.gate_proj.weight") == (
            "mtp.layers.0.mlp.gate_proj.weight")
        # The base module's packed primary must not mark the draft module as
        # quantized, or the draft loads the base model's weights.
        assert not weights.has("layers.0.mlp.gate_proj.qweight")
        assert not weights.has("layers.0.mlp.gate_proj.scales")
        assert weights.module_quant_type(
            "layers.0.mlp.gate_proj") == quantization.QUANT_FP16
    finally:
        weights.close()


def test_base_projections_keep_their_overrides(checkpoint):
    weights = Weights(checkpoint, quant=_QUANT, conversion=qwen3_5_weights)
    try:
        assert weights.module_quant_type(
            "layers.0.mlp.gate_proj") == quantization.QUANT_NVFP4
        assert weights.is_nvfp4("layers.0.mlp.gate_proj")
        assert weights.module_quant_type(
            "layers.0.mlp.up_proj") == quantization.QUANT_NVFP4
    finally:
        weights.close()
