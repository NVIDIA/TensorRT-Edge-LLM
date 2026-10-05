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
"""``--no-quantize-activations`` plumbing.

The flag decides whether a quantized dense Linear emits its activation Q/DQ, and
it reaches the layer by two hops: the export CLI sets the default that every
later ``QuantConfig`` picks up, and ``make_linear`` copies that onto each layer
it builds. A layer that silently keeps the class default would export the
checkpoint's own recipe while the caller believes it asked for weight-only, so
both hops are covered here.
"""

import onnx
import pytest
import torch

from tensorrt_edgellm.config import (QUANT_FP8, QUANT_INT8_SQ, QUANT_MXFP8,
                                     QUANT_NVFP4, ModelConfig, QuantConfig,
                                     set_default_quantize_activations)
from tensorrt_edgellm.models.linear import (LinearBase, NVFP4LinearMethod,
                                            TPMode, make_linear)
from tensorrt_edgellm.onnx.dynamo_translations import \
    build_custom_translation_table
from tensorrt_edgellm.onnx.export import _permissive_inline_opset

# (quant_type, group_size) for the four recipes that quantize activations. INT4
# AWQ / GPTQ are weight-only by construction and have no Q to drop.
_ACTIVATION_QUANT_RECIPES = [
    (QUANT_NVFP4, 16),
    (QUANT_FP8, 1),
    (QUANT_MXFP8, 32),
    (QUANT_INT8_SQ, 1),
]


@pytest.fixture
def restore_default():
    yield
    set_default_quantize_activations(True)


def _model_config(quant: QuantConfig) -> ModelConfig:
    config = ModelConfig(model_type="llama",
                         hidden_size=64,
                         num_hidden_layers=1,
                         num_attention_heads=4,
                         num_key_value_heads=4,
                         intermediate_size=128,
                         head_dim=16,
                         rms_norm_eps=1e-6,
                         vocab_size=32,
                         rope_theta=1e4,
                         max_position_embeddings=128,
                         default_attention_scale=1.0)
    config.quant = quant
    return config


def test_default_quantizes_activations():
    assert QuantConfig().quantize_activations is True
    assert LinearBase.quantize_activations is True


def test_cli_default_reaches_later_quant_configs(restore_default):
    """The export CLI sets this once; configs parsed afterwards must see it."""
    set_default_quantize_activations(False)
    assert QuantConfig().quantize_activations is False


def test_explicit_value_beats_the_cli_default(restore_default):
    set_default_quantize_activations(False)
    assert QuantConfig(quantize_activations=True).quantize_activations is True


@pytest.mark.parametrize("quant_type,group_size", _ACTIVATION_QUANT_RECIPES)
@pytest.mark.parametrize("quantize_activations", [True, False])
def test_make_linear_propagates_to_the_layer(quant_type, group_size,
                                             quantize_activations):
    config = _model_config(
        QuantConfig(quant_type=quant_type,
                    group_size=group_size,
                    quantize_activations=quantize_activations))
    layer = make_linear(config, 64, 128)
    assert layer.quantize_activations is quantize_activations


def test_make_linear_uses_module_group_size_for_mixed_precision():
    config = _model_config(
        QuantConfig(quant_type=QUANT_FP8,
                    group_size=1,
                    layer_overrides={"lm_head": QUANT_NVFP4},
                    layer_group_sizes={"lm_head": 16},
                    is_mixed_precision=True))

    layer = make_linear(config, 64, 128, module_name="lm_head")

    assert layer.group_size == 16
    assert tuple(layer.weight_scale.shape) == (128, 4)


def test_row_parallel_nvfp4_rejects_weight_only():
    """The fused TP plugin quantizes inside the kernel, so it cannot express this.

    Silently keeping W4A4 on row-parallel linears alone would make a TP engine
    disagree with itself layer by layer.
    """

    class _Stub(LinearBase):
        quantize_activations = False

    with pytest.raises(NotImplementedError, match="no-quantize-activations"):
        NVFP4LinearMethod(group_size=16).apply_linear_allreduce(
            _Stub(), torch.zeros(1, 64, dtype=torch.float16))


def test_tp2_dense_nvfp4_token_matrix_exports_rank2_gemm(tmp_path):
    """Qwen3 TP2 MLP flattening must not broadcast the dense NVFP4 weight."""
    config = _model_config(QuantConfig(quant_type=QUANT_NVFP4,
                                       group_size=16)).for_rank(0, 2)
    layer = make_linear(config,
                        64,
                        config.intermediate_size,
                        tp_mode=TPMode.COL).eval()
    with _permissive_inline_opset():
        program = torch.onnx.export(
            layer, (torch.zeros(7, 64, dtype=torch.float16), ),
            dynamo=True,
            opset_version=24,
            custom_translation_table=build_custom_translation_table(),
            optimize=False)

    output = tmp_path / "tp2-dense-nvfp4-rank2.onnx"
    program.save(str(output))
    graph = onnx.shape_inference.infer_shapes(onnx.load(str(output))).graph
    values = {
        value.name: value
        for value in (*graph.input, *graph.value_info, *graph.output)
    }
    gemms = [node for node in graph.node if node.op_type == "Gemm"]
    assert len(gemms) == 1
    gemm = gemms[0]
    assert [
        len(values[name].type.tensor_type.shape.dim)
        for name in (*gemm.input[:2], gemm.output[0])
    ] == [2, 2, 2]
    producers = {name: node for node in graph.node for name in node.output}
    assert producers[gemm.input[1]].op_type != "Unsqueeze"


def test_golden_linears_share_the_switch():
    """The golden must be able to follow the export into the same regime.

    They are separate processes driven by one script flag; if the golden could
    not be switched, the comparison would be measuring the flag itself.
    """
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import golden_quant_linears as gql

    for cls in (gql._GoldenNVFP4Linear, gql._GoldenFP8Linear,
                gql._GoldenMXFP8Linear, gql._GoldenINT8SQLinear):
        assert issubclass(cls, gql._GoldenLinearBase)
        assert cls.quantize_activations is True
