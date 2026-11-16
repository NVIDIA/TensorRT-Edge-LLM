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
"""W4A4 export of the TRT-native MoE arm (``TRT_MoE`` quantization_mode 2).

No CI lane has a TensorRT with the mode-2 importer, so the exported ONNX
contract is pinned here, GPU-free: the mode-2 graphs of a tiny gated (Qwen3)
and non-gated (Nemotron-H) MoE, the W4A8 graph when the flag is off, the
flag validation and the one host-side rescaling of the weight scales.
"""

import math

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto as TP
from onnx import numpy_helper

import tensorrt_edgellm.onnx.cli as export_script
from tensorrt_edgellm.checkpoint.repacking import \
    stack_nvfp4_moe_projection_with_global_scale
from tensorrt_edgellm.config import (LAYER_ATTN, LAYER_MAMBA, LAYER_MOE,
                                     QUANT_NVFP4, QUANT_NVFP4_A16, MambaConfig,
                                     ModelConfig, QuantConfig)
from tensorrt_edgellm.models import ops
from tensorrt_edgellm.models.linear import make_linear
from tensorrt_edgellm.models.nemotron_h.modeling_nemotron_h import (
    NemotronHCausalLM, NemotronHMoEMLP)
from tensorrt_edgellm.models.qwen3_moe.modeling_qwen3_moe import \
    Qwen3SparseMoeBlock
from tensorrt_edgellm.onnx.dynamo_translations import \
    build_custom_translation_table
from tensorrt_edgellm.onnx.export import (_OPSET_VERSION, _export_model,
                                          _fix_nvfp4_weight_dtype,
                                          _permissive_inline_opset)

_NVFP4 = QuantConfig(quant_type=QUANT_NVFP4, group_size=16)
# Fake ModelOpt calibration: gate == up input_scale per expert (the FC1 input
# is shared, so the export takes the max), one down_proj.input_scale each.
_FC1_SCALES = [1.2e-3, 1.5e-3, 0.9e-3, 1.1e-3]
_DOWN_SCALES = [1e-3, 2e-3, 3e-3, 4e-3]
# The export rounds the shared FC1 scale to an fp16-representable value.
_FC1_DBLQ = float(torch.tensor(max(_FC1_SCALES)).to(torch.float16))


@pytest.fixture
def native_moe_flags():
    """Set (native, w4a4) for one test and switch both off afterwards."""
    yield ops.set_trt_native_moe
    ops.set_trt_native_moe(False)


def _qwen3_config(quant: QuantConfig = _NVFP4) -> ModelConfig:
    config = ModelConfig(model_type="qwen3_moe",
                         hidden_size=32,
                         num_hidden_layers=1,
                         num_attention_heads=2,
                         num_key_value_heads=1,
                         intermediate_size=64,
                         head_dim=16,
                         rms_norm_eps=1e-6,
                         vocab_size=32,
                         rope_theta=1e4,
                         max_position_embeddings=128,
                         default_attention_scale=1.0)
    config.quant = quant
    config.num_experts = 4
    config.num_experts_per_tok = 2
    config.moe_intermediate_size = 32
    return config


def _nemotron_config(quant: QuantConfig = _NVFP4) -> ModelConfig:
    """Mamba + MoE + attention, as in test_nemotron_h_ragged_export."""
    config = ModelConfig(model_type="nemotron_h",
                         hidden_size=16,
                         num_hidden_layers=3,
                         num_attention_heads=2,
                         num_key_value_heads=1,
                         intermediate_size=32,
                         head_dim=8,
                         rms_norm_eps=1e-6,
                         vocab_size=32,
                         rope_theta=10_000.0,
                         max_position_embeddings=4096,
                         default_attention_scale=1.0 / math.sqrt(8.0),
                         layer_types=[LAYER_MAMBA, LAYER_MOE, LAYER_ATTN],
                         mamba_cfg=MambaConfig(num_heads=2,
                                               head_dim=8,
                                               ssm_state_size=8,
                                               conv_dim=32,
                                               conv_kernel=4,
                                               n_groups=1))
    config.quant = quant
    config.n_routed_experts = 4
    config.num_experts_per_tok = 2
    config.moe_intermediate_size = 32
    config.moe_shared_expert_intermediate_size = 16
    return config


def _fill_nvfp4_linear(linear, rng, weight_scale_2: float,
                       input_scale: float) -> None:
    """Give an NVFP4 linear the buffers a ModelOpt checkpoint would load."""
    out, half = linear.weight.shape
    codes = rng.integers(0, 256, size=(out, half), dtype=np.uint8)
    linear.weight.copy_(torch.from_numpy(codes.view(np.int8)))
    sf = rng.uniform(0.25, 4.0, size=(out, half * 2 // 16)).astype(np.float32)
    # The loader keeps the checkpoint's float8_e4m3fn dtype for weight_scale.
    linear.weight_scale = torch.from_numpy(sf).to(torch.float8_e4m3fn)
    linear.weight_scale_2.fill_(weight_scale_2)
    linear.input_scale.fill_(input_scale)


def _fill_experts(experts, rng, projections) -> None:
    """Distinct weight_scale_2 per expert and projection, calibrated
    input_scale per expert (see _FC1_SCALES / _DOWN_SCALES)."""
    for e, expert in enumerate(experts):
        for p, name in enumerate(projections):
            _fill_nvfp4_linear(
                getattr(expert, name), rng, 1e-4 * (e + 1) * (p + 1),
                _DOWN_SCALES[e] if name == "down_proj" else _FC1_SCALES[e])


def _attrs(node) -> dict:
    return {
        attr.name: onnx.helper.get_attribute_value(attr)
        for attr in node.attribute
    }


def _producers(graph) -> dict:
    return {name: node for node in graph.node for name in node.output}


def _single(graph, op_type: str):
    nodes = [node for node in graph.node if node.op_type == op_type]
    assert len(nodes) == 1, f"{len(nodes)} {op_type} nodes"
    return nodes[0]


def _init_bytes(init) -> bytes:
    if init.raw_data:
        return init.raw_data
    return numpy_helper.to_array(init).view(np.uint8).tobytes()


def _assert_mode2_attributes(moe) -> dict:
    attrs = _attrs(moe)
    assert attrs["quantization_mode"] == ops.TRT_MOE_QUANT_DYNAMIC_DBLQ == 2
    assert attrs["quantization_dtype"] == ops.TRT_DATATYPE_FP4 == 10
    assert list(attrs["quantization_block_shape"]) == [1, 1, 1, 16]
    assert attrs["dyn_q_output_scale_dtype"] == ops.TRT_DATATYPE_FP8 == 6
    assert "swiglu_limit" not in attrs
    assert len(moe.input) == 10
    return attrs


def _assert_fc2_scale_input(graph, moe, expected) -> None:
    """Input 9: fp32 [E], down_proj.input_scale verbatim (W4A8: * 6)."""
    inits = {init.name: init for init in graph.initializer}
    scale = inits[moe.input[9]]
    assert scale.data_type == TP.FLOAT
    np.testing.assert_array_equal(numpy_helper.to_array(scale),
                                  np.asarray(expected, np.float32))


def _assert_hidden_chain(graph, name: str, expected_dblq: float) -> None:
    """Input 0: TRT_FP4DynamicQuantize -> DequantizeLinear(Cast(dblq)) ->
    TRT_BlockDequantize([1, 1, 16], fp16), one fp16-representable dblq."""
    producers = _producers(graph)
    inits = {init.name: init for init in graph.initializer}
    block_dq = producers[name]
    assert block_dq.op_type == "TRT_BlockDequantize"
    assert list(_attrs(block_dq)["block_shape"]) == [1, 1, 16]
    assert _attrs(block_dq)["output_dtype"] == TP.FLOAT16
    dynq = producers[block_dq.input[0]]
    assert dynq.op_type == "TRT_FP4DynamicQuantize"
    assert _attrs(dynq)["block_size"] == 16
    assert _attrs(dynq)["scale_type"] == TP.FLOAT8E4M3FN
    assert _attrs(dynq)["axis"] in (-1, 2)
    scale_dq = producers[block_dq.input[1]]
    assert scale_dq.op_type == "DequantizeLinear" and scale_dq.domain == ""
    assert scale_dq.input[0] == dynq.output[1]
    cast = producers[scale_dq.input[1]]
    assert cast.op_type == "Cast" and _attrs(cast)["to"] == TP.FLOAT16
    assert cast.input[0] == dynq.input[1]
    dblq = inits[dynq.input[1]]
    assert dblq.data_type == TP.FLOAT
    assert numpy_helper.to_array(dblq).reshape(-1).tolist() == [expected_dblq]


def _assert_w4a4_weight(graph, name: str, linears) -> None:
    """Weights 3-5: Transpose(TRT_BlockDequantize(fp4 weight,
    DequantizeLinear(fp8 block scales, fp16 global, axis=0))). The codes and
    the E4M3 block scales are the checkpoint bytes verbatim, the global is
    fp16(weight_scale_2) per expert."""
    producers = _producers(graph)
    inits = {init.name: init for init in graph.initializer}
    transpose = producers[name]
    assert transpose.op_type == "Transpose"
    # [E, out, in] -> [E, in, out], the layout TRT_MoE takes
    assert list(_attrs(transpose)["perm"]) == [0, 2, 1]
    block_dq = producers[transpose.input[0]]
    assert block_dq.op_type == "TRT_BlockDequantize"
    assert list(_attrs(block_dq)["block_shape"]) == [1, 1, 16]
    assert _attrs(block_dq)["output_dtype"] == TP.FLOAT16
    weight = inits[block_dq.input[0]]
    assert weight.data_type == TP.FLOAT4E2M1  # post-export reinterpretation
    assert _init_bytes(weight) == torch.stack([
        linear.weight.view(torch.int8) for linear in linears
    ]).numpy().tobytes()
    scale_dq = producers[block_dq.input[1]]
    assert scale_dq.op_type == "DequantizeLinear" and scale_dq.domain == ""
    assert _attrs(scale_dq)["axis"] == 0
    fp8_scales = inits[scale_dq.input[0]]
    assert fp8_scales.data_type == TP.FLOAT8E4M3FN
    assert list(
        fp8_scales.dims) == [len(linears), *linears[0].weight_scale.shape]
    assert _init_bytes(fp8_scales) == torch.stack([
        linear.weight_scale.view(torch.uint8) for linear in linears
    ]).numpy().tobytes()
    fp16_global = inits[scale_dq.input[1]]
    assert fp16_global.data_type == TP.FLOAT16
    expected = torch.tensor(
        [float(linear.weight_scale_2) for linear in linears])
    np.testing.assert_array_equal(numpy_helper.to_array(fp16_global),
                                  expected.to(torch.float16).numpy())


def _qwen3_block(native_moe_flags, native: bool, w4a4: bool):
    native_moe_flags(native, w4a4)
    block = Qwen3SparseMoeBlock(_qwen3_config())
    experts = list(block.experts)  # _prepare_moe_weights drops the modules
    _fill_experts(experts, np.random.default_rng(0),
                  ("gate_proj", "up_proj", "down_proj"))
    block._prepare_moe_weights()
    return block.eval(), experts


def _export_block(block, path) -> onnx.GraphProto:
    with _permissive_inline_opset():
        program = torch.onnx.export(
            block, (torch.randn(5, block.hidden_size, dtype=torch.float16), ),
            dynamo=True,
            opset_version=_OPSET_VERSION,
            custom_translation_table=build_custom_translation_table(),
            optimize=False)
    program.save(str(path))
    _fix_nvfp4_weight_dtype(str(path))
    return onnx.load(str(path)).graph


def test_qwen3_w4a4_export_pins_the_gated_mode2_graph(tmp_path,
                                                      native_moe_flags):
    block, experts = _qwen3_block(native_moe_flags, True, True)
    graph = _export_block(block, tmp_path / "qwen3-w4a4.onnx")

    moe = _single(graph, "TRT_MoE")
    attrs = _assert_mode2_attributes(moe)
    assert attrs["activation_type"] == 1  # kSILU, gated
    assert attrs["swiglu_alpha"] == 1.0 and attrs["swiglu_beta"] == 0.0
    assert moe.input[3]  # gate weights
    inits = {init.name: init for init in graph.initializer}
    for i in (6, 7, 8):  # explicit zero biases keep the FC2 scale at input 9
        assert not numpy_helper.to_array(inits[moe.input[i]]).any()
    _assert_hidden_chain(graph, moe.input[0], _FC1_DBLQ)
    for idx, proj in ((3, "gate_proj"), (4, "up_proj"), (5, "down_proj")):
        _assert_w4a4_weight(graph, moe.input[idx],
                            [getattr(expert, proj) for expert in experts])
    _assert_fc2_scale_input(graph, moe, _DOWN_SCALES)


def test_qwen3_w4a8_export_is_unchanged_without_the_flag(
        tmp_path, native_moe_flags):
    block, _ = _qwen3_block(native_moe_flags, True, False)
    graph = _export_block(block, tmp_path / "qwen3-w4a8.onnx")

    moe = _single(graph, "TRT_MoE")
    attrs = _attrs(moe)
    assert attrs["quantization_mode"] == 1 and attrs["quantization_dtype"] == 6
    assert "quantization_block_shape" not in attrs
    assert "dyn_q_output_scale_dtype" not in attrs
    assert _producers(graph)[moe.input[0]].op_type == "DequantizeLinear"
    assert not any(node.op_type == "TRT_FP4DynamicQuantize"
                   for node in graph.node)
    # The FP8 form carries amax / 448 = input_scale * 6.
    _assert_fc2_scale_input(graph, moe,
                            (torch.tensor(_DOWN_SCALES) * 6.0).numpy())


def test_nemotron_h_w4a4_export_pins_the_non_gated_mode2_graph(
        tmp_path, native_moe_flags):
    native_moe_flags(True, True)
    model = NemotronHCausalLM(_nemotron_config())
    [mixer] = [m for m in model.modules() if isinstance(m, NemotronHMoEMLP)]
    experts = list(mixer.experts)  # prepare_for_export drops the modules
    _fill_experts(experts, np.random.default_rng(1), ("up_proj", "down_proj"))
    output = tmp_path / "nemotron-h-w4a4.onnx"
    _export_model(model, str(output), optimize=False)
    graph = onnx.load(str(output)).graph

    moe = _single(graph, "TRT_MoE")
    attrs = _assert_mode2_attributes(moe)
    assert attrs["activation_type"] == 2  # kRELU2, non-gated only
    assert "swiglu_alpha" not in attrs and "swiglu_beta" not in attrs
    assert [moe.input[i] for i in (3, 6, 7, 8)] == ["", "", "", ""]
    _assert_hidden_chain(graph, moe.input[0], _FC1_DBLQ)
    for idx, proj in ((4, "up_proj"), (5, "down_proj")):
        _assert_w4a4_weight(graph, moe.input[idx],
                            [getattr(expert, proj) for expert in experts])
    _assert_fc2_scale_input(graph, moe, _DOWN_SCALES)


def test_w4a4_needs_the_native_arm_and_nvfp4_experts(native_moe_flags, capsys):
    with pytest.raises(SystemExit) as excinfo:
        export_script.main(
            ["/no/such/model", "/no/such/output", "--trt-native-moe-w4a4"])
    assert excinfo.value.code == 2
    assert ("--trt-native-moe-w4a4 requires --trt-native-moe"
            in capsys.readouterr().err)
    native_moe_flags(False, True)  # the switch itself folds the same rule
    assert not ops.is_trt_native_moe_w4a4_enabled()
    native_moe_flags(True, True)
    # FP16 experts have only the unquantized form; --no-quantize-activations
    # turns NVFP4 into NVFP4-A16, which has no TRT_MoE form.
    for quant in (QuantConfig(),
                  QuantConfig(quant_type=QUANT_NVFP4_A16, group_size=16)):
        with pytest.raises(ValueError, match="--trt-native-moe-w4a4"):
            Qwen3SparseMoeBlock(_qwen3_config(quant))
        with pytest.raises(ValueError, match="--trt-native-moe-w4a4"):
            NemotronHMoEMLP(_nemotron_config(quant), "backbone.layers.1.mixer")


def test_stacker_rescales_a_weight_scale_2_below_fp16_normal():
    """The per-expert global is exported as fp16, so a ModelOpt weight_scale_2
    below 2**-14 is multiplied by 2**k and the E4M3 block scales are divided
    by 2**k (exact: a power of two only moves the exponent)."""
    rng = np.random.default_rng(2)
    experts = torch.nn.ModuleList()
    for weight_scale_2 in (1e-5, 1e-3):  # k = 3, untouched
        expert = torch.nn.Module()
        expert.up_proj = make_linear(_qwen3_config(), 32, 32)
        _fill_nvfp4_linear(expert.up_proj, rng, weight_scale_2, 1e-3)
        experts.append(expert)

    weight, block_scale, global_scale = (
        stack_nvfp4_moe_projection_with_global_scale(experts, "up_proj"))

    sf = [expert.up_proj.weight_scale.to(torch.float32) for expert in experts]
    assert torch.equal(block_scale[0].to(torch.float32), sf[0] / 8)
    assert torch.equal(block_scale[1].to(torch.float32), sf[1])
    assert global_scale.dtype == torch.float16
    assert global_scale.tolist() == pytest.approx([8e-5, 1e-3], rel=1e-3)
    assert torch.equal(
        weight,
        torch.stack(
            [expert.up_proj.weight.view(torch.int8) for expert in experts]))
