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

import copy
import itertools
import json
import os

import numpy as np
import onnx
import pytest
import test_lilicorr_loading as synthetic_checkpoint
import torch
import torch.nn.functional as F
from onnx import numpy_helper
from test_lilicorr_loading import _write_base, _write_draft

from tensorrt_edgellm.config import make_lilicorr_draft_config
from tensorrt_edgellm.models.dflash2.modeling_dflash2_draft import (
    DFlash2DecoderLayer, DFlash2DraftModel)
from tensorrt_edgellm.models.lilicorr.modeling_lilicorr_draft import (
    LILICORR_EMBEDDING_TABLE_INPUT, LILICORR_FC_INPUT_SCALE,
    LILICORR_SUBLAYER_OUTPUT_SCALE, LiLiCorrDraftModel, select_anchor_rows)
from tensorrt_edgellm.models.linear import Q8_0Linear
from tensorrt_edgellm.onnx import cli as export_script
from tensorrt_edgellm.onnx.export import _export_model

_HIDDEN = 64
_VOCAB = 128
_BLOCK = 8
_SLOTS = _BLOCK - 1
_TOPK = 8
_NUM_LAYERS = 5

# Unquantized, dimension-scaled copy of the LiLiCorr draft config.json.
_LILICORR_CONFIG = {
    "architectures": ["LiLiCorrDraftModel"],
    "model_type": "qwen3",
    "block_size": _BLOCK,
    "dflash_config": {
        "mask_token_id": 990,
        "target_layer_ids": [1, 18, 35, 51, 68, 85],
        "use_swa": True,
        "swa_window_size": 1024,
        "causal": False,
        "attention_sink_bias": True,
        "projector_type": "lilicorr",
        "lilicorr_enabled": True,
        "lilicorr_candidate_topk": _TOPK,
        "lilicorr_hidden_size": 32,
        "lilicorr_num_layers": 2,
        "lilicorr_num_heads": 4,
        "lilicorr_mlp_ratio": 2.0,
        "lilicorr_factor_dim": 24,
        "lilicorr_vector_eps": 1e-3,
        "lilicorr_logit_scale": 3.0,
        "conv_kernel_size": 2,
        "conv_group_size": 16,
    },
    "hidden_size": _HIDDEN,
    "num_hidden_layers": _NUM_LAYERS,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "vocab_size": _VOCAB,
    "max_position_embeddings": 262144,
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "layer_types": ["sliding_attention"] * _NUM_LAYERS,
    "sliding_window": 1024,
    "use_sliding_window": True,
}

# Runtime dflash_config the LiLiCorr draft export must write.
_RUNTIME_DFLASH_CONFIG = {
    "version": 2,
    "target_layer_ids": [1, 18, 35, 51, 68, 85],
    "block_size": 8,
    "mask_token_id": 990,
    "is_causal": False,
    "conv_kernel_size": 2,
    "conv_group_size": 16,
    "supports_probabilistic_sampling": True,
    "projector_type": "lilicorr",
    "lilicorr_candidate_topk": 8,
}

_DRAFT_INPUT_NAMES = (["inputs_embeds", "dflash_target_hidden_concat"] + [
    f"past_key_values_{i}" for i in range(_NUM_LAYERS)
] + [
    "rope_rotary_cos_sin", "positions", "query_start_offsets", "query_lengths",
    "past_lengths", "attention_sequence_lengths", "state_indices",
    "execution_phase_marker", "context_sequence_count_carrier",
    "kv_page_table", "dflash_delta_rope_cos_sin", "dflash_delta_positions",
    "dflash_delta_token_to_sequence", "attention_position_ids",
    "packed_attention_mask", LILICORR_EMBEDDING_TABLE_INPUT
])


def _tiny_model(tmp_path, seed: int = 0) -> LiLiCorrDraftModel:
    (tmp_path / "config.json").write_text(json.dumps(_LILICORR_CONFIG))
    config = make_lilicorr_draft_config(str(tmp_path),
                                        lambda head_dim: head_dim**-0.5)
    model = LiLiCorrDraftModel(config).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in itertools.chain(model.named_parameters(),
                                            model.named_buffers()):
            if (not tensor.is_floating_point()
                    or name.endswith("lattice_attention_bias")):
                continue
            values = torch.randn(tensor.shape, generator=g) * 0.1
            if name.endswith("norm.weight"):
                values += 1.0
            tensor.copy_(values.to(tensor.dtype))
        # Identity-like head: logits = [hidden, 0, ..., 0].
        model.lm_head.weight.zero_()
        model.lm_head.weight[:_HIDDEN].copy_(torch.eye(_HIDDEN))
    return model


def _fixed_final_norm(model, seed: int = 1) -> torch.Tensor:
    """Replace the eager final-norm output with a seeded tensor.

    The eager plugin proxies return uninitialized memory, so the drafter
    hidden is pinned to make the candidate path deterministic.
    """
    rows = 2 * _BLOCK
    fixed = torch.randn(rows,
                        _HIDDEN,
                        generator=torch.Generator().manual_seed(seed))
    model.norm.register_forward_hook(
        lambda _module, _args, output: fixed.to(output.dtype))
    return fixed


def _ragged_delta_args(model, owners):
    """Export args with padded sequence-major delta rows owned by ``owners``."""
    spec = model.onnx_export_spec()
    args = dict(zip(spec.input_names, spec.args))
    rows = len(owners)
    g = torch.Generator().manual_seed(2)
    args["dflash_target_hidden_concat"] = torch.randn(rows,
                                                      6 * _HIDDEN,
                                                      generator=g).to(
                                                          torch.float16)
    args["dflash_delta_rope_cos_sin"] = torch.zeros(
        rows, args["dflash_delta_rope_cos_sin"].shape[1])
    args["dflash_delta_positions"] = torch.arange(rows, dtype=torch.int32)
    args["dflash_delta_token_to_sequence"] = torch.tensor(owners,
                                                          dtype=torch.int32)
    args[LILICORR_EMBEDDING_TABLE_INPUT] = torch.randn(_VOCAB,
                                                       _HIDDEN,
                                                       generator=g).to(
                                                           torch.float16)
    return spec, args


def _expected_candidates(fixed: torch.Tensor):
    hidden = fixed.to(torch.float16).reshape(2, _BLOCK, _HIDDEN)[:, 1:]
    logits = torch.cat([
        hidden.float(),
        torch.zeros(2, _SLOTS, _VOCAB - _HIDDEN),
    ],
                       dim=-1)
    values, ids = torch.topk(logits, _TOPK, dim=-1)
    log_probs = values - torch.logsumexp(logits, dim=-1, keepdim=True)
    return hidden, ids, log_probs


def test_select_anchor_rows_uses_last_valid_row_per_sequence():
    """Anchor is the last owned row of each sequence; empty sequences give zeros."""
    h_delta = torch.randn(8, 16).to(torch.float16)
    owners = torch.tensor([0, 0, 0, -1, 1, -1, -1, -1], dtype=torch.int32)

    anchor, valid = select_anchor_rows(h_delta, owners, 2)

    assert torch.equal(anchor, h_delta[[2, 4]])
    assert valid.dtype == torch.bool
    assert valid.tolist() == [True, True]

    anchor, valid = select_anchor_rows(h_delta, owners, 3)

    assert torch.equal(anchor[:2], h_delta[[2, 4]])
    assert valid.tolist() == [True, True, False]
    assert torch.count_nonzero(anchor[2]) == 0


def test_select_anchor_rows_keeps_nan_out_of_empty_sequences():
    """A non-finite row 0 must not leak into the zero anchor of a sequence without delta rows."""
    h_delta = torch.randn(4, 16).to(torch.float16)
    h_delta[0] = float("nan")
    owners = torch.tensor([-1, -1, 1, -1], dtype=torch.int32)

    anchor, valid = select_anchor_rows(h_delta, owners, 2)

    assert valid.tolist() == [False, True]
    assert torch.equal(anchor[0], torch.zeros_like(anchor[0]))
    assert torch.equal(anchor[1], h_delta[2])


def test_candidates_are_full_vocab_logprobs(tmp_path):
    """Support ids are the FP32 top-K and the head sees full-vocabulary log-probs."""
    model = _tiny_model(tmp_path)
    fixed = _fixed_final_norm(model)
    captured = []
    model.lilicorr.register_forward_pre_hook(
        lambda _module, args: captured.append(args))
    spec = model.onnx_export_spec()

    with torch.no_grad():
        outputs = spec.wrapped(*spec.args)

    hidden, ids, log_probs = _expected_candidates(fixed)
    assert outputs[0].dtype == torch.int32
    assert torch.equal(outputs[0], ids.to(torch.int32))
    head_log_probs = captured[0][1]
    assert head_log_probs.dtype == torch.float32
    torch.testing.assert_close(head_log_probs, log_probs)
    assert torch.equal(captured[0][2], hidden)


def test_export_io_contract(tmp_path):
    """Draft ONNX keeps DFlash2 inputs and emits only ids, transition scores and KV."""
    model = _tiny_model(tmp_path)
    output_path = tmp_path / "lilicorr_draft.onnx"

    _export_model(model, str(output_path), optimize=False)

    onnx.checker.check_model(str(output_path))
    graph = onnx.load(str(output_path), load_external_data=False).graph
    assert [value.name for value in graph.input] == _DRAFT_INPUT_NAMES
    outputs = {value.name: value.type.tensor_type for value in graph.output}
    assert list(outputs) == [
        "spec_proposal_support_ids", "spec_lilicorr_transition_scores"
    ] + [f"present_key_values_{i}" for i in range(_NUM_LAYERS)]
    ids = outputs["spec_proposal_support_ids"]
    scores = outputs["spec_lilicorr_transition_scores"]
    assert ids.elem_type == onnx.TensorProto.INT32
    assert scores.elem_type == onnx.TensorProto.FLOAT
    # The slot extent of the ids follows the dynamic token-major block width.
    assert len(ids.shape.dim) == 3
    assert ids.shape.dim[2].dim_value == _TOPK
    assert [dim.dim_value
            for dim in scores.shape.dim[1:]] == [_SLOTS, _TOPK, _TOPK]
    assert ids.shape.dim[0].dim_param == "num_sequences"
    assert scores.shape.dim[0].dim_param == "num_sequences"
    assert not hasattr(model, "candidate_selector")
    assert not any("candidate_selector" in init.name or "codebook" in init.name
                   for init in graph.initializer)


def test_eager_wrapper_matches_head_reference(tmp_path):
    """Graph scores equal LiLiCorrHead on the anchor, gathered runtime embeddings and log-prob inputs."""
    model = _tiny_model(tmp_path)
    fixed = _fixed_final_norm(model)
    owners = [0, 0, -1, 1, -1, -1]
    spec, args = _ragged_delta_args(model, owners)

    with torch.no_grad():
        outputs = spec.wrapped(*args.values())
        hidden, ids, log_probs = _expected_candidates(fixed)
        h_delta = model.hidden_norm(
            F.linear(args["dflash_target_hidden_concat"].float(),
                     model.fc.weight.float())).to(torch.float16)
        embeddings = args[LILICORR_EMBEDDING_TABLE_INPUT][ids]
        expected = model.lilicorr(embeddings, log_probs, hidden,
                                  h_delta[[1, 3]], torch.tensor([True, True]))

    assert torch.equal(outputs[0], ids.to(torch.int32))
    scores = outputs[1]
    assert scores.dtype == torch.float32
    assert scores.shape == (2, _SLOTS, _TOPK, _TOPK)
    assert (scores - expected).abs().max() <= 1e-4


def test_native_fc_runs_downscaled_and_restores_scale_in_fp32(tmp_path):
    """The packed fc sees FP16 target hidden divided by its input scale; hidden_norm sees fc output times that scale in FP32."""
    model = _tiny_model(tmp_path)
    _fixed_final_norm(model)
    owners = [0, 0, -1, 1, -1, -1]
    spec, args = _ragged_delta_args(model, owners)
    model.fc_native_precision = True
    seen = {}
    fc_out = torch.randn(len(owners),
                         _HIDDEN,
                         generator=torch.Generator().manual_seed(4)).to(
                             torch.float16)

    def fc_stub(_module, inputs, _output):
        seen["fc_input"] = inputs[0]
        return fc_out

    model.fc.register_forward_hook(fc_stub)
    model.hidden_norm.register_forward_pre_hook(
        lambda _module, inputs: seen.setdefault("norm_input", inputs[0]))
    head_args = []
    model.lilicorr.register_forward_pre_hook(
        lambda _module, inputs: head_args.append(inputs))

    with torch.no_grad():
        spec.wrapped(*args.values())
        restored = fc_out.float() * LILICORR_FC_INPUT_SCALE
        expected_anchor = model.hidden_norm(restored).to(torch.float16)[[1, 3]]

    target_hidden = args["dflash_target_hidden_concat"]
    assert seen["fc_input"].dtype == torch.float16
    assert torch.equal(seen["fc_input"],
                       target_hidden / LILICORR_FC_INPUT_SCALE)
    assert seen["norm_input"].dtype == torch.float32
    assert torch.equal(seen["norm_input"], restored)
    assert torch.equal(head_args[0][3], expected_anchor)


def test_dflash2_projection_rejects_native_fc(tmp_path):
    """The DFlash2 dense fc projection refuses a native-precision fc rather than matmul packed weights."""
    model = _tiny_model(tmp_path)
    model.fc_native_precision = True

    with pytest.raises(NotImplementedError, match="packed fc"):
        DFlash2DraftModel._project_target_hidden(
            model, torch.zeros(2, 6 * _HIDDEN, dtype=torch.float16))


def test_export_spec_prepares_q8_0_lm_head(tmp_path):
    """A Q8_0 lm_head is switched to INT8 storage before export."""
    model = _tiny_model(tmp_path)
    model.lm_head = Q8_0Linear(_HIDDEN, _VOCAB)

    model.onnx_export_spec()

    assert model.lm_head.weight.dtype == torch.int8


@pytest.fixture(scope="module")
def cli_draft_export(tmp_path_factory):
    root = tmp_path_factory.mktemp("lilicorr_cli_export")
    _write_base(root / "base")
    _write_draft(root / "draft")
    out_dir = root / "onnx"
    export_script.main([
        str(root / "base"),
        str(out_dir), "--dflash-draft", "--dflash-draft-dir",
        str(root / "draft")
    ])
    return out_dir / "dflash_draft"


def _strictly_inferred_types(model: onnx.ModelProto) -> dict:
    """Re-derive standard-op tensor types from initializers and inputs.

    Exporter-declared ``value_info`` is dropped except for custom-op outputs,
    whose types ONNX cannot infer, so stale declarations cannot hide a
    downgraded initializer. Strict inference raises on mixed-type operands.
    """
    graph = model.graph
    custom_outputs = {
        name
        for node in graph.node if node.domain not in ("", "ai.onnx")
        for name in node.output
    }
    kept = [
        value for value in graph.value_info if value.name in custom_outputs
    ]
    del graph.value_info[:]
    graph.value_info.extend(kept)
    inferred = onnx.shape_inference.infer_shapes(model,
                                                 check_type=True,
                                                 strict_mode=True)
    types = {
        value.name: value.type.tensor_type.elem_type
        for value in list(inferred.graph.value_info) +
        list(inferred.graph.output)
    }
    types.update(
        (init.name, init.data_type) for init in inferred.graph.initializer)
    return types


def test_cli_export_keeps_head_fp32_islands(cli_draft_export):
    """Production CLI export keeps every head FP32 island type-consistent.

    TensorRT rejects mixed FP16/FP32 operands, e.g. a rank feature downgraded
    to FP16 inside the FP32 feature stack.
    """
    model = onnx.load(str(cli_draft_export / "model.onnx"))

    types = _strictly_inferred_types(model)

    layer_norms = [
        node for node in model.graph.node
        if node.op_type == "LayerNormalization"
    ]
    assert layer_norms
    for node in layer_norms:
        assert [types[name] for name in node.input
                ] == [onnx.TensorProto.FLOAT] * len(node.input)
    feature_stack = next(node for node in model.graph.node
                         if node.output[0] == layer_norms[0].input[0])
    assert feature_stack.op_type == "Concat"
    assert len(feature_stack.input) == 5
    assert all(types[name] == onnx.TensorProto.FLOAT
               for name in feature_stack.input)
    assert types["spec_lilicorr_transition_scores"] == onnx.TensorProto.FLOAT


def test_cli_export_keeps_nvfp4_linears_packed(cli_draft_export):
    """fc and the NVFP4 head linears export as W4A16 plugins; no dense fc or folded token table is emitted."""
    model = onnx.load(str(cli_draft_export / "model.onnx"))
    graph = model.graph

    plugins = [
        node for node in graph.node if node.op_type == "Nvfp4A16GemmPlugin"
    ]
    # 5 layers x 3 backbone MLP + fc + 3 head projections + 2 lattice layers x 3.
    assert len(plugins) == 15 + 1 + 3 + 2 * 3
    # The plugin accepts token-major [M, K] or batched [B, M, K] activations only.
    inferred = onnx.shape_inference.infer_shapes(model)
    ranks = {
        value.name: len(value.type.tensor_type.shape.dim)
        for value in list(inferred.graph.value_info) +
        list(inferred.graph.input)
    }
    assert all(ranks[node.input[0]] in (2, 3) for node in plugins)
    float_dims = [
        tuple(init.dims) for init in graph.initializer
        if init.data_type in (onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT)
    ]
    assert (_HIDDEN, 6 * _HIDDEN) not in float_dims
    assert (6 * _HIDDEN, _HIDDEN) not in float_dims
    assert not any(dims[:1] == (_VOCAB, ) for dims in float_dims)


def test_cli_export_gathers_candidates_from_runtime_embedding(
        cli_draft_export):
    """The draft takes the FP16 target embedding table [V, H] as an input and gathers candidate rows from it."""
    model = onnx.load(str(cli_draft_export / "model.onnx"),
                      load_external_data=False)
    graph = model.graph

    table = next(value for value in graph.input
                 if value.name == LILICORR_EMBEDDING_TABLE_INPUT)
    tensor_type = table.type.tensor_type
    assert tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert [dim.dim_value
            for dim in tensor_type.shape.dim] == [_VOCAB, _HIDDEN]
    consumers = [
        node for node in graph.node
        if LILICORR_EMBEDDING_TABLE_INPUT in node.input
    ]
    assert [node.op_type for node in consumers] == ["Gather"]
    assert consumers[0].input[0] == LILICORR_EMBEDDING_TABLE_INPUT


def test_cli_export_writes_no_selector_sidecar(cli_draft_export):
    """The draft CLI export writes the LiLiCorr dflash_config and no selector sidecar."""
    draft_out = cli_draft_export
    assert (draft_out / "model.onnx").is_file()
    assert not (draft_out / "dflash2_selector.safetensors").exists()
    config = json.loads((draft_out / "config.json").read_text())
    assert config["dflash_config"] == _RUNTIME_DFLASH_CONFIG


@pytest.mark.parametrize("projection", ["up_proj", "down_proj"])
def test_input_scaling_rejects_biased_projections(tmp_path, monkeypatch,
                                                  projection):
    """Input scaling is only exact for bias-free o_proj/up_proj/down_proj, so a biased MLP projection is rejected."""
    (tmp_path / "config.json").write_text(json.dumps(_LILICORR_CONFIG))
    config = make_lilicorr_draft_config(str(tmp_path),
                                        lambda head_dim: head_dim**-0.5)
    base_init = DFlash2DecoderLayer.__init__

    def init_with_bias(self, layer_config, layer_idx):
        base_init(self, layer_config, layer_idx)
        linear = getattr(self.mlp, projection)
        linear.bias = torch.nn.Parameter(torch.zeros(linear.weight.shape[0]))

    monkeypatch.setattr(DFlash2DecoderLayer, "__init__", init_with_bias)

    with pytest.raises(ValueError, match="bias-free"):
        LiLiCorrDraftModel(config)


def _constant_values(model: onnx.ModelProto) -> dict:
    values = {
        init.name: numpy_helper.to_array(init)
        for init in model.graph.initializer
    }
    for node in model.graph.node:
        if node.op_type == "Constant":
            values[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
    return values


def _scalar_multiplier(node, constants):
    """Return the scalar constant factor of a Mul node, or None."""
    if node is None or node.op_type != "Mul":
        return None
    factors = [
        constants[name] for name in node.input
        if name in constants and constants[name].size == 1
    ]
    return float(factors[0]) if len(factors) == 1 else None


def test_cli_export_restores_sublayer_scale_in_fp32_residual(cli_draft_export):
    """o_proj/up_proj inputs are divided by scale; each fused-residual conv adds residual/scale and its output is multiplied by scale."""
    model = onnx.load(str(cli_draft_export / "model.onnx"))
    constants = _constant_values(model)
    producers = {out: node for node in model.graph.node for out in node.output}
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    def projection_inputs(weight_fragment):
        return [
            node.input[0] for node in model.graph.node if any(
                name.startswith("_model.layers.") and weight_fragment in name
                for name in node.input[1:])
        ]

    for fragment in ("self_attn.o_proj.", "mlp.up_proj."):
        inputs = projection_inputs(fragment)
        assert len(inputs) == _NUM_LAYERS
        assert all(
            _scalar_multiplier(producers.get(name), constants) ==
            pytest.approx(1.0 / LILICORR_SUBLAYER_OUTPUT_SCALE)
            for name in inputs)
    down_inputs = projection_inputs("mlp.down_proj.")
    assert len(down_inputs) == _NUM_LAYERS
    assert all(
        _scalar_multiplier(producers.get(name), constants) is None
        for name in down_inputs)

    finishes = [
        node for node in model.graph.node
        if node.op_type == "DFlash2GroupedDynamicConvPlugin"
        and len(node.input) == 4
    ]
    assert len(finishes) == 2 * _NUM_LAYERS
    for node in finishes:
        assert _scalar_multiplier(producers.get(node.input[3]),
                                  constants) == pytest.approx(
                                      1.0 / LILICORR_SUBLAYER_OUTPUT_SCALE)
        assert [
            _scalar_multiplier(consumer, constants)
            for consumer in consumers[node.output[0]]
        ] == [pytest.approx(LILICORR_SUBLAYER_OUTPUT_SCALE)]


@pytest.fixture(scope="module")
def direct_builder_checkpoints(tmp_path_factory):
    """The synthetic base and draft at dims the direct-builder plugins accept.

    The attention plugin needs head_dim 128 and the W4A16 GEMM K % 64 == 0.
    """
    hidden, head_hidden, factor = 128, 64, 32
    draft_config = copy.deepcopy(synthetic_checkpoint._DRAFT_CONFIG)
    draft_config.update(hidden_size=hidden, head_dim=128)
    draft_config["dflash_config"].update(lilicorr_hidden_size=head_hidden,
                                         lilicorr_factor_dim=factor)
    dims = {
        "_H":
        hidden,
        "_HEAD_DIM":
        128,
        "_D":
        head_hidden,
        "_LILICORR_MLP":
        2 * head_hidden,
        "_FACTOR":
        factor,
        "_CONV_PROJ_OUT":
        2 * 2 * (hidden // 16),
        "_DRAFT_CONFIG":
        draft_config,
        "_BASE_CONFIG":
        dict(synthetic_checkpoint._BASE_CONFIG,
             hidden_size=hidden,
             head_dim=128),
    }
    root = tmp_path_factory.mktemp("lilicorr_direct_builder")
    with pytest.MonkeyPatch.context() as patch:
        for name, value in dims.items():
            patch.setattr(synthetic_checkpoint, name, value)
        _write_base(root / "base")
        _write_draft(root / "draft")
    return root, hidden


def _o_proj_input_scales(network, trt) -> dict:
    """Map each o_proj weight to the scalar its activation input is multiplied by (None if unscaled)."""
    producers = {}
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        for output in range(layer.num_outputs):
            producers[layer.get_output(output).name] = layer
    scales = {}
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        weight = layer.get_input(1) if layer.num_inputs > 1 else None
        if (layer.type != trt.LayerType.MATRIX_MULTIPLY or weight is None
                or not weight.name.endswith("self_attn.o_proj.weight")):
            continue
        scale = None
        product = producers.get(layer.get_input(0).name)
        if product is not None and product.type == trt.LayerType.ELEMENTWISE:
            product.__class__ = trt.IElementWiseLayer
            if product.op == trt.ElementWiseOperation.PROD:
                for operand in (product.get_input(0), product.get_input(1)):
                    source = producers.get(operand.name)
                    if (source is not None
                            and source.type == trt.LayerType.CONSTANT):
                        source.__class__ = trt.IConstantLayer
                        scale = float(np.asarray(source.weights).item())
        scales[weight.name] = scale
    return scales


def _onnx_static_dims(value) -> list:
    return [
        dim.dim_value if dim.WhichOneof("value") == "dim_value" else None
        for dim in value.type.tensor_type.shape.dim
    ]


def test_direct_builder_draft_matches_the_onnx_draft_contract(
        direct_builder_checkpoints, tmp_path, monkeypatch):
    """A checkpoint-direct LiLiCorr draft engine has the ONNX draft's bindings and dflash_config, no selector sidecar, and whole-block proposal profiles."""
    trt = pytest.importorskip("tensorrt")
    plugin_path = os.environ.get("EDGELLM_PLUGIN_PATH",
                                 "build/lib/libNvInfer_edgellm_plugin.so")
    if not os.path.isfile(plugin_path):
        pytest.skip("Edge-LLM plugin library is not available")
    from tensorrt_edgellm.builder.core import contracts
    from tensorrt_edgellm.builder.core.artifacts.runtime_config import \
        build_runtime_config
    from tensorrt_edgellm.builder.core.builder import (BuildArgs, build_engine,
                                                       load_device_config)
    from tensorrt_edgellm.builder.models import registry
    from tensorrt_edgellm.builder.models.dflash2 import \
        artifacts as dflash2_artifacts
    from tensorrt_edgellm.dflash import DFlashVersion

    root, hidden = direct_builder_checkpoints
    export_script.main([
        str(root / "base"),
        str(tmp_path / "onnx"), "--dflash-draft", "--dflash-draft-dir",
        str(root / "draft")
    ])
    onnx_dir = tmp_path / "onnx" / "dflash_draft"
    graph = onnx.load(str(onnx_dir / "model.onnx"),
                      load_external_data=False).graph
    trt_builder = trt.Builder

    o_proj_scales = {}

    class FastBuilder:
        """Builds at optimization level 0 and records each layer's inputs."""

        def __init__(self, logger):
            self._builder = trt_builder(logger)

        def __getattr__(self, name):
            return getattr(self._builder, name)

        def create_builder_config(self):
            config = self._builder.create_builder_config()
            config.builder_optimization_level = 0
            config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
            return config

        def build_serialized_network(self, network, config):
            o_proj_scales.update(_o_proj_input_scales(network, trt))
            return self._builder.build_serialized_network(network, config)

    monkeypatch.setattr(trt, "Builder", FastBuilder)
    engine_dir = tmp_path / "engine"
    args = BuildArgs(model_dir=str(root / "draft"),
                     engine_dir=str(engine_dir),
                     spec_role="draft",
                     spec_type="dflash",
                     dflash_version=DFlashVersion.V2,
                     target_model_dir=str(root / "base"),
                     max_batch_size=2,
                     max_input_len=64,
                     max_kv_cache_capacity=256,
                     max_spec_verify_tokens=_BLOCK,
                     max_spec_draft_tokens=_BLOCK,
                     plugin_path=plugin_path)

    cfg = load_device_config(args)
    result = build_engine(args, cfg)
    # The synthetic checkpoint has no tokenizer for the shared runtime files.
    monkeypatch.setattr(dflash2_artifacts, "write_runtime_artifacts",
                        lambda *args, **kwargs: None)
    registry.artifact_writer_for("qwen3", "dflash", contracts.SpecRole.DRAFT,
                                 DFlashVersion.V2).write_artifacts(
                                     None, cfg, args, str(engine_dir))

    engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(
        open(result.engine_path, "rb").read())
    checkpoint_inputs = {
        binding["engine_name"]
        for binding in result.checkpoint_weight_bindings
    }
    bindings = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if name not in checkpoint_inputs:
            bindings[name] = (engine.get_tensor_mode(name),
                              engine.get_tensor_dtype(name),
                              tuple(engine.get_tensor_shape(name)))
    expected = [(trt.TensorIOMode.INPUT, value) for value in graph.input]
    expected += [(trt.TensorIOMode.OUTPUT, value) for value in graph.output]
    trt_dtypes = {
        onnx.TensorProto.FLOAT: trt.float32,
        onnx.TensorProto.FLOAT16: trt.float16,
        onnx.TensorProto.INT32: trt.int32,
    }
    assert set(bindings) == {value.name for _, value in expected}
    for mode, value in expected:
        engine_mode, dtype, shape = bindings[value.name]
        static_dims = _onnx_static_dims(value)
        assert engine_mode == mode, value.name
        assert dtype == trt_dtypes[
            value.type.tensor_type.elem_type], value.name
        assert len(shape) == len(static_dims), value.name
        assert all(want is None or got == want
                   for got, want in zip(shape, static_dims)), value.name
    # TensorRT keeps an unused input binding, so check the table is read.
    layers = json.loads(
        engine.create_engine_inspector().get_engine_information(
            trt.LayerInformationFormat.JSON))["Layers"]
    assert any(tensor["Name"] == LILICORR_EMBEDDING_TABLE_INPUT
               for layer in layers for tensor in layer.get("Inputs", []))
    # FP16 range guard: every o_proj reads its input at 1/1024.
    assert o_proj_scales == {
        f"layers.{index}.self_attn.o_proj.weight":
        pytest.approx(1.0 / LILICORR_SUBLAYER_OUTPUT_SCALE)
        for index in range(_NUM_LAYERS)
    }
    for profile in range(engine.num_optimization_profiles):
        minimum = engine.get_tensor_profile_shape("inputs_embeds", profile)[0]
        assert tuple(minimum) == (_BLOCK, hidden)
    onnx_config = json.loads((onnx_dir / "config.json").read_text())
    assert (build_runtime_config(
        cfg, args)["dflash_config"] == onnx_config["dflash_config"])
    assert not (engine_dir / "dflash2_selector.safetensors").exists()
