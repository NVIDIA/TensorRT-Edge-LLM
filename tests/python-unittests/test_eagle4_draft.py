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
"""EAGLE-4 drafter config parsing, ONNX export and quantize draft, on a tiny seeded drafter."""

import json
import os
import types

import onnx
import pytest
import torch
import transformers
from safetensors.torch import load_file, save_file
from torch import nn

from tensorrt_edgellm import model as model_module
from tensorrt_edgellm.model import load_model_config
from tensorrt_edgellm.onnx import cli as export_script

_HIDDEN = 16
_VOCAB = 64
_HEAD_DIM = 8
_ENGINE_HEAD_DIM = 128
_RANK = 16
_CANDIDATES = 8
_WINDOW = 4096
_BASE_LAYERS = 8
_LAYER_LINEARS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                  "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                  "mlp.down_proj")
# The published drafter schema: aux IDs nested in eagle_config name decoder-layer outputs 1 to 3 (HF hidden-state
# indices 2 to 4), and the low-rank head sits next to a dense lm_head.
_DRAFTER_CONFIG = {
    "architectures": ["LlamaForCausalLMEagle4"],
    "model_type": "llama",
    "eagle_config": {
        "eagle_aux_hidden_state_layer_ids": [1, 2, 3],
        "use_aux_hidden_state": True
    },
    "efficient_lm_head_config": {
        "num_candidates": _CANDIDATES,
        "rank": _RANK,
        "type": "low_rank"
    },
    "fc_norm": True,
    "norm_output": True,
    "hidden_size": _HIDDEN,
    "intermediate_size": 2 * _HIDDEN,
    "head_dim": _HEAD_DIM,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 1,
    "num_aux_hidden_states": 3,
    "target_hidden_size": _HIDDEN,
    "vocab_size": _VOCAB,
    "draft_vocab_size": _VOCAB,
    "rms_norm_eps": 1e-6,
    "layer_types": ["sliding_attention"],
    "sliding_window": _WINDOW,
    "rope_parameters": {
        "rope_theta": 1e7,
        "rope_type": "default"
    },
    "torch_dtype": "bfloat16",
}


def _write_config(directory, config: dict) -> str:
    os.makedirs(str(directory), exist_ok=True)
    with open(os.path.join(str(directory), "config.json"), "w") as f:
        json.dump(config, f)
    return str(directory)


def _write_drafter(directory, head_dim: int = _HEAD_DIM) -> str:
    """Writes the drafter's config.json and seeded bfloat16 weights under its checkpoint tensor names."""
    generator = torch.Generator().manual_seed(0)
    shapes = {
        "efficient_lm_head.weight": (_VOCAB, _RANK),
        "efficient_lm_head_down_proj.weight": (_RANK, _HIDDEN),
        "lm_head.weight": (_VOCAB, _HIDDEN),
        "embed_tokens.weight": (_VOCAB, _HIDDEN),
        "fc.weight": (_HIDDEN, 3 * _HIDDEN),
        "layers.0.self_attn.q_proj.weight": (2 * head_dim, 2 * _HIDDEN),
        "layers.0.self_attn.k_proj.weight": (2 * head_dim, 2 * _HIDDEN),
        "layers.0.self_attn.v_proj.weight": (2 * head_dim, 2 * _HIDDEN),
        "layers.0.self_attn.o_proj.weight": (_HIDDEN, 2 * head_dim),
        "layers.0.mlp.gate_proj.weight": (2 * _HIDDEN, _HIDDEN),
        "layers.0.mlp.up_proj.weight": (2 * _HIDDEN, _HIDDEN),
        "layers.0.mlp.down_proj.weight": (_HIDDEN, 2 * _HIDDEN),
    }
    tensors = {
        name: torch.randn(shape, generator=generator) / shape[1]**0.5
        for name, shape in shapes.items()
    }
    for norm in ("fc_norm.0", "fc_norm.1", "fc_norm.2", "layers.0.hidden_norm",
                 "layers.0.input_layernorm",
                 "layers.0.post_attention_layernorm", "norm"):
        tensors[f"{norm}.weight"] = 1.0 + 0.1 * torch.randn(
            _HIDDEN, generator=generator)
    path = _write_config(directory, {**_DRAFTER_CONFIG, "head_dim": head_dim})
    save_file(
        {
            name: tensor.to(torch.bfloat16).contiguous()
            for name, tensor in tensors.items()
        }, os.path.join(path, "model.safetensors"))
    return path


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return _write_drafter(tmp_path_factory.mktemp("eagle4"))


def _attribute(node, name):
    (attribute, ) = [item for item in node.attribute if item.name == name]
    return onnx.helper.get_attribute_value(attribute)


@pytest.mark.parametrize(
    ("draft_overrides", "base_model_type", "base_layers", "message"), [
        ({}, "qwen3_5_text", _BASE_LAYERS, None),
        ({
            "architectures": ["LlamaForCausalLM"],
            "eagle_config": {
                "eagle_aux_hidden_state_layer_ids": [1, 2, 3]
            },
        }, "qwen3_5_text", _BASE_LAYERS, "no EAGLE draft marker"),
        ({
            "_torchspec_version": "0.1",
            "eagle_config": {
                "eagle_aux_hidden_state_layer_ids": [2, 1, 3],
                "use_aux_hidden_state": True,
            },
        }, "qwen3_5_text", _BASE_LAYERS, "ascending"),
        ({}, "qwen3_5_text", 4, "below the base num_hidden_layers=4"),
        ({}, "gemma4_text", _BASE_LAYERS, "Gemma4 EAGLE base"),
        ({
            "fc_norm": False,
            "norm_before_fc": True
        }, "qwen3_5_text", _BASE_LAYERS, "not support norm_before_fc"),
        ({
            "model_type": "gemma4_text"
        }, "qwen3_5_text", _BASE_LAYERS, "Gemma4 draft layer"),
    ],
    ids=[
        "drafter", "no-marker", "torchspec-order", "id-past-base",
        "gemma4-base", "norm-before-fc", "gemma4-draft"
    ])
def test_eagle4_drafter_config(tmp_path, draft_overrides, base_model_type,
                               base_layers, message):
    """The drafter config yields its norm flags, the candidate head and its aux layers as HF hidden-state indices (nested layer outputs + 1), which its EAGLE base inherits; parsing and pairing reject a drafter that would export as dense EAGLE3, sets an EAGLE3 draft option or layer type the EAGLE-4 graph does not implement, reads aux layers the base cannot provide, or pairs with a Gemma4 base."""
    draft_dir = _write_config(tmp_path, {**_DRAFTER_CONFIG, **draft_overrides})
    base = types.SimpleNamespace(model_type=base_model_type,
                                 num_hidden_layers=base_layers)

    if message is not None:
        with pytest.raises(ValueError, match=message):
            load_model_config(draft_dir)
            model_module._inherit_eagle_base_layer_ids(base, draft_dir)
        return
    config = load_model_config(draft_dir)
    model_module._inherit_eagle_base_layer_ids(base, draft_dir)

    assert config.is_eagle4_draft
    assert config.eagle_fc_norm and config.eagle_norm_output
    assert (config.draft_candidate_rank,
            config.draft_candidate_count) == (_RANK, _CANDIDATES)
    assert config.eagle3_target_layer_ids == [2, 3, 4]
    assert base.eagle3_target_layer_ids == [2, 3, 4]


@pytest.mark.parametrize("dense_head", [False, True],
                         ids=["candidate-head", "dense-head"])
def test_eagle4_draft_export(checkpoint, tmp_path, dense_head):
    """The draft ONNX replaces logits with the candidate head plugin's ids (INT32) and log-probs (FP32), [rows, Kc] each, and records the head in its runtime config; --draft-dense-head exports full-vocabulary logits from lm_head instead. Both keep the drafter's sliding window."""
    export_script.main([checkpoint, str(tmp_path)] +
                       (["--draft-dense-head"] if dense_head else []))
    exported = os.path.join(str(tmp_path), "llm")
    graph = onnx.load(os.path.join(exported, "model.onnx")).graph
    with open(os.path.join(exported, "config.json"), encoding="utf-8") as f:
        eagle_config = json.load(f)["eagle3_config"]
    outputs = {output.name: output.type.tensor_type for output in graph.output}
    heads = [
        node for node in graph.node
        if node.op_type == "LowRankCandidateHeadPlugin"
    ]
    (attention, ) = [
        node for node in graph.node if node.op_type == "AttentionPlugin"
    ]

    assert "hidden_states" in outputs
    assert _attribute(attention, "sliding_window_size") == _WINDOW
    if dense_head:
        assert outputs["logits"].shape.dim[1].dim_value == _VOCAB
        assert not heads
        assert "candidate_head" not in eagle_config
        return
    (head, ) = heads
    assert "logits" not in outputs
    assert _attribute(head, "num_candidates") == _CANDIDATES
    assert list(head.output) == [
        "spec_proposal_support_ids", "spec_proposal_support_log_probs"
    ]
    assert [outputs[name].elem_type for name in head.output
            ] == [onnx.TensorProto.INT32, onnx.TensorProto.FLOAT]
    assert [outputs[name].shape.dim[1].dim_value
            for name in head.output] == [_CANDIDATES, _CANDIDATES]
    assert eagle_config["candidate_head"] == {
        "type": "low_rank",
        "rank": _RANK,
        "num_candidates": _CANDIDATES
    }


_ONNX_TO_TRT_DTYPE = {
    onnx.TensorProto.FLOAT16: "HALF",
    onnx.TensorProto.FLOAT: "FLOAT",
    onnx.TensorProto.INT32: "INT32",
    onnx.TensorProto.INT64: "INT64",
}

# The export lists these draft inputs although no node reads them; the runtime
# binds tree metadata only to engines that declare it.
_UNREAD_EXPORT_INPUTS = ("tree_parent_ids", "tree_depths", "valid_tree_counts")


def _export_io(onnx_path: str) -> dict:
    """Graph I/O as (mode, TensorRT dtype, dims), with -1 for symbolic dims."""
    graph = onnx.load(onnx_path, load_external_data=False).graph

    def describe(value, mode):
        tensor = value.type.tensor_type
        return (mode, _ONNX_TO_TRT_DTYPE[tensor.elem_type],
                tuple(dim.dim_value or -1 for dim in tensor.shape.dim))

    io = {
        value.name: describe(value, "INPUT")
        for value in graph.input if value.name not in _UNREAD_EXPORT_INPUTS
    }
    io.update(
        {value.name: describe(value, "OUTPUT")
         for value in graph.output})
    return io


def test_eagle4_direct_draft_engine_matches_export(tmp_path, monkeypatch):
    """The checkpoint-direct EAGLE-4 draft matches its export: engine I/O, plugin fields, eagle3_config, every checkpoint tensor bound and the final-norm state fed back."""
    trt = pytest.importorskip("tensorrt")
    plugin_path = os.environ.get("EDGELLM_PLUGIN_PATH",
                                 "build/lib/libNvInfer_edgellm_plugin.so")
    if not os.path.isfile(plugin_path):
        pytest.skip("Edge-LLM plugin library is not available")
    if not torch.cuda.is_available():
        pytest.skip("building a TensorRT engine needs CUDA")
    from tensorrt_edgellm.builder.core.artifacts.runtime_config import \
        build_runtime_config
    from tensorrt_edgellm.builder.core.builder import (BuildArgs, build_engine,
                                                       load_device_config)
    from tensorrt_edgellm.builder.models import registry
    from tensorrt_edgellm.builder.models.eagle4.modeling_eagle4_draft import \
        Eagle4DraftModel
    from tensorrt_edgellm.builder.ops import backend
    trt_builder = trt.Builder
    operation = backend.Net.operation
    head = Eagle4DraftModel._head
    mark_outputs = Eagle4DraftModel.mark_outputs
    plugin_fields = {}
    graph_tensors = {}

    def record_operation(net, name, attributes, inputs):
        plugin_fields[name] = dict(attributes)
        return operation(net, name, attributes, inputs)

    def record_head(model, normalized):
        graph_tensors["head_input"] = normalized
        return head(model, normalized)

    def record_outputs(model, outputs):
        graph_tensors["outputs"] = dict(outputs)
        return mark_outputs(model, outputs)

    class FastBuilder:
        """Builds at optimization level 0; the test reads I/O, not tactics."""

        def __init__(self, logger):
            self._builder = trt_builder(logger)

        def __getattr__(self, name):
            return getattr(self._builder, name)

        def create_builder_config(self):
            config = self._builder.create_builder_config()
            config.builder_optimization_level = 0
            return config

    monkeypatch.setattr(trt, "Builder", FastBuilder)
    monkeypatch.setattr(backend.Net, "operation", record_operation)
    monkeypatch.setattr(Eagle4DraftModel, "_head", record_head)
    monkeypatch.setattr(Eagle4DraftModel, "mark_outputs", record_outputs)
    # The attention plugin has no Blackwell decode kernel for the 8-wide heads
    # of the export-only drafter.
    checkpoint = _write_drafter(tmp_path / "draft", head_dim=_ENGINE_HEAD_DIM)
    export_script.main([checkpoint, str(tmp_path / "onnx")])
    exported = os.path.join(str(tmp_path), "onnx", "llm")
    with open(os.path.join(exported, "config.json"), encoding="utf-8") as f:
        exported_eagle3_config = json.load(f)["eagle3_config"]
    target = _write_config(
        tmp_path / "target", {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": _HIDDEN,
            "intermediate_size": 2 * _HIDDEN,
            "num_hidden_layers": _BASE_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": _ENGINE_HEAD_DIM,
            "vocab_size": _VOCAB,
        })
    args = BuildArgs(model_dir=checkpoint,
                     engine_dir=str(tmp_path / "engine"),
                     spec_role="draft",
                     spec_type="eagle3",
                     target_model_dir=target,
                     max_batch_size=2,
                     max_input_len=64,
                     max_kv_cache_capacity=256,
                     max_spec_draft_tokens=8,
                     plugin_path=plugin_path)

    result = build_engine(args)

    with open(result.engine_path, "rb") as f:
        engine = trt.Runtime(trt.Logger(
            trt.Logger.ERROR)).deserialize_cuda_engine(f.read())
    weights = {
        binding["engine_name"]
        for binding in result.checkpoint_weight_bindings
    }
    direct = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if name not in weights:
            direct[name] = (engine.get_tensor_mode(name).name,
                            engine.get_tensor_dtype(name).name,
                            tuple(engine.get_tensor_shape(name)))
    cfg = load_device_config(args)
    runtime = build_runtime_config(cfg, args)
    writer = registry.artifact_writer_for(cfg.root_model_type,
                                          args.spec_type,
                                          args.resolved_spec_role,
                                          args.dflash_version,
                                          eagle4_draft=args.eagle4_draft)
    writer.runtime_config.update_llm_config(runtime, {}, cfg, args)
    bound = {
        key
        for binding in result.checkpoint_weight_bindings
        for key in binding["checkpoint_keys"]
    }
    # The draft reads the base embedding.
    unbound = {"embed_tokens.weight"}

    graph = onnx.load(os.path.join(exported, "model.onnx"),
                      load_external_data=False).graph
    exported_plugins = {node.op_type: node for node in graph.node}
    exported_io = _export_io(os.path.join(exported, "model.onnx"))
    assert direct.keys() == exported_io.keys()
    for name, (mode, dtype, dims) in exported_io.items():
        engine_mode, engine_dtype, engine_dims = direct[name]
        assert (engine_mode, engine_dtype,
                len(engine_dims)) == (mode, dtype, len(dims)), name
        # Profiles pin some symbolic dims to one value in the engine.
        assert all(dim in (-1, engine_dim)
                   for dim, engine_dim in zip(dims, engine_dims)), name
    attention = exported_plugins["AttentionPlugin"]
    for field in ("num_q_heads", "num_kv_heads", "head_size",
                  "sliding_window_size", "enable_tree_attention"):
        assert plugin_fields["attention"][field] == _attribute(
            attention, field), field
    assert plugin_fields["attention"]["attention_scale"] == pytest.approx(
        _attribute(attention, "attention_scale"))
    assert plugin_fields["low_rank_candidate_head"] == {
        "num_candidates":
        _attribute(exported_plugins["LowRankCandidateHeadPlugin"],
                   "num_candidates")
    }
    assert runtime["eagle3_config"] == exported_eagle3_config
    assert bound == set(
        load_file(os.path.join(checkpoint, "model.safetensors"))) - unbound
    # norm_output: the final-norm state feeds both the head and the next step.
    assert graph_tensors["outputs"]["hidden_states"] is graph_tensors[
        "head_input"]


def _fp8_checkpoint(directory, lm_head: bool, kv_cache: bool) -> str:
    """The drafter with FP8 decoder-layer linears, optionally an FP8 lm_head and an FP8 KV cache."""
    _write_drafter(directory)
    path = os.path.join(str(directory), "model.safetensors")
    tensors = load_file(path)
    modules = [f"layers.0.{name}" for name in _LAYER_LINEARS]
    for module in modules + (["lm_head"] if lm_head else []):
        weight = tensors[f"{module}.weight"].float()
        scale = weight.abs().max() / 448.0
        tensors[f"{module}.weight"] = (weight / scale).to(torch.float8_e4m3fn)
        tensors[f"{module}.weight_scale"] = scale.reshape(())
        tensors[f"{module}.input_scale"] = torch.tensor(1.0)
    quantization = {
        "quant_algo": "FP8",
        "exclude_modules": ["fc"] if lm_head else ["fc", "lm_head"]
    }
    if kv_cache:
        quantization["kv_cache_quant_algo"] = "FP8"
        for name in ("k", "v"):
            tensors[f"layers.0.self_attn.{name}_proj.{name}_scale"] = (
                torch.tensor(1.0))
    save_file(tensors, path)
    with open(os.path.join(str(directory), "hf_quant_config.json"), "w") as f:
        json.dump({"quantization": quantization}, f)
    return str(directory)


@pytest.mark.parametrize(("lm_head", "kv_cache", "option"),
                         [(True, False, "lm_head"),
                          (False, True, "--kv-cache-quantization")],
                         ids=["quantized-lm-head", "fp8-kv-cache"])
def test_candidate_draft_rejects_quantization_its_graph_cannot_run(
        tmp_path, lm_head, kv_cache, option):
    """A candidate-head draft checkpoint that quantizes lm_head or the KV cache fails at export, naming what to drop: the head gathers fp16 lm_head rows and the graph writes an fp16 KV cache."""
    checkpoint = _fp8_checkpoint(tmp_path, lm_head, kv_cache)

    with pytest.raises(ValueError, match=option):
        model_module.AutoModel.from_pretrained(checkpoint)


class _FakeBase(nn.Module):
    """An embedding and seeded hidden states shaped like an HF base output."""

    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(_VOCAB, _HIDDEN)
        self.generator = torch.Generator().manual_seed(5)

    def forward(self, input_ids, output_hidden_states):
        shape = (*input_ids.shape, _HIDDEN)
        return {
            "hidden_states":
            tuple(
                torch.randn(shape, generator=self.generator).to(
                    input_ids.device, torch.float16)
                for _ in range(_BASE_LAYERS + 1))
        }


class _FakeTokenizer:
    pad_token = eos_token = "<pad>"

    def __call__(self, texts, **kwargs):
        return {"input_ids": torch.randint(0, _VOCAB - 1, (len(texts), 12))}


def test_eagle4_quantize_draft_keeps_fc_and_heads_unquantized(
        checkpoint, tmp_path, monkeypatch):
    """quantize draft quantizes the decoder-layer linears and keeps fc (recorded in exclude_modules), lm_head and the candidate head factors at their checkpoint values."""
    if not torch.cuda.is_available():
        pytest.skip("modelopt calibration of the fp16 draft needs CUDA")
    pytest.importorskip("modelopt.torch.quantization")
    from tensorrt_edgellm.quantization.models import eagle4_draft
    base_dir = _write_config(
        tmp_path / "base", {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": _HIDDEN,
            "intermediate_size": 2 * _HIDDEN,
            "num_hidden_layers": _BASE_LAYERS,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": _HIDDEN // 2,
            "vocab_size": _VOCAB,
        })
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: _FakeTokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda *args, **kwargs: _FakeBase())
    out_dir = str(tmp_path / "quantized")

    eagle4_draft.quantize_and_export_eagle4_draft(
        base_dir,
        checkpoint,
        out_dir,
        text_dataset=lambda: iter("abcd"),
        num_samples=4,
        quantization="nvfp4",
        device="cuda")

    output = load_file(os.path.join(out_dir, "model.safetensors"))
    source = load_file(os.path.join(checkpoint, "model.safetensors"))
    with open(os.path.join(out_dir, "hf_quant_config.json")) as f:
        quantization = json.load(f)["quantization"]
    assert quantization["quant_algo"] == "NVFP4"
    assert "fc" in quantization["exclude_modules"]
    for name in _LAYER_LINEARS:
        assert output[f"layers.0.{name}.weight"].dtype == torch.uint8, name
    for name in ("fc.weight", "lm_head.weight"):
        assert torch.equal(output[name], source[name].to(torch.float16)), name
    for name in ("efficient_lm_head.weight",
                 "efficient_lm_head_down_proj.weight"):
        assert torch.equal(output[name], source[name]), name
