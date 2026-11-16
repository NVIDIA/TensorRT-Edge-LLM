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
"""Direct action graphs preserve the two-pool runtime and precision contract."""

import ctypes
import json
import os

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from tensorrt_edgellm.builder import cli, ops
from tensorrt_edgellm.builder.core import (builder, bundle, contracts,
                                           numpy_dtypes)
from tensorrt_edgellm.builder.core import weights as checkpoint_weights
from tensorrt_edgellm.builder.models.alpamayo import \
    modeling_alpamayo_action as dense
from tensorrt_edgellm.builder.models.alpamayo import \
    modeling_alpamayo_action_paged as paged
from tensorrt_edgellm.builder.models.alpamayo import runtime_config
from tensorrt_edgellm.builder.ops import backend, ragged, scope


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize(
    "reuse_option", [None, True, False],
    ids=["reuse-default", "reuse-explicit", "dense-explicit"])
def test_direct_action_selects_contract_and_prefix_scales(
        tmp_path, fp8, reuse_option):
    """Reuse defaults on; explicit opt-out selects dense action without changing action TP."""
    safetensors = pytest.importorskip("safetensors.numpy")
    root = {
        "expert_cfg": {
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 128,
            "hidden_size": 128,
            "intermediate_size": 128
        }
    }
    values = {
        "expert.layers.0.self_attn.k_proj.weight": np.zeros((128, 128),
                                                            np.float16),
        "expert.layers.0.input_layernorm.weight": np.ones(128, np.float16)
    }
    if fp8:
        (tmp_path / "hf_quant_config.json").write_text(
            json.dumps({"quantization": {
                "kv_cache_quant_algo": "FP8"
            }}))
        values.update({
            "model.layers.0.self_attn.k_proj.k_scale":
            np.array(0.25, np.float32),
            "model.layers.0.self_attn.v_proj.v_scale":
            np.array(0.5, np.float32)
        })
    safetensors.save_file(values, str(tmp_path / "model.safetensors"))
    checkpoint = bundle.BundleConfig(str(tmp_path), root, "alpamayo_r1")
    command = [
        "--model-dir",
        str(tmp_path), "--engine-dir",
        str(tmp_path / "engine"), "--components", "action"
    ]
    reuse = reuse_option is not False
    if reuse_option is not None:
        command.append(f"--reuse-llm-kv-cache={str(reuse_option).lower()}")
    args = cli._build_args(cli._parser().parse_args(command), "action")
    assert args.reuse_llm_kv_cache is reuse
    assert args.tp_size == 1
    if reuse_option is None:
        assert builder.BuildArgs("", "").reuse_llm_kv_cache is True
    weights = checkpoint_weights.Weights(str(tmp_path), component="action")
    try:
        ctx = ops.BuildContext(None, None, weights, ops.BuildOptions(),
                               checkpoint, args)
        model = dense.AlpamayoActionModel.from_config(ctx)
        assert type(model) is (paged.PagedAlpamayoActionModel
                               if reuse else dense.AlpamayoActionModel)
        exported = runtime_config.component_runtime_config(
            checkpoint, contracts.Component.ACTION, args)
        expected = [[0.25, 0.5]] if fp8 and reuse else None
        assert exported.get("kv_cache_scales") == expected
        if reuse:
            assert model.kv_scales == expected
            assert model.layers[0].kv_scales == (expected[0]
                                                 if expected else None)
    finally:
        weights.close()


def _cuda(result):
    assert int(result[0]) == 0, result[0]
    return result[1] if len(result) == 2 else result[1:]


@pytest.fixture(scope="module", params=["fp16", "fp8", "int8"])
def direct_attention_engine(request):
    runtime = pytest.importorskip("cuda.bindings.runtime")
    status, count = runtime.cudaGetDeviceCount()
    if int(status) or count == 0:
        pytest.skip("Direct TensorRT execution requires a CUDA device")
    plugin_path = os.environ.get("EDGELLM_PLUGIN_PATH",
                                 "build/lib/libNvInfer_edgellm_plugin.so")
    if not os.path.isfile(plugin_path):
        pytest.skip("Edge-LLM plugin library is not available")
    handle = ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
    precision = request.param
    properties = _cuda(runtime.cudaGetDeviceProperties(0))
    if precision == "fp8" and (properties.major, properties.minor) in ((8, 0),
                                                                       (8, 6),
                                                                       (8, 7)):
        pytest.skip("Ampere uses INT8 bindings for FP8 prefix bytes")
    logger = trt.Logger(trt.Logger.WARNING)
    trt_builder = trt.Builder(logger)
    network = trt_builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    net = backend.Net(trt_builder,
                      network,
                      fp8_kv_cache_as_int8=precision == "int8")
    args = builder.BuildArgs("",
                             "",
                             component="action",
                             reuse_llm_kv_cache=True,
                             max_batch_size=2,
                             max_kv_cache_capacity=256)
    ctx = ops.BuildContext(net, None, None, ops.BuildOptions(), None, args)
    model = paged.PagedAlpamayoActionModel.__new__(
        paged.PagedAlpamayoActionModel)
    ops.NetworkModule.__init__(model, ctx)
    model.num_layers = model.num_kv_heads = 1
    model.head_dim = 128
    model.diffusion_tokens = 64
    model.kv_scales = None if precision == "fp16" else [[0.25, 0.5]]
    layer = paged.PagedAlpamayoActionDecoderLayer(ctx, "expert.layers.0", 1, 1,
                                                  128, 1e-6)
    layer.kv_scales = model.kv_scales[0] if model.kv_scales else None
    with scope.build_scope(ctx):
        inputs = model.input_tensors()
        query, key, value = [
            model.add_input(name, trt.float16, (-1, 64, 1, 128))
            for name in ("query", "key", "value")
        ]
        metadata = ragged.RaggedDecoderInputs(
            None, inputs["query_start_offsets"], inputs["query_lengths"],
            inputs["past_lengths"], inputs["attention_sequence_lengths"], None,
            None, inputs["execution_phase_marker"],
            inputs["context_sequence_count_carrier"],
            inputs["action_page_table"])
        output, present = layer.attention(query, key, value,
                                          inputs["prefix_pools"][0],
                                          inputs["action_pools"][0],
                                          inputs["rope"].reshape((-1, 128)),
                                          metadata, inputs["kv_page_table"],
                                          inputs["context_mask_selector"])
        model.mark_outputs({
            "attention_output": output,
            "present_key_values_0": present
        })
    config = trt_builder.create_builder_config()
    config.set_preview_feature(trt.PreviewFeature.ALIASED_PLUGIN_IO_10_03,
                               True)
    builder._setup_component_profile(trt_builder, config, network, args)
    plan = trt_builder.build_serialized_network(network, config)
    assert plan is not None
    trt_runtime = trt.Runtime(logger)
    engine = trt_runtime.deserialize_cuda_engine(plan)
    assert engine is not None
    assert engine.get_tensor_dtype("past_key_values_0") == {
        "fp16": trt.float16,
        "fp8": trt.DataType.FP8,
        "int8": trt.int8
    }[precision]
    assert engine.get_tensor_profile_shape("past_key_values_0",
                                           0) == [(2, 4, 128, 1, 128)] * 3
    assert engine.get_tensor_profile_shape("action_key_values_0",
                                           0) == [(2, 2, 128, 1, 128)] * 3
    yield engine, precision
    del engine, trt_runtime, handle


@pytest.mark.parametrize("batch_size", [1, 2])
def test_direct_paged_attention_reads_prefix_and_writes_action(
        direct_attention_engine, batch_size):
    """Direct-built attention reads scaled prefix pages, preserves them, and writes private action KV."""
    from cuda.bindings import runtime

    engine, precision = direct_attention_engine
    rng = np.random.default_rng(123)
    lengths = np.array([127, 129], np.int32)[:batch_size]
    prefix_map = np.array([[[2, 0], [6, 4]], [[3, 1], [7, 5]]],
                          np.int32)[:batch_size]
    action_map = np.array([[[1], [3]], [[0], [2]]], np.int32)[:batch_size]
    prefix = rng.normal(0, 0.2, (2, 4, 128, 1, 128)).astype(np.float16)
    if precision != "fp16":
        prefix = numpy_dtypes.f32_to_fp8_e4m3_bytes(prefix.astype(np.float32))
        decoded = numpy_dtypes.fp8_e4m3_bytes_to_f32(prefix)
        decoded[0] *= 0.25
        decoded[1] *= 0.5
        decoded = decoded.astype(np.float16)
    else:
        decoded = prefix
    q, k, v = [
        rng.normal(0, 0.2, (batch_size, 64, 1, 128)).astype(np.float16)
        for _ in range(3)
    ]
    angles = rng.uniform(-1, 1, (batch_size, 64, 64)).astype(np.float32)
    cos, sin = np.cos(angles), np.sin(angles)
    rope = np.concatenate((cos, sin), axis=-1)

    def rotate(x):
        a, b = x[..., :64].astype(np.float32), x[..., 64:].astype(np.float32)
        return np.concatenate((a * cos[:, :, None] - b * sin[:, :, None],
                               b * cos[:, :, None] + a * sin[:, :, None]),
                              axis=-1).astype(np.float16)

    qr, kr = rotate(q), rotate(k)
    expected = []
    for row, length in enumerate(lengths):
        keys = decoded[0, prefix_map[row, 0]].reshape(-1, 128)[:length]
        values = decoded[1, prefix_map[row, 1] - 4].reshape(-1, 128)[:length]
        keys = np.concatenate((keys, kr[row, :, 0])).astype(np.float32)
        values = np.concatenate((values, v[row, :, 0])).astype(np.float32)
        scores = qr[row, :, 0].astype(np.float32) @ keys.T / np.sqrt(128)
        probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
        expected.append(probs / probs.sum(axis=-1, keepdims=True) @ values)
    feeds = {
        "query": q,
        "key": k,
        "value": v,
        "rope_rotary_cos_sin": rope,
        "past_key_values_0": prefix,
        "action_key_values_0": np.zeros((2, 2, 128, 1, 128), np.float16),
        "kv_page_table": prefix_map,
        "action_page_table": action_map,
        "past_lengths": lengths,
        "query_lengths": np.full(batch_size, 64, np.int32),
        "attention_sequence_lengths": lengths + 64,
        "query_start_offsets": np.arange(batch_size + 1, dtype=np.int32) * 64,
        "context_mask_selector": np.zeros(batch_size, np.int32),
        "execution_phase_marker": np.zeros(6, np.int32),
        "context_sequence_count_carrier": np.zeros(0, np.int32),
        "noise_trajectory": np.zeros((batch_size, 64, 2), np.float32),
        "time_steps_t0": np.zeros(1, np.float32),
        "time_steps_t1": np.ones(1, np.float32)
    }
    context = engine.create_execution_context()
    stream = _cuda(runtime.cudaStreamCreate())
    device = {}
    output = np.empty((batch_size, 64, 128), np.float16)
    prefix_after = np.empty_like(prefix)
    action_after = np.empty_like(feeds["action_key_values_0"])
    try:
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                data = feeds[name]
                context.set_input_shape(name, data.shape)
                device[name] = _cuda(runtime.cudaMalloc(max(data.nbytes, 1)))
                if data.nbytes:
                    _cuda(
                        runtime.cudaMemcpyAsync(
                            device[name], data.ctypes.data, data.nbytes,
                            runtime.cudaMemcpyKind.cudaMemcpyHostToDevice,
                            stream))
        device["attention_output"] = _cuda(runtime.cudaMalloc(output.nbytes))
        for name, ptr in device.items():
            context.set_tensor_address(name, ptr)
        context.set_tensor_address("present_key_values_0",
                                   device["action_key_values_0"])
        assert context.execute_async_v3(int(stream))
        for host, name in [(output, "attention_output"),
                           (prefix_after, "past_key_values_0"),
                           (action_after, "action_key_values_0")]:
            _cuda(
                runtime.cudaMemcpyAsync(
                    host.ctypes.data, device[name], host.nbytes,
                    runtime.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
        _cuda(runtime.cudaStreamSynchronize(stream))
        np.testing.assert_allclose(output,
                                   np.stack(expected),
                                   rtol=0.02,
                                   atol=0.005)
        np.testing.assert_array_equal(prefix_after, prefix)
        for row in range(batch_size):
            np.testing.assert_allclose(action_after[0, action_map[row, 0,
                                                                  0], :64],
                                       kr[row],
                                       rtol=0,
                                       atol=0.001)
            np.testing.assert_array_equal(
                action_after[1, action_map[row, 1, 0] - 2, :64], v[row])
    finally:
        for ptr in device.values():
            _cuda(runtime.cudaFree(ptr))
        _cuda(runtime.cudaStreamDestroy(stream))
