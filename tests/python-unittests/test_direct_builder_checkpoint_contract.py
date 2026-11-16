# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

safetensors_numpy = pytest.importorskip("safetensors.numpy")

from tensorrt_edgellm.builder.core import (contracts, quantization,
                                           weight_policy)
from tensorrt_edgellm.builder.core.artifacts.embeddings import \
    externalizes_embedding
from tensorrt_edgellm.builder.core.artifacts.external_weights import (
    _share_tied_embedding, checkpoint_identity, patch_external_weight_config)
from tensorrt_edgellm.builder.core.safetensors_np import (SafetensorsStore,
                                                          _ShardFile)
from tensorrt_edgellm.builder.core.weights import (LinearWeights,
                                                   ParameterSpec, Weights,
                                                   component_quantization)
from tensorrt_edgellm.builder.models.qwen3 import weights as qwen3_weights
from tensorrt_edgellm.builder.models.qwen3_omni import weights as omni_weights
from tensorrt_edgellm.builder.ops import linear as linear_ops
from tensorrt_edgellm.builder.ops.linear import Linear
from tensorrt_edgellm.builder.weight_packing import nvfp4 as nvfp4_pack
from tensorrt_edgellm.models.sm12x_nvfp4 import \
    supports_sm12x_nvfp4_gemm_allreduce


def _checkpoint(path, values):
    path.mkdir()
    safetensors_numpy.save_file(
        {"model.embed_tokens.weight": np.asarray(values, dtype=np.float16)},
        str(path / "model.safetensors"),
    )


def _binding(engine_name, key="model.embed_tokens.weight", **extra):
    return {
        "engine_name": engine_name,
        "checkpoint_keys": [key],
        "source_layout": "fp16",
        "dtype": "F16",
        "shape": [8, 4],
        **extra,
    }


def test_default_policy_externalizes_supported_weights():
    policy = weight_policy.WeightPolicy.from_request(None)

    assert policy.wants(weight_policy.EXTERNAL_WEIGHT_NVFP4_MOE)
    assert policy.wants(weight_policy.EXTERNAL_WEIGHT_INT4_FFN)
    assert policy.externalizes_embedding
    assert policy.externalizes_fp16("lm_head")
    assert policy.externalizes_fp16("model.layers.0.self_attn.q_proj")

    with_fp16 = weight_policy.WeightPolicy.from_request(
        (weight_policy.EXTERNAL_WEIGHT_FP16, ))
    assert not with_fp16.wants(weight_policy.EXTERNAL_WEIGHT_NVFP4_MOE)
    assert with_fp16.externalizes_fp16("model.layers.0.self_attn.q_proj")


def test_checkpoint_identity_rejects_same_shape_different_content(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _checkpoint(first, np.arange(32).reshape(8, 4))
    _checkpoint(second, np.arange(32, 64).reshape(8, 4))
    bindings = [_binding("lm_head.weight")]

    first_identity = checkpoint_identity(bindings, str(first))
    second_identity = checkpoint_identity(bindings, str(second))
    first_tensor = first_identity["sources"]["component"]["tensors"][
        "model.embed_tokens.weight"]
    second_tensor = second_identity["sources"]["component"]["tensors"][
        "model.embed_tokens.weight"]

    assert first_tensor["dtype"] == second_tensor["dtype"]
    assert first_tensor["shape"] == second_tensor["shape"]
    assert first_tensor["bytes"] == second_tensor["bytes"]
    assert first_tensor["samples"] != second_tensor["samples"]
    assert set(first_identity["sources"]["component"]["files"]) == {
        "model.safetensors"
    }


def test_checkpoint_identity_reads_bounded_payload(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    _checkpoint(checkpoint, np.zeros((1 << 18, 4), dtype=np.float16))
    sampled_bytes = 0
    original = _ShardFile.raw_slice

    def counted(self, name, offset, length):
        nonlocal sampled_bytes
        sampled_bytes += length
        return original(self, name, offset, length)

    monkeypatch.setattr(_ShardFile, "raw_slice", counted)
    with SafetensorsStore(str(checkpoint)) as store:
        identity = store.tensor_identity("model.embed_tokens.weight")

    assert len(identity["samples"]) == 3
    assert sampled_bytes == 48


def test_checkpoint_identity_records_shard_index(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    key = "model.embed_tokens.weight"
    shard_name = "model-00001-of-00001.safetensors"
    safetensors_numpy.save_file(
        {key: np.arange(32, dtype=np.float16).reshape(8, 4)},
        str(checkpoint / shard_name),
    )
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {
            key: shard_name
        }}))

    identity = checkpoint_identity([_binding("lm_head.weight")],
                                   str(checkpoint))

    assert set(identity["sources"]["component"]["files"]) == {
        "model.safetensors.index.json",
        shard_name,
    }


def test_checkpoint_identity_sampling_cost_is_model_bounded(
        tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    tensors = {
        f"model.layers.{index}.weight": np.full((32, 32),
                                                index,
                                                dtype=np.float16)
        for index in range(64)
    }
    safetensors_numpy.save_file(tensors, str(checkpoint / "model.safetensors"))
    bindings = [
        _binding(f"layer_{index}.weight", key=key)
        for index, key in enumerate(tensors)
    ]
    sampled_bytes = 0
    original = _ShardFile.raw_slice

    def counted(self, name, offset, length):
        nonlocal sampled_bytes
        sampled_bytes += length
        return original(self, name, offset, length)

    monkeypatch.setattr(_ShardFile, "raw_slice", counted)
    identity = checkpoint_identity(bindings, str(checkpoint))
    recorded = identity["sources"]["component"]["tensors"]

    assert len(recorded) == len(tensors)
    assert sum(bool(tensor["samples"]) for tensor in recorded.values()) == 16
    assert sampled_bytes == 16 * 3 * 16


@pytest.mark.parametrize(("tied", "aliases"), ((True, True), (False, False)))
def test_embedding_alias_requires_model_tying(tied, aliases):
    bindings = [
        _binding("lm_head.weight"),
        _binding("__embedding__", role="embedding", embedding_scale=1.0),
    ]

    _share_tied_embedding(bindings, tied)

    assert (bindings[1].get("storage_alias_of") == "lm_head.weight") is aliases


@pytest.mark.parametrize("refit", [False, True])
def test_embedding_alias_respects_refit_lifetime(refit):
    """Persistent embeddings never alias transient refit storage."""
    # Given tied embedding and head, when assigning aliases, then transient refit storage is never shared.
    head = _binding("lm_head.weight",
                    **({
                        "destination": "refit"
                    } if refit else {}))
    embedding = _binding("__embedding__", role="embedding")

    _share_tied_embedding([head, embedding], True)

    assert embedding.get("storage_alias_of") == (None if refit else
                                                 head["engine_name"])


@pytest.mark.parametrize(
    ("candidate", "embedding"),
    [
        (
            _binding("lm_head.weight"),
            _binding("__embedding__", role="embedding", embedding_scale=2.0),
        ),
        (
            _binding("lm_head.weight", embedding_scale=2.0),
            _binding("__embedding__", role="embedding", embedding_scale=1.0),
        ),
        (
            _binding("lm_head.weight"),
            _binding("__embedding__",
                     key="other.embed_tokens.weight",
                     role="embedding",
                     embedding_scale=1.0),
        ),
    ],
)
def test_embedding_with_different_transform_does_not_alias(
        candidate, embedding):
    bindings = [candidate, embedding]

    _share_tied_embedding(bindings, True)

    assert "storage_alias_of" not in bindings[1]


@pytest.mark.parametrize(
    ("tied", "expected"),
    ((True, "model.embed_tokens.weight"), (False, "lm_head.weight")),
)
def test_lm_head_checkpoint_resolution_follows_model_tying(
        tmp_path, tied, expected):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    safetensors_numpy.save_file(
        {
            "model.embed_tokens.weight": np.zeros((8, 4), dtype=np.float16),
            "lm_head.weight": np.ones((8, 4), dtype=np.float16),
        },
        str(checkpoint / "model.safetensors"),
    )

    def resolve_candidates(name, **_):
        return (("model.embed_tokens.weight", )
                if name == "lm_head.weight" else ())

    weights = Weights(
        str(checkpoint),
        conversion=SimpleNamespace(resolve_candidates=resolve_candidates),
        tie_word_embeddings=tied)
    try:
        assert weights.checkpoint_key("lm_head.weight") == expected
    finally:
        weights.close()


def test_runtime_config_records_checkpoint_identity(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")
    identity = {
        "version": 1,
        "sources": {
            "component": {
                "build_source": "/checkpoint",
                "tensors": {},
            },
        },
    }

    patch_external_weight_config(str(config_path), [_binding("weight")],
                                 identity)

    config = json.loads(config_path.read_text())
    assert config["checkpoint_identity"] == identity


def test_runtime_config_requires_checkpoint_identity(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}")

    with pytest.raises(ValueError, match="requires a checkpoint identity"):
        patch_external_weight_config(str(config_path), [_binding("weight")],
                                     {})


def test_runtime_config_publishes_marker_and_clears_it_on_ordinary_rebuild(
        tmp_path):
    """Runtime artifacts track the selected transport without stale markers."""
    # Given an artifact config, when switching weight transport, then marker and binding destination track the build.
    path = tmp_path / "config.json"
    path.write_text("{}")
    identity = {"version": 1, "sources": {}}
    bindings = [_binding("w", destination="refit")]

    patch_external_weight_config(str(path),
                                 bindings,
                                 identity,
                                 "/checkpoint",
                                 weightless_engine=True)

    config = json.loads(path.read_text())
    assert config["weightless_engine"] is True
    assert config["checkpoint_weight_bindings"] == bindings
    assert config["checkpoint_identity"] == identity
    assert config["checkpoint_dir"] == "/checkpoint"

    patch_external_weight_config(str(path), [_binding("w")], identity)

    config = json.loads(path.read_text())
    assert "weightless_engine" not in config
    assert "destination" not in config["checkpoint_weight_bindings"][0]


def test_unknown_quantization_algorithm_is_not_fp16():
    with pytest.raises(ValueError, match="unsupported.*algorithm"):
        quantization.algorithm_to_type("UNSUPPORTED_QUANT")


@pytest.mark.parametrize("algorithm", ("W4A8_MXFP4_FP8", "W4A8_MXFP4_MXFP8"))
def test_mxfp4_checkpoint_is_not_misclassified_as_fp8(algorithm):
    with pytest.raises(ValueError, match="MXFP4 weight layouts"):
        quantization.algorithm_to_type(algorithm)


def test_nvfp4_weight_layout_takes_precedence_over_activation_precision():
    assert (quantization.algorithm_to_type("W4A8_NVFP4_FP8") ==
            quantization.QUANT_NVFP4)


def test_unimplementable_mtp_config_is_rejected_rather_than_ignored():
    """MTP config fields no implementation reads stop a Qwen4-Exp draft build."""
    # Given MTP config fields that no implementation reads -- a predictor rope
    # that disagrees with the decoder's, a named source layer, a dedicated
    # embedding table -- when a draft build configures itself, then each one
    # stops the build. Ignoring them is what makes them dangerous: the released
    # checkpoint holds them all at their off value, so a future one that sets
    # any of them would silently get the behavior it did not ask for.
    from tensorrt_edgellm.builder.models.qwen4_exp import configuration

    def config(section=None, **text):
        mtp = {"num_hidden_layers": 1, "layer_types": ["full_attention"]}
        mtp.update(section or {})
        return SimpleNamespace(
            raw_component={"text_config": {
                "mtp": mtp,
                **text
            }},
            rope_theta=1e7,
            mtp_num_hidden_layers=1,
        )

    with pytest.raises(ValueError, match="rope_theta"):
        configuration.configure_draft(config({"rope_theta": 5e5}))
    with pytest.raises(ValueError, match="mtp_use_hidden_state_from_layer"):
        configuration.configure_draft(
            config({"mtp_use_hidden_state_from_layer": 44}))
    with pytest.raises(ValueError, match="mtp_use_dedicated_embeddings"):
        configuration.configure_draft(
            config(mtp_use_dedicated_embeddings=True))

    # The released values are all off, and configure_draft accepts them.
    accepted = config(
        {
            "rope_theta": 1e7,
            "mtp_use_hidden_state_from_layer": None
        },
        mtp_use_dedicated_embeddings=False)
    configuration.configure_draft(accepted)
    assert accepted.num_hidden_layers == 1


def test_expert_projection_inherits_the_bank_group_size():
    """An expert projection takes the group size its bank is listed with."""
    # Given a mixed-precision checkpoint whose quantized_layers names expert banks
    # rather than the projections under them, when a projection asks for its group
    # size, then it inherits the bank's -- falling through to the dominant group
    # size would decode a group-128 FP8 bank as if it were group 16.
    quant = quantization.QuantConfig(
        quant_type=quantization.QUANT_NVFP4,
        group_size=16,
        layer_overrides={
            "mtp.layers.0.mlp.experts": quantization.QUANT_FP8_BLOCK
        },
        layer_group_sizes={"mtp.layers.0.mlp.experts": 128},
        is_mixed_precision=True,
    )

    assert quant.module_group_size("mtp.layers.0.mlp.experts.7.up_proj") == 128
    assert quant.module_group_size("mtp.layers.0.self_attn.q_proj") == 16


@pytest.mark.parametrize("with_sidecar", [False, True])
def test_direct_builder_selects_nvfp4_a16_for_fp16_activations(
        tmp_path, with_sidecar):
    """Calibrated FP4 and plain FP16 payloads select executable frontends."""
    import torch
    from safetensors import torch as safetensors_torch

    from tensorrt_edgellm import config
    from tensorrt_edgellm.models import linear

    calibrated = "model.layers.0.mlp.up_proj"
    weight_only = "model.layers.0.mlp.down_proj"
    plain = "model.layers.1.mlp.down_proj"
    embedded = {
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "config_groups": {
                "nvfp4": {
                    "targets": [calibrated, weight_only, plain],
                    "weights": {
                        "type": "float",
                        "num_bits": 4,
                        "group_size": 16
                    },
                    "input_activations": {
                        "type": "float",
                        "num_bits": 4
                    },
                },
            },
        },
    }
    tensors = {"lm_head.weight": torch.ones((64, 64), dtype=torch.float16)}
    plain_weight = (torch.arange(4096).reshape(64, 64).float() / 4096).half()
    tensors[plain + ".weight"] = plain_weight
    for prefix in (calibrated, weight_only):
        tensors.update({
            prefix + ".weight":
            torch.full((64, 32), 0x22, dtype=torch.uint8),
            prefix + ".weight_scale":
            torch.ones((64, 4)).to(torch.float8_e4m3fn),
            prefix + ".weight_scale_2":
            torch.tensor(0.5),
        })
    tensors[calibrated + ".input_scale"] = torch.tensor(2.)
    safetensors_torch.save_file(tensors, str(tmp_path / "model.safetensors"))
    if with_sidecar:
        (tmp_path / "hf_quant_config.json").write_text(
            json.dumps({
                "quantization": {
                    "quant_algo": "MIXED_PRECISION",
                    "quantized_layers": {
                        **{
                            name: {
                                "quant_algo": "W4A16_NVFP4",
                                "group_size": 16
                            }
                            for name in (calibrated, weight_only)
                        },
                        plain: {
                            "quant_algo": "FP16"
                        },
                    },
                    "exclude_modules": ["lm_head"],
                },
            }))

    try:
        for enabled in (True, False):
            config.set_default_quantize_activations(enabled)
            onnx_quant = config._parse_quant(str(tmp_path), embedded)
            onnx_model = SimpleNamespace(quant=onnx_quant,
                                         tie_word_embeddings=False,
                                         tp_size=1,
                                         mapping=config.Mapping())
            direct_quant = quantization.parse_quantization(
                str(tmp_path), embedded, {}, quantize_activations=enabled)
            weights = Weights(str(tmp_path), quant=direct_quant)
            try:
                descriptor = weights.linear_descriptor(
                    plain, weights.module_quant_type(plain))
                np.testing.assert_array_equal(descriptor.weight,
                                              plain_weight.numpy())
                module = linear.make_linear(
                    onnx_model, 64, 64, module_name="layers.1.mlp.down_proj")
                module.load_state_dict(
                    {"weight": torch.from_numpy(descriptor.weight.copy())})
                inputs = (torch.arange(128).reshape(2, 64).float() / 64).half()
                torch.testing.assert_close(module(inputs),
                                           torch.nn.functional.linear(
                                               inputs, plain_weight),
                                           rtol=0,
                                           atol=0)
                expected = quantization.QUANT_NVFP4 if enabled else quantization.QUANT_NVFP4_A16
                assert weights.module_quant_type(calibrated) == expected
                assert config.module_quant_type("layers.0.mlp.up_proj",
                                                onnx_model) == expected
                assert weights.module_quant_type(
                    weight_only) == quantization.QUANT_NVFP4_A16
                assert config.module_quant_type(
                    "layers.0.mlp.down_proj",
                    onnx_model) == config.QUANT_NVFP4_A16
                assert weights.module_quant_type(
                    "lm_head") == quantization.QUANT_FP16
                assert config.module_quant_type(
                    "lm_head", onnx_model) == config.QUANT_FP16
            finally:
                weights.store.close()
    finally:
        config.set_default_quantize_activations(True)


def test_direct_builder_packs_gated_nvfp4_a16_experts():
    hidden = 128
    intermediate = 128

    def raw(out_features, in_features, alpha=1.0):
        return {
            "packed": np.zeros((out_features, in_features // 2), np.uint8),
            "sf": np.ones((out_features, in_features // 16), np.uint8),
            "alpha": alpha,
        }

    def load_expert(_):
        return {
            "gate": raw(intermediate, hidden),
            "up": raw(intermediate, hidden),
            "down": raw(hidden, intermediate),
        }

    packed = nvfp4_pack.pack_nvfp4_a16_gated_experts(load_expert, 2, hidden,
                                                     intermediate)

    assert packed["fc1_qweights"].shape == (2, hidden // 16,
                                            8 * 2 * intermediate)
    assert packed["fc1_block_scales"].shape == (2, hidden // 16,
                                                2 * intermediate)
    assert packed["fc2_qweights"].shape == (2, intermediate // 16, 8 * hidden)
    assert packed["fc2_block_scales"].shape == (2, intermediate // 16, hidden)


def test_direct_builder_binds_calibrated_moe_input_scales():
    weights = object.__new__(Weights)
    weights.nvfp4_checkpoint_names = lambda prefix: (
        prefix + ".weight", prefix + ".weight_scale", prefix +
        ".weight_scale_2", False, prefix + ".input_scale")
    weights.checkpoint_binding = lambda keys, *layout, **extra: {
        "keys": keys,
        "layout": layout,
        "extra": extra,
    }

    binding = weights.nvfp4_moe_input_scale_binding("layers.0.mlp.experts", 2,
                                                    ("up_proj", "gate_proj"))

    assert binding["keys"] == [
        f"layers.0.mlp.experts.{expert}.{projection}.input_scale"
        for expert in range(2) for projection in ("up_proj", "gate_proj")
    ]
    assert binding["layout"] == ("plugin", "nvfp4_moe_input_scale")
    assert binding["extra"]["keys_per_expert"] == 2
    assert binding["extra"]["require_equal"] is True


def test_mixed_precision_preserves_module_group_sizes():
    dominant, group_size, overrides, group_sizes = (
        quantization._parse_mixed_precision(
            {
                "model.layers.0.self_attn.o_proj": {
                    "quant_algo": "FP8",
                    "group_size": 1
                },
                "model.layers.1.self_attn.o_proj": {
                    "quant_algo": "FP8",
                    "group_size": 1
                },
                "model.layers.0.mlp.gate_proj": {
                    "quant_algo": "NVFP4",
                    "group_size": 16
                },
            }, None))

    assert dominant == quantization.QUANT_FP8
    assert group_size == 1
    assert overrides["layers.0.mlp.gate_proj"] == quantization.QUANT_NVFP4
    assert group_sizes["layers.0.mlp.gate_proj"] == 16

    config = quantization.QuantConfig(quant_type=dominant,
                                      group_size=group_size,
                                      layer_overrides=overrides,
                                      layer_group_sizes=group_sizes,
                                      is_mixed_precision=True)
    assert config.module_group_size("model.layers.0.mlp.gate_proj") == 16
    assert config.module_group_size("model.layers.0.self_attn.o_proj") == 1
    assert config.is_quantized is True


def test_unknown_compressed_tensor_format_is_not_fp16(tmp_path):
    embedded = {
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
        },
    }

    with pytest.raises(ValueError, match="unsupported compressed-tensors"):
        quantization.parse_quantization(str(tmp_path), embedded, {})


@pytest.mark.parametrize(
    ("weight_bits", "activation_bits", "expected"),
    ((4, 4, quantization.QUANT_INT4_GPTQ),
     (8, 8, quantization.QUANT_INT8_GPTQ)),
)
def test_direct_compressed_integer_contract_lowers_to_a16(
        weight_bits, activation_bits, expected):
    quant_type, _ = quantization._compressed_group_contract(
        {
            "input_activations": {
                "type": "int",
                "num_bits": activation_bits,
                "dynamic": False,
            },
            "weights": {
                "type": "int",
                "num_bits": weight_bits,
                "group_size": 128,
            },
        }, "pack-quantized")

    assert quant_type == expected


def test_compressed_w8a8_uses_dense_int8_weights_as_w8a16(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    prefix = "model.layers.0.mlp.up_proj"
    dense = np.arange(-16, 16, dtype=np.int8).reshape(4, 8)
    scales = np.arange(1, 5, dtype=np.float32).reshape(4, 1) / 16
    safetensors_numpy.save_file(
        {
            prefix + ".weight": dense,
            prefix + ".weight_scale": scales,
        },
        str(checkpoint / "model.safetensors"),
    )
    weights = Weights(
        str(checkpoint),
        quant=quantization.QuantConfig(
            quant_type=quantization.QUANT_INT8_GPTQ,
            checkpoint_format=(
                quantization.CHECKPOINT_FORMAT_COMPRESSED_TENSORS),
        ),
    )
    try:
        descriptor = weights._linear_unreduced(prefix,
                                               quantization.QUANT_INT8_GPTQ)
    finally:
        weights.close()

    assert descriptor.quant_type == quantization.QUANT_INT8_GPTQ
    np.testing.assert_array_equal(descriptor.weight, dense)
    np.testing.assert_array_equal(descriptor.weight_scale, scales)
    assert descriptor.group_size == dense.shape[1]


def test_nvfp4_fallback_preserves_checkpoint_weights_as_w4a16():
    packed = np.arange(64, dtype=np.uint8).reshape(8, 8)
    scales = np.arange(8, dtype=np.uint8).reshape(8, 1)
    weights = object.__new__(Weights)
    weights._w4a16_fallback_types = frozenset((quantization.QUANT_NVFP4, ))
    weights._w8a16_fallback_types = frozenset()
    weights._mxfp8_conversion_types = frozenset()
    weights.opt_f16 = lambda _: None
    weights.module_quant_group_size = lambda _: 16
    weights.linear_nvfp4_raw = lambda _: {
        "packed": packed,
        "weight_scale": scales,
        "weight_scale_2": 0.25,
        "input_scale": 2.0,
        "bias": None,
    }

    descriptor = weights._linear_unreduced("model.layers.0.mlp.up_proj",
                                           quantization.QUANT_NVFP4)

    assert descriptor.quant_type == quantization.QUANT_NVFP4_A16
    assert descriptor.weight is packed
    assert descriptor.weight_scale is scales
    assert descriptor.weight_scale_2 == 0.25
    assert descriptor.input_scale is None


def test_fp8_fallback_requantizes_dense_weight_as_w8a16():
    dense = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(4, 8)
    weights = object.__new__(Weights)
    weights._w4a16_fallback_types = frozenset()
    weights._w8a16_fallback_types = frozenset((quantization.QUANT_FP8, ))
    weights._mxfp8_conversion_types = frozenset()
    weights.opt_f16 = lambda _: None
    weights._fp8_dense_f32 = lambda *_: dense

    descriptor = weights._linear_unreduced("model.layers.0.mlp.up_proj",
                                           quantization.QUANT_FP8)

    assert descriptor.quant_type == quantization.QUANT_INT8_GPTQ
    assert descriptor.weight.dtype == np.int8
    assert descriptor.weight.shape == dense.shape
    assert descriptor.weight_scale.shape == (dense.shape[0], 1)


def test_fallback_decoders_use_module_group_size():
    weights = object.__new__(Weights)
    weights.group_size = 1
    weights.module_quant_group_size = lambda _: 2
    tensors = {
        "block.weight": np.ones((4, 4), dtype=np.float32),
        "block.weight_scale": np.array([[1, 2], [3, 4]], dtype=np.float32),
        "mx.weight": np.ones((2, 4), dtype=np.float32),
        "mx.weight_scale": np.array([[127, 128], [129, 130]], dtype=np.uint8),
    }
    weights.f32 = lambda name: tensors[name].astype(np.float32)
    weights.array = tensors.__getitem__

    block = weights._fp8_dense_f32("block", quantization.QUANT_FP8_BLOCK)
    mxfp8 = weights._mxfp8_dense_f32("mx")

    np.testing.assert_array_equal(
        block,
        np.repeat(np.repeat(tensors["block.weight_scale"], 2, axis=0),
                  2,
                  axis=1),
    )
    np.testing.assert_array_equal(
        mxfp8,
        np.repeat(np.exp2(tensors["mx.weight_scale"].astype(np.int16) - 127),
                  2,
                  axis=1),
    )


def test_block_fp8_scale_inv_resolves_inside_owned_module(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    prefix = "model.layers.0.mlp.up_proj"
    safetensors_numpy.save_file(
        {
            prefix + ".weight": np.zeros((4, 4), dtype=np.float16),
            prefix + ".weight_scale_inv": np.ones((1, 1), dtype=np.float32),
        },
        str(checkpoint / "model.safetensors"),
    )
    weights = Weights(
        str(checkpoint),
        quant=quantization.QuantConfig(
            quant_type=quantization.QUANT_FP8_BLOCK,
            group_size=4,
            checkpoint_format=quantization.CHECKPOINT_FORMAT_FP8_SCALE_INV,
        ),
    )
    try:
        assert weights.checkpoint_key(prefix +
                                      ".weight_scale") == (prefix +
                                                           ".weight_scale_inv")
    finally:
        weights.close()


def test_compressed_gptq_metadata_preserves_provider_layout(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    prefix = "model.layers.0.mlp.up_proj"
    safetensors_numpy.save_file(
        {
            prefix + ".weight_packed": np.zeros((64, 16), dtype=np.int32),
            prefix + ".weight_scale": np.ones((64, 1), dtype=np.float16),
            prefix + ".weight_g_idx": np.tile(np.array([1, 0], dtype=np.int32),
                                              64),
            prefix + ".bias": np.zeros((64, ), dtype=np.float16),
        },
        str(checkpoint / "model.safetensors"),
    )
    weights = Weights(
        str(checkpoint),
        quant=quantization.QuantConfig(
            quant_type=quantization.QUANT_INT4_GPTQ,
            group_size=64,
            gptq_zero_point_offset=0,
            checkpoint_format=(
                quantization.CHECKPOINT_FORMAT_COMPRESSED_TENSORS),
        ),
    )
    try:
        descriptor = weights.linear_metadata(prefix,
                                             quantization.QUANT_INT4_GPTQ)
        _, _, _, g_idx = weights.gptq_tensors(prefix)
    finally:
        weights.close()

    assert descriptor.logical_out_features == 64
    assert descriptor.logical_in_features == 128
    assert descriptor.weight.shape == (16, 512)
    assert descriptor.weight_scale.shape == (1, 64)
    assert descriptor.bias.shape == (64, )
    assert descriptor.bias_recipe["checkpoint_keys"] == [prefix + ".bias"]
    assert descriptor.zero_correction is None
    assert descriptor.weight_recipe["checkpoint_keys"] == [
        prefix + ".weight_packed",
        prefix + ".weight_zero_point",
        prefix + ".weight_g_idx",
    ]
    assert descriptor.weight_recipe["extra"]["checkpoint_layout"] == (
        "output-major")
    assert descriptor.weight_recipe["extra"]["preserve_zero_points"] is True
    assert descriptor.activation_permutation.shape == (128, )
    np.testing.assert_array_equal(
        g_idx, np.tile(np.array([1, 0], dtype=np.int32), 64))


def test_compressed_gptq_expert_projection_uses_provider_names(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    experts = "model.layers.0.mlp.experts"
    prefix = experts + ".0.up_proj"
    packed = np.arange(64 * 16, dtype=np.int32).reshape(64, 16)
    zero_points = np.arange(64 * 2, dtype=np.int32).reshape(64, 2)
    scales = np.arange(64 * 2, dtype=np.float16).reshape(64, 2)
    group_index = np.arange(128, dtype=np.int32) // 64
    safetensors_numpy.save_file(
        {
            prefix + ".weight_packed": packed,
            prefix + ".weight_zero_point": zero_points,
            prefix + ".weight_scale": scales,
            prefix + ".weight_g_idx": group_index,
        },
        str(checkpoint / "model.safetensors"),
    )
    weights = Weights(
        str(checkpoint),
        quant=quantization.QuantConfig(
            quant_type=quantization.QUANT_INT4_GPTQ,
            group_size=64,
            gptq_zero_point_offset=0,
            checkpoint_format=(
                quantization.CHECKPOINT_FORMAT_COMPRESSED_TENSORS),
        ),
    )
    try:
        qweight, qzeros, loaded_scales = weights.gptq_expert_projection(
            experts, 0, "up_proj")
    finally:
        weights.close()

    np.testing.assert_array_equal(qweight, packed.T)
    np.testing.assert_array_equal(qzeros, zero_points.T)
    np.testing.assert_array_equal(loaded_scales, scales.T)


def test_declared_nvfp4_layer_without_payload_is_not_fp16():
    weights = object.__new__(Weights)
    weights._weightless_engine = False
    weights.is_nvfp4 = lambda prefix: False

    with pytest.raises(ValueError, match="selects NVFP4"):
        weights.linear_descriptor("model.layers.0.mlp.up_proj",
                                  quantization.QUANT_NVFP4)


@pytest.mark.parametrize("quant_type", ["fp8", "nvfp4"])
def test_quantized_refit_reads_metadata_not_matrix_payload(
        tmp_path, monkeypatch, quant_type):
    """Native refit preserves checkpoint layouts without loading matrix payloads."""
    prefix = "model.layers.0.self_attn.q_proj"
    tensors = {
        "weight": ("F8_E4M3" if quant_type == "fp8" else "U8",
                   np.full((16, 16 if quant_type == "fp8" else 8), 0x22,
                           np.uint8)),
        "weight_scale":
        ("F32", np.ones(1, np.float32)) if quant_type == "fp8" else
        ("F8_E4M3", np.full((16, 1), 0x38, np.uint8)),
        "weight_scale_2": ("F32", np.ones(1, np.float32)),
        "input_scale": ("F32", np.ones(1, np.float32)),
        "bias": ("F16", np.zeros(16, np.float16)),
    }
    header, payload = {}, bytearray()
    for suffix, (dtype, value) in tensors.items():
        start = len(payload)
        payload.extend(value.tobytes())
        header[prefix + "." + suffix] = {
            "dtype": dtype,
            "shape": list(value.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    (tmp_path / "model.safetensors"
     ).write_bytes(len(encoded).to_bytes(8, "little") + encoded + payload)
    weights = Weights(str(tmp_path),
                      policy=weight_policy.WeightPolicy.from_request(),
                      weightless_engine=True)
    original = weights.store._raw_array

    def read_small_scales(name, *args, **kwargs):
        assert name not in (prefix + ".weight", prefix + ".bias")
        if quant_type == "nvfp4":
            assert name != prefix + ".weight_scale"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(weights.store, "_raw_array", read_small_scales)
    try:
        descriptor = weights.linear_descriptor(prefix, quant_type)
    finally:
        weights.close()
    assert isinstance(descriptor.weight, ParameterSpec)
    assert isinstance(descriptor.bias, ParameterSpec)
    assert descriptor.weight.shape == tensors["weight"][1].shape
    assert descriptor.weight_recipe["checkpoint_keys"] == [prefix + ".weight"]
    if quant_type == "nvfp4":
        assert isinstance(descriptor.weight_scale, ParameterSpec)
        assert descriptor.weight_scale.shape == (16, 1)
        assert descriptor.scale_recipe["checkpoint_keys"] == [
            prefix + ".weight_scale"
        ]


@pytest.mark.parametrize(
    "mode, source_shape, expected_shape, shard_axis",
    (
        ("column", (12288, 4096), (6144, 4096), 0),
        ("row", (4096, 12288), (4096, 6144), 1),
    ),
)
def test_tp_external_fp16_shard_recipe_is_rank_neutral(mode, source_shape,
                                                       expected_shape,
                                                       shard_axis):
    descriptor = LinearWeights(
        quantization.QUANT_FP16,
        ParameterSpec(source_shape, np.float16),
        weight_recipe={"checkpoint_keys": ["weight"]},
    )

    rank0 = Weights.shard_linear(descriptor, mode, 2, 0)
    rank1 = Weights.shard_linear(descriptor, mode, 2, 1)

    assert rank0.weight.shape == expected_shape
    assert rank0.weight_recipe == rank1.weight_recipe
    assert rank0.weight_recipe["extra"]["tp_shard"] == {
        "axis": shard_axis,
        "size": 2,
    }


def test_tp_runtime_kv_metadata_uses_rank_local_head_counts():
    # Keep the TensorRT-dependent runtime-config import out of module scope so
    # non-TensorRT test jobs can still collect this shared test module.
    from tensorrt_edgellm.builder.core.artifacts.runtime_config import \
        _tp_rank_overrides

    global_config = {
        "num_attention_heads":
        24,
        "num_key_value_heads":
        4,
        "intermediate_size":
        17408,
        "recurrent_state_num_heads":
        48,
        "conv_dim":
        10240,
        "kv_layer_configs": [
            {
                "num_kv_heads": 4,
                "head_dim": 128,
            },
            None,
            {
                "num_kv_heads": 8,
                "head_dim": 256,
            },
        ],
    }

    overrides = _tp_rank_overrides(global_config, 2)

    assert overrides["recurrent_state_num_heads"] == 24
    assert overrides["conv_dim"] == 5120
    assert overrides["kv_layer_configs"] == [
        {
            "num_kv_heads": 2,
            "head_dim": 128,
        },
        None,
        {
            "num_kv_heads": 4,
            "head_dim": 256,
        },
    ]
    assert global_config["kv_layer_configs"][0]["num_kv_heads"] == 4


def test_gemma4_multimodal_ple_runtime_config_includes_pad_token():
    from tensorrt_edgellm.builder.models.gemma4 import runtime_config

    config = {}
    root = {
        "audio_token_id": 258881,
        "text_config": {
            "pad_token_id": 0,
        },
    }
    cfg = SimpleNamespace(hidden_size_per_layer_input=256)

    runtime_config.update_llm_config(config, root, cfg, SimpleNamespace())

    assert config["pad_token_id"] == 0


def test_gemma4_multimodal_ple_runtime_config_requires_pad_token():
    from tensorrt_edgellm.builder.models.gemma4 import runtime_config

    config = {}
    root = {"image_token_id": 258880, "text_config": {}}
    cfg = SimpleNamespace(hidden_size_per_layer_input=256)

    with pytest.raises(ValueError, match="requires text_config.pad_token_id"):
        runtime_config.update_llm_config(config, root, cfg, SimpleNamespace())


def test_gemma4_vision_linear_applies_checkpoint_clipping(monkeypatch):
    from tensorrt_edgellm.builder.models.gemma4 import modeling_gemma4_layers

    prefix = "vision_tower.encoder.layers.0.self_attn.q_proj"
    events = []

    class FakeWeights:

        @staticmethod
        def has(name):
            return name in {
                prefix + ".linear.weight",
                prefix + ".input_min",
                prefix + ".output_min",
            }

        @staticmethod
        def f16(name):
            return name

    class FakeLinear:

        def __init__(self, _ctx, linear_prefix, **_kwargs):
            self.prefix = linear_prefix

        def __call__(self, hidden):
            events.append(("linear", self.prefix))
            return hidden

    class FakeTensor:

        def maximum(self, value):
            events.append(("maximum", value))
            return self

        def minimum(self, value):
            events.append(("minimum", value))
            return self

    monkeypatch.setattr(modeling_gemma4_layers, "Linear", FakeLinear)
    projection = modeling_gemma4_layers.Gemma4ClippableLinear(
        SimpleNamespace(weights=FakeWeights()), prefix, rank=2)

    assert projection(FakeTensor()) is not None
    assert events == [
        ("maximum", prefix + ".input_min"),
        ("minimum", prefix + ".input_max"),
        ("linear", prefix + ".linear"),
        ("maximum", prefix + ".output_min"),
        ("minimum", prefix + ".output_max"),
    ]


def test_tp_segmented_column_shard_preserves_qkv_layout():
    weight = np.arange(20 * 4, dtype=np.float16).reshape(20, 4)
    scale = np.arange(20 * 2, dtype=np.uint8).reshape(20, 2)
    descriptor = LinearWeights(
        quantization.QUANT_NVFP4,
        weight,
        weight_scale=scale,
        group_size=4,
        logical_out_features=20,
        logical_in_features=8,
    )

    rank0 = Weights.shard_linear(descriptor,
                                 "column",
                                 2,
                                 0,
                                 output_segments=(4, 4, 12))
    rank1 = Weights.shard_linear(descriptor,
                                 "column",
                                 2,
                                 1,
                                 output_segments=(4, 4, 12))

    assert np.array_equal(rank0.weight,
                          weight[[0, 1, 4, 5, 8, 9, 10, 11, 12, 13]])
    assert np.array_equal(rank1.weight,
                          weight[[2, 3, 6, 7, 14, 15, 16, 17, 18, 19]])
    assert np.array_equal(rank0.weight_scale,
                          scale[[0, 1, 4, 5, 8, 9, 10, 11, 12, 13]])
    assert np.array_equal(rank1.weight_scale,
                          scale[[2, 3, 6, 7, 14, 15, 16, 17, 18, 19]])


def test_tp_nvfp4_row_parallel_shards_scale_on_input_axis():
    descriptor = LinearWeights(
        quantization.QUANT_NVFP4,
        np.zeros((4096, 2048), dtype=np.uint8),
        weight_scale=np.zeros((4096, 256), dtype=np.uint8),
        group_size=16,
    )

    sharded = Weights.shard_linear(descriptor, "row", 2, 0)

    assert sharded.weight.shape == (4096, 1024)
    assert sharded.weight_scale.shape == (4096, 128)


def test_tp_nvfp4_a16_row_parallel_shards_raw_weight_and_scale():
    descriptor = LinearWeights(
        quantization.QUANT_NVFP4_A16,
        np.zeros((4096, 2048), dtype=np.uint8),
        weight_scale=np.zeros((4096, 256), dtype=np.uint8),
        weight_scale_2=0.5,
        group_size=16,
    )

    sharded = Weights.shard_linear(descriptor, "row", 2, 1)

    assert sharded.quant_type == quantization.QUANT_NVFP4_A16
    assert sharded.weight.shape == (4096, 1024)
    assert sharded.weight_scale.shape == (4096, 128)
    assert sharded.in_features == 2048
    assert sharded.weight_scale_2 == 0.5


def test_tp_external_nvfp4_shard_recipe_is_rank_neutral():
    descriptor = LinearWeights(
        quantization.QUANT_NVFP4,
        ParameterSpec((4096, 2048), np.uint8),
        weight_scale=ParameterSpec((4096, 256), np.uint8),
        group_size=16,
        weight_recipe={"checkpoint_keys": ["weight"]},
        scale_recipe={"checkpoint_keys": ["weight_scale"]},
        logical_out_features=4096,
        logical_in_features=4096,
    )

    rank0 = Weights.shard_linear(descriptor, "row", 2, 0)
    rank1 = Weights.shard_linear(descriptor, "row", 2, 1)

    assert rank0.weight.shape == (4096, 1024)
    assert rank0.weight_scale.shape == (4096, 128)
    assert rank0.weight_recipe == rank1.weight_recipe
    assert rank0.scale_recipe == rank1.scale_recipe
    assert rank0.weight_recipe["extra"]["tp_shard"] == {
        "axis": 1,
        "size": 2,
    }
    assert rank0.scale_recipe["extra"]["tp_shard"] == {
        "axis": 1,
        "size": 2,
    }


@pytest.mark.parametrize(
    "sm12x,qualified,expected_raw_scale",
    [
        (False, False, False),
        (True, True, True),
    ],
)
def test_tp_nvfp4_row_projection_uses_target_scale_contract(
        monkeypatch, sm12x, qualified, expected_raw_scale):
    descriptor = LinearWeights(
        quantization.QUANT_NVFP4,
        np.zeros((4096, 2048), dtype=np.uint8),
        bias=np.zeros((4096, ), dtype=np.float16),
        weight_scale=np.zeros((4096, 256), dtype=np.uint8),
        weight_scale_2=0.5,
        input_scale=0.25,
        group_size=16,
    )

    class StubWeights:

        @staticmethod
        def uses_w4a16_fallback(quant_type):
            return False

        @staticmethod
        def shard_linear(full, mode, tp_size, tp_rank, output_segments):
            assert full is descriptor
            assert (mode, tp_size, tp_rank, output_segments) == ("row", 2, 1,
                                                                 ())
            return descriptor

    ctx = SimpleNamespace(
        cfg=SimpleNamespace(tp_size=2, tp_rank=1),
        weights=StubWeights(),
        options=SimpleNamespace(
            sm12x=sm12x,
            sm12x_nvfp4_gemm_allreduce=qualified,
        ),
    )
    projection = Linear(ctx, "model.layers.0.self_attn.o_proj", tp_mode="row")
    monkeypatch.setattr(projection, "quant_type", lambda: "nvfp4")
    monkeypatch.setattr(projection, "has_adapter", lambda: False)

    def full_descriptor(quant_type, use_fused_nvfp4_tp):
        assert quant_type == "nvfp4"
        assert use_fused_nvfp4_tp
        return descriptor

    monkeypatch.setattr(projection, "_full_descriptor", full_descriptor)
    captured = {}

    def fused(hidden_states, weights, bias, tp_size, rank, **kwargs):
        captured.update(hidden_states=hidden_states,
                        weights=weights,
                        bias=bias,
                        tp_size=tp_size,
                        rank=rank,
                        kwargs=kwargs)
        return "fused"

    monkeypatch.setattr(linear_ops.F, "fused_nvfp4_gemm_all_reduce", fused)

    assert projection.forward("hidden", rank=3) == "fused"
    assert captured["hidden_states"] == "hidden"
    assert captured["weights"].bias is None
    assert captured["bias"] is descriptor.bias
    assert captured["tp_size"] == 2
    assert captured["rank"] == 3
    assert captured["kwargs"]["raw_activation_scale"] is expected_raw_scale


@pytest.mark.parametrize(
    "model_type,hidden,layers,head_dim,intermediate,q_heads,kv_heads",
    [
        ("qwen3", 2560, 36, 128, 9728, 32, 8),
        ("qwen3", 4096, 36, 128, 12288, 32, 8),
        ("qwen3", 5120, 40, 128, 17408, 40, 8),
        ("qwen3", 5120, 64, 128, 25600, 64, 8),
        ("qwen3_vl_text", 4096, 36, 128, 12288, 32, 8),
        ("qwen3_5_text", 5120, 64, 256, 17408, 24, 4),
    ],
)
def test_sm12x_nvfp4_gemm_qualification_accepts_supported_tp2_models(
        model_type, hidden, layers, head_dim, intermediate, q_heads, kv_heads):
    config = SimpleNamespace(model_type=model_type,
                             tp_size=2,
                             hidden_size=hidden,
                             num_hidden_layers=layers,
                             head_dim=head_dim,
                             intermediate_size=intermediate // 2,
                             num_attention_heads=q_heads // 2,
                             num_key_value_heads=kv_heads // 2)
    assert supports_sm12x_nvfp4_gemm_allreduce(config)


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "tp_size": 1
        },
        {
            "model_type": "qwen2"
        },
        {
            "hidden_size": 4097
        },
        {
            "num_hidden_layers": 35
        },
    ],
)
def test_sm12x_nvfp4_gemm_qualification_rejects_unvalidated_models(overrides):
    values = {
        "model_type": "qwen3",
        "tp_size": 2,
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "head_dim": 128,
        "intermediate_size": 6144,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
    }
    values.update(overrides)
    assert not supports_sm12x_nvfp4_gemm_allreduce(SimpleNamespace(**values))


def test_sm12x_tp_nvfp4_row_projection_requires_model_capability(monkeypatch):
    descriptor = LinearWeights(
        quantization.QUANT_NVFP4,
        np.zeros((4096, 2048), dtype=np.uint8),
        weight_scale=np.zeros((4096, 256), dtype=np.uint8),
        weight_scale_2=0.5,
        input_scale=0.25,
        group_size=16,
    )

    class StubWeights:

        @staticmethod
        def uses_w4a16_fallback(quant_type):
            return False

        @staticmethod
        def shard_linear(full, mode, tp_size, tp_rank, output_segments):
            assert (mode, tp_size, tp_rank, output_segments) == ("row", 2, 0,
                                                                 ())
            return full

    ctx = SimpleNamespace(
        cfg=SimpleNamespace(tp_size=2, tp_rank=0),
        weights=StubWeights(),
        options=SimpleNamespace(sm12x=True, sm12x_nvfp4_gemm_allreduce=False),
    )
    projection = Linear(ctx, "model.layers.0.self_attn.o_proj", tp_mode="row")
    monkeypatch.setattr(projection, "quant_type", lambda: "nvfp4")
    monkeypatch.setattr(projection, "has_adapter", lambda: False)
    monkeypatch.setattr(projection, "_uses_lora", lambda: False)
    monkeypatch.setattr(projection, "_apply_static_adapter",
                        lambda hidden, output, mode, rank: output)

    def full_descriptor(quant_type, use_fused_nvfp4_tp):
        assert quant_type == "nvfp4"
        assert not use_fused_nvfp4_tp
        return descriptor

    monkeypatch.setattr(projection, "_full_descriptor", full_descriptor)
    monkeypatch.setattr(linear_ops.F, "linear_from_weights",
                        lambda hidden, weights, rank, name, out_dtype: "gemm")
    monkeypatch.setattr(linear_ops.F, "all_reduce", lambda output, tp_size:
                        (output, tp_size))

    assert projection.forward("hidden", rank=3) == ("gemm", 2)


def test_tp_nvfp4_resource_rejects_signed_packed_storage():
    prefix = "model.layers.0.self_attn.o_proj"
    dtypes = {
        prefix + ".weight": "I8",
        prefix + ".weight_scale": "F8_E4M3",
        prefix + ".weight_scale_2": "F32",
    }

    class DtypeOnlyStore:

        @staticmethod
        def has(name):
            return name in dtypes

        @staticmethod
        def dtype(name):
            return dtypes[name]

    weights = object.__new__(Weights)
    weights.store = DtypeOnlyStore()
    weights.conversion = None
    weights.vocab_map = None
    weights.component = "llm"
    weights.spec_type = "none"
    weights.spec_role = "none"
    weights.tie_word_embeddings = False

    assert weights.linear_nvfp4_tp_metadata(prefix) is None


def test_tp_default_bakes_small_fp16_parameters():
    from tensorrt_edgellm.builder.core.builder import BuildArgs

    default = BuildArgs("/checkpoint", "/engine", tp_size=2).weight_policy
    assert default.wants(weight_policy.EXTERNAL_WEIGHT_EMBEDDING)
    assert default.wants(weight_policy.EXTERNAL_WEIGHT_LM_HEAD)
    assert default.wants(weight_policy.EXTERNAL_WEIGHT_NVFP4_TP)
    assert not default.wants(weight_policy.EXTERNAL_WEIGHT_FP16)

    fully_external = BuildArgs(
        "/checkpoint",
        "/engine",
        tp_size=2,
        externalize_weights=(weight_policy.EXTERNAL_WEIGHT_ALL, ),
    ).weight_policy
    assert fully_external.wants(weight_policy.EXTERNAL_WEIGHT_EMBEDDING)
    assert fully_external.wants(weight_policy.EXTERNAL_WEIGHT_LM_HEAD)
    assert fully_external.wants(weight_policy.EXTERNAL_WEIGHT_FP16)


@pytest.mark.parametrize("trt_version, disable_cuda_tile", [
    ("10.16.0", False),
    ("11.0.1.7", False),
    ("11.1.0", True),
    ("11.3.1.7", True),
    ("12.0.0", True),
])
def test_compile_workarounds_are_scoped_to_the_build(monkeypatch, trt_version,
                                                     disable_cuda_tile):
    from tensorrt_edgellm.builder.core import builder

    def sm80():
        return (8, 0)

    monkeypatch.setattr(builder, "trt",
                        SimpleNamespace(__version__=trt_version))
    monkeypatch.setattr(builder, "_active_cuda_compute_capability", sm80)
    monkeypatch.delenv("__LUNOWUD", raising=False)

    with builder._compile_workarounds(max_batch_size=4) as flags:
        assert "-mlir:autotune:num_threads=1" in flags
        assert ("-kgen:codegen:cuda_tile=0" in flags) == disable_cuda_tile
        assert os.environ["__LUNOWUD"] == flags
    assert "__LUNOWUD" not in os.environ

    monkeypatch.setenv("__LUNOWUD", "-user:flag")
    with builder._compile_workarounds(max_batch_size=1) as flags:
        assert flags.startswith("-user:flag ")
        assert "-peep:fc_h_fusion=off" in flags
    assert os.environ["__LUNOWUD"] == "-user:flag"


@pytest.mark.parametrize("use_trt_native_moe", [False, True])
def test_compile_workarounds_keep_cuda_tile_for_the_native_moe_layer(
        monkeypatch, use_trt_native_moe):
    """TensorRT's native MoE layer exists only in the CUDA Tile code generator:
    the cuda_tile=0 workaround is skipped for such builds while every other
    flag stays."""
    from tensorrt_edgellm.builder.core import builder

    def sm100():
        return (10, 0)

    monkeypatch.setattr(builder, "trt",
                        SimpleNamespace(__version__="11.3.0.89"))
    monkeypatch.setattr(builder, "_active_cuda_compute_capability", sm100)
    monkeypatch.delenv("__LUNOWUD", raising=False)

    with builder._compile_workarounds(
            max_batch_size=4, use_trt_native_moe=use_trt_native_moe) as flags:
        assert ("-kgen:codegen:cuda_tile=0" in flags) is not use_trt_native_moe
        assert "-mlir:autotune:num_threads=1" in flags
        assert "-peep:match_dual_gemm=off" in flags
        assert os.environ["__LUNOWUD"] == flags
    assert "__LUNOWUD" not in os.environ


@pytest.mark.parametrize("family", ["qwen3_tts", "qwen3_omni"])
def test_talker_runtime_config_keeps_the_dialect_speakers(family):
    """A dialect speaker names its own language rows, so the checkpoint's
    spk_is_dialect map reaches the talker's runtime config unchanged."""
    import importlib
    runtime_config = importlib.import_module(
        f"tensorrt_edgellm.builder.models.{family}.runtime_config")
    # Given a talker checkpoint with one dialect speaker and one plain speaker
    dialects = {"eric": "sichuan_dialect", "ryan": False}
    root = {
        "talker_config": {
            "spk_is_dialect": dialects,
            "codec_language_id": {
                "sichuan_dialect": 2062
            },
        }
    }
    config = {}
    # When its runtime config is written
    runtime_config.update_talker_config(config, root, cfg=None, args=None)
    # Then the dialect map arrives as the checkpoint spells it
    assert config["spk_is_dialect"] == dialects


_GPTQ = quantization.QuantConfig(quant_type=quantization.QUANT_INT4_GPTQ,
                                 group_size=128)
_SPLIT_OMNI_CHECKPOINT = (
    "thinker.model.layers.0.mlp.down_proj.qweight",
    "talker.code_predictor.model.layers.0.mlp.down_proj.qweight",
    "talker.model.layers.0.mlp.down_proj.weight",
)


@pytest.mark.parametrize(
    "names,weights,component,root,keeps",
    [
        pytest.param(_SPLIT_OMNI_CHECKPOINT,
                     omni_weights,
                     "talker",
                     _GPTQ,
                     False,
                     id="talker_beside_a_quantized_thinker"),
        pytest.param(_SPLIT_OMNI_CHECKPOINT,
                     omni_weights,
                     "llm",
                     _GPTQ,
                     True,
                     id="quantized_thinker"),
        # Unprefixed names count, as Weights._resolve tries them, and FP8-block scales mark
        # a quantized weight.
        pytest.param(
            ("model.layers.0.mlp.down_proj.weight",
             "model.layers.0.mlp.down_proj.weight_scale_inv"),
            omni_weights,
            "talker",
            quantization.QuantConfig(quant_type=quantization.QUANT_FP8_BLOCK),
            True,
            id="unprefixed_fp8_block"),
        pytest.param(("model.layers.0.mlp.down_proj.weight", ),
                     qwen3_weights,
                     "llm",
                     _GPTQ,
                     True,
                     id="conversion_not_opted_in"),
    ])
def test_a_component_shipped_unquantized_builds_in_fp16(
        tmp_path, names, weights, component, root, keeps):
    """An opted-in conversion builds a component the checkpoint ships unquantized in fp16; any other keeps the root quantization."""
    (tmp_path / "ckpt").mkdir()
    safetensors_numpy.save_file(
        {name: np.zeros((2, 2), dtype=np.float16)
         for name in names}, str(tmp_path / "ckpt" / "model.safetensors"))
    result = component_quantization(str(tmp_path / "ckpt"), weights, component,
                                    root, "none", "none")
    assert (result is root) == keeps
    assert result.quant_type == (root.quant_type
                                 if keeps else quantization.QUANT_FP16)


def test_the_omni_llm_build_writes_its_embedding_file():
    """A Qwen3-Omni llm build writes embedding.safetensors under the default weight policy."""
    # Given an llm build under the default weight policy, which moves the embedding into a
    # checkpoint binding for a family that does not opt out,
    # When the Qwen3-Omni weight conversion decides,
    # Then it goes to embedding.safetensors, the file the segments talker reads the thinker's
    # table from.
    args = SimpleNamespace(
        weight_policy=weight_policy.WeightPolicy.from_request(None),
        resolved_spec_role=contracts.SpecRole.NONE,
        spec_type="none")
    assert not externalizes_embedding(args, omni_weights)
