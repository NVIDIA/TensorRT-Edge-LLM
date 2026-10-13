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
"""Focused DiffusionGemma frontend and reference-sampler tests."""

import json
import math
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from tensorrt_edgellm.checkpoint.checkpoint_utils import (
    _runtime_embedding_scale, build_runtime_llm_config_dict)
from tensorrt_edgellm.config import (ModelConfig, _is_diffusion_gemma_config,
                                     module_quant_type)
from tensorrt_edgellm.models.diffusion_gemma import (
    DiffusionGemmaBackbone, make_diffusion_gemma_key_remap)
from tensorrt_edgellm.models.gemma4.modeling_gemma4_text import (
    _gemma4_dense_moe_routing, _gemma4_uses_swa_kv_cache)
from tensorrt_edgellm.scripts import export as export_script


@dataclass(frozen=True)
class EntropyBoundSamplerConfig:
    entropy_threshold: float = 0.005
    entropy_bound: float = 0.1
    stability_window: int = 2


def entropy_bound_accept_mask(
    logits: torch.Tensor,
    previous_tokens: torch.Tensor | None,
    stable_counts: torch.Tensor | None,
    config: EntropyBoundSamplerConfig,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference DiffusionGemma block sampler used by correctness tests."""
    if logits.ndim != 3:
        raise ValueError(
            "entropy_bound_accept_mask expects logits with shape [B, C, V].")
    scaled_logits = logits.float() / max(float(temperature), 1e-6)
    probs = torch.softmax(scaled_logits, dim=-1)
    tokens = torch.argmax(probs, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-20))).sum(dim=-1)

    if previous_tokens is None:
        previous_tokens = torch.full_like(tokens, -1)
    if stable_counts is None:
        stable_counts = torch.zeros_like(tokens, dtype=torch.int32)

    same_as_previous = tokens == previous_tokens
    next_stable_counts = torch.where(same_as_previous, stable_counts + 1,
                                     torch.ones_like(stable_counts))

    sorted_entropy, sorted_idx = torch.sort(entropy, dim=-1)
    cumsum_entropy = torch.cumsum(sorted_entropy, dim=-1)
    cummax_entropy = torch.cummax(sorted_entropy, dim=-1).values
    sorted_accept = ((cumsum_entropy - cummax_entropy)
                     <= float(config.entropy_bound))
    accept_mask = torch.zeros_like(sorted_accept, dtype=torch.bool)
    accept_mask.scatter_(1, sorted_idx, sorted_accept)
    return tokens, accept_mask, next_stable_counts, entropy


def soft_token_embeds(logits: torch.Tensor,
                      embedding_weight: torch.Tensor,
                      temperature: float = 1.0) -> torch.Tensor:
    scaled_logits = logits.float() / max(float(temperature), 1e-6)
    probs = torch.softmax(scaled_logits, dim=-1)
    return torch.matmul(probs.to(embedding_weight.dtype), embedding_weight)


def _load_model_config(tmp_path):
    return ModelConfig.from_pretrained(
        str(tmp_path), lambda head_dim: 1.0 / math.sqrt(float(head_dim)))


def _write_minimal_diffusion_gemma_config(tmp_path,
                                          *,
                                          ple=False,
                                          heterogeneous_attention=False):
    config = {
        "model_type": "diffusion_gemma",
        "architectures": ["DiffusionGemmaForCausalLM"],
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 32,
        "num_experts": 4,
        "top_k_experts": 2,
        "moe_intermediate_size": 8,
        "self_conditioning_size": 24,
        "rms_norm_eps": 1e-6,
        "vocab_size": 128,
        "max_position_embeddings": 256,
        "layer_types": ["full_attention", "full_attention"],
        "attention_scaling": 1.0,
        "final_logit_softcapping": 30.0,
        "encoder_layer_scalars": [1.0, 2.0],
        "decoder_layer_scalars": [3.0, 4.0],
        "torch_dtype": "float16",
    }
    if ple:
        config["hidden_size_per_layer_input"] = 4
        config["vocab_size_per_layer_input"] = 32
    if heterogeneous_attention:
        config.update({
            "global_head_dim": 8,
            "num_global_key_value_heads": 1,
            "layer_types": ["sliding_attention", "full_attention"],
            "sliding_window": 32,
            "rope_parameters": {
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.5,
                },
            },
        })
    generation_config = {
        "max_denoising_steps": 8,
        "max_new_tokens": 8,
        "sampler_config": {
            "_cls_name": "EntropyBound",
            "entropy_bound": 0.02,
        },
        "stability_threshold": 3,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "generation_config.json").write_text(
        json.dumps(generation_config))


def test_model_config_detects_diffusion_gemma(tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path)
    cfg = _load_model_config(tmp_path)

    assert cfg.model_type == "diffusion_gemma"
    assert cfg.is_diffusion_gemma
    assert cfg.attention_k_eq_v
    assert cfg.attention_scaling == 1.0
    assert cfg.final_logit_softcapping == 30.0
    assert cfg.attention_layer_types == ["full_attention", "full_attention"]
    assert cfg.embedding_scale == 4.0
    assert _runtime_embedding_scale(SimpleNamespace(config=cfg)) == 4.0
    assert cfg.self_conditioning_size == 24
    assert cfg.num_experts == 4
    assert cfg.num_experts_per_tok == 2
    assert cfg.moe_intermediate_size == 8
    assert cfg.enable_moe_block
    assert cfg.encoder_layer_scalars == [1.0, 2.0]
    assert cfg.decoder_layer_scalars == [3.0, 4.0]
    assert cfg.diffusion is not None
    assert cfg.diffusion.canvas_length == 256
    assert cfg.diffusion.max_denoising_steps == 8
    assert cfg.diffusion.entropy_bound == 0.02
    assert cfg.diffusion.stability_window == 3


def test_diffusion_config_detection_allows_null_architectures():
    assert not _is_diffusion_gemma_config(
        {
            "model_type": "qwen3_tts",
            "architectures": None,
        }, {})
    assert _is_diffusion_gemma_config(
        {
            "model_type": "gemma4_text",
            "architectures": None,
        }, {"architectures": ["DiffusionGemmaForCausalLM"]})


def test_model_config_keeps_kv_shared_layer_validation():
    base_kwargs = {
        "model_type": "gemma4_text",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 32,
        "rms_norm_eps": 1e-6,
        "vocab_size": 32,
        "rope_theta": 10000.0,
        "max_position_embeddings": 128,
        "default_attention_scale": 0.5,
        "num_kv_shared_layers": 2,
    }
    with pytest.raises(ValueError, match="num_kv_shared_layers"):
        ModelConfig(**base_kwargs)

    cfg = ModelConfig(**base_kwargs, shares_target_kv=True)
    assert cfg.num_kv_shared_layers == 2
    assert cfg.attention_scaling == 0.5


def test_diffusion_config_does_not_use_max_new_tokens_as_denoising_steps(
        tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path)
    generation_config_path = tmp_path / "generation_config.json"
    generation_config = json.loads(generation_config_path.read_text())
    generation_config.pop("max_denoising_steps")
    generation_config["max_new_tokens"] = 512
    generation_config_path.write_text(json.dumps(generation_config))

    cfg = _load_model_config(tmp_path)

    assert cfg.diffusion is not None
    assert cfg.diffusion.max_denoising_steps == 48


def test_runtime_config_emits_diffusion_engine_metadata(tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path)
    cfg = _load_model_config(tmp_path)

    dllm = SimpleNamespace(
        config=cfg,
        diffusion_engine_role="dllm",
        diffusion_unified_conditioning=True,
    )
    runtime_cfg = build_runtime_llm_config_dict(dllm)

    assert runtime_cfg["model"] == "diffusion_gemma_text"
    assert runtime_cfg["engine_role"] == "dllm"
    assert runtime_cfg["decoding_strategy"] == "block_diffusion"
    assert runtime_cfg["context_mask_selector_enabled"] is True
    assert runtime_cfg["diffusion_unified_conditioning"] is True
    assert runtime_cfg["diffusion_engines"] == {
        "dllm": {
            "path": "dllm.engine",
            "role": "dllm",
        },
    }
    assert runtime_cfg["attention_scaling"] == 1.0
    assert runtime_cfg["final_logit_softcapping"] == 30.0
    assert runtime_cfg["attention_k_eq_v"] is True
    assert runtime_cfg["num_experts"] == 4
    assert runtime_cfg["num_experts_per_tok"] == 2
    assert runtime_cfg["moe_intermediate_size"] == 8
    assert runtime_cfg["enable_moe_block"] is True
    assert runtime_cfg["attention_layer_types"] == [
        "full_attention", "full_attention"
    ]

    stale_dllm = SimpleNamespace(config=cfg, diffusion_engine_role="dllm")
    try:
        build_runtime_llm_config_dict(stale_dllm)
    except ValueError as exc:
        assert "unified self-conditioning" in str(exc)
    else:
        raise AssertionError(
            "Expected dllm config to require unified conditioning")


def test_backbone_export_spec_has_dynamic_batch_dims(tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path)
    cfg = _load_model_config(tmp_path)
    model = DiffusionGemmaBackbone(cfg)

    spec = model.onnx_export_spec()
    shapes_by_name = dict(zip(spec.input_names, spec.dynamic_shapes))

    assert model.diffusion_unified_conditioning is True
    assert spec.input_names[:5] == [
        "inputs_embeds",
        "phase_is_encoder",
        "canvas_ids",
        "prev_self_conditioning_embeds",
        "self_conditioning_temperature",
    ]
    assert 0 in shapes_by_name["inputs_embeds"]
    assert shapes_by_name["phase_is_encoder"] == {}
    assert list(shapes_by_name["canvas_ids"].keys()) == [0]
    assert list(shapes_by_name["prev_self_conditioning_embeds"].keys()) == [0]
    assert shapes_by_name["self_conditioning_temperature"] == {}
    assert list(shapes_by_name["past_key_values_0"].keys()) == [1]
    assert 0 in shapes_by_name["rope_rotary_cos_sin"]
    assert "context_lengths" not in shapes_by_name
    assert "kvcache_start_index" not in shapes_by_name
    for name in ("positions", "query_start_offsets", "query_lengths",
                 "past_lengths", "attention_sequence_lengths", "state_indices",
                 "execution_phase_marker"):
        assert name in shapes_by_name
    assert list(shapes_by_name["kv_page_table"].keys()) == [0, 2]
    assert list(shapes_by_name["execution_phase_marker"].keys()) == [0]
    assert 0 in shapes_by_name["select_token_indices"]
    assert list(shapes_by_name["context_mask_selector"].keys()) == [0]
    assert "prev_logits" not in shapes_by_name
    assert spec.output_names[:3] == [
        "logits",
        "next_self_conditioning_embeds",
        "present_key_values_0",
    ]


def test_diffusion_sliding_attention_keeps_full_kv_pool(tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path,
                                          heterogeneous_attention=True)
    cfg = _load_model_config(tmp_path)

    assert not _gemma4_uses_swa_kv_cache(cfg)
    model = DiffusionGemmaBackbone(cfg)
    spec = model.onnx_export_spec()
    assert "swa_kv_page_table" not in spec.input_names
    assert "swa_kv_cache_mode" not in spec.input_names
    runtime_cfg = build_runtime_llm_config_dict(model)
    assert runtime_cfg["layer_types"] == ["attention", "attention"]
    assert runtime_cfg["kv_layer_configs"] == [
        {
            "num_kv_heads": 2,
            "head_dim": 4,
        },
        {
            "num_kv_heads": 1,
            "head_dim": 8,
        },
    ]
    assert all("kv_cache_capacity" not in layer
               for layer in runtime_cfg["kv_layer_configs"])


def test_unified_backbone_export_spec_has_conditioning_bindings(tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path)
    cfg = _load_model_config(tmp_path)
    model = DiffusionGemmaBackbone(cfg)
    model.enable_unified_conditioning()

    spec = model.onnx_export_spec()
    shapes_by_name = dict(zip(spec.input_names, spec.dynamic_shapes))

    assert spec.input_names[:5] == [
        "inputs_embeds",
        "phase_is_encoder",
        "canvas_ids",
        "prev_self_conditioning_embeds",
        "self_conditioning_temperature",
    ]
    assert list(shapes_by_name["canvas_ids"].keys()) == [0]
    assert list(shapes_by_name["prev_self_conditioning_embeds"].keys()) == [0]
    assert shapes_by_name["self_conditioning_temperature"] == {}
    assert "kv_page_table" in shapes_by_name
    assert spec.input_names.index("kv_page_table") < spec.input_names.index(
        "select_token_indices")
    assert "prev_logits" not in shapes_by_name
    assert "prev_logits_valid" not in shapes_by_name
    assert spec.output_names[:3] == [
        "logits",
        "next_self_conditioning_embeds",
        "present_key_values_0",
    ]


def test_unified_backbone_token_major_wrapper_executes(tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path)
    model = DiffusionGemmaBackbone(_load_model_config(tmp_path)).eval()
    spec = model.onnx_export_spec()

    outputs = spec.wrapped(*spec.args)
    args = dict(zip(spec.input_names, spec.args))

    assert outputs[0].ndim == 2
    assert outputs[1].ndim == 2
    assert outputs[2].shape == spec.args[spec.input_names.index(
        "past_key_values_0")].shape
    assert args["positions"].shape[0] == args["inputs_embeds"].shape[0]
    assert "token_to_sequence" not in args
    assert args["query_start_offsets"].shape[
        0] == args["query_lengths"].shape[0] + 1


def test_unified_backbone_dynamic_export_succeeds(tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path)
    model = DiffusionGemmaBackbone(_load_model_config(tmp_path)).eval()
    spec = model.onnx_export_spec()

    exported = torch.export.export(spec.wrapped,
                                   spec.args,
                                   dynamic_shapes=spec.dynamic_shapes,
                                   strict=False)

    assert exported is not None


def test_unified_backbone_with_ple_dynamic_export_succeeds(tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path, ple=True)
    model = DiffusionGemmaBackbone(_load_model_config(tmp_path)).eval()
    spec = model.onnx_export_spec()

    exported = torch.export.export(spec.wrapped,
                                   spec.args,
                                   dynamic_shapes=spec.dynamic_shapes,
                                   strict=False)

    assert exported is not None


def test_unified_backbone_with_heterogeneous_attention_exports(tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path,
                                          heterogeneous_attention=True)
    model = DiffusionGemmaBackbone(_load_model_config(tmp_path)).eval()
    spec = model.onnx_export_spec()

    exported = torch.export.export(spec.wrapped,
                                   spec.args,
                                   dynamic_shapes=spec.dynamic_shapes,
                                   strict=False)

    assert exported is not None


def test_unified_backbone_with_ple_and_heterogeneous_attention_exports(
        tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path,
                                          ple=True,
                                          heterogeneous_attention=True)
    model = DiffusionGemmaBackbone(_load_model_config(tmp_path)).eval()
    spec = model.onnx_export_spec()

    exported = torch.export.export(spec.wrapped,
                                   spec.args,
                                   dynamic_shapes=spec.dynamic_shapes,
                                   strict=False)

    assert exported is not None


def test_unified_backbone_dynamic_export_flattens_moe_plugin_output(tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path,
                                          heterogeneous_attention=True)
    model = DiffusionGemmaBackbone(_load_model_config(tmp_path)).eval()

    class RankThreeMoeOutput(torch.nn.Module):

        def forward(self, expert_input, residual):
            del residual
            return expert_input.unsqueeze(0)

    for layer in model.model.layers:
        layer.moe_block = RankThreeMoeOutput()
    spec = model.onnx_export_spec()

    exported = torch.export.export(spec.wrapped,
                                   spec.args,
                                   dynamic_shapes=spec.dynamic_shapes,
                                   strict=False)

    assert exported is not None


def test_unified_backbone_rejects_reduced_vocab(tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path)
    cfg = _load_model_config(tmp_path)
    cfg.reduced_vocab_size = 16

    with pytest.raises(ValueError, match="reduced vocabulary"):
        DiffusionGemmaBackbone(cfg)


def test_unified_backbone_conditioning_matches_reference(tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path)
    cfg = _load_model_config(tmp_path)
    backbone = DiffusionGemmaBackbone(cfg).eval()
    backbone.enable_unified_conditioning()

    torch.manual_seed(11)
    with torch.no_grad():
        backbone.model.embed_tokens.weight.uniform_(-0.2, 0.2)
        for parameter in backbone.self_conditioning.parameters():
            parameter.uniform_(-0.2, 0.2)

    canvas_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    prev_logits = torch.randn(2, 3, cfg.vocab_size, dtype=torch.float32)
    temperature = torch.tensor([0.7], dtype=torch.float32)

    scaled_embedding = backbone._scaled_embedding_weight()
    token_embeds = torch.nn.functional.embedding(canvas_ids.long(),
                                                 scaled_embedding)
    soft_embeds = soft_token_embeds(prev_logits,
                                    scaled_embedding,
                                    temperature=float(temperature.item()))

    next_soft = backbone._next_self_conditioning_embeds(
        prev_logits, temperature)
    soft_embeds = soft_embeds.to(next_soft.dtype)
    assert torch.allclose(next_soft, soft_embeds, atol=2e-3, rtol=2e-3)

    actual = backbone._unified_conditioned_inputs(canvas_ids, next_soft)
    expected = backbone.self_conditioning(token_embeds, soft_embeds)
    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3)

    zero_soft_embeds = torch.zeros_like(next_soft)
    first_actual = backbone._unified_conditioned_inputs(
        canvas_ids, zero_soft_embeds)
    first_expected = backbone.self_conditioning(token_embeds, zero_soft_embeds)
    assert torch.allclose(first_actual, first_expected, atol=2e-3, rtol=2e-3)


def test_diffusion_gemma_key_remap_splits_backbone_and_self_conditioning():
    unified_remap = make_diffusion_gemma_key_remap(
        include_backbone=True,
        include_self_conditioning=True,
    )
    backbone_remap = make_diffusion_gemma_key_remap(
        include_backbone=True,
        include_self_conditioning=False,
    )
    sc_remap = make_diffusion_gemma_key_remap(include_backbone=False,
                                              include_self_conditioning=True)
    assert (backbone_remap("model.decoder.layers.0.self_attn.q_proj.weight") ==
            "model.layers.0.self_attn.q_proj.weight")
    assert (unified_remap("model.decoder.self_conditioning.pre_norm.weight") ==
            "self_conditioning.pre_norm.weight")
    assert (backbone_remap("model.decoder.layers.0.router.proj.weight") ==
            "model.layers.0.router.proj.weight")
    assert (backbone_remap(
        "model.encoder.language_model.layers.0."
        "layer_scalar") == "model.layers.0.encoder_layer_scalar")
    assert (backbone_remap("model.decoder.layers.0.layer_scalar") ==
            "model.layers.0.decoder_layer_scalar")
    assert backbone_remap(
        "model.encoder.vision_tower.patch_embedding.weight") is None
    assert (sc_remap("model.decoder.self_conditioning.pre_norm.weight") ==
            "self_conditioning.pre_norm.weight")
    assert sc_remap("model.encoder.language_model.embed_tokens.weight") is None


def test_diffusion_gemma_key_remap_preserves_gemma4_nvfp4_moe_layout():
    backbone_remap = make_diffusion_gemma_key_remap(
        include_backbone=True,
        include_self_conditioning=False,
        nvfp4_moe=True,
    )

    assert (backbone_remap("model.decoder.layers.0.router.proj.weight") ==
            "model.layers.0.moe_block.router.proj.weight")
    assert (backbone_remap("model.decoder.layers.0.router.per_expert_scale") ==
            "model.layers.0.moe_block.router.per_expert_scale")
    assert (backbone_remap("model.decoder.layers.0.experts.7.gate_proj.weight")
            == "model.layers.0.moe_block.experts._experts.7.gate_proj.weight")
    assert (backbone_remap(
        "model.decoder.layers.0.experts.7.down_proj.weight_scale_2") ==
            "model.layers.0.moe_block.experts._experts.7.down_proj."
            "weight_scale_2")


def test_diffusion_gemma_nvfp4_excludes_decoder_attention_from_quant(tmp_path):
    _write_minimal_diffusion_gemma_config(tmp_path)
    quant_config = {
        "quantization": {
            "quant_algo":
            "NVFP4",
            "kv_cache_quant_algo":
            "FP8",
            "group_size":
            16,
            "exclude_modules": [
                "lm_head",
                "*self_attn*",
                "*mlp*",
                "model.decoder.self_conditioning.*",
            ],
        }
    }
    (tmp_path / "hf_quant_config.json").write_text(json.dumps(quant_config))

    cfg = _load_model_config(tmp_path)

    assert cfg.quant.quant_type == "nvfp4"
    # KV-cache precision is inferred from checkpoint scale tensors, not the
    # sidecar field alone; this synthetic fixture has no safetensors.
    assert cfg.quant.kv_cache_quant is None
    assert module_quant_type("layers.0.self_attn.q_proj", cfg) == "fp16"
    assert module_quant_type("layers.0.mlp.gate_proj", cfg) == "fp16"
    assert module_quant_type("self_conditioning.gate_proj", cfg) == "fp16"
    assert module_quant_type("self_conditioning.up_proj", cfg) == "fp16"
    assert module_quant_type("self_conditioning.down_proj", cfg) == "fp16"
    assert module_quant_type("layers.0.moe_block.experts._experts.0.gate_proj",
                             cfg) == "nvfp4"
    assert module_quant_type("", cfg) == "nvfp4"


def test_diffusion_gemma_nvfp4_externalize_appends_required_moe_sidecar(
        tmp_path):
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {
            "quant_algo": "NVFP4"
        }}))

    assert export_script._diffusion_gemma_backbone_externalize_weights(
        str(tmp_path), []) == ["nvfp4_moe"]
    assert export_script._diffusion_gemma_backbone_externalize_weights(
        str(tmp_path), ["lm_head"]) == ["lm_head", "nvfp4_moe"]
    assert export_script._diffusion_gemma_backbone_externalize_weights(
        str(tmp_path), ["nvfp4_moe"]) == ["nvfp4_moe"]


def test_gemma4_dense_moe_routing_matches_vllm_semantics():
    logits = torch.tensor([
        [2.0, 0.0, 1.0, -1.0],
        [0.0, 3.0, 2.0, 1.0],
    ],
                          dtype=torch.float32)
    per_expert_scale = torch.tensor([1.0, 0.5, 2.0, 3.0], dtype=torch.float32)

    weights, expert_ids = _gemma4_dense_moe_routing(logits, 2,
                                                    per_expert_scale)

    ref_ids = torch.topk(logits, k=2, dim=-1).indices
    ref_weights = torch.softmax(logits, dim=-1).gather(1, ref_ids)
    ref_weights = ref_weights / ref_weights.sum(dim=-1, keepdim=True)
    ref_weights = ref_weights * per_expert_scale[ref_ids]

    biased = torch.softmax(logits + torch.log(per_expert_scale),
                           dim=-1).gather(1, ref_ids)
    biased = biased / biased.sum(dim=-1, keepdim=True)

    assert torch.equal(expert_ids, ref_ids)
    assert torch.allclose(weights, ref_weights)
    assert not torch.allclose(weights, biased)


def test_entropy_bound_reference_sampler_and_soft_embeds():
    logits = torch.tensor([[
        [10.0, -10.0, -10.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]])
    previous = torch.tensor([[0, 1, 2]])
    stable = torch.tensor([[1, 1, 1]], dtype=torch.int32)

    tokens, accept, next_stable, entropy = entropy_bound_accept_mask(
        logits,
        previous,
        stable,
        EntropyBoundSamplerConfig(entropy_threshold=0.01,
                                  entropy_bound=0.1,
                                  stability_window=2),
    )
    assert tokens.tolist() == [[0, 0, 0]]
    assert accept.tolist() == [[True, True, False]]
    assert next_stable.tolist() == [[2, 1, 1]]
    assert entropy.shape == tokens.shape

    embedding = torch.eye(3, 4, dtype=torch.float16)
    soft = soft_token_embeds(logits, embedding)
    assert list(soft.shape) == [1, 3, 4]
    assert torch.argmax(soft[0, 0]).item() == 0

    temp_logits = torch.tensor([[[2.0, 0.0, -1.0]]])
    temp_embedding = torch.eye(3, 4, dtype=torch.float16)
    _, _, _, cold_entropy = entropy_bound_accept_mask(
        temp_logits, None, None, EntropyBoundSamplerConfig(), temperature=0.5)
    _, _, _, hot_entropy = entropy_bound_accept_mask(
        temp_logits, None, None, EntropyBoundSamplerConfig(), temperature=2.0)
    assert hot_entropy[0, 0] > cold_entropy[0, 0]
    cold_soft = soft_token_embeds(temp_logits, temp_embedding, temperature=0.5)
    hot_soft = soft_token_embeds(temp_logits, temp_embedding, temperature=2.0)
    assert hot_soft[0, 0, 0] < cold_soft[0, 0, 0]


def test_entropy_bound_stability_window_one_counts_first_observation():
    logits = torch.tensor([[[0.01, 0.0, 0.0]]])
    cfg = EntropyBoundSamplerConfig(entropy_threshold=0.01,
                                    entropy_bound=-1.0,
                                    stability_window=1)

    tokens, accept, stable, _ = entropy_bound_accept_mask(
        logits, None, None, cfg)
    assert tokens.tolist() == [[0]]
    assert accept.tolist() == [[False]]
    assert stable.tolist() == [[1]]


def test_entropy_bound_reference_sampler_trajectory():
    logits_steps = [
        torch.tensor([[
            [20.0, -20.0, -20.0],
            [0.0, 0.0, 0.0],
        ]]),
        torch.tensor([[
            [20.0, -20.0, -20.0],
            [0.0, 4.0, 0.0],
        ]]),
        torch.tensor([[
            [0.0, 20.0, 0.0],
            [0.0, 4.0, 0.0],
        ]]),
    ]

    previous = None
    stable = None
    token_trajectory = []
    accept_trajectory = []
    stable_trajectory = []
    for logits in logits_steps:
        tokens, accept, stable, _ = entropy_bound_accept_mask(
            logits,
            previous,
            stable,
            EntropyBoundSamplerConfig(entropy_threshold=0.01,
                                      entropy_bound=0.1,
                                      stability_window=2),
        )
        token_trajectory.append(tokens)
        accept_trajectory.append(accept)
        stable_trajectory.append(stable)
        previous = tokens

    assert torch.stack(token_trajectory).squeeze(1).tolist() == [[0,
                                                                  0], [0, 1],
                                                                 [1, 1]]
    assert torch.stack(accept_trajectory).squeeze(1).tolist() == [
        [True, True],
        [True, True],
        [True, True],
    ]
    assert torch.stack(stable_trajectory).squeeze(1).tolist() == [[1,
                                                                   1], [2, 1],
                                                                  [1, 2]]
