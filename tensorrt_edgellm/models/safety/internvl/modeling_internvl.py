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
"""Safety decoder export with batched inputs, packed QKV and contiguous KV storage."""

import dataclasses
import math

import torch
from torch import nn
from torch.nn import functional as F

from .... import config as config_module
from ....checkpoint import checkpoint_utils, loader
from ... import linear


@torch.library.custom_op("trt::safety_attention_plugin", mutates_args=())
def safety_attention_plugin(
    qkv: torch.Tensor,
    past_key_value: torch.Tensor,
    context_lengths: torch.Tensor,
    rope_rotary_cos_sin: torch.Tensor,
    kvcache_start_index: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Export placeholder for the five-input contiguous AttentionPlugin ABI."""
    return _safety_attention_outputs(qkv, past_key_value, num_q_heads,
                                     num_kv_heads, head_size)


def _safety_attention_outputs(qkv, past_key_value, num_q_heads, num_kv_heads,
                              head_size):
    if qkv.ndim != 3 or past_key_value.ndim != 5:
        raise ValueError("Safety attention requires rank-3 QKV and rank-5 KV")
    torch._check(qkv.shape[2] == (num_q_heads + 2 * num_kv_heads) * head_size)
    torch._check(past_key_value.shape[0] == qkv.shape[0])
    torch._check(past_key_value.shape[1] == 2)
    torch._check(past_key_value.shape[2] == num_kv_heads)
    torch._check(past_key_value.shape[4] == head_size)
    if qkv.dtype != torch.float16 or past_key_value.dtype != torch.float16:
        raise ValueError("Safety attention requires FP16 QKV and KV")
    return (qkv.new_empty(qkv.shape[0], qkv.shape[1], num_q_heads,
                          head_size), torch.empty_like(past_key_value))


@safety_attention_plugin.register_fake
def _safety_attention_fake(qkv, past_key_value, context_lengths,
                           rope_rotary_cos_sin, kvcache_start_index,
                           num_q_heads, num_kv_heads, head_size):
    return _safety_attention_outputs(qkv, past_key_value, num_q_heads,
                                     num_kv_heads, head_size)


def validate_safety_model(model):
    """Reject features not represented by the safety vanilla decoder ABI."""
    if type(model) is not SafetyCausalLM:
        raise ValueError("Expected a Safety InternVL decoder")
    validate_safety_config(model.config)


def validate_safety_config(config):
    """Validate the supported InternVL text configuration before constructing layers."""
    if config.num_hidden_layers < 1:
        raise ValueError("Decoder layer count must be positive")
    if config.model_type not in ("qwen2", "internvl_chat"):
        raise ValueError(
            f"Unsupported InternVL text model type: {config.model_type!r}")
    if checkpoint_utils._determine_spec_decode_type(config) != "none":
        raise ValueError("Expected a vanilla InternVL checkpoint")
    if config.mapping.tp_size != 1:
        raise ValueError("InternVL Safety export requires a single device")
    if (config.quant.quant_type not in ("fp16", "fp8")
            or any(precision not in ("fp16", "fp8")
                   for precision in config.quant.layer_overrides.values())
            or config.quant.kv_cache_quant):
        raise ValueError(
            "InternVL Safety export requires FP16/FP8 linear layers and FP16 KV"
        )
    if (config.has_qk_norm or config.has_value_norm or config.attention_k_eq_v
            or config.attention_sink_bias or config.num_kv_shared_layers
            or config.use_vision_bidirectional_attention
            or config.attn_output_gate or config.sliding_window_size > 0 or
            not math.isclose(config.attention_scaling, config.head_dim**-0.5)):
        raise ValueError("Unsupported InternVL attention configuration")
    if (config.num_deepstack_features or config.num_experts
            or config.hidden_activation != "silu"
            or config.final_logit_softcapping is not None
            or config.output_multiplier != 1.0):
        raise ValueError("Unsupported InternVL decoder configuration")


@dataclasses.dataclass
class OnnxSpec:
    wrapped: nn.Module
    args: tuple
    input_names: list[str]
    output_names: list[str]
    dynamic_shapes: list


class RMSNorm(nn.Module):
    """FP32 variance with FP16 input and output."""

    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size,
                                              dtype=torch.float16))
        self.variance_epsilon = eps

    def forward(self, hidden):
        dtype = hidden.dtype
        values = hidden.float()
        variance = values.pow(2).mean(-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight.to(dtype) * normalized.to(dtype)


class SafetyAttention(nn.Module):

    def __init__(self, config, layer_index):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        prefix = f"layers.{layer_index}.self_attn"
        self.q_proj = linear.make_linear(config,
                                         config.hidden_size,
                                         self.num_heads * self.head_dim,
                                         bias=config.attention_bias,
                                         module_name=f"{prefix}.q_proj")
        self.k_proj = linear.make_linear(config,
                                         config.hidden_size,
                                         self.num_kv_heads * self.head_dim,
                                         bias=config.attention_bias,
                                         module_name=f"{prefix}.k_proj")
        self.v_proj = linear.make_linear(config,
                                         config.hidden_size,
                                         self.num_kv_heads * self.head_dim,
                                         bias=config.attention_bias,
                                         module_name=f"{prefix}.v_proj")
        self.o_proj = linear.make_linear(config,
                                         self.num_heads * self.head_dim,
                                         config.hidden_size,
                                         module_name=f"{prefix}.o_proj")

    def forward(self, hidden, past_key_value, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index):
        qkv = torch.cat(
            (self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)),
            dim=-1)
        output, present = safety_attention_plugin(
            qkv, past_key_value, context_lengths, rope_rotary_cos_sin,
            kvcache_start_index, self.num_heads, self.num_kv_heads,
            self.head_dim)
        return self.o_proj(output.flatten(2)), present


class SafetyMLP(nn.Module):

    def __init__(self, config, layer_index):
        super().__init__()
        prefix = f"layers.{layer_index}.mlp"
        self.gate_proj = linear.make_linear(config,
                                            config.hidden_size,
                                            config.intermediate_size,
                                            module_name=f"{prefix}.gate_proj")
        self.up_proj = linear.make_linear(config,
                                          config.hidden_size,
                                          config.intermediate_size,
                                          module_name=f"{prefix}.up_proj")
        self.down_proj = linear.make_linear(config,
                                            config.intermediate_size,
                                            config.hidden_size,
                                            module_name=f"{prefix}.down_proj")

    def forward(self, hidden):
        return self.down_proj(
            F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class SafetyDecoderLayer(nn.Module):

    def __init__(self, config, layer_index):
        super().__init__()
        self.self_attn = SafetyAttention(config, layer_index)
        self.mlp = SafetyMLP(config, layer_index)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                config.rms_norm_eps)

    def forward(self, hidden, past_key_value, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index):
        output, present = self.self_attn(self.input_layernorm(hidden),
                                         past_key_value, rope_rotary_cos_sin,
                                         context_lengths, kvcache_start_index)
        hidden = hidden + output
        return hidden + self.mlp(
            self.post_attention_layernorm(hidden)), present


class SafetyTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            SafetyDecoderLayer(config, index)
            for index in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, hidden, past_key_values, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index):
        present = []
        for index, layer in enumerate(self.layers):
            hidden, kv = layer(hidden, past_key_values[index],
                               rope_rotary_cos_sin, context_lengths,
                               kvcache_start_index)
            present.append(kv)
        return self.norm(hidden), present


class SafetyCausalLM(nn.Module):
    """InternVL text model with checkpoint-compatible names and contiguous KV."""

    def __init__(self, config):
        super().__init__()
        validate_safety_config(config)
        self.config = config
        self.model = SafetyTransformer(config)
        self.lm_head = linear.make_linear(config,
                                          config.hidden_size,
                                          config.vocab_size,
                                          module_name="lm_head")

    def tie_weights(self):
        if self.config.tie_word_embeddings and isinstance(
                self.lm_head, linear.FP16Linear):
            self.lm_head.weight = nn.Parameter(
                self.model.embed_tokens.weight.detach().clone(),
                requires_grad=False)

    def forward(self, inputs_embeds, past_key_values, rope_rotary_cos_sin,
                context_lengths, kvcache_start_index, last_token_ids):
        hidden, present = self.model(inputs_embeds, past_key_values,
                                     rope_rotary_cos_sin, context_lengths,
                                     kvcache_start_index)
        selected = torch.ops.trt.gather_nd(hidden, last_token_ids)
        return (self.lm_head(selected).float(), *present)

    def onnx_export_spec(self):
        return safety_onnx_export_spec(self)


def load_safety_model_config(model_dir):
    return config_module.ModelConfig.from_pretrained(
        model_dir, default_attention_scale=lambda head_dim: head_dim**-0.5)


def load_safety_llm(model_dir,
                    *,
                    num_decoder_layers=None,
                    reduced_vocab_dir=None):
    """Load the Safety model directly from checkpoint tensors."""
    config = load_safety_model_config(model_dir)
    if num_decoder_layers is not None:
        if not 1 <= num_decoder_layers <= config.num_hidden_layers:
            raise ValueError(
                "--num-decoder-layer exceeds the checkpoint layer count")
        config = dataclasses.replace(config,
                                     num_hidden_layers=num_decoder_layers)
    model = SafetyCausalLM(config).half().eval()
    loader.load_weights(model, model_dir, device="cpu", mapping=config.mapping)
    if reduced_vocab_dir:
        from ....vocab_reduction.onnx_export import \
            apply_reduced_vocab_from_dir
        apply_reduced_vocab_from_dir(model, reduced_vocab_dir)
    return model


def safety_onnx_export_spec(model):
    """Create the batched Safety decoder export specification."""
    validate_safety_model(model)
    config = model.config
    device = next(model.parameters()).device
    batch_size, seq_len, capacity, rope_len = 2, 3, 8, 16
    kv_names = [
        f"past_key_values_{i}" for i in range(config.num_hidden_layers)
    ]
    names = [
        "inputs_embeds", *kv_names, "rope_rotary_cos_sin", "context_lengths",
        "kvcache_start_index", "last_token_ids"
    ]
    source = (
        f"def forward(self, {', '.join(names)}):\n"
        f"    return self.decoder(inputs_embeds, ({', '.join(kv_names)},), "
        "rope_rotary_cos_sin, context_lengths, kvcache_start_index, last_token_ids)\n"
    )
    namespace = {}
    exec(source, namespace)  # noqa: S102

    class FlatWrapper(nn.Module):

        def __init__(self):
            super().__init__()
            self.decoder = model

    FlatWrapper.forward = namespace["forward"]
    batch = torch.export.Dim("batch", min=1)
    seq = torch.export.Dim("seq_len", min=1)
    kv_capacity = torch.export.Dim("kv_capacity", min=1)
    rope_batch = torch.export.Dim("rope_batch", min=1)
    rope_length = torch.export.Dim("rope_len", min=1)
    args = [
        torch.zeros(batch_size,
                    seq_len,
                    config.hidden_size,
                    dtype=torch.float16,
                    device=device)
    ]
    shapes = [{0: batch, 1: seq}]
    for _ in kv_names:
        args.append(
            torch.zeros(batch_size,
                        2,
                        config.num_key_value_heads,
                        capacity,
                        config.head_dim,
                        dtype=torch.float16,
                        device=device))
        shapes.append({0: batch, 3: kv_capacity})
    args.extend(
        (torch.zeros(batch_size,
                     rope_len,
                     int(config.head_dim * config.partial_rotary_factor),
                     dtype=torch.float32,
                     device=device),
         torch.full((batch_size, ), seq_len, dtype=torch.int32, device=device),
         torch.zeros(batch_size, dtype=torch.int32, device=device),
         torch.full((batch_size, 1),
                    seq_len - 1,
                    dtype=torch.int64,
                    device=device)))
    shapes.extend(({
        0: rope_batch,
        1: rope_length
    }, {
        0: batch
    }, {
        0: batch
    }, {
        0: batch
    }))
    return OnnxSpec(FlatWrapper(), tuple(args), names, [
        "logits",
        *[f"present_key_values_{i}" for i in range(config.num_hidden_layers)]
    ], shapes)
