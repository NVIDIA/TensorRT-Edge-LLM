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
"""Qwen4-Exp (Qwen3.8-Flash-Next) checkpoint-direct graph."""

import logging
from typing import Dict

import tensorrt as trt

from ...core import config
from ...ops import (GatedDeltaNet, GatedMLP, Linear, Module, NetworkModule,
                    RaggedDecoderInputs, add_ragged_decoder_inputs)
from ...ops import functional as F
from .modeling_qwen4_exp_ple import Qwen4ExpPle
from .modeling_qwen4_exp_qsa import Qwen4ExpQsaAttention
from .modeling_qwen4_exp_residual import Qwen4ExpGatedResidual
from .modeling_qwen4_exp_sparse_moe import Qwen4ExpSparseMoeBlock

LOGGER = logging.getLogger("builder.qwen4_exp")

__all__ = ["Qwen4ExpDecoderLayer", "Qwen4ExpForCausalLM"]

# The runtime binds all of these on every speculative engine; QSA reads only
# the tree and the layout marker.
_SPECULATIVE_INPUTS = (
    ("attention_position_ids", (-1, )),
    ("packed_attention_mask", (-1, -1)),
    ("tree_parent_ids", (-1, )),
    ("tree_depths", (-1, )),
    ("spec_verify_layout", (-1, )),
)


def speculative_inputs(add_input, engine_role: str) -> Dict[str, object]:
    """Declare the INT32 speculative inputs; ``None`` for a vanilla engine."""
    speculative = engine_role in ("base", "draft")
    return {
        name: add_input(name, trt.int32, shape) if speculative else None
        for name, shape in _SPECULATIVE_INPUTS
    }


class Qwen4ExpDecoderLayer(Module):
    """One Qwen4-Exp block pair over the multi-stream residual.

    Both the mixer and the feed-forward block are wrapped by their own
    read/write gate pair, so neither ever sees more than ``hidden_size``.
    """

    def __init__(self, ctx, prefix: str, layer_type: str,
                 has_ple: bool) -> None:
        super().__init__(ctx, prefix)
        self.layer_type = layer_type
        if layer_type == config.LAYER_GDN:
            self.mixer = GatedDeltaNet(ctx, self.key("linear_attn"))
        elif layer_type == config.LAYER_ATTN:
            self.mixer = Qwen4ExpQsaAttention(ctx, self.key("self_attn"))
        else:
            raise ValueError(
                f"unsupported Qwen4-Exp layer type {layer_type!r}")
        self.mlp = (Qwen4ExpSparseMoeBlock(ctx, self.key("mlp"))
                    if ctx.cfg.num_experts > 0 else GatedMLP(
                        ctx, self.key("mlp")))
        self.ple = Qwen4ExpPle(ctx, self.key("ple")) if has_ple else None
        self.attn_hyper_connection = Qwen4ExpGatedResidual(
            ctx, self.key("attn_hyper_connection"))
        self.mlp_hyper_connection = Qwen4ExpGatedResidual(
            ctx, self.key("mlp_hyper_connection"))

    def forward(self,
                hyper_states,
                ragged: RaggedDecoderInputs,
                ple_cache_pool=None,
                ple_row_map=None,
                ngram_conv_state=None,
                past_key_value=None,
                rope=None,
                conv_state=None,
                recurrent_state=None,
                tree_parent_ids=None,
                tree_depths=None):
        ngram_conv_state_out = None
        ngram_intermediate = None
        if self.ple is not None:
            injection, ngram_conv_state_out, ngram_intermediate = self.ple(
                hyper_states,
                ple_cache_pool,
                ple_row_map,
                ngram_conv_state,
                ragged,
                tree_parent_ids=tree_parent_ids,
                tree_depths=tree_depths,
                collect_intermediate=self.cfg.engine_role == "base")
            # A bare addition: unlike every other write to the streams, the
            # per-stream gate is already inside the injection.
            hyper_states = hyper_states + injection

        mixed, injection_weights = self.attn_hyper_connection(hyper_states)
        if self.layer_type == config.LAYER_GDN:
            mixer_out, conv_out, recurrent_out, conv_steps, recurrent_steps = (
                self.mixer(
                    mixed,
                    conv_state,
                    recurrent_state,
                    ragged,
                    tree_parent_ids=tree_parent_ids,
                    tree_depths=tree_depths,
                    collect_intermediate=self.cfg.engine_role == "base"))
            present = (conv_out, recurrent_out, conv_steps, recurrent_steps)
        else:
            mixer_out, present_kv = self.mixer(mixed, past_key_value, rope,
                                               ragged, tree_parent_ids,
                                               tree_depths)
            present = (present_kv, )
        hyper_states = self.attn_hyper_connection.combine(
            hyper_states, mixer_out, injection_weights)

        mixed, injection_weights = self.mlp_hyper_connection(hyper_states)
        hyper_states = self.mlp_hyper_connection.combine(
            hyper_states, self.mlp(mixed), injection_weights)
        return hyper_states, present, (ngram_conv_state_out,
                                       ngram_intermediate)


class Qwen4ExpForCausalLM(NetworkModule):
    """Qwen4-Exp text model: tile into the streams, run layers, fold back.

    There is no final norm member: ``hyper_connection_mixer`` is both the final
    normalization and the stream reduction, and the checkpoint carries no
    separate final-norm weight.
    """

    def __init__(self, ctx) -> None:
        super().__init__(ctx)
        cfg = ctx.cfg
        if cfg.engine_role not in ("llm", "base"):
            raise NotImplementedError(
                f"Qwen4-Exp has no {cfg.engine_role!r} engine")
        if cfg.engine_role == "base" and cfg.spec_decode_type != "mtp":
            raise NotImplementedError(
                "Qwen4-Exp speculative base supports only --spec-type mtp, "
                f"not {cfg.spec_decode_type!r}")
        ple_layers = set(cfg.ngram_layers)
        self.layers = [
            Qwen4ExpDecoderLayer(ctx, f"model.layers.{index}", layer_type,
                                 index in ple_layers)
            for index, layer_type in enumerate(cfg.layer_types)
        ]
        self.hyper_connection_mixer = Qwen4ExpGatedResidual(
            ctx, "model.hyper_connection_mixer", use_combine=False)
        self.lm_head = Linear(ctx, "lm_head")

    def input_tensors(self) -> Dict[str, object]:
        cfg = self.cfg
        gdn = cfg.gdn_cfg
        kv_dtype = (trt.DataType.FP8
                    if cfg.kv_cache_quant == "fp8" else trt.float16)
        result = {
            "inputs_embeds":
            self.add_input("inputs_embeds", trt.float16,
                           (-1, cfg.hidden_size)),
            # The paged pool is widened by the indexer state, which rides as a
            # tail on each row rather than in a cache of its own.
            "past_key_values": [
                self.add_input(f"past_key_values_{index}", kv_dtype,
                               (2, -1, F.KV_PAGE_SIZE, cfg.num_key_value_heads,
                                cfg.qsa_pool_head_dim))
                for index in range(cfg.num_attn_layers)
            ],
            # Declared even for a stack with no sparse-attention layer: the
            # runtime binds the rotary table unconditionally.
            "rope":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, cfg.rotary_dim)),
            "conv_states": [
                self.add_input(f"conv_state_{index}", trt.float16,
                               (-1, gdn.conv_dim, gdn.conv_kernel))
                for index in range(cfg.num_gdn_layers)
            ],
            "recurrent_states": [
                self.add_input(f"recurrent_state_{index}", trt.float32,
                               (-1, gdn.num_value_heads, gdn.key_head_dim,
                                gdn.value_head_dim))
                for index in range(cfg.num_gdn_layers)
            ],
            "ple_cache_pool":
            None,
            "ple_row_map":
            None,
            "ngram_conv_states": [],
        }
        if cfg.ngram_layers:
            # The runtime owns the pool; slot 0 is the padding row, so its capacity is
            # row_count + 1 and stays dynamic here.
            result["ple_cache_pool"] = self.add_input(
                "ple_cache_pool", trt.fp8,
                (-1, cfg.ple_embed_dim // cfg.ngram_heads))
            result["ple_row_map"] = self.add_input("ple_row_map", trt.int32,
                                                   (-1, cfg.ngram_heads))
            result["ngram_conv_states"] = [
                self.add_input(
                    f"ngram_conv_state_{index}", trt.float16,
                    (-1, cfg.hc_hidden_size, cfg.ngram_conv_state_len))
                for index in range(len(cfg.ngram_layers))
            ]
        result.update(add_ragged_decoder_inputs(self.add_input).as_dict())
        result.update(speculative_inputs(self.add_input, cfg.engine_role))
        return result

    def forward(self, **io):
        cfg = self.cfg
        ragged = RaggedDecoderInputs.from_dict(io)
        hyper_states = F.concatenate([io["inputs_embeds"]] * cfg.hc_count, 1)
        present_kv = []
        present_conv = []
        present_recurrent = []
        intermediate_conv = []
        intermediate_recurrent = []
        present_ple_conv = []
        intermediate_ple_conv = []
        attention_index = 0
        state_index = 0
        ple_index = 0
        for layer_index, layer in enumerate(self.layers):
            LOGGER.info("building layer %d/%d", layer_index + 1,
                        len(self.layers))
            has_ple = layer.ple is not None
            hyper_states, present, ple_states = layer(
                hyper_states,
                ragged,
                ple_cache_pool=io["ple_cache_pool"],
                ple_row_map=io["ple_row_map"],
                ngram_conv_state=(io["ngram_conv_states"][ple_index]
                                  if has_ple else None),
                past_key_value=(io["past_key_values"][attention_index]
                                if layer.layer_type == config.LAYER_ATTN else
                                None),
                rope=io["rope"],
                conv_state=(io["conv_states"][state_index]
                            if layer.layer_type == config.LAYER_GDN else None),
                recurrent_state=(io["recurrent_states"][state_index]
                                 if layer.layer_type == config.LAYER_GDN else
                                 None),
                tree_parent_ids=io["tree_parent_ids"],
                tree_depths=io["tree_depths"])
            if has_ple:
                present_ple_conv.append(ple_states[0])
                if ple_states[1] is not None:
                    intermediate_ple_conv.append(ple_states[1])
                ple_index += 1
            if layer.layer_type == config.LAYER_GDN:
                present_conv.append(present[0])
                present_recurrent.append(present[1])
                if present[2] is not None:
                    intermediate_conv.append(present[2])
                    intermediate_recurrent.append(present[3])
                state_index += 1
            else:
                present_kv.append(present[0])
                attention_index += 1

        hidden_states = self.hyper_connection_mixer(hyper_states)
        selected = F.gather_token_rows(hidden_states, ragged.logits_indices)
        outputs = {"logits": F.cast(self.lm_head(selected), trt.float32)}
        if cfg.engine_role == "base":
            # The MTP draft owns a multi-stream bus of its own and resumes this
            # one, so what it needs is the state the mixer consumed, not the
            # single stream the mixer produced.
            outputs["hidden_states"] = hyper_states
        for index, tensor in enumerate(present_kv):
            outputs[f"present_key_values_{index}"] = tensor
        for index, tensor in enumerate(present_conv):
            outputs[f"present_conv_state_{index}"] = tensor
        for index, tensor in enumerate(present_recurrent):
            outputs[f"present_recurrent_state_{index}"] = tensor
        for index, tensor in enumerate(present_ple_conv):
            outputs[f"present_ngram_conv_state_{index}"] = tensor
        for index, tensor in enumerate(intermediate_conv):
            outputs[f"intermediate_conv_state_{index}"] = tensor
        for index, tensor in enumerate(intermediate_recurrent):
            outputs[f"intermediate_recurrent_state_{index}"] = tensor
        for index, tensor in enumerate(intermediate_ple_conv):
            outputs[f"intermediate_ngram_conv_state_{index}"] = tensor
        return outputs
