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

import math

import torch

from tensorrt_edgellm.config import LAYER_ATTN, ModelConfig
from tensorrt_edgellm.models.eagle3.modeling_eagle3_draft import \
    Eagle3DraftModel
from tensorrt_edgellm.models.qwen3_5.modeling_qwen3_5_mtp import \
    Qwen3_5MtpDraftModel


def _config(model_type: str) -> ModelConfig:
    return ModelConfig(model_type=model_type,
                       hidden_size=16,
                       num_hidden_layers=1,
                       num_attention_heads=2,
                       num_key_value_heads=1,
                       intermediate_size=32,
                       head_dim=8,
                       rms_norm_eps=1e-6,
                       vocab_size=32,
                       rope_theta=10_000.0,
                       max_position_embeddings=4096,
                       default_attention_scale=1.0 / math.sqrt(8.0),
                       layer_types=[LAYER_ATTN],
                       attn_output_gate=model_type == "qwen3_5_text",
                       target_hidden_size=16,
                       eagle3_target_layer_ids=[0, 1, 2],
                       draft_vocab_size=32,
                       has_qk_norm=True)


def _assert_unified_draft_contract(spec):
    inputs = dict(zip(spec.input_names, spec.args))
    assert inputs["inputs_embeds"].ndim == 2
    assert inputs["rope_rotary_cos_sin"].ndim == 2
    assert inputs["hidden_states_input"].ndim == 2
    assert inputs["hidden_states_from_draft"].ndim == 2
    assert inputs["attention_position_ids"].ndim == 1
    assert inputs["tree_parent_ids"].ndim == 1
    assert inputs["tree_depths"].ndim == 1
    assert "context_lengths" not in inputs
    assert "kvcache_start_index" not in inputs
    assert "last_token_ids" not in inputs
    assert inputs["context_sequence_count_carrier"].shape == inputs[
        "query_lengths"].shape
    assert inputs["execution_phase_marker"].shape == (2, )
    shapes = dict(zip(spec.input_names, spec.dynamic_shapes))
    context_dim = shapes["context_sequence_count_carrier"][0]
    assert context_dim != shapes["query_lengths"][0]
    assert context_dim.min == 0
    outputs = spec.wrapped(*spec.args)
    assert outputs[0].ndim == 2
    assert outputs[1].ndim == 2
    torch.export.export(spec.wrapped,
                        spec.args,
                        dynamic_shapes=spec.dynamic_shapes,
                        strict=False)


def test_eagle3_draft_uses_unified_token_major_contract():
    _assert_unified_draft_contract(
        Eagle3DraftModel(_config("qwen3")).onnx_export_spec())


def test_qwen35_mtp_draft_uses_unified_token_major_contract():
    _assert_unified_draft_contract(
        Qwen3_5MtpDraftModel(_config("qwen3_5_text")).onnx_export_spec())
