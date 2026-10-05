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

import math

import onnx
import pytest
import torch
from torch import nn

from tensorrt_edgellm.config import LAYER_ATTN, ModelConfig
from tensorrt_edgellm.models.qwen3_omni.modeling_qwen3_omni_text import (
    Qwen3OmniDenseTransformer, Qwen3OmniLanguageModel)
from tensorrt_edgellm.onnx.export import _export_model


def _config(accept_hidden_layer: int) -> ModelConfig:
    config = ModelConfig(model_type="qwen3",
                         hidden_size=16,
                         num_hidden_layers=2,
                         num_attention_heads=2,
                         num_key_value_heads=1,
                         intermediate_size=32,
                         head_dim=8,
                         rms_norm_eps=1e-6,
                         vocab_size=32,
                         rope_theta=10_000.0,
                         max_position_embeddings=4096,
                         default_attention_scale=1.0 / math.sqrt(8.0),
                         layer_types=[LAYER_ATTN, LAYER_ATTN])
    config.accept_hidden_layer = accept_hidden_layer
    return config


class _AddLayer(nn.Module):

    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment

    def forward_ragged(self, hidden_states, **_kwargs):
        return hidden_states + self.increment, None


class _AddNorm(nn.Module):

    def forward(self, hidden_states):
        return hidden_states + 100.0


def _run_transformer(accept_hidden_layer: int):
    transformer = Qwen3OmniDenseTransformer(_config(accept_hidden_layer))
    transformer.layers = nn.ModuleList([_AddLayer(1.0), _AddLayer(1.0)])
    transformer.norm = _AddNorm()
    inputs = torch.zeros((2, 16), dtype=torch.float16)
    deepstack = (torch.full_like(inputs, 10.0), )
    normed, _ = transformer.forward_ragged(inputs, (None, None),
                                           rope_rotary_cos_sin=None,
                                           positions=None,
                                           query_start_offsets=None,
                                           query_lengths=None,
                                           past_lengths=None,
                                           attention_sequence_lengths=None,
                                           state_indices=None,
                                           execution_phase_marker=None,
                                           context_sequence_count_carrier=None,
                                           kv_page_table=None,
                                           deepstack_embeds=deepstack,
                                           target_layer_ids=[0])
    return transformer, normed


def test_dense_omni_ragged_captures_accept_hidden_after_deepstack():
    transformer, normed = _run_transformer(accept_hidden_layer=1)

    torch.testing.assert_close(transformer.target_hidden_concat,
                               torch.full_like(normed, 1.0))
    torch.testing.assert_close(transformer.emitted_hidden_states,
                               torch.full_like(normed, 11.0))
    torch.testing.assert_close(transformer.last_pre_norm_hidden_states,
                               transformer.emitted_hidden_states)
    torch.testing.assert_close(normed, torch.full_like(normed, 112.0))


@pytest.mark.parametrize("accept_hidden_layer", [-1, 0, 3])
def test_dense_omni_ragged_invalid_accept_layer_falls_back_to_final_norm(
        accept_hidden_layer):
    transformer, normed = _run_transformer(accept_hidden_layer)

    torch.testing.assert_close(transformer.emitted_hidden_states, normed)


def test_dense_omni_ragged_onnx_hidden_carrier_is_rank_two(tmp_path):
    output = tmp_path / "qwen3-omni-dense-ragged.onnx"

    _export_model(Qwen3OmniLanguageModel(_config(accept_hidden_layer=1)),
                  str(output),
                  optimize=False)

    hidden_output = onnx.load(str(output),
                              load_external_data=False).graph.output[1]
    assert len(hidden_output.type.tensor_type.shape.dim) == 2
