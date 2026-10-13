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
"""EAGLE3 feedback must match the hidden-state indices used in calibration."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_edgellm.checkpoint.checkpoint_utils import \
    build_runtime_llm_config_dict
from tensorrt_edgellm.config import ModelConfig
from tensorrt_edgellm.models.default.modeling_default import CausalLM


class _HiddenStates(torch.nn.Module):

    def __init__(self, num_layers):
        super().__init__()
        self.all_hidden_states = tuple(
            torch.full((2, 3, 16), float(index), dtype=torch.float16)
            for index in range(num_layers + 1))

    def forward(self, *args, **kwargs):
        if not kwargs['output_hidden_states']:
            pytest.fail('EAGLE3 feedback requires all hidden states')
        return self.all_hidden_states[-1], (), self.all_hidden_states


@pytest.mark.parametrize('num_layers', [8, 32, 36])
@pytest.mark.parametrize('ragged', [False, True])
def test_eagle_feedback_matches_calibration_layers(num_layers, ragged):
    # Given distinct layer outputs, when producing dense or ragged EAGLE3 feedback,
    # then the high feature must match HF hidden_states[-4].
    config = ModelConfig(model_type='qwen3',
                         hidden_size=16,
                         num_hidden_layers=1,
                         num_attention_heads=2,
                         num_key_value_heads=1,
                         intermediate_size=32,
                         head_dim=8,
                         rms_norm_eps=1e-6,
                         vocab_size=32,
                         rope_theta=10000.0,
                         max_position_embeddings=128,
                         default_attention_scale=8**-0.5,
                         eagle_base=True)
    model = CausalLM(config)
    model.model = _HiddenStates(num_layers)
    states = model.model.all_hidden_states
    # HF includes the embedding input and the final normalized output.
    expected = torch.cat(
        [states[2], states[(len(states) - 1) // 2], states[-4]],
        dim=-1).to(torch.float16)
    if ragged:
        actual = model._ragged_emitted_hidden()
        expected = expected.reshape(-1, expected.shape[-1])
    else:
        _, actual, _ = model(states[0], (), torch.empty(0), torch.empty(0),
                             torch.empty(0), torch.empty(0),
                             torch.tensor([[0], [0]], dtype=torch.int64))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('num_layers', [8, 32, 36])
@pytest.mark.parametrize('target_layers', [(), (1, 3, 5)])
def test_experimental_feedback_matches_hf_layers(monkeypatch, num_layers,
                                                 target_layers):
    from experimental.builder.ops.functional import speculative

    # The experimental frontend stores block outputs, without the embedding input.
    hf_states = _HiddenStates(num_layers).all_hidden_states
    block_states = hf_states[1:]
    config = SimpleNamespace(spec_decode_type='eagle3',
                             eagle3_target_layer_ids=target_layers)
    monkeypatch.setattr(speculative.F, 'concatenate',
                        lambda values, axis: torch.cat(values, dim=axis))
    actual = speculative.hidden_state_feedback(block_states[-1], block_states,
                                               config)
    layer_ids = target_layers or (2, num_layers // 2, num_layers - 3)
    expected = torch.cat([hf_states[index] for index in layer_ids], dim=-1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('num_layers', [8, 32, 36])
@pytest.mark.parametrize('target_layers', [(), (5, 1, 3)])
@pytest.mark.parametrize('component', ['base', 'draft'])
def test_experimental_pairing_feedback_layers(monkeypatch, num_layers,
                                              target_layers, component):
    from experimental.builder.models.eagle3 import configuration

    raw_component = {'target_layer_ids': list(target_layers)}
    config = SimpleNamespace(num_hidden_layers=num_layers,
                             raw_component=raw_component)
    if component == 'base':
        bundle = SimpleNamespace(component_dict=lambda _: raw_component)
        monkeypatch.setattr(configuration.BundleConfig, 'from_pretrained',
                            lambda _: bundle)
        configuration.configure_base(config, paired_draft_dir='draft')
    else:
        target = SimpleNamespace(num_hidden_layers=num_layers, hidden_size=16)
        configuration.configure_draft(config, paired_target=target)

    hf_indices = list(range(num_layers + 1))
    expected = list(target_layers) or [
        hf_indices[2], hf_indices[num_layers // 2], hf_indices[-4]
    ]
    if config.eagle3_target_layer_ids != expected:
        pytest.fail(
            f'Expected feedback layers {expected}, got {config.eagle3_target_layer_ids}'
        )


@pytest.mark.parametrize('num_layers', [8, 32, 36])
@pytest.mark.parametrize('target_layers', [(), (1, 3, 5)])
def test_runtime_config_records_feedback_layers(num_layers, target_layers):
    config = ModelConfig(model_type='qwen3',
                         hidden_size=16,
                         num_hidden_layers=num_layers,
                         num_attention_heads=2,
                         num_key_value_heads=1,
                         intermediate_size=32,
                         head_dim=8,
                         rms_norm_eps=1e-6,
                         vocab_size=32,
                         rope_theta=10000.0,
                         max_position_embeddings=128,
                         default_attention_scale=8**-0.5,
                         eagle_base=True,
                         eagle3_target_layer_ids=target_layers)
    actual = build_runtime_llm_config_dict(SimpleNamespace(config=config))
    expected = list(target_layers or (2, num_layers // 2, num_layers - 3))
    if actual['eagle_hidden_state_layers'] != expected:
        pytest.fail(
            f'Expected feedback layers {expected}, got {actual["eagle_hidden_state_layers"]}'
        )
