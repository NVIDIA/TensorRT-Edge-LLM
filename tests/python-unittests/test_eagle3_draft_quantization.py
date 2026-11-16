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
"""Calibration model of the EAGLE3 draft quantizer."""

import json

import pytest
import torch
from transformers import AutoConfig

_HEAD_DIM = 16
_ROPE_THETA = 1.0e6


def test_calibration_rope_theta_follows_checkpoint(tmp_path):
    """The draft calibration model rotates with the checkpoint's top-level rope_theta, which transformers 5 moves into rope_parameters."""
    pytest.importorskip("modelopt.torch.quantization")
    from tensorrt_edgellm.quantization.models import eagle3_draft
    (tmp_path / "config.json").write_text(
        json.dumps({
            "architectures": ["LlamaForCausalLMEagle3"],
            "model_type": "llama",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": _HEAD_DIM,
            "rms_norm_eps": 1e-6,
            "vocab_size": 256,
            "draft_vocab_size": 256,
            "target_hidden_size": 64,
            "rope_theta": _ROPE_THETA,
        }))
    model = eagle3_draft.Eagle3DraftModel(
        AutoConfig.from_pretrained(str(tmp_path)))
    expected = 1.0 / _ROPE_THETA**(
        torch.arange(0, _HEAD_DIM, 2, dtype=torch.float32) / _HEAD_DIM)

    torch.testing.assert_close(model.rotary_emb.inv_freq, expected)
