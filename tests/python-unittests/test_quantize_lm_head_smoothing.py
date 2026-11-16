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
"""AWQ smoothing stays on the backbone, off a quantized lm_head."""

import modelopt.torch.quantization as mtq
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from tensorrt_edgellm.quantization.quantization_configs import (
    build_quant_config, split_lm_head_calibration)

_HIDDEN, _VOCAB = 128, 256


def _calibrate(model):
    torch.manual_seed(1)
    for _ in range(4):
        model(torch.randint(0, _VOCAB, (2, 16)))


def _tiny_llama() -> LlamaForCausalLM:
    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=_VOCAB,
                         hidden_size=_HIDDEN,
                         intermediate_size=2 * _HIDDEN,
                         num_hidden_layers=1,
                         num_attention_heads=4,
                         num_key_value_heads=4,
                         tie_word_embeddings=False)
    return LlamaForCausalLM(config).eval()


def test_awq_backbone_leaves_lm_head_unsmoothed():
    """The backbone gets a pre_quant_scale; the NVFP4 lm_head gets none and is
    still calibrated."""
    model = _tiny_llama()
    quant_cfg = build_quant_config("int4_awq", "nvfp4")

    lm_head_cfg = split_lm_head_calibration(quant_cfg, "nvfp4")
    mtq.quantize(model, quant_cfg, forward_loop=_calibrate)
    mtq.quantize(model, lm_head_cfg, forward_loop=_calibrate)

    layer = model.model.layers[0].mlp.down_proj
    assert layer.input_quantizer.pre_quant_scale is not None
    head = model.lm_head
    assert head.input_quantizer.pre_quant_scale is None
    assert head.weight_quantizer.is_enabled
    assert head.weight_quantizer.amax is not None
    assert layer.weight_quantizer.is_enabled
