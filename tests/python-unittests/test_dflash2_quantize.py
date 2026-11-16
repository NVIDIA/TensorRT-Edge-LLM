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

from types import SimpleNamespace

import torch

from tensorrt_edgellm.quantization.models import dflash_draft
from tensorrt_edgellm.quantization.quantization_configs import \
    build_quant_config


def _config():
    return SimpleNamespace(
        architectures=["DFlash2DraftModel"],
        model_type="qwen3",
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=1,
        vocab_size=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        dflash_config={
            "target_layer_ids": [0],
            "block_size": 4,
            "mask_token_id": 31,
            "conv_kernel_size": 2,
            "conv_group_size": 4,
            "selector_rank": 4,
            "selector_top_k": 2,
        },
    )


def test_dflash2_calibration_model_materializes_all_trained_modules():
    model = dflash_draft.DFlashCalibDraftModel(_config())

    assert model.is_dflash2
    assert model.layers[0].self_attn.is_causal is False
    assert model.layers[0].attention_conv.kernel_projection.out_features == 8
    assert model.layers[0].mlp_conv.base_kernel.shape == (2, 2, 8)
    assert model.candidate_selector.hidden_projection.out_features == 4
    assert model.candidate_selector.predecessor_codebook.shape == (32, 4)


def test_dflash2_grouped_conv_resets_history_at_each_batch_block():
    conv = dflash_draft.DFlash2CalibGroupedConv(8, 4, 2, 4)
    with torch.no_grad():
        conv.base_kernel.zero_()
        conv.base_kernel[:, 1].fill_(1.0)
        conv.kernel_projection.weight.zero_()
    hidden = torch.arange(64, dtype=torch.float32).reshape(2, 4, 8)

    pre, post = conv.prepare(hidden)

    assert torch.equal(pre[:, 0], torch.zeros_like(pre[:, 0]))
    assert torch.equal(pre[:, 1:], hidden[:, :-1])
    assert post.shape == (2, 4, 2, 2)


def test_dflash2_post_conv_fuses_fp32_residual_before_dtype_narrowing():
    conv = dflash_draft.DFlash2CalibGroupedConv(8, 4, 2, 4).half()
    with torch.no_grad():
        conv.base_kernel.zero_()
        conv.base_kernel[1, 0].fill_(1.0)
        conv.kernel_projection.weight.zero_()
    hidden = torch.full((1, 4, 8), 2.0, dtype=torch.float16)
    coefficients = torch.zeros((1, 4, 2, 2), dtype=torch.float16)
    residual = torch.full((1, 4, 8), 70000.0, dtype=torch.float32)

    output = conv.finish(hidden, coefficients, residual)

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.full_like(output, 70002.0))


def test_dflash_sensitive_projection_exclusions_support_modelopt_schema():
    cfg = build_quant_config("nvfp4", lm_head_quantization="nvfp4")

    dflash_draft._disable_dflash_fc_quantization(cfg)

    section = cfg["quant_cfg"]
    names = (
        "fc.input_quantizer",
        "fc.weight_quantizer",
        "fc.output_quantizer",
        "candidate_selector.hidden_projection.input_quantizer",
        "candidate_selector.hidden_projection.weight_quantizer",
        "candidate_selector.hidden_projection.output_quantizer",
    )
    if isinstance(section, dict):
        assert all(section[name]["enable"] is False for name in names)
    else:
        tail = section[-len(names):]
        assert [entry["quantizer_name"] for entry in tail] == list(names)
        assert all(entry["enable"] is False for entry in tail)


def test_dflash2_calibration_keeps_residual_stream_fp32():
    model = dflash_draft.DFlashCalibDraftModel(_config()).half()
    residual_dtypes = []
    handle = model.layers[0].register_forward_hook(
        lambda _module, _inputs, output: residual_dtypes.append(output.dtype))
    proposal = torch.randn(1, 4, 8, dtype=torch.float16)
    target_hidden = torch.randn(1, 4, 8, dtype=torch.float16)

    output = model(proposal, target_hidden)
    handle.remove()

    assert residual_dtypes == [torch.float32]
    assert output.dtype == torch.float16
    assert torch.isfinite(output).all()
