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

from tensorrt_edgellm.config import (QUANT_FP8, QUANT_FP16, QUANT_NVFP4,
                                     ModelConfig, QuantConfig,
                                     make_mtp_draft_config, module_quant_type)
from tensorrt_edgellm.model import AutoModel, _mtp_key_remap
from tensorrt_edgellm.models.linear import (FP8Linear, NVFP4LinearMethod,
                                            make_linear)
from tensorrt_edgellm.onnx import export as onnx_export
from tensorrt_edgellm.scripts import export as export_script
from tensorrt_edgellm.scripts.export import _export_llm, _export_mtp_draft


def _mixed_mtp_base_config() -> ModelConfig:
    return ModelConfig(
        model_type="qwen3_5_text",
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        head_dim=8,
        rms_norm_eps=1e-6,
        vocab_size=64,
        rope_theta=10000.0,
        max_position_embeddings=128,
        default_attention_scale=8**-0.5,
        quant=QuantConfig(
            quant_type=QUANT_FP8,
            group_size=1,
            excluded=["mtp.fc", "mtp.layers.0.self_attn.q_proj"],
            layer_overrides={
                "lm_head": QUANT_NVFP4,
                "model.layers.0.mlp.down_proj": QUANT_FP8,
            },
            is_mixed_precision=True,
        ),
        mtp_num_hidden_layers=1,
    )


def test_unquantized_mtp_body_preserves_quantized_lm_head():
    base = _mixed_mtp_base_config()
    base.mtp_tree_base = True
    draft = make_mtp_draft_config(base)

    assert module_quant_type("fc", draft) == QUANT_FP16
    assert module_quant_type("layers.0.self_attn.q_proj", draft) == QUANT_FP16
    assert module_quant_type("lm_head", draft) == QUANT_NVFP4
    assert draft.quant.group_size == 16
    assert draft.mtp_tree_base


def test_mixed_precision_nvfp4_uses_format_block_size():
    config = _mixed_mtp_base_config()

    head = make_linear(config, 32, 64, module_name="lm_head")
    body = make_linear(config,
                       32,
                       64,
                       module_name="model.layers.0.mlp.down_proj")

    assert isinstance(head.quant_method, NVFP4LinearMethod)
    assert head.group_size == 16
    assert head.weight.shape == (64, 16)
    assert head.weight_scale.shape == (64, 2)
    assert isinstance(body, FP8Linear)


def test_mtp_key_remap_keeps_quantized_lm_head_sidecars():
    for key in ("lm_head.weight", "lm_head.weight_scale",
                "lm_head.weight_scale_2", "lm_head.input_scale"):
        assert _mtp_key_remap(key, tie_word_embeddings=False) == key

    assert _mtp_key_remap("model.layers.0.mlp.down_proj.weight",
                          tie_word_embeddings=False) is None


def test_mtp_draft_export_always_enables_tree_mode(monkeypatch, tmp_path):
    load_args = {}
    model = object()

    def fake_from_pretrained(*args, **kwargs):
        load_args.update(kwargs)
        return model

    monkeypatch.setattr(AutoModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(onnx_export, "export_onnx",
                        lambda *args, **kwargs: None)

    _export_mtp_draft("checkpoint", str(tmp_path))

    assert load_args["mtp_draft"]
    assert load_args["mtp_tree_base"]


def test_mtp_base_export_always_enables_tree_mode(monkeypatch, tmp_path):
    load_args = {}
    model = object()

    def fake_from_pretrained(*args, **kwargs):
        load_args.update(kwargs)
        return model

    monkeypatch.setattr(AutoModel, "from_pretrained", fake_from_pretrained)
    monkeypatch.setattr(onnx_export, "export_onnx",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(export_script, "_patch_multimodal_token_ids",
                        lambda *args, **kwargs: None)

    _export_llm("checkpoint", str(tmp_path), mtp_base=True)

    assert load_args["mtp_base"]
    assert load_args["mtp_tree_base"]
