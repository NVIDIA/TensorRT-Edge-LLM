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

import glob
import json
import os
import tempfile

import pytest

pytest.importorskip("transformers")
pytest.importorskip("modelopt.torch.quantization")
# tensorrt_edgellm.quantization imports datasets at package import time.
pytest.importorskip("datasets")

import torch  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from tensorrt_edgellm.quantization.quantize import _untie_lm_head  # noqa: E402


def _tiny_llama(tie: bool) -> LlamaForCausalLM:
    config = LlamaConfig(vocab_size=32,
                         hidden_size=16,
                         intermediate_size=32,
                         num_hidden_layers=1,
                         num_attention_heads=2,
                         num_key_value_heads=2,
                         tie_word_embeddings=tie)
    return LlamaForCausalLM(config)


def test_untie_lm_head_gives_independent_weight():
    model = _tiny_llama(tie=True)
    embed = model.get_input_embeddings().weight
    assert model.get_output_embeddings().weight is embed

    _untie_lm_head(model)

    head = model.get_output_embeddings().weight
    assert head is not model.get_input_embeddings().weight
    assert torch.equal(head, model.get_input_embeddings().weight)
    assert model.config.tie_word_embeddings is False
    head.data.mul_(2.0)
    assert not torch.equal(head, model.get_input_embeddings().weight)


def test_untie_lm_head_is_noop_when_untied():
    model = _tiny_llama(tie=False)
    head = model.get_output_embeddings().weight

    _untie_lm_head(model)

    assert model.get_output_embeddings().weight is head


def test_untie_lm_head_ignores_models_without_embedding_accessors():
    _untie_lm_head(torch.nn.Linear(4, 4))


def test_int4_awq_lm_head_export_keeps_original_embeddings():
    export_hf_checkpoint = pytest.importorskip(
        "modelopt.torch.export").export_hf_checkpoint
    import modelopt.torch.quantization as mtq
    from safetensors.torch import load_file

    from tensorrt_edgellm.quantization.quantization_configs import \
        build_quant_config

    torch.manual_seed(0)
    config = LlamaConfig(vocab_size=256,
                         hidden_size=128,
                         intermediate_size=256,
                         num_hidden_layers=1,
                         num_attention_heads=4,
                         num_key_value_heads=4,
                         tie_word_embeddings=True)
    config.architectures = ["LlamaForCausalLM"]
    model = LlamaForCausalLM(config).eval()
    original = model.get_input_embeddings().weight.detach().clone()
    calib = [torch.randint(0, 256, (4, 64)) for _ in range(4)]

    def forward_loop(m):
        for input_ids in calib:
            m(input_ids=input_ids)

    _untie_lm_head(model)
    mtq.quantize(model,
                 build_quant_config("int4_awq", "int4_awq"),
                 forward_loop=forward_loop)
    # AWQ must have smoothed the head, otherwise this test proves nothing.
    assert model.lm_head.input_quantizer.pre_quant_scale is not None

    with tempfile.TemporaryDirectory() as export_dir:
        with torch.inference_mode():
            export_hf_checkpoint(model, export_dir=export_dir)
        state = {}
        for path in glob.glob(os.path.join(export_dir, "*.safetensors")):
            state.update(load_file(path))
        with open(os.path.join(export_dir, "config.json")) as f:
            exported_config = json.load(f)

    assert torch.equal(state["model.embed_tokens.weight"].float(),
                       original.float())
    assert "lm_head.pre_quant_scale" in state
    assert exported_config["tie_word_embeddings"] is False
