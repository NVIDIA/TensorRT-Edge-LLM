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

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("tensorrt")

from tensorrt_edgellm.config import QUANT_NVFP4, ModelConfig, QuantConfig
from tensorrt_edgellm.models.muse_glimmer import MuseGlimmerForCausalLM
from tensorrt_edgellm.models.muse_glimmer.modeling_muse_glimmer_text import \
    MuseGlimmerRuntimeEmbedding


def _tiny_base_config():
    return {
        "model_type": "muse_glimmer",
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.19611613513818404,
            "final_logit_softcapping": 20.0,
            "layer_types": ["sliding_attention"] * 3 + ["full_attention"],
            "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0.0],
            "sliding_window": 2048,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 500000.0,
            },
            "tie_word_embeddings": False,
        },
    }


def _write_config(path, config):
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def test_muse_glimmer_nvfp4_embedding_materializes_fp16():
    config = SimpleNamespace(
        vocab_size=4,
        hidden_size=16,
        quant=QuantConfig(
            quant_type=QUANT_NVFP4,
            group_size=16,
            layer_overrides={"embed_tokens": QUANT_NVFP4},
            layer_group_sizes={"embed_tokens": 16},
            is_mixed_precision=True,
        ),
    )
    embedding = MuseGlimmerRuntimeEmbedding(config)
    embedding.weight.zero_()
    embedding.weight_scale.fill_(1)
    embedding.weight_scale_2.fill_(1)

    weight = embedding.runtime_weight()

    assert weight.shape == (4, 16)
    assert weight.dtype == torch.float16
    assert torch.count_nonzero(weight) == 0


def test_muse_glimmer_onnx_base_uses_ragged_tree_contract(tmp_path):
    model_dir = _write_config(tmp_path / "tiny", _tiny_base_config())
    config = ModelConfig.from_pretrained(model_dir, lambda h: h**-0.5)
    config = replace(config,
                     dflash_base=True,
                     dflash_tree_base=True,
                     dflash_target_layer_ids=[0, 1],
                     dflash2_target_layer_ids=[0, 1])
    model = MuseGlimmerForCausalLM(config).to(dtype=torch.float16)

    spec = model.onnx_export_spec()
    outputs = spec.wrapped(*spec.args)

    assert {
        "query_start_offsets", "query_lengths", "past_lengths",
        "attention_sequence_lengths", "state_indices", "logits_indices",
        "execution_phase_marker", "context_sequence_count_carrier",
        "kv_page_table", "attention_position_ids", "packed_attention_mask",
        "tree_parent_ids", "tree_depths", "valid_tree_counts"
    } <= set(spec.input_names)
    assert not ({"context_lengths", "kvcache_start_index", "last_token_ids"}
                & set(spec.input_names))
    assert spec.args[0].ndim == 2
    assert spec.output_names[:2] == ["logits", "hidden_states"]
    assert outputs[0].shape == (3, 128)
    assert outputs[1].shape == (4, 128)
