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
"""Direct-builder contract tests for Domino speculative decoding."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt")

from tensorrt_edgellm.builder.core import contracts
from tensorrt_edgellm.builder.core.artifacts.runtime_config import \
    build_runtime_config
from tensorrt_edgellm.builder.core.artifacts.tensors import save_safetensors
from tensorrt_edgellm.builder.core.builder import BuildArgs
from tensorrt_edgellm.builder.core.config import DeviceConfig
from tensorrt_edgellm.builder.core.safetensors_np import SafetensorsStore
from tensorrt_edgellm.builder.models import registry
from tensorrt_edgellm.builder.models.domino import \
    artifacts as domino_artifacts
from tensorrt_edgellm.builder.models.domino import modeling_domino_draft

HIDDEN, VOCAB, GRU, PROJ = 64, 160, 8, 4
TARGET_LAYERS = [1, 9, 17, 25, 33]
DRAFT_CONFIG = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "qwen3",
    "vocab_size": VOCAB,
    "block_size": 16,
    "num_hidden_layers": 5,
    "hidden_size": HIDDEN,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1_000_000,
    "max_position_embeddings": 4096,
    "tie_word_embeddings": False,
    "dflash_config": {
        "projector_type": "domino",
        "mask_token_id": 151,
        "shift_label": True,
        "target_layer_ids": TARGET_LAYERS,
        "pure_draft_prefix_len": 1,
        "gru_hidden_dim": GRU,
        "emb_dim": PROJ,
    },
}


def test_domino_runtime_metadata_carries_projector_contract(tmp_path):
    """Base and draft config.json carry the dflash_config the runtime parses."""
    draft_dir = tmp_path / "draft"
    draft_dir.mkdir()
    (draft_dir / "config.json").write_text(json.dumps(DRAFT_CONFIG))
    target = DeviceConfig(model_type="qwen3",
                          root_model_type="qwen3",
                          hidden_size=HIDDEN,
                          num_hidden_layers=36,
                          num_attention_heads=4,
                          num_key_value_heads=2,
                          head_dim=16,
                          intermediate_size=128,
                          vocab_size=VOCAB,
                          rms_norm_eps=1e-6,
                          rope_theta=1_000_000.0,
                          max_position_embeddings=4096)
    registry.configure_for_build(
        target,
        contracts.SpecRole.BASE,
        "domino",
        paired_draft_dir=str(draft_dir),
        build_args=SimpleNamespace(num_decoder_layers=0))
    base_args = BuildArgs(model_dir="target",
                          engine_dir="engine",
                          spec_role="base",
                          spec_type="domino",
                          draft_model_dir=str(draft_dir))

    assert build_runtime_config(target, base_args)["dflash_config"] == {
        "version": 1,
        "target_layer_ids": TARGET_LAYERS,
        "block_size": 16,
        "mask_token_id": 151,
        "supports_probabilistic_sampling": False,
        "projector_type": "domino",
        "gru_hidden_dim": GRU,
        "proj_dim": PROJ,
        "prefix_len": 1,
        "pool_size": VOCAB,
        "shift_label": True,
        "heads_file": "domino_heads.safetensors",
    }

    draft = DeviceConfig.from_pretrained(str(draft_dir))
    registry.configure_for_build(draft,
                                 contracts.SpecRole.DRAFT,
                                 "domino",
                                 paired_target=target)
    draft_args = BuildArgs(model_dir=str(draft_dir),
                           engine_dir="engine",
                           spec_role="draft",
                           spec_type="domino",
                           target_model_dir="target")
    draft_metadata = build_runtime_config(draft, draft_args)

    assert draft_metadata["base_model_hidden_size"] == len(
        TARGET_LAYERS) * HIDDEN
    assert draft_metadata["dflash_config"]["projector_type"] == "domino"


def test_domino_heads_sidecar_precomputes_gru_input_table(tmp_path):
    """The sidecar holds embed_tokens @ W_ih^T, W_hh, and the GRU-state half
    of embed_proj.0, matching the ONNX export sidecar."""
    rng = np.random.default_rng(0)
    weight_ih = rng.standard_normal((3 * GRU, HIDDEN)).astype(np.float16)
    weight_hh = rng.standard_normal((3 * GRU, GRU)).astype(np.float16)
    embed_proj = rng.standard_normal((PROJ, HIDDEN + GRU)).astype(np.float16)
    embedding = rng.standard_normal((VOCAB, HIDDEN)).astype(np.float16)
    for name in ("draft", "target", "engine"):
        (tmp_path / name).mkdir()
    save_safetensors(
        str(tmp_path / "draft" / "model.safetensors"), {
            "prefix_gru.weight_ih_l0": weight_ih,
            "prefix_gru.weight_hh_l0": weight_hh,
            "embed_proj.0.weight": embed_proj,
        })
    save_safetensors(str(tmp_path / "target" / "model.safetensors"),
                     {"model.language_model.embed_tokens.weight": embedding})
    args = SimpleNamespace(model_dir=str(tmp_path / "draft"),
                           target_model_dir=str(tmp_path / "target"),
                           resolved_component=contracts.Component.LLM)
    config = SimpleNamespace(hidden_size=HIDDEN,
                             vocab_size=VOCAB,
                             domino_gru_hidden_dim=GRU,
                             domino_proj_dim=PROJ)

    domino_artifacts._write_heads_sidecar(config, args,
                                          str(tmp_path / "engine"))

    expected_table = (
        embedding.astype(np.float32) @ weight_ih.astype(np.float32).T).astype(
            np.float16)
    with SafetensorsStore(str(tmp_path / "engine")) as store:
        np.testing.assert_array_equal(store.get_f16("gru_input_table"),
                                      expected_table)
        np.testing.assert_array_equal(store.get_f16("gru_weight_hh"),
                                      weight_hh)
        np.testing.assert_array_equal(store.get_f16("proj_state_weight"),
                                      embed_proj[:, HIDDEN:])


@pytest.mark.parametrize(("vocab_size", "pool_size", "expected"), [
    (151936, 2048, 16),
    (248320, 2048, 16),
    (160, 2048, 1),
    (151937, 2048, 1),
])
def test_domino_pool_topk_chunks_only_when_exact_and_large_enough(
        vocab_size, pool_size, expected):
    """The two-level TopK is used only when it reproduces the exact set."""
    assert modeling_domino_draft.pool_topk_chunks(vocab_size,
                                                  pool_size) == expected
