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

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt")

from experimental.builder.core import contracts, quantization
from experimental.builder.models import registry
from experimental.builder.models.nemotron_h import configuration, weights
from experimental.builder.ops.functional import moe


def test_nemotron_h_mtp_uses_family_direct_components():
    definition = registry.definition_for("nemotron_h", contracts.Component.LLM,
                                         "mtp", contracts.SpecRole.DRAFT)

    assert definition.module == "nemotron_h.modeling_nemotron_h_mtp"
    assert definition.class_name == "NemotronHMtpDraftModel"
    assert (registry.weight_conversion_for(
        "nemotron_h", "mtp",
        contracts.SpecRole.DRAFT).__name__.endswith("nemotron_h.weights"))


def test_nemotron_h_mtp_config_expands_one_predictor_to_two_blocks():
    cfg = SimpleNamespace(
        mtp_num_hidden_layers=1,
        raw_component={"mtp_layers_block_type": ["attention", "moe"]},
        quant=quantization.QuantConfig(
            quant_type=quantization.QUANT_NVFP4,
            group_size=16,
            kv_cache_quant="fp8",
        ),
        tie_word_embeddings=False,
        mamba_cfg=object(),
        gdn_cfg=object(),
        mtp_base=True,
    )

    configuration.configure_draft(cfg)

    assert cfg.num_hidden_layers == 2
    assert cfg.layer_types == ["attention", "moe"]
    assert cfg.attention_layer_types == ["full_attention", "moe"]
    assert cfg.mamba_cfg is None
    assert cfg.gdn_cfg is None
    assert cfg.quant.module_type("layers.0.mixer.q_proj") == "fp16"
    assert cfg.quant.module_type("lm_head") == "nvfp4"
    assert cfg.quant.module_group_size("lm_head") == 16
    assert cfg.quant.kv_cache_quant == "fp8"


class _FakeWeights:

    def __init__(self):
        self.values = {}
        for expert in range(2):
            prefix = f"layers.1.mixer.experts.{expert}"
            self.values[prefix + ".up_proj.weight"] = np.full((3, 4),
                                                              expert + 1,
                                                              dtype=np.float16)
            self.values[prefix + ".down_proj.weight"] = np.full(
                (4, 3), expert + 2, dtype=np.float16)

    def f16(self, name):
        return self.values[name]

    def checkpoint_binding(self, names, source_layout, assemble, **extra):
        return {
            "checkpoint_keys": tuple(names),
            "source_layout": source_layout,
            "assemble": assemble,
            "extra": extra,
        }


def test_nemotron_h_mtp_experts_are_zero_padded_for_relu2_plugin():
    source = _FakeWeights()
    packed = weights.prepare_mtp_fp16_experts(source, "layers.1.mixer.experts",
                                              2, 4, 3)

    assert packed["padded_intermediate"] == 128
    assert packed["fc1_weights"].shape == (2, 128, 4)
    assert packed["fc2_weights"].shape == (2, 4, 128)
    np.testing.assert_array_equal(
        packed["fc1_weights"][:, :3],
        np.stack((np.ones((3, 4), np.float16), np.full((3, 4), 2,
                                                       np.float16))))
    assert not np.any(packed["fc1_weights"][:, 3:])
    assert not np.any(packed["fc2_weights"][:, :, 3:])


def test_nemotron_h_mtp_expert_bindings_select_non_gated_assembler():
    bindings = weights.mtp_fp16_expert_bindings(_FakeWeights(),
                                                "layers.1.mixer.experts", 2)

    assert bindings["fc1_weights"]["assemble"] == "fp16_moe_fc1_relu2"
    assert bindings["fc2_weights"]["assemble"] == "fp16_moe_fc2"
    assert len(bindings["fc1_weights"]["checkpoint_keys"]) == 2
    assert len(bindings["fc2_weights"]["checkpoint_keys"]) == 2


def test_nemotron_h_mtp_checkpoint_prefix_mapping():
    candidates = weights.resolve_candidates(
        "layers.0.enorm.weight",
        component="llm",
        spec_type="mtp",
        spec_role="draft",
        quant_type="fp16",
    )

    assert candidates[0] == "mtp.layers.0.enorm.weight"


def test_fp16_moe_emits_nemotron_routing_contract(monkeypatch):
    captured = {}

    monkeypatch.setattr(moe,
                        "parameter",
                        lambda name, value, kind, recipe=None:
                        (name, value, kind, recipe))

    def fake_operation(name, inputs, **attributes):
        captured.update(name=name, inputs=inputs, attributes=attributes)
        return "output"

    monkeypatch.setattr(moe, "operation", fake_operation)
    expert_weights = {
        "fc1_weights": np.zeros((2, 128, 4), np.float16),
        "fc2_weights": np.zeros((2, 4, 128), np.float16),
    }
    bindings = {"fc1_weights": {}, "fc2_weights": {}}

    result = moe.fp16_moe("router",
                          "hidden",
                          expert_weights,
                          2,
                          1,
                          4,
                          128,
                          weight_prefix="experts",
                          weight_bindings=bindings,
                          activation_type=moe.MoeActivation.RELU2,
                          routing_mode=moe.MoeRouting.SIGMOID_GROUP_TOPK,
                          n_group=2,
                          topk_group=1,
                          routed_scaling_factor=2.5,
                          e_score_correction_bias="correction")

    assert result == "output"
    assert captured["inputs"][-1] == "correction"
    assert captured["attributes"]["activation_type"] == 4
    assert captured["attributes"]["routing_mode"] == 1
    assert captured["attributes"]["n_group"] == 2
    assert captured["attributes"]["topk_group"] == 1
    assert captured["attributes"]["routed_scaling_factor"] == 2.5
