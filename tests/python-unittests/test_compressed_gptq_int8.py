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

import pytest
import torch

from tensorrt_edgellm import config
from tensorrt_edgellm.checkpoint.repacking import unpack_gptq_int8


def test_compressed_w8a16_config_is_not_treated_as_fp16(tmp_path):
    quant = config._parse_quant(
        str(tmp_path), {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "input_activations": None,
                        "weights": {
                            "type": "int",
                            "num_bits": 8,
                            "group_size": None,
                        },
                    },
                },
                "ignore": ["lm_head"],
            },
        })

    assert quant.quant_type == config.QUANT_INT8_GPTQ
    assert quant.group_size == 0
    assert quant.gptq_zero_point_offset == 0
    assert "lm_head" in quant.excluded


@pytest.mark.parametrize(
    ("weight_bits", "activation_bits", "expected"),
    ((4, 4, config.QUANT_INT4_GPTQ), (8, 8, config.QUANT_INT8_GPTQ)),
)
def test_compressed_integer_activation_contract_lowers_to_a16(
        tmp_path, weight_bits, activation_bits, expected):
    quant = config._parse_quant(
        str(tmp_path), {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "targets": ["Linear"],
                        "input_activations": {
                            "type": "int",
                            "num_bits": activation_bits,
                            "dynamic": False,
                        },
                        "weights": {
                            "type": "int",
                            "num_bits": weight_bits,
                            "group_size": 128,
                        },
                    },
                },
            },
        })

    assert quant.quant_type == expected


def test_onnx_moe_frontends_reject_w8a16_experts():
    from tensorrt_edgellm.models.nemotron_h.modeling_nemotron_h import \
        NemotronHMoEMLP
    from tensorrt_edgellm.models.qwen3_moe.modeling_qwen3_moe import \
        Qwen3SparseMoeBlock

    model_config = SimpleNamespace(quant=config.QuantConfig(
        quant_type=config.QUANT_INT8_GPTQ))
    with pytest.raises(ValueError, match="no W8A16"):
        Qwen3SparseMoeBlock(model_config)
    with pytest.raises(ValueError, match="W8A16.*not implemented"):
        NemotronHMoEMLP(model_config, "backbone.layers.0.mixer")


def test_compressed_kv_cache_scheme_is_parsed(tmp_path):
    quant = config._parse_quant(
        str(tmp_path), {
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "input_activations": None,
                        "weights": {
                            "type": "int",
                            "num_bits": 8,
                            "symmetric": True,
                        },
                    },
                },
                "kv_cache_scheme": {
                    "type": "float",
                    "num_bits": 8,
                },
            },
        })

    assert quant.kv_cache_quant == "fp8"


def test_unpack_gptq_int8_preserves_signed_weight_values():
    values = torch.tensor([
        [0, 128],
        [1, 129],
        [127, 255],
        [255, 0],
        [128, 64],
        [129, 65],
        [130, 66],
        [131, 67],
    ],
                          dtype=torch.int64)
    qweight = torch.zeros(2, 2, dtype=torch.int64)
    for byte in range(4):
        qweight |= values[byte::4] << (8 * byte)

    weight, permutation, correction, group_size = unpack_gptq_int8(
        qweight.to(torch.int32),
        torch.empty(1, 0, dtype=torch.int32),
        torch.ones(1, 2, dtype=torch.float16),
        zero_point_offset=0,
    )

    assert torch.equal(weight, (values - 128).to(torch.int8).t())
    assert torch.equal(permutation, torch.arange(8))
    assert correction is None
    assert group_size == 8
