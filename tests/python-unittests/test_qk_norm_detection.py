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
"""QK-norm checkpoint-key detection (`_detect_has_qk_norm`).

The Qwen3 ``.q_norm.weight`` spelling is model-agnostic, while the HunYuan
``.query_layernorm.weight`` spelling must stay scoped to the model families
whose loader remaps it onto ``q_norm`` — a foreign family reusing that key
name for an unrelated norm must not trip the fused QK-norm path.
"""

import json

from tensorrt_edgellm.config import _detect_has_qk_norm


def _write_index(tmp_path, keys):
    index = {
        "weight_map": {
            key: "model-00001-of-00001.safetensors"
            for key in keys
        }
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    return str(tmp_path)


def test_q_norm_key_detected_for_any_model_type(tmp_path):
    model_dir = _write_index(tmp_path,
                             ["model.layers.0.self_attn.q_norm.weight"])
    assert _detect_has_qk_norm(model_dir, "qwen3")
    assert _detect_has_qk_norm(model_dir)


def test_query_layernorm_key_detected_only_for_hunyuan(tmp_path):
    model_dir = _write_index(
        tmp_path, ["model.layers.0.self_attn.query_layernorm.weight"])
    assert _detect_has_qk_norm(model_dir, "hunyuan_v1_dense")
    # Foreign families without the query_layernorm -> q_norm remap must not
    # report QK-norm (the gammas could never be loaded).
    assert not _detect_has_qk_norm(model_dir, "some_future_model")
    assert not _detect_has_qk_norm(model_dir)


def test_no_norm_keys_detected(tmp_path):
    model_dir = _write_index(tmp_path,
                             ["model.layers.0.self_attn.q_proj.weight"])
    assert not _detect_has_qk_norm(model_dir, "hunyuan_v1_dense")
