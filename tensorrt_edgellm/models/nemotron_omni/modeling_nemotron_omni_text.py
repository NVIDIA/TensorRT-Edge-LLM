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
"""
Nemotron-Omni LLM backbone.

The Nemotron-Omni LLM is the NemotronH hybrid decoder (Mamba2 + Attention +
MoE). Its implementation lives in ``models/nemotron_h/`` and is dispatched by
the default ``AutoModel`` pipeline from ``llm_config.architectures =
["NemotronHForCausalLM"]`` in the HF checkpoint's ``config.json``.

This module re-exports it under the Nemotron-Omni namespace so all three
components (LLM, visual, audio) are exported together by
:mod:`tensorrt_edgellm.onnx.cli` when the checkpoint lives in one HF repo.

For export:
    from tensorrt_edgellm.model import AutoModel
    from tensorrt_edgellm.onnx.export import export_onnx
    model = AutoModel.from_pretrained(model_dir)
    export_onnx(model, output_path, model_dir=model_dir)
"""
from ..nemotron_h.modeling_nemotron_h import \
    NemotronHCausalLM as NemotronOmniLanguageModel

__all__ = ["NemotronOmniLanguageModel"]
