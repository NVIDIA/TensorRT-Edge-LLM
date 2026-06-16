# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Checkpoint loader and ONNX exporter for causal LMs.

Supports common HF architectures and FP8, NVFP4, INT4 (AWQ / GPTQ), INT8 SmoothQuant,
and mixed-precision checkpoints when described by ``config.json`` / ``hf_quant_config.json``.

Quick start::

    from tensorrt_edgellm import AutoModel, export_onnx

    model = AutoModel.from_pretrained("/path/to/checkpoint")
    export_onnx(model, "output/model.onnx", model_dir="/path/to/checkpoint")

Config: :func:`checkpoint.checkpoint_utils.load_checkpoint_config_dicts` /
:func:`checkpoint.checkpoint_utils.load_config_dict`. Weights: :func:`checkpoint.loader.load_weights`.
Export sidecars: :func:`checkpoint.checkpoint_utils.write_runtime_artifacts`.
"""

from ._version import __version__
from .checkpoint.checkpoint_utils import (load_checkpoint_config_dicts,
                                          load_config_dict)
from .checkpoint.loader import load_weights
from .config import ModelConfig, QuantConfig
from .model import AutoModel, register_model
# Register model-type-specific implementations
from .models.gemma4.modeling_gemma4_text import Gemma4CausalLM
from .models.nemotron_h.modeling_nemotron_h import NemotronHCausalLM
from .models.qwen3_5.modeling_qwen3_5_text import Qwen3_5CausalLM
from .models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeCausalLM
from .models.qwen3_asr.modeling_qwen3_asr_text import Qwen3ASRLanguageModel
from .models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeCausalLM
from .models.qwen3_omni.modeling_qwen3_omni_text import Qwen3OmniLanguageModel
from .onnx.export import export_onnx

register_model("nemotron_h", NemotronHCausalLM)
# Gemma 4: ``gemma4`` is the multimodal root model_type, ``gemma4_text`` the
# promoted text sub-config; register both so either is dispatched here.
register_model("gemma4", Gemma4CausalLM)
register_model("gemma4_text", Gemma4CausalLM)
register_model("qwen3_5_text", Qwen3_5CausalLM)
register_model("qwen3_5_moe_text", Qwen3_5MoeCausalLM)
register_model("qwen3_5_moe", Qwen3_5MoeCausalLM)
register_model("qwen3_moe", Qwen3MoeCausalLM)
register_model("NemotronH_Nano_VL_V2", NemotronHCausalLM)
register_model("NemotronH_Nano_Omni_Reasoning_V3", NemotronHCausalLM)
# Qwen3-Omni thinker needs an extra ``hidden_states`` ONNX output for the
# Talker handoff. Cover every model_type string that can appear in the
# thinker's config.json across HF / exported variants.
register_model("qwen3_omni", Qwen3OmniLanguageModel)
register_model("qwen3_omni_thinker", Qwen3OmniLanguageModel)
register_model("qwen3_omni_text", Qwen3OmniLanguageModel)
register_model("qwen3_asr", Qwen3ASRLanguageModel)
register_model("qwen3_asr_thinker", Qwen3ASRLanguageModel)

__all__ = [
    "__version__",
    "AutoModel",
    "export_onnx",
    "load_checkpoint_config_dicts",
    "load_config_dict",
    "load_weights",
    "ModelConfig",
    "QuantConfig",
    "register_model",
]
