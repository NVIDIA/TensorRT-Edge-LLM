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
Shared model infrastructure (linear layers, ops) and model variants.

Model variants live in sub-packages named after the model family:
  ``default/``       - standard decoder transformer + Mamba hybrid
  ``gemma4/``        - Gemma 4 (gemma4_text) decoder (config parsing + ONNX-export modeling; runtime WIP)
  ``nemotron_h/``    - Nemotron-H hybrid (Mamba2 + attention)
  ``nemotron_omni/`` - Nemotron-Omni: RADIO visual + Parakeet audio (LLM reuses nemotron_h)
  ``qwen3_vl/``      - Qwen3-VL visual encoder + LLM
  ``qwen3_5/``       - Qwen3.5 visual encoder + LLM (no deepstack)
  ``qwen3_5_moe/``   - Qwen3.5 hybrid sparse MoE decoder
  ``qwen2_5_vl/``    - Qwen2.5-VL visual encoder + LLM
  ``qwen3_asr/``     - Qwen3-ASR audio encoder + LLM
  ``qwen3_moe/``     - Qwen3 sparse MoE decoder
  ``qwen3_omni/``    - Qwen3-Omni: visual + audio (re-exports) + LLM
  ``qwen3_tts/``     - Qwen3-TTS: audio encoder + talker + LLM
  ``internvl3/``     - InternVL3 visual encoder + LLM
  ``internvl3_5/``   - InternVL3.5 visual encoder + LLM (re-exports from internvl3)
  ``phi4mm/``        - Phi-4 Multimodal visual encoder + LLM

"""
