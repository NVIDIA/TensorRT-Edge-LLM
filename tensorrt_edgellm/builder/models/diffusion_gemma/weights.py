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
"""DiffusionGemma checkpoint mapping and expert packing."""

import re

from ..gemma4 import weights as gemma4_weights

# DiffusionGemma and Gemma4 checkpoints use the same gated expert tensors and
# the same NVFP4 MoE plugin layout. Keep the checkpoint-name mapping below
# model-specific while sharing that exact packing contract.
pack_dense_nvfp4_experts = gemma4_weights.pack_dense_nvfp4_experts
repack_nvfp4_experts = gemma4_weights.repack_nvfp4_experts
nvfp4_expert_specs = gemma4_weights.nvfp4_expert_specs
nvfp4_expert_bindings = gemma4_weights.nvfp4_expert_bindings


def resolve_candidates(name: str, *, component: str, spec_type: str,
                       spec_role: str, quant_type: str):
    del spec_type, spec_role, quant_type
    if component == "dllm":
        encoder_scalar = re.fullmatch(
            r"model\.layers\.(\d+)\.encoder_layer_scalar", name)
        if encoder_scalar:
            return (f"model.encoder.language_model.layers."
                    f"{encoder_scalar.group(1)}.layer_scalar", )
        decoder_scalar = re.fullmatch(
            r"model\.layers\.(\d+)\.decoder_layer_scalar", name)
        if decoder_scalar:
            return (f"model.decoder.layers."
                    f"{decoder_scalar.group(1)}.layer_scalar", )
        if name.startswith("model."):
            return ("model.decoder." + name[len("model."):], )
        if name.startswith("self_conditioning."):
            return ("model.decoder." + name, )
        if name == "lm_head.weight":
            return ("model.decoder.embed_tokens.weight", )
    if component == "visual":
        if name.startswith(("vision_tower.", "embed_vision.")):
            return ("model.encoder." + name, )
    return ()


def normalize_checkpoint_name(name: str) -> str:
    if name.startswith("model.decoder."):
        return "model." + name[len("model.decoder."):]
    if name.startswith("model.encoder."):
        return name[len("model.encoder."):]
    return name
