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
"""Qwen4-Exp (Qwen3.8-Flash-Next) model components."""

from .modeling_qwen4_exp_mtp import Qwen4ExpMtpDraftModel
from .modeling_qwen4_exp_text import (Qwen4ExpBackbone, Qwen4ExpCausalLM,
                                      Qwen4ExpDecoderLayer,
                                      Qwen4ExpGatedResidual, Qwen4ExpPle,
                                      Qwen4ExpRMSNorm)

__all__ = [
    "Qwen4ExpBackbone",
    "Qwen4ExpCausalLM",
    "Qwen4ExpDecoderLayer",
    "Qwen4ExpGatedResidual",
    "Qwen4ExpMtpDraftModel",
    "Qwen4ExpPle",
    "Qwen4ExpRMSNorm",
]
