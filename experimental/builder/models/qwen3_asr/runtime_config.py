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
"""Qwen3-ASR runtime configuration artifacts."""

from typing import Any, Dict

from ...core import contracts


def component_runtime_config(bundle, component: contracts.Component, args):
    """Return Qwen3-ASR component runtime config."""
    if component != contracts.Component.AUDIO:
        return None
    root = bundle.root
    audio = dict(bundle.component_dict(component))
    result: Dict[str, Any] = {
        "model_type": "qwen3_asr_thinker",
        "audio_config": audio,
        "builder_config": {
            "min_time_steps": args.min_time_steps,
            "max_time_steps": args.max_time_steps,
            "min_code_len": args.min_code_len,
            "opt_code_len": args.opt_code_len,
            "max_code_len": args.max_code_len,
        },
    }
    thinker = root.get("thinker_config") or {}
    for key in ("audio_token_id", "audio_start_token_id", "audio_end_token_id",
                "user_token_id"):
        if key in thinker:
            result[key] = thinker[key]
        elif key in root:
            result[key] = root[key]
    text = thinker.get("text_config") or root.get("text_config") or {}
    rope_theta = text.get("rope_theta")
    if rope_theta is None:
        rope_theta = (text.get("rope_parameters") or {}).get("rope_theta")
    if rope_theta is not None:
        result["text_config"] = {"rope_theta": rope_theta}
    return result
