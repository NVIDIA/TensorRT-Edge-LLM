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
"""Alpamayo tokenizer artifacts."""

import json
import os
import tempfile
from typing import Any, Dict

from .configuration import vlm_reference

_SPECIAL_TOKEN_NAMES = (
    "prompt_start",
    "prompt_end",
    "image_start",
    "image_pre_tkn",
    "image_end",
    "traj_history_start",
    "traj_history_pre_tkn",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
    "traj_future_start",
    "traj_future_pre_tkn",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "vectorized_wm",
    "vectorized_wm_start",
    "vectorized_wm_end",
    "vectorized_wm_pre_tkn",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
)

_TRAJECTORY_TOKENS = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}


def vlm_file(root: Dict[str, Any], model_dir: str, filename: str) -> str:
    reference = vlm_reference(model_dir, root)
    if os.path.isdir(reference):
        path = os.path.join(reference, filename)
    else:
        try:
            import huggingface_hub
        except ImportError as error:
            raise RuntimeError(
                "Alpamayo runtime artifacts require huggingface_hub or a "
                "local base VLM checkpoint") from error
        path = huggingface_hub.hf_hub_download(reference, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Alpamayo base VLM asset {filename!r} not found at {path!r}")
    return path


def prepare_runtime_model(root: Dict[str, Any], args):
    """Create Alpamayo tokenizer/config artifacts without editing checkpoint."""
    try:
        import transformers
    except ImportError as error:
        raise RuntimeError(
            "Alpamayo tokenizer generation requires transformers") from error
    reference = vlm_reference(args.model_dir, root)
    tokenizer = transformers.AutoTokenizer.from_pretrained(reference)
    trajectory_start = int(root.get("traj_token_start_idx", len(tokenizer)))
    if len(tokenizer) != trajectory_start:
        raise ValueError("Alpamayo base tokenizer length does not match "
                         f"traj_token_start_idx: {len(tokenizer)} != "
                         f"{trajectory_start}")
    tokenizer.add_tokens([
        f"<i{value}>" for value in range(int(root.get("traj_vocab_size", 0)))
    ])
    tokenizer.add_tokens([f"<|{name}|>" for name in _SPECIAL_TOKEN_NAMES],
                         special_tokens=True)
    expected_vocabulary = int(root.get("vocab_size", len(tokenizer)))
    if len(tokenizer) != expected_vocabulary:
        raise ValueError("Alpamayo runtime tokenizer size does not match "
                         f"vocab_size: {len(tokenizer)} != "
                         f"{expected_vocabulary}")
    for name, expected in (root.get("traj_token_ids") or {}).items():
        token = _TRAJECTORY_TOKENS.get(name)
        if token is None:
            continue
        actual = int(tokenizer.convert_tokens_to_ids(token))
        if actual != int(expected):
            raise ValueError(f"Alpamayo token {token!r} has ID {actual}, "
                             f"expected {expected}")

    artifacts = tempfile.TemporaryDirectory(prefix="alpamayo-runtime-")
    runtime_root = dict(root)
    runtime_root["vlm_name_or_path"] = reference
    with open(os.path.join(artifacts.name, "config.json"), "w") as config_file:
        json.dump(runtime_root, config_file, indent=2)
    tokenizer.save_pretrained(artifacts.name)
    with open(os.path.join(artifacts.name, "generation_config.json"),
              "w") as generation_file:
        json.dump({"eos_token_id": int(tokenizer.eos_token_id)},
                  generation_file,
                  indent=2)
    return artifacts
