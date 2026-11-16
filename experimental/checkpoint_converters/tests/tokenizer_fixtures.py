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
"""Shared synthetic filesystems for Flax converter unit tests."""

import json

_SPECIAL_TOKENS = {
    0: "<pad>",
    1: "<eos>",
    2: "<bos>",
    3: "<unk>",
    4: "<mask>",
    46: "<|tool>",
    47: "<tool|>",
    48: "<|tool_call>",
    49: "<tool_call|>",
    50: "<|tool_response>",
    51: "<tool_response|>",
    52: '<|"|>',
    98: "<|think|>",
    100: "<|channel>",
    101: "<channel|>",
    105: "<|turn>",
    106: "<turn|>",
    255999: "<|image>",
    256000: "<|audio>",
    258880: "<|image|>",
    258881: "<|audio|>",
    258882: "<image|>",
    258883: "<audio|>",
    258884: "<|video|>",
}


def _compatible_tokenizer_assets():
    return {
        "generation_config.json": {
            "bos_token_id": 2,
            "eos_token_id": [1, 106, 50],
            "pad_token_id": 0,
        },
        "tokenizer.json": {
            "version":
            "1.0",
            "added_tokens": [{
                "id": token_id,
                "content": content,
                "special": True,
            } for token_id, content in _SPECIAL_TOKENS.items()],
            "normalizer": {},
            "pre_tokenizer": {},
            "post_processor": {},
            "decoder": {},
            "model": {
                "type": "BPE",
                "unk_token": "<unk>",
                "byte_fallback": True,
                "merges": ["a b"],
                "vocab": {
                    f"token_{token_id}": token_id
                    for token_id in range(262144)
                },
            },
        },
        "tokenizer_config.json": {
            "audio_token": "<|audio|>",
            "boa_token": "<|audio>",
            "boi_token": "<|image>",
            "bos_token": "<bos>",
            "eoa_token": "<audio|>",
            "eoi_token": "<image|>",
            "eos_token": "<eos>",
            "image_token": "<|image|>",
            "mask_token": "<mask>",
            "pad_token": "<pad>",
            "tokenizer_class": "GemmaTokenizer",
            "unk_token": "<unk>",
            "backend": "tokenizers",
            "extra_special_tokens": ["<|video|>"],
        },
    }


_TOKENIZER_ASSETS = {
    name: json.dumps(value)
    for name, value in _compatible_tokenizer_assets().items()
}
_TOKENIZER_ASSETS["chat_template.jinja"] = "<|turn>user\ncontent<turn|>\n"


def create_conversion_paths(tmp_path):
    """Create a safe request filesystem with compatible Gemma4 tokenizer assets."""
    checkpoint = tmp_path / "orbax"
    checkpoint.mkdir()
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    for name, value in _TOKENIZER_ASSETS.items():
        (tokenizer / name).write_text(value, encoding="utf-8")
    return checkpoint, tokenizer, tmp_path / "output"
