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
"""Gemma4 conversion request, revision, and tokenizer contract."""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from ...schema import (ConversionError, ConversionRequest, RevisionMismatch,
                       UnsupportedConversion, load_json_asset)
from .variant import A4B_26B_IT_VARIANT, E2B_IT_VARIANT, VARIANTS

GEMMA_REVISION: Final = "0f7e7420e84c6789d96ba465608d2a008ecb443d"
GEMMA_REVISION_SHORT: Final = GEMMA_REVISION[:7]
HF_REVISIONS: Final = MappingProxyType({
    E2B_IT_VARIANT.name:
    "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
    A4B_26B_IT_VARIANT.name:
    "4d7ae4984b7db7de8f8457170b3f1a419ee76d52",
})
SUPPORTED_VARIANT: Final = E2B_IT_VARIANT.name
SUPPORTED_VARIANTS: Final = tuple(VARIANTS)
SOURCE_DTYPE: Final = "float32"
TARGET_DTYPE: Final = "bfloat16"
SUPPORTED_DTYPE: Final = TARGET_DTYPE
ROUNDING_POLICY: Final = "round_to_nearest_even"
TOKENIZER_JSON_FILES: Final = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
CHAT_TEMPLATE_FILE: Final = "chat_template.jinja"
TOKENIZER_FILES: Final = TOKENIZER_JSON_FILES + (CHAT_TEMPLATE_FILE, )
_TOKENIZER_SPECIAL_TOKENS: Final = {
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
    52: "<|\"|>",
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
_TOKENIZER_CONFIG_FIELDS: Final = {
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
}


def _installed_gemma_revision() -> str | None:
    try:
        distribution = importlib.metadata.distribution("gemma")
    except importlib.metadata.PackageNotFoundError:
        return None
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return None
    try:
        metadata = json.loads(direct_url)
    except json.JSONDecodeError:
        return None
    vcs_info = metadata.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return None
    commit_id = vcs_info.get("commit_id")
    return commit_id if isinstance(commit_id, str) else None


def validate_tokenizer_asset(name: str, value: dict[str, Any]) -> None:
    """Validate one Gemma4 tokenizer JSON object."""
    if name == "generation_config.json":
        expected = {
            "bos_token_id": 2,
            "eos_token_id": [1, 106, 50],
            "pad_token_id": 0
        }
        if any(
                value.get(key) != expected_value
                for key, expected_value in expected.items()):
            raise ConversionError(
                "generation_config.json has incompatible Gemma4 token IDs")
        return
    if name == "tokenizer_config.json":
        if any(
                value.get(key) != expected_value
                for key, expected_value in _TOKENIZER_CONFIG_FIELDS.items()):
            raise ConversionError(
                "tokenizer_config.json has incompatible Gemma4 special tokens")
        if (value.get("backend") != "tokenizers"
                or value.get("extra_special_tokens") != ["<|video|>"]):
            raise ConversionError(
                "tokenizer_config.json has an incompatible Gemma4 backend")
        return
    model = value.get("model")
    added_tokens = value.get("added_tokens")
    required_sections = ("normalizer", "pre_tokenizer", "post_processor",
                         "decoder")
    if (value.get("version") != "1.0" or not isinstance(model, dict)
            or model.get("type") != "BPE" or model.get("unk_token") != "<unk>"
            or model.get("byte_fallback") is not True
            or not isinstance(model.get("merges"), list) or not model["merges"]
            or any(not isinstance(value.get(section), dict)
                   for section in required_sections)
            or not isinstance(added_tokens, list)):
        raise ConversionError(
            "tokenizer.json has an incompatible Gemma4 schema")
    vocab = model.get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ConversionError(
            "tokenizer.json has an invalid Gemma4 vocabulary")
    vocab_ids = tuple(vocab.values())
    if (any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            or token_id < 0 for token_id in vocab_ids)
            or len(vocab_ids) != 262144 or min(vocab_ids) != 0
            or max(vocab_ids) != 262143 or len(set(vocab_ids)) != 262144):
        raise ConversionError(
            "tokenizer.json vocabulary does not match Gemma4")
    added_token_map = {
        token.get("id"): token.get("content")
        for token in added_tokens
        if isinstance(token, dict) and token.get("special") is True
    }
    if any(
            added_token_map.get(token_id) != content
            for token_id, content in _TOKENIZER_SPECIAL_TOKENS.items()):
        raise ConversionError(
            "tokenizer.json has incompatible Gemma4 special-token IDs")


def process_tokenizer_asset(tokenizer_path: Path, name: str) -> dict[str, Any]:
    """Safely load and validate one allowlisted tokenizer asset."""
    asset = tokenizer_path / name
    if asset.is_symlink() or not asset.is_file():
        raise ConversionError(
            f"tokenizer asset must be a regular non-symlink file: {asset}")
    value, _ = load_json_asset(asset)
    validate_tokenizer_asset(name, value)
    return value


def validate_tokenizer_assets(tokenizer_path: Path) -> None:
    """Validate the exact Gemma4 tokenizer asset set."""
    actual_names = {entry.name for entry in tokenizer_path.iterdir()}
    expected_names = set(TOKENIZER_FILES)
    if actual_names != expected_names:
        raise ConversionError(
            "tokenizer directory must contain exactly the required assets; "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}")
    for name in TOKENIZER_JSON_FILES:
        process_tokenizer_asset(tokenizer_path, name)
    chat_template = tokenizer_path / CHAT_TEMPLATE_FILE
    if (chat_template.is_symlink() or not chat_template.is_file()
            or not chat_template.read_bytes()):
        raise ConversionError(
            f"tokenizer asset must be a nonempty regular non-symlink file: {chat_template}"
        )


def check_installed_revision(expected_revision: str) -> None:
    """Verify the installed Gemma package source revision."""
    installed_revision = _installed_gemma_revision()
    if installed_revision != expected_revision:
        raise RevisionMismatch(
            "installed Gemma provenance does not match required revision "
            f"{expected_revision}; found {installed_revision or 'unknown'}")


@dataclass(frozen=True)
class Gemma4ConversionRequest(ConversionRequest):
    """Validated conversion request for one supported Gemma4 IT variant."""

    variant_name: str = SUPPORTED_VARIANT
    output_dtype: str = TARGET_DTYPE
    text_only: bool = True
    gemma_revision: str = GEMMA_REVISION

    def validate_family(self) -> None:
        """Validate Gemma4 tokenizer, representation, and provenance rules."""
        if self.variant_name not in SUPPORTED_VARIANTS:
            raise UnsupportedConversion(
                f"unsupported variant {self.variant_name!r}; "
                f"supported: {SUPPORTED_VARIANTS!r}")
        if self.output_dtype != TARGET_DTYPE or not self.text_only:
            raise UnsupportedConversion(
                "Gemma4 supports only canonical FP32 source to text-only BF16 target conversion"
            )
        if self.gemma_revision != GEMMA_REVISION:
            raise RevisionMismatch(
                f"Gemma revision {self.gemma_revision!r} does not match "
                f"pinned revision {GEMMA_REVISION!r}")
        check_installed_revision(GEMMA_REVISION)
        validate_tokenizer_assets(self.tokenizer_path)
