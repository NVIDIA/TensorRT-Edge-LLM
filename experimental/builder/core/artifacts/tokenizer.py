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
"""Tokenizer artifact lookup, conversion, and copying."""

import json
import logging
import os
import shutil
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

RUNTIME_TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
)


def find_token_id(model_dir: str, token: str) -> Optional[int]:
    """Resolve one token from Hugging Face tokenizer artifacts."""
    tokenizer_path = os.path.join(model_dir, "tokenizer.json")
    if os.path.isfile(tokenizer_path):
        with open(tokenizer_path) as tokenizer_file:
            tokenizer = json.load(tokenizer_file)
        for entry in tokenizer.get("added_tokens", []):
            if entry.get("content") == token:
                return int(entry["id"])
        for token_id, entry in tokenizer.get("added_tokens_decoder",
                                             {}).items():
            if entry.get("content") == token:
                return int(token_id)
    added_tokens_path = os.path.join(model_dir, "added_tokens.json")
    if os.path.isfile(added_tokens_path):
        with open(added_tokens_path) as added_tokens_file:
            added_tokens = json.load(added_tokens_file)
        if token in added_tokens:
            return int(added_tokens[token])
    return None


def write_tokenizer_json_if_missing(model_dir: str, engine_dir: str) -> None:
    """Convert BPE vocabulary artifacts to the runtime tokenizer format."""
    destination = os.path.join(engine_dir, "tokenizer.json")
    if os.path.isfile(destination):
        return
    vocab = os.path.join(model_dir, "vocab.json")
    merges = os.path.join(model_dir, "merges.txt")
    if not os.path.isfile(vocab) or not os.path.isfile(merges):
        return
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "tokenizer conversion requires the Transformers package"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    with tempfile.TemporaryDirectory() as temporary_dir:
        tokenizer.save_pretrained(temporary_dir)
        generated = os.path.join(temporary_dir, "tokenizer.json")
        if not os.path.isfile(generated):
            raise RuntimeError(
                "tokenizer conversion did not produce tokenizer.json")
        shutil.copy2(generated, destination)
    logger.info("Generated tokenizer.json from vocab.json and merges.txt")


def copy_tokenizer_artifacts(model_dir: str, output_dir: str) -> None:
    """Copy tokenizer files needed by runtime if they exist."""
    for filename in RUNTIME_TOKENIZER_FILENAMES:
        source = os.path.join(model_dir, filename)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(output_dir, filename))
            logger.info("Copied %s", filename)
