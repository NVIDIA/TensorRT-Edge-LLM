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
"""Direct-builder chat template extraction for thinking-aware templates."""

import json

import pytest

transformers = pytest.importorskip("transformers")

from experimental.builder.core.artifacts.chat_template import \
    process_chat_template

_REASONING = ("Reasoning effort is set to xhigh. Please think carefully "
              "through the task.")


class _Qwen38LikeTokenizer:
    """Mimics Qwen3.8: an undefined ``enable_thinking`` means thinking on,
    and thinking mode injects reasoning instructions into the system block."""

    chat_template = "fake"
    bos_token = None

    def apply_chat_template(self, messages, **kwargs):
        del kwargs["tokenize"]
        add_generation_prompt = kwargs.get("add_generation_prompt", False)
        thinking = kwargs.get("enable_thinking", None) is not False
        instructions = _REASONING if thinking else ""

        output = ""
        rest = messages
        if messages[0]["role"] == "system":
            content = messages[0]["content"]
            if instructions:
                content = instructions + "\n\n" + content
            output += f"<|im_start|>system\n{content}<|im_end|>\n"
            rest = messages[1:]
        elif instructions:
            output += f"<|im_start|>system\n{instructions}<|im_end|>\n"
        for message in rest:
            output += f"<|im_start|>{message['role']}\n"
            if message["role"] == "assistant":
                output += "<think>\n\n</think>\n\n"
            output += f"{message['content']}<|im_end|>\n"
        if add_generation_prompt:
            output += "<|im_start|>assistant\n"
            output += "<think>\n" if thinking else "<think>\n\n</think>\n\n"
        return output


def test_thinking_aware_template_yields_clean_role_prefixes(
        monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    out_dir = tmp_path / "out"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3_5"}))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: _Qwen38LikeTokenizer())

    process_chat_template(str(model_dir), str(out_dir))

    data = json.loads((out_dir / "processed_chat_template.json").read_text())
    assert data["roles"]["system"] == {
        "prefix": "<|im_start|>system\n",
        "suffix": "<|im_end|>\n",
    }
    assert data["roles"]["user"] == {
        "prefix": "<|im_start|>user\n",
        "suffix": "<|im_end|>\n",
    }
    assert data["roles"]["assistant"]["prefix"] == (
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert data["generation_prompt"] == (
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert data[
        "generation_prompt_thinking"] == "<|im_start|>assistant\n<think>\n"
    assert data["default_system_prompt"] == ""
    serialized = json.dumps(data)
    assert "placeholder" not in serialized
    assert _REASONING not in serialized
