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

import json
from pathlib import Path

import pytest

from tensorrt_edgellm.chat_template import write_chat_template


def _checkpoint(tmp_path, model_type, template=None):
    model_dir = tmp_path / model_type
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(
        {"model_type": model_type}),
                                           encoding="utf-8")
    if template is not None:
        (model_dir / "tokenizer_config.json").write_text(json.dumps(
            {"chat_template": template}),
                                                         encoding="utf-8")
    return model_dir


def test_copies_embedded_provider_template_verbatim(tmp_path):
    source = "{% for message in messages %}{{ message.content }}{% endfor %}"
    model_dir = _checkpoint(tmp_path, "qwen3", "tokenizer template")
    (model_dir / "processor_config.json").write_text(json.dumps(
        {"chat_template": source}),
                                                     encoding="utf-8")
    output_dir = tmp_path / "engine"
    output_dir.mkdir()

    result = write_chat_template(str(model_dir), str(output_dir))

    destination = output_dir / "chat_template.jinja"
    assert result == str(destination)
    assert destination.read_text(encoding="utf-8") == source
    assert not (output_dir / "chat_template.inja").exists()
    assert not (output_dir / "chat_template.model").exists()


def test_copies_standalone_provider_template_verbatim(tmp_path):
    embedded = "embedded"
    source = b"{% macro content(message) %}{{ message.content }}{% endmacro %}\r\n"
    model_dir = _checkpoint(tmp_path, "gemma4", embedded)
    (model_dir / "chat_template.jinja").write_bytes(source)

    result = Path(write_chat_template(str(model_dir),
                                      str(tmp_path / "engine")))

    assert result.read_bytes() == source
    assert not (result.parent / "chat_template.inja").exists()


def test_rejects_standalone_named_provider_templates(tmp_path):
    model_dir = _checkpoint(tmp_path, "qwen3")
    (model_dir / "chat_template.jinja").write_bytes(b"default\r\n")
    additional = model_dir / "additional_chat_templates"
    additional.mkdir()
    (additional / "tool_use.jinja").write_bytes(b"tool-use\r\n")

    with pytest.raises(ValueError, match="unsupported named chat templates"):
        write_chat_template(str(model_dir), str(tmp_path / "engine"))


@pytest.mark.parametrize(
    "encoded",
    ({
        "default": "default",
        "tool_use": "tools"
    }, [{
        "name": "tool_use",
        "template": "tools"
    }, {
        "name": "default",
        "template": "default"
    }]),
)
def test_rejects_embedded_named_provider_templates(tmp_path, encoded):
    model_dir = _checkpoint(tmp_path, "qwen3")
    (model_dir / "tokenizer_config.json").write_text(json.dumps(
        {"chat_template": encoded}),
                                                     encoding="utf-8")

    with pytest.raises(ValueError, match="defines named chat templates"):
        write_chat_template(str(model_dir), str(tmp_path / "engine"))


def test_materializes_provider_jinja_without_json_runtime_artifact(tmp_path):
    source = "{% for message in messages %}{{ message.content }}{% endfor %}"
    model_dir = _checkpoint(tmp_path, "qwen3_omni_moe")
    (model_dir / "chat_template.json").write_text(json.dumps(
        {"chat_template": source}),
                                                  encoding="utf-8")
    output_dir = tmp_path / "engine"
    output_dir.mkdir()
    (output_dir / "chat_template.json").write_text("stale", encoding="utf-8")
    (output_dir / "tokenizer_config.json").write_text(json.dumps({
        "bos_token":
        "<s>",
        "chat_template":
        source,
    }),
                                                      encoding="utf-8")

    result = Path(write_chat_template(str(model_dir), str(output_dir)))

    assert result.name == "chat_template.jinja"
    assert result.read_text(encoding="utf-8") == source
    assert not (output_dir / "chat_template.json").exists()
    tokenizer_config = json.loads(
        (output_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert tokenizer_config == {"bos_token": "<s>"}


def test_rejects_duplicate_named_provider_templates(tmp_path):
    model_dir = _checkpoint(tmp_path, "qwen3")
    encoded = [{
        "name": "default",
        "template": "first"
    }, {
        "name": "default",
        "template": "second"
    }]
    (model_dir / "tokenizer_config.json").write_text(json.dumps(
        {"chat_template": encoded}),
                                                     encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate named"):
        write_chat_template(str(model_dir), str(tmp_path / "engine"))


def test_malformed_provider_config_fails(tmp_path):
    model_dir = _checkpoint(tmp_path, "qwen3")
    (model_dir / "processor_config.json").write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="failed to read"):
        write_chat_template(str(model_dir), str(tmp_path / "engine"))


def test_records_phi_raw_processor_contract(tmp_path):
    model_dir = _checkpoint(tmp_path, "phi4mm", "{{ messages[0].content }}")
    (model_dir / "processor_config.json").write_text(json.dumps({
        "auto_map": {
            "AutoProcessor": "processing_phi4mm.Phi4MMProcessor"
        },
        "processor_class":
        "Phi4MMProcessor",
    }),
                                                     encoding="utf-8")

    result = Path(write_chat_template(str(model_dir),
                                      str(tmp_path / "engine")))

    assert (result.parent / "chat_template.processor").read_text(
        encoding="utf-8") == "phi4mm\n"


def test_records_legacy_internvl_raw_processor_contract(tmp_path):
    model_dir = _checkpoint(tmp_path, "internvl_chat",
                            "{{ messages[0].content }}")

    result = Path(write_chat_template(str(model_dir),
                                      str(tmp_path / "engine")))

    assert (result.parent / "chat_template.processor").read_text(
        encoding="utf-8") == "internvl\n"


def test_preserves_explicit_raw_processor_when_rebuilding_artifacts(tmp_path):
    model_dir = _checkpoint(tmp_path, "phi4mm")
    (model_dir / "chat_template.jinja").write_text("{{ messages[0].content }}",
                                                   encoding="utf-8")
    (model_dir / "chat_template.processor").write_text("phi4mm\n",
                                                       encoding="utf-8")

    write_chat_template(str(model_dir), str(model_dir))

    assert (model_dir / "chat_template.processor").read_text(
        encoding="utf-8") == "phi4mm\n"
    assert not (model_dir / "chat_template.inja").exists()


@pytest.mark.parametrize(
    ("model_type", "manual_family"),
    (("alpamayo_r1", "alpamayo"), ("qwen3_asr", "qwen3_asr"),
     ("qwen3_tts", "qwen3_tts")),
)
def test_model_without_jinja_uses_explicit_native_renderer(
        tmp_path, model_type, manual_family):
    model_dir = _checkpoint(tmp_path, model_type)
    output_dir = tmp_path / "engine"

    result = write_chat_template(str(model_dir), str(output_dir))

    assert result == str(output_dir / "chat_template.model")
    assert (output_dir /
            "chat_template.model").read_text() == f"{manual_family}\n"
    assert not (output_dir / "chat_template.jinja").exists()
    assert not (output_dir / "chat_template.inja").exists()


def test_provider_template_takes_precedence_over_native_contract(tmp_path):
    source = "{{ messages[0].content }}"
    model_dir = _checkpoint(tmp_path, "qwen3_asr", source)

    result = Path(write_chat_template(str(model_dir),
                                      str(tmp_path / "engine")))

    assert result.name == "chat_template.jinja"
    assert result.read_text(encoding="utf-8") == source
    assert not (result.parent / "chat_template.inja").exists()


def test_unknown_model_provider_template_is_not_rewritten(tmp_path):
    source = "{{ bos_token + messages[0].content }}"
    model_dir = _checkpoint(tmp_path, "future_model", source)

    result = Path(write_chat_template(str(model_dir),
                                      str(tmp_path / "engine")))

    assert result.read_text(encoding="utf-8") == source
    assert not (result.parent / "chat_template.inja").exists()


def test_removes_legacy_compiled_artifacts(tmp_path):
    source = """{% macro render(message) %}{{ message.content }}{% endmacro %}
{% for message in messages %}{{ render(message) }}{% endfor %}"""
    model_dir = _checkpoint(tmp_path, "future_model", source)
    output_dir = tmp_path / "engine"
    callback_dir = output_dir / "chat_template.inja.d"
    callback_dir.mkdir(parents=True)
    (output_dir / "chat_template.inja").write_text("stale", encoding="utf-8")
    (callback_dir / "macro.inja").write_text("stale", encoding="utf-8")
    (output_dir / "processed_chat_template.json").write_text("stale",
                                                             encoding="utf-8")

    result = Path(write_chat_template(str(model_dir), str(output_dir)))

    assert result.read_text(encoding="utf-8") == source
    assert not (output_dir / "chat_template.inja").exists()
    assert not callback_dir.exists()
    assert not (output_dir / "processed_chat_template.json").exists()


def test_model_without_provider_or_native_contract_fails(tmp_path):
    model_dir = _checkpoint(tmp_path, "future_model")

    with pytest.raises(ValueError, match="does not provide a chat template"):
        write_chat_template(str(model_dir), str(tmp_path / "engine"))
