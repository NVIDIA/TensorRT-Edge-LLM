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
"""Copy the checkpoint's chat template into the runtime artifacts."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["write_chat_template"]

_RUNTIME_ARTIFACTS = ("chat_template.jinja", "chat_template.model",
                      "chat_template.processor")
_UNSUPPORTED_NAMED_TEMPLATE_DIR = "additional_chat_templates"
_LEGACY_INJA_ARTIFACTS = ("chat_template.inja", "chat_template.inja.d")
_PROVIDER_JSON_METADATA = "chat_template.json"
_LEGACY_RUNTIME_JSON_ARTIFACT = "processed_chat_template.json"

# These checkpoints do not publish a Jinja template. Their prompt contracts
# depend on model-specific runtime data, so they use explicit native renderers.
_MANUAL_BY_MODEL_TYPE = {
    "alpamayo_r1": "alpamayo",
    "qwen3_asr": "qwen3_asr",
    "qwen3_asr_thinker": "qwen3_asr",
    "qwen3_tts": "qwen3_tts",
    "qwen3_tts_talker": "qwen3_tts",
    "qwen3_tts_code_predictor": "qwen3_tts",
    "qwen3_tts_code2wav": "qwen3_tts",
}

# These provider processors require message content to be normalized before
# their tokenizer-owned Jinja template is rendered. The C++ runtime implements
# the same documented contract without loading Python or Transformers.
_RAW_PROCESSOR_BY_CLASS = {
    "Phi4MMProcessor": "phi4mm",
    # Nemotron-Omni's tokenizer-owned template renders content as a string and
    # expects the media sentinels (<image>/<audio>/<video>) already inlined; the
    # C++ runtime inlines them without loading Python or Transformers.
    "NemotronH_Nano_Omni_Reasoning_V3Processor": "nemotron_omni",
}
_RAW_PROCESSOR_BY_MODEL_TYPE = {
    # Legacy InternVL's model.chat() path inserts these media sentinels before
    # applying its text-only tokenizer template.
    "internvl_chat": "internvl",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"failed to read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _provider_template(value: Any, source: Path) -> str | None:
    """Return the checkpoint's single default provider template."""
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{source} contains an empty chat template")
        return value

    if isinstance(value, dict):
        named = value
    elif isinstance(value, list):
        if any(not isinstance(item, dict)
               or not isinstance(item.get("name"), str) or not isinstance(
                   item.get("template"), str) or not item["template"].strip()
               for item in value):
            raise ValueError(
                f"{source} contains an invalid named chat template")
        named = {item["name"]: item["template"] for item in value}
        if len(named) != len(value):
            raise ValueError(
                f"{source} contains duplicate named chat templates")
    else:
        raise ValueError(
            f"{source} has an invalid chat_template value of type "
            f"{type(value).__name__}")

    if set(named) != {"default"}:
        raise ValueError(
            f"{source} defines named chat templates, which are unsupported; "
            "the checkpoint must provide one default template")
    template = named["default"]
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"{source} contains an invalid default chat template")
    return template


def _embedded_provider_template(model_dir: Path) -> str | None:
    # Transformers may serialize provider-owned Jinja in a JSON metadata
    # container. It is an import source only and is never a runtime artifact.
    for filename in ("chat_template.json", "processor_config.json",
                     "tokenizer_config.json"):
        path = model_dir / filename
        template = _provider_template(
            _read_json(path).get("chat_template"), path)
        if template is not None:
            return template
    return None


def _read_provider_template(path: Path) -> str:
    template = path.read_text(encoding="utf-8")
    if not template.strip():
        raise ValueError(f"{path} contains an empty chat template")
    return template


def _file_provider_template(model_dir: Path) -> Path | None:
    """Return a standalone provider Jinja file when present."""
    default = model_dir / "chat_template.jinja"
    if default.is_file():
        _read_provider_template(default)
    additional = model_dir / _UNSUPPORTED_NAMED_TEMPLATE_DIR
    if additional.exists():
        raise ValueError(
            f"{additional} contains unsupported named chat templates; "
            "the checkpoint must provide one default template")
    return default if default.is_file() else None


def _raw_processor(model_dir: Path) -> str | None:
    processor_classes = set()
    processors = set()
    marker = model_dir / "chat_template.processor"
    if marker.is_file():
        processor = marker.read_text(encoding="utf-8").strip()
        supported = (set(_RAW_PROCESSOR_BY_CLASS.values())
                     | set(_RAW_PROCESSOR_BY_MODEL_TYPE.values()))
        if processor not in supported:
            raise ValueError(f"{marker} names unsupported raw processor "
                             f"{processor!r}")
        processors.add(processor)

    for filename in ("processor_config.json", "preprocessor_config.json",
                     "tokenizer_config.json"):
        config = _read_json(model_dir / filename)
        processor_class = config.get("processor_class")
        if isinstance(processor_class, str):
            processor_classes.add(processor_class)
        auto_map = config.get("auto_map")
        auto_processor = (auto_map.get("AutoProcessor") if isinstance(
            auto_map, dict) else None)
        if isinstance(auto_processor, str):
            processor_classes.add(auto_processor.rsplit(".", 1)[-1])

    processors.update(_RAW_PROCESSOR_BY_CLASS[name]
                      for name in processor_classes
                      if name in _RAW_PROCESSOR_BY_CLASS)
    model_type = _get_model_type(model_dir)
    if model_type in _RAW_PROCESSOR_BY_MODEL_TYPE:
        processors.add(_RAW_PROCESSOR_BY_MODEL_TYPE[model_type])
    if len(processors) > 1:
        raise ValueError(f"checkpoint declares conflicting raw processors: "
                         f"{sorted(processors)}")
    return next(iter(processors), None)


def _get_model_type(model_dir: str | Path) -> str:
    return str(
        _read_json(Path(model_dir) / "config.json").get("model_type", ""))


def _write_raw_processor(output: Path, raw_processor: str | None) -> None:
    if raw_processor is not None:
        (output / "chat_template.processor").write_text(raw_processor + "\n",
                                                        encoding="utf-8")


def _remove_json_runtime_templates(source: Path, output: Path) -> None:
    """Remove JSON copies after materializing the runtime Jinja artifact."""
    (output / _LEGACY_RUNTIME_JSON_ARTIFACT).unlink(missing_ok=True)
    if source.resolve() == output.resolve():
        return

    (output / _PROVIDER_JSON_METADATA).unlink(missing_ok=True)
    for filename in ("processor_config.json", "tokenizer_config.json"):
        path = output / filename
        config = _read_json(path)
        if "chat_template" not in config:
            continue
        del config["chat_template"]
        path.write_text(  # NOSONAR - the artifact filename is fixed above.
            json.dumps(config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")


def _clear_runtime_templates(source: Path, output: Path,
                             provider_file: Path | None) -> None:
    preserved = ({provider_file.resolve()}
                 if provider_file is not None else set())
    for stale in _RUNTIME_ARTIFACTS:
        stale_path = output / stale
        if stale_path.exists() and stale_path.resolve() not in preserved:
            stale_path.unlink()

    additional = output / _UNSUPPORTED_NAMED_TEMPLATE_DIR
    source_additional = source / _UNSUPPORTED_NAMED_TEMPLATE_DIR
    copied_additional = (additional.exists() and additional.resolve()
                         != source_additional.resolve())
    if copied_additional:
        if additional.is_dir():
            shutil.rmtree(additional)
        else:
            additional.unlink()

    for stale in _LEGACY_INJA_ARTIFACTS:
        stale_path = output / stale
        if stale_path.is_dir():
            shutil.rmtree(stale_path)
        else:
            stale_path.unlink(missing_ok=True)
    _remove_json_runtime_templates(source, output)  # NOSONAR: fixed names


def write_chat_template(model_dir: str, output_dir: str) -> str:
    """Copy the provider Jinja template for direct Pantor Inja loading."""
    source = Path(model_dir)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    provider_file = _file_provider_template(source)
    embedded_template = (None if provider_file is not None else
                         _embedded_provider_template(source))
    raw_processor = (_raw_processor(source) if provider_file is not None
                     or embedded_template is not None else None)

    _clear_runtime_templates(source, output, provider_file)

    destination = output / "chat_template.jinja"
    if provider_file is not None:
        if destination.resolve() != provider_file.resolve():
            shutil.copyfile(provider_file, destination)
    elif embedded_template is not None:
        destination.write_text(  # NOSONAR - destination has a fixed filename.
            embedded_template,
            encoding="utf-8")

    if provider_file is not None or embedded_template is not None:
        _write_raw_processor(output, raw_processor)
        logger.info("Copied provider chat template to %s", destination)
        return str(destination)

    model_type = _get_model_type(source)
    manual = _MANUAL_BY_MODEL_TYPE.get(model_type)
    if manual is None:
        raise ValueError(
            f"model_type={model_type!r} does not provide a chat template")

    destination = output / "chat_template.model"
    destination.write_text(manual + "\n", encoding="utf-8")
    logger.info("Wrote native chat renderer %s to %s", manual, destination)
    return str(destination)
