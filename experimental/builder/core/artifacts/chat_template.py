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
"""Processed chat-template selection and extraction."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def load_root_config(model_dir: str) -> Dict[str, Any]:
    """Return config.json as a dict, or empty dict if unavailable."""
    try:
        with open(os.path.join(model_dir, "config.json")) as config_file:
            return json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return {}


def _chat_template_dir() -> Path:
    """Return the packaged Edge-LLM chat-template directory."""
    return Path(
        __file__).resolve().parents[4] / "tensorrt_edgellm" / "chat_templates"


def _write_fallback_processed_chat_template(model_dir: str,
                                            output_dir: str) -> None:
    """Write a minimal runtime chat template when tokenizer extraction fails."""
    os.makedirs(output_dir, exist_ok=True)
    destination = os.path.join(output_dir, "processed_chat_template.json")
    if os.path.exists(destination):
        return
    fallback = {
        "model_path": model_dir,
        "roles": {
            "system": {
                "prefix": "",
                "suffix": "\n",
            },
            "user": {
                "prefix": "User: ",
                "suffix": "\n",
            },
            "assistant": {
                "prefix": "Assistant: ",
                "suffix": "\n",
            },
        },
        "content_types": {},
        "generation_prompt": "Assistant: ",
        "default_system_prompt": "",
    }
    with open(destination, "w") as template_file:
        json.dump(fallback, template_file, indent=2)
    logger.info("Wrote fallback processed_chat_template.json")


def try_write_packaged_chat_template(model_dir: str,
                                     output_dir: str,
                                     model_tokenizer=None) -> bool:
    """Write a packaged runtime template selected by model tokenizer code."""
    template_file = getattr(model_tokenizer, "CHAT_TEMPLATE", None)
    if not template_file:
        return False
    template_path = _chat_template_dir() / template_file
    if not template_path.exists():
        logger.warning("Packaged template file not found: %s", template_path)
        return False
    data = json.loads(template_path.read_text())
    data["model_path"] = model_dir
    os.makedirs(output_dir, exist_ok=True)
    destination = os.path.join(output_dir, "processed_chat_template.json")
    with open(destination, "w") as template_file_obj:
        json.dump(data, template_file_obj, indent=2)
    logger.info("Wrote packaged chat template %s", template_file)
    return True


def _format_chat(tokenizer, messages, **kwargs) -> str:
    """Apply a tokenizer chat template and return text.

    Probes render with thinking disabled unless told otherwise: templates
    that rewrite the conversation in thinking mode (Qwen3.8 injects reasoning
    instructions into the system block and treats an undefined flag as
    enabled) would otherwise yield prefixes that never match a request.
    """
    kwargs.setdefault("enable_thinking", False)
    try:
        return tokenizer.apply_chat_template(messages,
                                             tokenize=False,
                                             **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages,
                                             tokenize=False,
                                             **kwargs)


def _format_chat_generation(tokenizer, messages, enable_thinking: bool) -> str:
    """Apply generation prompt formatting across tokenizer variants."""
    kwargs = {"add_generation_prompt": True}
    try:
        return _format_chat(tokenizer,
                            messages,
                            enable_thinking=enable_thinking,
                            **kwargs)
    except TypeError:
        return _format_chat(tokenizer, messages, **kwargs)


def _extract_prefix_suffix(text: str, placeholder: str):
    """Split text around a placeholder."""
    index = text.find(placeholder)
    if index == -1:
        return "", ""
    return text[:index], text[index + len(placeholder):]


def process_chat_template(model_dir: str,
                          output_dir: str,
                          model_tokenizer=None) -> None:
    """Extract a runtime chat template without importing the main package.

    The main ``tensorrt_edgellm`` package imports every registered model at
    module import time.  The checkpoint-direct builder only needs tokenizer
    behavior here, so keep this path narrow and avoid optional model-family
    imports such as Gemma4 vision/audio dependencies.
    """
    if try_write_packaged_chat_template(model_dir, output_dir,
                                        model_tokenizer):
        return
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.warning("transformers is unavailable; chat template skipped")
        return

    tokenizer = None
    for search_dir in (model_dir, output_dir):
        try:
            candidate = AutoTokenizer.from_pretrained(search_dir)
        except (OSError, ValueError, ImportError, KeyError, AttributeError):
            continue
        if getattr(candidate, "chat_template", None):
            tokenizer = candidate
            break
    if tokenizer is None:
        logger.debug("No chat template found in %s", model_dir)
        return

    system_text = "<placeholder_system_prompt>"
    user_text = "<placeholder_user_text>"
    assistant_text = "<placeholder_assistant_text>"
    system_message = {"role": "system", "content": system_text}
    user_message = {"role": "user", "content": user_text}
    assistant_message = {"role": "assistant", "content": assistant_text}

    try:
        user_formatted = _format_chat(tokenizer,
                                      [system_message, user_message],
                                      add_generation_prompt=False)

        system_formatted = None
        try:
            system_formatted = _format_chat(tokenizer, [system_message])
        except Exception:
            pass

        if system_formatted is not None:
            system_prefix, system_suffix = _extract_prefix_suffix(
                system_formatted, system_text)
        else:
            system_prefix, system_suffix = _extract_prefix_suffix(
                user_formatted, system_text)

        user_only_formatted = None
        try:
            user_only_formatted = _format_chat(tokenizer, [user_message])
        except Exception:
            pass
        if (user_only_formatted is not None
                and system_prefix not in (_extract_prefix_suffix(
                    user_only_formatted, user_text)[0] or "")):
            user_prefix, user_suffix = _extract_prefix_suffix(
                user_only_formatted, user_text)
        elif system_formatted is not None:
            user_prefix, user_suffix = _extract_prefix_suffix(
                user_formatted[len(system_formatted):], user_text)
        else:
            system_end = user_formatted.find(system_text) + len(system_text)
            user_prefix, user_suffix = _extract_prefix_suffix(
                user_formatted[system_end:], user_text)

        if user_prefix and user_prefix in system_suffix:
            system_suffix = system_suffix[:system_suffix.find(user_prefix)]

        assistant_formatted = _format_chat(
            tokenizer, [system_message, user_message, assistant_message])
        assistant_prefix, assistant_suffix = _extract_prefix_suffix(
            assistant_formatted[len(user_formatted):], assistant_text)

        generation_formatted = _format_chat_generation(
            tokenizer, [system_message, user_message], enable_thinking=False)
        generation_prompt = generation_formatted[len(user_formatted):]
        if not generation_prompt and generation_formatted != user_formatted:
            common = 0
            for index, (lhs, rhs) in enumerate(
                    zip(user_formatted, generation_formatted)):
                if lhs != rhs:
                    break
                common = index + 1
            generation_prompt = generation_formatted[common:]

        generation_prompt_thinking = None
        try:
            # Slice against a thinking-mode baseline: a template may render a
            # longer conversation in thinking mode, so the non-thinking
            # baseline offset would cut into the wrong place.
            thinking_base = _format_chat(tokenizer,
                                         [system_message, user_message],
                                         add_generation_prompt=False,
                                         enable_thinking=True)
            thinking_formatted = _format_chat_generation(
                tokenizer, [system_message, user_message],
                enable_thinking=True)
            candidate = thinking_formatted[len(thinking_base):]
            if candidate != generation_prompt:
                generation_prompt_thinking = candidate
        except (TypeError, ValueError, KeyError):
            pass

        default_system_prompt = ""
        if user_only_formatted is not None and system_prefix:
            system_start = user_only_formatted.find(system_prefix)
            if system_start != -1:
                content_start = system_start + len(system_prefix)
                content_end = user_only_formatted.find(system_suffix,
                                                       content_start)
                if content_end != -1:
                    candidate = user_only_formatted[content_start:content_end]
                    if candidate and candidate != system_text:
                        default_system_prompt = candidate
                    elif candidate == "":
                        empty_system = system_prefix + system_suffix
                        if not user_prefix.startswith(empty_system):
                            user_prefix = empty_system + user_prefix

        bos_token = getattr(tokenizer, "bos_token", None)
        prompt_prefix = ""
        if bos_token:
            bos_token = str(bos_token)
            if system_prefix.startswith(bos_token) or user_prefix.startswith(
                    bos_token):
                prompt_prefix = bos_token
                if system_prefix.startswith(bos_token):
                    system_prefix = system_prefix[len(bos_token):]
                if user_prefix.startswith(bos_token):
                    user_prefix = user_prefix[len(bos_token):]
                if assistant_prefix.startswith(bos_token):
                    assistant_prefix = assistant_prefix[len(bos_token):]

        data: Dict[str, Any] = {
            "model_path": model_dir,
            "roles": {
                "system": {
                    "prefix": system_prefix,
                    "suffix": system_suffix,
                },
                "user": {
                    "prefix": user_prefix,
                    "suffix": user_suffix,
                },
                "assistant": {
                    "prefix": assistant_prefix,
                    "suffix": assistant_suffix,
                },
            },
            "content_types": {},
            "generation_prompt": generation_prompt,
            "default_system_prompt": default_system_prompt,
        }
        if prompt_prefix:
            data["prompt_prefix"] = prompt_prefix
        if generation_prompt_thinking is not None:
            data["generation_prompt_thinking"] = generation_prompt_thinking

        os.makedirs(output_dir, exist_ok=True)
        destination = os.path.join(output_dir, "processed_chat_template.json")
        with open(destination, "w") as template_file:
            json.dump(data, template_file, indent=2)
        logger.info("Chat template saved to %s", destination)
    except Exception as error:
        logger.warning("Chat template extraction failed for %s: %s", model_dir,
                       error)


def write_processed_chat_template(model_dir: str,
                                  output_dir: str,
                                  model_tokenizer=None) -> None:
    """Generate processed_chat_template.json from tokenizer metadata."""
    destination = os.path.join(output_dir, "processed_chat_template.json")
    if not os.path.isfile(destination):
        process_chat_template(model_dir, output_dir, model_tokenizer)
        if not os.path.isfile(destination):
            _write_fallback_processed_chat_template(model_dir, output_dir)

    root_config = load_root_config(model_dir)
    if model_tokenizer is not None and hasattr(model_tokenizer,
                                               "patch_chat_template"):
        with open(destination) as template_file:
            template = json.load(template_file)
        model_tokenizer.patch_chat_template(template, root_config)
        with open(destination, "w") as template_file:
            json.dump(template, template_file, indent=2)
