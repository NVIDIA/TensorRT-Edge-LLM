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
"""Resolve CLI settings without loading CUDA or TensorRT."""

import argparse
import dataclasses
import json
import math
import os
import pathlib

from experimental.server import config as server_config
from experimental.server.runtime import engine, engine_build

RUNTIME_OPTIONS = {
    "cache_dir": str,
    "engine_cache_max_size_gb": float,
    "max_input_len": int,
    "max_kv_cache_capacity": int,
    "max_batch_size": int,
    "clear_engine_cache": bool,
    "draft_top_k": int,
    "draft_step": int,
    "verify_tree_size": int,
    "speculative_config": dict,
    "context_cache_config": dict,
}
BUILD_OPTIONS = {
    "tp_size": int,
    "max_image_tokens": int,
    "max_image_tokens_per_image": int,
    "plugin_path": str,
}
SAMPLING_OPTIONS = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "max_tokens": int,
    "enable_thinking": bool,
    "reasoning_effort": str,
    "disable_spec_decode": bool,
    "num_logprobs": int,
    "stop": list,
    "logit_bias": dict,
    "guided_decoding": dict,
    "skip_special_tokens": bool,
    "reuse_context": bool,
    "cache_generated_tokens": bool,
}
TEXT_OPTIONS = (
    "prompt",
    "prompt_file",
    "system",
    "system_file",
    "template",
    "json_schema",
)
BOOL_OPTIONS = ("verbose", "timings", "no_color", "no_metrics")
OPTION_TYPES = {
    **RUNTIME_OPTIONS,
    **BUILD_OPTIONS,
    **SAMPLING_OPTIONS,
    **dict.fromkeys(TEXT_OPTIONS, str),
    **dict.fromkeys(BOOL_OPTIONS, bool),
}
OPTION_HELP = {
    "max_image_tokens":
    "Maximum total visual tokens in the visual engine profile",
    "max_image_tokens_per_image":
    "Maximum visual tokens per image in the visual engine profile",
    "draft_top_k":
    "Candidate tokens retained per draft step. Defaults depend on the speculative method",
    "draft_step":
    "Number of draft steps. DFlash, JetSpec, and dSpark require 1",
    "verify_tree_size":
    "Number of tokens in the runtime verification tree. Must fit the engine profile",
    "disable_spec_decode":
    "Disable speculative decoding for generation with a speculative engine",
    "verbose":
    "Show detailed startup and runtime logs",
    "no_color":
    "Disable colored terminal output",
    "no_metrics":
    "Hide per-response performance metrics",
    "timings":
    "Compatibility option. Timings are shown by default",
    "temperature":
    "Sampling temperature. 0 selects greedy decoding",
    "top_p":
    "Nucleus sampling probability threshold",
    "top_k":
    "Limit sampling to the top K tokens",
    "max_tokens":
    "Maximum generated tokens per response, including thinking",
    "enable_thinking":
    "Enable the model's reasoning output",
    "skip_special_tokens":
    "Hide tokenizer-designated control tokens",
    "cache_dir":
    "Directory for cached checkpoints and engines",
    "engine_cache_max_size_gb":
    "Engine-cache size budget in GiB",
    "max_input_len":
    "Maximum prompt tokens, including conversation history",
    "max_kv_cache_capacity":
    "Maximum total tokens per sequence, including prompt and generated tokens",
    "max_batch_size":
    "Engine batch capacity. Interactive requests remain serial",
    "speculative_config":
    'JSON: {"method":"mtp","num_speculative_tokens":3}. Paired drafts also use "model"',
    "context_cache_config":
    'JSON context-cache settings, including "enabled" and "max_records"',
    "guided_decoding":
    'JSON with one mode: json_schema, json_object, regex, ebnf, or choice',
    "logit_bias":
    'JSON mapping token IDs to biases in [-100, 100]',
    "clear_engine_cache":
    "Remove compiled engine bundles before preparing this model. Downloaded checkpoints are preserved",
    "reasoning_effort":
    "Model-specific reasoning effort passed to the chat formatter",
    "num_logprobs":
    "Collect native token logprobs (not displayed by the text UI)",
    "tp_size":
    "Build tensor-parallel engines for this many ranks",
    "plugin_path":
    "Plugin library used for engine construction",
    "reuse_context":
    "Allow this request to look up prefixes when context caching is enabled",
    "cache_generated_tokens":
    "Include generated tokens when committing a context-cache record",
}


@dataclasses.dataclass
class Config:
    """Validated settings passed to the frontend and shared LLM API."""

    model: str
    runtime: dict
    sampling: engine.SamplingParams
    messages: list
    prompt: str | None = None
    verbose: bool = False
    timings: bool = False
    no_color: bool = False
    no_metrics: bool = False


def parser(prog="trtedge chat") -> argparse.ArgumentParser:
    """Build the trtedge command parser."""
    chat = argparse.ArgumentParser(
        prog=prog,
        description="Chat locally with NVIDIA TensorRT Edge-LLM.",
        usage="%(prog)s MODEL [options]")
    chat.add_argument(
        "model", help="Local checkpoint directory or Hugging Face model ID")
    groups = {
        name: chat.add_argument_group(name)
        for name in ("Input and configuration", "Generation",
                     "Thinking and speculation", "Engine and cache",
                     "Output and diagnostics")
    }
    inputs = groups["Input and configuration"]
    inputs.add_argument("--config",
                        help="JSON settings file (or EDGELLM_CONFIG)")
    for name, value_type in {
            **RUNTIME_OPTIONS,
            **BUILD_OPTIONS,
            **SAMPLING_OPTIONS
    }.items():
        if name in ("enable_thinking", "reasoning_effort",
                    "disable_spec_decode", "speculative_config", "draft_top_k",
                    "draft_step", "verify_tree_size"):
            group = groups["Thinking and speculation"]
        elif name in SAMPLING_OPTIONS:
            group = groups["Generation"]
        else:
            group = groups["Engine and cache"]
        option = "--" + name.replace("_", "-")
        help_text = OPTION_HELP.get(name)
        if value_type is bool:
            group.add_argument(option,
                               action=argparse.BooleanOptionalAction,
                               default=None,
                               help=help_text)
        elif value_type is list:
            group.add_argument(option,
                               action="append",
                               metavar="TEXT",
                               help="Stop string. Repeat for multiple values")
        elif value_type is dict:
            group.add_argument(option,
                               metavar="JSON",
                               help=help_text or "JSON configuration object")
        else:
            group.add_argument(option, type=value_type, help=help_text)
    inputs.add_argument("--prompt", "-p", help="Run one turn and exit")
    inputs.add_argument("--prompt-file",
                        help="Read a one-shot prompt from a UTF-8 file")
    inputs.add_argument("--system", help="System message")
    inputs.add_argument("--system-file",
                        help="Read a system message from a UTF-8 file")
    inputs.add_argument(
        "--template",
        metavar="PATH",
        help=("Path to a JSON file containing an array of initial "
              "role/content messages"))
    groups["Generation"].add_argument(
        "--json-schema", help="Constrain output using a JSON Schema file")
    for name in BOOL_OPTIONS:
        groups["Output and diagnostics"].add_argument("--" +
                                                      name.replace("_", "-"),
                                                      action="store_true",
                                                      default=None,
                                                      help=OPTION_HELP[name])
    chat.epilog = (
        "Settings: API defaults < JSON config < EDGELLM_<OPTION> < CLI. "
        "Use lowercase underscore keys in JSON (max_tokens) and uppercase environment variables (EDGELLM_MAX_TOKENS)."
    )
    return chat


def read_text(path: str) -> str:
    """Read a user-supplied UTF-8 text file."""
    return pathlib.Path(path).expanduser().read_text(encoding="utf-8")


def _value(name, value, value_type):
    if value_type in (dict, list):
        if isinstance(value, str):
            value = json.loads(value)
        if not isinstance(value, value_type):
            raise ValueError(f"{name} must be a JSON {value_type.__name__}")
        if value_type is list and any(not isinstance(item, str) or not item
                                      for item in value):
            raise ValueError(f"{name} must contain nonempty strings")
        return value
    if value_type is bool:
        if isinstance(value,
                      str) and value.lower() in ("true", "false", "1", "0"):
            return value.lower() in ("true", "1")
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be true or false")
        return value
    if value_type is str:
        if not isinstance(value, str):
            raise ValueError(f"{name} must be text")
        return value
    if isinstance(value, bool) or (value_type is int
                                   and isinstance(value, float)):
        raise ValueError(f"{name} must be a {value_type.__name__}")
    try:
        number = value_type(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid {name}: {value!r}") from error
    if value_type is int and not 0 <= number < (1 << 31):
        raise ValueError(f"{name} must fit in a nonnegative 32-bit integer")
    if value_type is float and not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    if (number < 0 or
        (name not in ("temperature", "top_k", "num_logprobs") and number == 0)
            or (name == "top_p" and number > 1)):
        raise ValueError(f"{name} is outside its supported range")
    return number


def _runtime_options(values, model):
    runtime = {
        name: values[name]
        for name in RUNTIME_OPTIONS if name in values
    }
    if "speculative_config" in runtime:
        if not runtime["speculative_config"]:
            raise ValueError("speculative_config must specify a method")
        runtime["speculative_config"] = server_config.SpeculativeConfig.parse(
            runtime["speculative_config"])
    if "context_cache_config" in runtime:
        runtime[
            "context_cache_config"] = server_config.ContextCacheConfig.parse(
                runtime["context_cache_config"])
    if any(name in values for name in BUILD_OPTIONS):
        defaults = server_config.ModelConfig(model=model)
        profile = {
            name: values.get(name, getattr(defaults, name))
            for name in ("max_input_len", "max_batch_size",
                         "max_kv_cache_capacity")
        }
        runtime["build_options"] = engine_build.BuildOptions(
            **profile,
            **{name: values[name]
               for name in BUILD_OPTIONS if name in values})
    return runtime


def _read_messages(path):
    messages = json.loads(read_text(path))
    if not isinstance(messages, list):
        raise ValueError("Template must be an array of role/content messages")
    for message in messages:
        if not isinstance(message, dict) or set(message) != {
                "role", "content"
        }:
            raise ValueError("Each template message needs role and content")
        if message["role"] not in ("system", "user", "assistant"):
            raise ValueError(
                "Template roles must be system, user, or assistant")
        if not isinstance(message["content"], str):
            raise ValueError("Template content must be text")
    return messages


def resolve(args: argparse.Namespace) -> Config:
    """Apply config, environment, and CLI precedence; retain API defaults."""
    values = {}
    config_path = args.config or os.environ.get("EDGELLM_CONFIG")
    if config_path:
        values = json.loads(read_text(config_path))
        if not isinstance(values, dict):
            raise ValueError("Config must be a JSON object")
        unknown = values.keys() - OPTION_TYPES.keys()
        if unknown:
            raise ValueError("Unknown config options: " +
                             ", ".join(sorted(unknown)))
    cli_values = vars(args)
    for name, value_type in OPTION_TYPES.items():
        env_value = os.environ.get("EDGELLM_" + name.upper())
        if env_value is not None:
            values[name] = env_value
        if cli_values[name] is not None:
            values[name] = cli_values[name]
        if name in values:
            values[name] = _value(name, values[name], value_type)

    model = args.model
    model_path = pathlib.Path(model).expanduser()
    if model_path.is_dir():
        model = str(model_path.resolve())
    elif model_path.is_absolute() or model.startswith(("./", "../", "~")):
        raise ValueError(
            f"Checkpoint directory does not exist: {model_path.resolve()}")
    for inline, file_option in (("prompt", "prompt_file"), ("system",
                                                            "system_file")):
        if inline in values and file_option in values:
            raise ValueError(f"Use either {inline} or {file_option}")
        if file_option in values:
            values[inline] = read_text(values[file_option])
    messages = _read_messages(
        values["template"]) if "template" in values else []
    if "system" in values:
        messages = [
            message for message in messages if message["role"] != "system"
        ]
        messages.insert(0, {"role": "system", "content": values["system"]})
    sampling = engine.SamplingParams(
        **{name: values[name]
           for name in SAMPLING_OPTIONS if name in values})
    sampling.logit_bias = engine._normalize_logit_bias(sampling.logit_bias)
    sampling.guided_decoding = engine._normalize_guided_decoding(
        sampling.guided_decoding)
    if "json_schema" in values:
        if sampling.guided_decoding is not None:
            raise ValueError("Use either json_schema or guided_decoding")
        schema = json.loads(read_text(values["json_schema"]))
        if not isinstance(schema, (dict, bool)):
            raise ValueError("JSON Schema must be an object or boolean")
        sampling.guided_decoding = ("json_schema", json.dumps(schema))
    return Config(
        model=model,
        runtime=_runtime_options(values, model),
        sampling=sampling,
        messages=messages,
        prompt=values.get("prompt"),
        **{name: values.get(name, False)
           for name in BOOL_OPTIONS},
    )
