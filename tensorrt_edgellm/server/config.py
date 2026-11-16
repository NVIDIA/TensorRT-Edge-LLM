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
"""Validated launch configuration for the inference server."""

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Union

_INT32_MAX = 2**31 - 1


class ServerConfigError(ValueError):
    """Raised when a launch option cannot be honored by Edge-LLM."""


# In-flight batching scheduling options at their defaults; any other value needs
# --enable-in-flight-batching.
IFB_SCHEDULING_DEFAULTS = {
    "ifb_scheduling_mode": "auto",
    "max_num_tokens": 0,
    "max_chunk_tokens": 0,
    "scheduling_policy": "fcfs",
    "preemption": "none",
    "priority_reserved_slots": 0,
    "max_pause_steps": 0,
}

# Option names in launch errors, keyed by the spellings the Python API and the
# C++ RequestEngine use; scheduling_option_flags() turns them into CLI flags.
_SCHEDULING_FLAG_NAMES = {
    spelled: "--" + name.replace("_", "-")
    for name in ("ifb_scheduling_mode", "max_num_tokens", "max_chunk_tokens",
                 "scheduling_policy", "priority_reserved_slots",
                 "max_pause_steps", "max_batch_size")
    for spelled in (name, re.sub(r"_(\w)", lambda m: m.group(1).upper(), name))
}
# "preemption" is also an English word, so only "preemption=..." names the
# option.
_SCHEDULING_NAME_RE = re.compile(
    r"(?<![\w-])(?:(" +
    "|".join(sorted(_SCHEDULING_FLAG_NAMES, key=len, reverse=True)) +
    r")\b|(preemption)(?==))(=)?")


def scheduling_option_flags(message: str) -> str:
    """Rewrite scheduling option names in a launch error as the server's
    command-line flags (``ifbSchedulingMode=mixed`` -> ``--ifb-scheduling-mode
    mixed``)."""

    def flag(match: "re.Match[str]") -> str:
        name = match.group(1) or match.group(2)
        spelled = _SCHEDULING_FLAG_NAMES.get(name, "--preemption")
        return spelled + (" " if match.group(3) else "")

    return _SCHEDULING_NAME_RE.sub(flag, message)


def validate_ifb_scheduling_options(
        enable_in_flight_batching: bool,
        options: Mapping[str, Any],
        max_batch_size: Optional[int] = None) -> None:
    """Reject scheduling options before the engine is built.

    Without in-flight batching every option must keep its default. With it,
    the options must be int32 integers and pass the RequestEngine's own
    validation for an engine built for ``max_batch_size`` (None when that is
    not known before the build, which leaves the check to the engine's
    startup). The step budgets and what the deployment supports are checked
    when the engine starts.
    """
    if not enable_in_flight_batching:
        changed = sorted(name
                         for name, default in IFB_SCHEDULING_DEFAULTS.items()
                         if options[name] != default)
        if changed:
            raise ServerConfigError(
                ", ".join(changed) +
                " only apply with --enable-in-flight-batching")
        return
    for name in ("max_num_tokens", "max_chunk_tokens",
                 "priority_reserved_slots", "max_pause_steps"):
        value = options[name]
        if (isinstance(value, bool) or not isinstance(value, int)
                or not -(2**31) <= value <= _INT32_MAX):
            raise ServerConfigError(f"{name} must be an int32 integer")
    if max_batch_size is None:
        return
    from .runtime.engine import _import_runtime

    try:
        _import_runtime().validate_scheduling_options(
            **options, max_batch_size=max_batch_size)
    except ValueError as exc:
        raise ServerConfigError(str(exc)) from exc


@dataclass(frozen=True)
class SpeculativeConfig:
    """Speculative model pair, runtime proposal length, and tree shape."""

    method: str
    num_speculative_tokens: Optional[int] = None
    draft_model: str = ""
    draft_top_k: Optional[int] = None
    verify_tokens: Optional[int] = None

    @classmethod
    def parse(
        cls, value: Union[str, Mapping[str, Any], "SpeculativeConfig", None]
    ) -> Optional["SpeculativeConfig"]:
        if not value:
            return None
        if isinstance(value, cls):
            raw = {
                "method": value.method,
                "num_speculative_tokens": value.num_speculative_tokens,
                # An absent draft model is stored as "", which re-parsing would
                # reject as a malformed model id. MTP is the only method that
                # legitimately has none, so re-parsing must round-trip.
                "model": value.draft_model or None,
                "draft_top_k": value.draft_top_k,
                "verify_tokens": value.verify_tokens,
            }
        elif isinstance(value, str):
            try:
                raw = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ServerConfigError(
                    "--speculative-config must be a JSON object") from exc
        else:
            raw = dict(value)
        if not isinstance(raw, dict):
            raise ServerConfigError(
                "--speculative-config must be a JSON object")

        supported_keys = {
            "method", "num_speculative_tokens", "model", "draft_top_k",
            "verify_tokens"
        }
        unsupported = sorted(set(raw) - supported_keys)
        if unsupported:
            raise ServerConfigError("unsupported --speculative-config keys: " +
                                    ", ".join(unsupported))

        method = raw.get("method")
        if not isinstance(method, str) or method not in {
                "eagle3", "mtp", "dflash", "jetspec", "domino", "dspark"
        }:
            raise ServerConfigError(
                "--speculative-config.method must be one of eagle3, mtp, "
                "dflash, jetspec, domino, or dspark")
        num_tokens = _optional_int_at_least(raw, "num_speculative_tokens", 1)
        draft_top_k = _optional_int_at_least(raw, "draft_top_k", 1)
        verify_tokens = _optional_int_at_least(raw, "verify_tokens", 2)
        draft_model = raw.get("model")
        if draft_model is not None and (not isinstance(draft_model, str)
                                        or not draft_model.strip()):
            raise ServerConfigError(
                "--speculative-config.model must be a checkpoint path or "
                "Hugging Face model ID")
        if method in {"eagle3", "dflash", "jetspec", "domino", "dspark"
                      } and not draft_model:
            raise ServerConfigError(
                f"--speculative-config.method {method!r} requires "
                "speculative-config.model")
        return cls(method=method,
                   num_speculative_tokens=num_tokens,
                   draft_model=draft_model or "",
                   draft_top_k=draft_top_k,
                   verify_tokens=verify_tokens)


def _optional_int_at_least(raw: Mapping[str, Any], key: str,
                           minimum: int) -> Optional[int]:
    value = raw.get(key)
    if value is not None and (isinstance(value, bool) or
                              not isinstance(value, int) or value < minimum):
        raise ServerConfigError(
            f"--speculative-config.{key} must be an integer >= {minimum}")
    return value


#: Requests that may wait behind the running batch before the server refuses
#: (HTTP 429) and the direct Python API blocks. One number for both entry points.
DEFAULT_MAX_QUEUED_REQUESTS = 16


@dataclass(frozen=True)
class ContextCacheConfig:
    """Deployment-scoped KV-cache reuse configuration."""

    enabled: bool = False
    max_records: int = 1024
    # Total snapshots on a model with recurrent layers; 0 keeps 4 x max batch size.
    max_snapshots: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ServerConfigError("context_cache.enabled must be boolean")
        for name in ("max_records", "max_snapshots"):
            value = getattr(self, name)
            if isinstance(value,
                          bool) or not isinstance(value, int) or value < 0:
                raise ServerConfigError(
                    f"context_cache.{name} must be a non-negative integer")
            if value > _INT32_MAX:
                raise ServerConfigError(
                    f"context_cache.{name} must fit a signed 32-bit integer")
        if not self.enabled and (self.max_records != 1024
                                 or self.max_snapshots):
            raise ServerConfigError(
                "context-cache tuning requires context reuse to be enabled")

    @classmethod
    def parse(
        cls,
        value: Optional[Union[Mapping[str, Any], "ContextCacheConfig"]],
    ) -> "ContextCacheConfig":
        """Normalize an HLAPI mapping without accepting ignored keys."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ServerConfigError(
                "context_cache_config must be a mapping or ContextCacheConfig")
        raw = dict(value)
        supported = {"enabled", "max_records", "max_snapshots"}
        unsupported = sorted(set(raw) - supported)
        if unsupported:
            raise ServerConfigError("unsupported context-cache keys: " +
                                    ", ".join(unsupported))
        return cls(**raw)


@dataclass(frozen=True)
class PleRuntimeConfig:
    """Runtime PLE residency and source selection; independent of engine builds."""

    residency: str = "resident"
    cache_slots: int = 0
    file_path: str = ""

    def __post_init__(self) -> None:
        if self.residency not in ("resident", "cached"):
            raise ServerConfigError("PLE residency must be resident or cached")
        if (isinstance(self.cache_slots, bool)
                or not isinstance(self.cache_slots, int)
                or not 0 <= self.cache_slots <= _INT32_MAX):
            raise ServerConfigError(
                "PLE cache_slots must be a non-negative signed 32-bit integer")
        if self.residency == "cached" and self.cache_slots < 2:
            raise ServerConfigError(
                "Cached PLE requires cache_slots >= 2, including the zero slot"
            )
        if not isinstance(self.file_path, str):
            raise ServerConfigError("PLE file_path must be a string")

    @classmethod
    def parse(
        cls, value: Optional[Union[Mapping[str, Any], "PleRuntimeConfig"]]
    ) -> "PleRuntimeConfig":
        """Normalize Python settings without accepting ignored keys."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ServerConfigError(
                "ple_config must be a mapping or PleRuntimeConfig")
        raw = dict(value)
        unsupported = sorted(
            set(raw) - {"residency", "cache_slots", "file_path"})
        if unsupported:
            raise ServerConfigError("unsupported PLE keys: " +
                                    ", ".join(unsupported))
        return cls(**raw)


@dataclass(frozen=True)
class ModelConfig:
    """Checkpoint, build-cache, and runtime-profile options."""

    model: str
    cache_dir: str = ""
    engine_cache_max_size_gb: float = 50.0
    clear_engine_cache: bool = False
    max_input_len: int = 4096
    max_batch_size: int = 1
    max_kv_cache_capacity: int = 8192
    max_spec_verify_tokens: Optional[int] = None
    max_spec_draft_tokens: Optional[int] = None
    max_image_tokens: Optional[int] = None
    max_image_tokens_per_image: Optional[int] = None
    draft_top_k: Optional[int] = None
    draft_step: Optional[int] = None
    proposal_tokens: Optional[int] = None
    verify_tokens: Optional[int] = None
    speculative_config: Optional[SpeculativeConfig] = None
    context_cache_config: ContextCacheConfig = field(
        default_factory=ContextCacheConfig)
    ple_config: PleRuntimeConfig = field(default_factory=PleRuntimeConfig)
    enable_in_flight_batching: bool = False
    ifb_scheduling_mode: str = "auto"
    max_num_tokens: int = 0
    max_chunk_tokens: int = 0
    scheduling_policy: str = "fcfs"
    preemption: str = "none"
    priority_reserved_slots: int = 0
    max_pause_steps: int = 0

    def __post_init__(self) -> None:
        if (isinstance(self.engine_cache_max_size_gb, bool)
                or not math.isfinite(self.engine_cache_max_size_gb)
                or self.engine_cache_max_size_gb <= 0):
            raise ServerConfigError(
                "engine_cache_max_size_gb must be positive")
        validate_ifb_scheduling_options(self.enable_in_flight_batching,
                                        self.ifb_scheduling_kwargs(),
                                        self.max_batch_size)

    def ifb_scheduling_kwargs(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in IFB_SCHEDULING_DEFAULTS}

    def llm_kwargs(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "cache_dir": self.cache_dir,
            "engine_cache_max_size_gb": self.engine_cache_max_size_gb,
            "clear_engine_cache": self.clear_engine_cache,
            "max_input_len": self.max_input_len,
            "max_batch_size": self.max_batch_size,
            "max_kv_cache_capacity": self.max_kv_cache_capacity,
            "max_spec_verify_tokens": self.max_spec_verify_tokens,
            "max_spec_draft_tokens": self.max_spec_draft_tokens,
            "max_image_tokens": self.max_image_tokens,
            "max_image_tokens_per_image": self.max_image_tokens_per_image,
            "draft_top_k": self.draft_top_k,
            "draft_step": self.draft_step,
            "proposal_tokens": self.proposal_tokens,
            "verify_tokens": self.verify_tokens,
            "speculative_config": self.speculative_config,
            "context_cache_config": self.context_cache_config,
            "ple_config": self.ple_config,
            "enable_in_flight_batching": self.enable_in_flight_batching,
            **self.ifb_scheduling_kwargs(),
        }


@dataclass(frozen=True)
class ApiConfig:
    """HTTP protocol and request-lifecycle options."""

    host: str = "0.0.0.0"
    port: int = 8000
    served_model_name: str = ""
    api_key: str = ""
    reasoning_parser: str = "auto"
    tool_call_parser: str = "auto"
    enable_auto_tool_choice: bool = False
    max_queued_requests: int = DEFAULT_MAX_QUEUED_REQUESTS
    queue_timeout: float = 600.0
    allowed_local_media_path: str = ""
    log_level: str = "info"
    #: A request whose priority is below this is refused with HTTP 400. At
    #: most 0, so a request without a priority (0) is always accepted.
    min_request_priority: int = 0

    def __post_init__(self) -> None:
        if self.max_queued_requests < 0:
            raise ServerConfigError("max_queued_requests must be non-negative")
        if self.queue_timeout <= 0:
            raise ServerConfigError("queue_timeout must be positive")
        if (isinstance(self.min_request_priority, bool)
                or not isinstance(self.min_request_priority, int)
                or not -(2**31) <= self.min_request_priority <= 0):
            raise ServerConfigError(
                "min_request_priority must be an int32 integer <= 0: a "
                "positive minimum would refuse every request without a "
                "priority, including every /v1/responses and /v1/messages "
                "request")


@dataclass(frozen=True)
class ServerConfig:
    """Complete validated launch configuration."""

    model: ModelConfig
    api: ApiConfig


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _bounded_port(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65535:
        raise argparse.ArgumentTypeError("must be in [1, 65535]")
    return parsed


def create_argument_parser(
        prog="tensorrt-edgellm-serve") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="TensorRT Edge-LLM OpenAI-compatible server",
    )
    parser.add_argument(
        "model",
        help="Hugging Face model ID or local checkpoint directory",
    )

    api = parser.add_argument_group("HTTP server")
    api.add_argument("--host",
                     default="0.0.0.0",
                     help="Address to bind (default: %(default)s)")
    api.add_argument("--port",
                     type=_bounded_port,
                     default=8000,
                     help="TCP port to listen on (default: %(default)s)")
    api.add_argument(
        "--served-model-name",
        default="",
        help=("API model name. Defaults to the Hugging Face model ID or local "
              "checkpoint directory name"))
    api.add_argument(
        "--api-key",
        default="",
        help=("Key required for /v1/ routes via Bearer authorization or "
              "x-api-key. Unset disables authentication"))
    api.add_argument("--reasoning-parser",
                     choices=("auto", "none", "qwen3", "deepseek_r1",
                              "muse_glimmer", "nemotron"),
                     default="auto",
                     help="Parser for reasoning output (default: %(default)s)")
    api.add_argument(
        "--tool-call-parser",
        choices=("auto", "generic", "hermes", "qwen3_xml", "nemotron",
                 "openai"),
        default="auto",
        help="Parser for model-generated tool calls (default: %(default)s)")
    api.add_argument("--enable-auto-tool-choice",
                     action="store_true",
                     help="Allow the model to select tools automatically")
    api.add_argument(
        "--min-request-priority",
        type=int,
        default=0,
        help="Lowest request priority the server accepts, at most 0; a "
        "request with a lower priority gets HTTP 400. The default 0 refuses "
        "the urgent (negative) priorities, and with them the slots "
        "--priority-reserved-slots keeps; set it below 0 (for example -1) "
        "to let clients use them. /v1/responses and /v1/messages requests "
        "carry no priority and run at 0.",
    )
    api.add_argument(
        "--max-queued-requests",
        type=int,
        default=16,
        help="Maximum number of waiting requests (default: %(default)s)")
    api.add_argument(
        "--queue-timeout",
        type=float,
        default=600.0,
        help="Maximum queue wait in seconds (default: %(default)s)")
    api.add_argument(
        "--allowed-local-media-path",
        default="",
        help=("Root directory allowed for local media files. Unset disables "
              "local files"))
    api.add_argument("--log-level",
                     choices=("debug", "info", "warning", "error"),
                     default="info",
                     help="Server logging level (default: %(default)s)")

    model = parser.add_argument_group("Model build and runtime")
    model.add_argument(
        "--cache-dir",
        default="",
        help=(
            "Root for downloaded checkpoints and complete engine bundles. A "
            "matching bundle is reused. A cache miss builds all components."),
    )
    model.add_argument(
        "--engine-cache-max-size-gb",
        type=_positive_float,
        default=50.0,
        help=
        ("Maximum disk space in GiB for compiled engine bundles. Older inactive "
         "profiles are evicted least-recently-used."),
    )
    model.add_argument(
        "--clear-engine-cache",
        action="store_true",
        help=("Remove compiled engine bundles before preparing this model. "
              "Downloaded checkpoints are preserved."),
    )
    model.add_argument(
        "--max-input-len",
        type=_positive_int,
        default=4096,
        help=("Maximum prompt tokens, including conversation history "
              "(default: %(default)s)"))
    model.add_argument(
        "--max-batch-size",
        type=_positive_int,
        default=1,
        help=("Maximum number of sequences in an engine batch (default: "
              "%(default)s)"))
    model.add_argument(
        "--max-kv-cache-capacity",
        type=_positive_int,
        default=8192,
        help=("Maximum total tokens per sequence, including prompt and "
              "generated tokens. The KV cache pool holds --max-batch-size "
              "times this many tokens (default: %(default)s)"))
    model.add_argument(
        "--max-spec-verify-tokens",
        type=_positive_int,
        help=("Maximum verification tokens compiled into the base engine; "
              "also caps the jump-forward width (default: the builder "
              "default)"))
    model.add_argument(
        "--max-spec-draft-tokens",
        type=_positive_int,
        help=("Maximum draft tokens compiled into the draft engine "
              "(default: the builder default)"))
    model.add_argument(
        "--max-image-tokens",
        type=_positive_int,
        help="Maximum total visual tokens in the visual engine profile")
    model.add_argument(
        "--max-image-tokens-per-image",
        type=_positive_int,
        help="Maximum visual tokens per image in the visual engine profile")
    model.add_argument(
        "--draft-top-k",
        type=_positive_int,
        help=("Candidate tokens retained per draft step. Defaults depend on "
              "the speculative method"))
    model.add_argument("--spec-draft-step", type=_positive_int)
    model.add_argument("--spec-proposal-tokens", type=_positive_int)
    model.add_argument("--spec-verify-tokens", type=_positive_int)
    model.add_argument(
        "--speculative-config",
        default="",
        help=(
            "Speculative decoding JSON, e.g. "
            "{\"method\":\"mtp\",\"num_speculative_tokens\":3}. Paired drafts "
            "also use \"model\""))
    model.add_argument(
        "--enable-context-reuse",
        action="store_true",
        help=("Reuse matching text prefixes across requests in this trusted "
              "server instance."),
    )
    model.add_argument(
        "--enable-in-flight-batching",
        action="store_true",
        help="Admit queued requests into the running batch at generation "
        "boundaries instead of serving one request at a time. Text, "
        "multimodal-input and speculative-decoding deployments, with or "
        "without context reuse: a standalone TTS deployment refuses the flag "
        "at startup, and a Qwen3-Omni bundle serves text only (its speech "
        "endpoints refuse).",
    )
    model.add_argument(
        "--ifb-scheduling-mode",
        default="auto",
        help="With --enable-in-flight-batching, one of auto (default), mixed, "
        "prefillFirst or singleCohort: mixed runs decode and chunked "
        "prefill in one step and needs an engine whose layers support "
        "compact execution; prefillFirst lets a pending prefill run alone; "
        "auto picks mixed when supported and logs why when it does not; "
        "singleCohort never runs compact steps and reproduces the outputs "
        "of in-flight batching without them.",
    )
    model.add_argument(
        "--max-num-tokens",
        type=_non_negative_int,
        default=0,
        help="Token budget of a mixed step with decoding requests; 0 derives "
        "it from the engine's batch size and input length.",
    )
    model.add_argument(
        "--max-chunk-tokens",
        type=_non_negative_int,
        default=0,
        help="Per-request prefill budget of a mixed step with decoding "
        "requests, a multiple of the KV page size (128 tokens); 0 derives it "
        "from --max-num-tokens. "
        "Capped at --max-input-len, the most one prompt chunk holds.",
    )
    model.add_argument(
        "--scheduling-policy",
        default="fcfs",
        help="Admission order of queued requests: arrival (fcfs), or the "
        "request's priority field first, lower values first (priority).",
    )
    model.add_argument(
        "--preemption",
        default="none",
        help="none (default) or pause: with pause, lower-priority requests "
        "pause while a higher-priority "
        "prompt is prefilled. Requires --scheduling-policy priority and the "
        "mixed scheduling mode.",
    )
    model.add_argument(
        "--priority-reserved-slots",
        type=_non_negative_int,
        default=0,
        help="Batch slots kept for requests with a negative priority; other "
        "requests are admitted only while more slots than this are free. "
        "Requires --scheduling-policy priority. HTTP clients reach these "
        "slots only when --min-request-priority is below 0 (the default 0 "
        "refuses negative priorities; the server warns at startup).",
    )
    model.add_argument(
        "--max-pause-steps",
        type=_non_negative_int,
        default=0,
        help="With --preemption pause: the most steps in a row a request "
        "stays paused; it is then not paused for one step before it can be "
        "paused again. A generating request then produces a token; one whose "
        "prompt is still being processed may get no prompt budget in that "
        "step. 0 (default) for no limit, so sustained urgent traffic can keep "
        "less urgent requests paused.",
    )
    model.add_argument("--ple-residency",
                       choices=("resident", "cached"),
                       default="resident")
    model.add_argument(
        "--ple-cache-slots",
        type=_non_negative_int,
        default=0,
        help=
        "Cached PLE pool slots, including reserved slot 0; selected at runtime. "
        "Distinct valid rows across each execution batch must fit in slots - 1; "
        "increase slots or reduce the execution chunk/batch if needed. "
        "Smaller I/O batches do not relax this requirement.")
    model.add_argument(
        "--ple-file",
        default="",
        help=
        "Original checkpoint file containing PLE; overrides the checkpoint source."
    )
    model.add_argument(
        "--context-cache-max-records",
        type=_non_negative_int,
        default=1024,
        help="Maximum retained context-cache records (default: %(default)s)")
    model.add_argument(
        "--context-cache-max-snapshots",
        type=_non_negative_int,
        default=0,
        help=("Total snapshots kept for context reuse on a model with "
              "recurrent layers (default: 4 x max batch size)"))
    return parser


def parse_server_config(argv: Optional[Sequence[str]] = None,
                        *,
                        prog="tensorrt-edgellm-serve") -> ServerConfig:
    parser = create_argument_parser(prog=prog)
    args = parser.parse_args(argv)

    if args.max_queued_requests < 0:
        raise ServerConfigError("--max-queued-requests must be non-negative")
    if args.queue_timeout <= 0:
        raise ServerConfigError("--queue-timeout must be positive")

    speculative = SpeculativeConfig.parse(args.speculative_config)
    draft_step = args.spec_draft_step
    proposal_tokens = args.spec_proposal_tokens
    if (speculative and speculative.method in {"eagle3", "mtp"}
            and speculative.num_speculative_tokens is not None):
        draft_step = speculative.num_speculative_tokens
    elif speculative and speculative.num_speculative_tokens is not None:
        proposal_tokens = speculative.num_speculative_tokens
    context_cache = ContextCacheConfig(
        enabled=args.enable_context_reuse,
        max_records=args.context_cache_max_records,
        max_snapshots=args.context_cache_max_snapshots,
    )
    model = ModelConfig(
        model=args.model,
        cache_dir=args.cache_dir,
        engine_cache_max_size_gb=args.engine_cache_max_size_gb,
        clear_engine_cache=args.clear_engine_cache,
        max_input_len=args.max_input_len,
        max_batch_size=args.max_batch_size,
        max_kv_cache_capacity=args.max_kv_cache_capacity,
        max_spec_verify_tokens=args.max_spec_verify_tokens,
        max_spec_draft_tokens=args.max_spec_draft_tokens,
        max_image_tokens=args.max_image_tokens,
        max_image_tokens_per_image=args.max_image_tokens_per_image,
        draft_top_k=(speculative.draft_top_k
                     if speculative and speculative.draft_top_k is not None
                     else args.draft_top_k),
        draft_step=draft_step,
        proposal_tokens=proposal_tokens,
        verify_tokens=(speculative.verify_tokens
                       if speculative and speculative.verify_tokens is not None
                       else args.spec_verify_tokens),
        speculative_config=speculative,
        context_cache_config=context_cache,
        ple_config=PleRuntimeConfig(residency=args.ple_residency,
                                    cache_slots=args.ple_cache_slots,
                                    file_path=args.ple_file),
        enable_in_flight_batching=args.enable_in_flight_batching,
        ifb_scheduling_mode=args.ifb_scheduling_mode,
        max_num_tokens=args.max_num_tokens,
        max_chunk_tokens=args.max_chunk_tokens,
        scheduling_policy=args.scheduling_policy,
        preemption=args.preemption,
        priority_reserved_slots=args.priority_reserved_slots,
        max_pause_steps=args.max_pause_steps,
    )
    api = ApiConfig(
        host=args.host,
        port=args.port,
        served_model_name=args.served_model_name,
        api_key=args.api_key,
        reasoning_parser=args.reasoning_parser,
        tool_call_parser=args.tool_call_parser,
        enable_auto_tool_choice=args.enable_auto_tool_choice,
        max_queued_requests=args.max_queued_requests,
        queue_timeout=args.queue_timeout,
        allowed_local_media_path=args.allowed_local_media_path,
        log_level=args.log_level,
        min_request_priority=args.min_request_priority,
    )
    return ServerConfig(model=model, api=api)
