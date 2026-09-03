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
"""Validated launch configuration for the experimental inference server."""

import argparse
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Union

_INT32_MAX = 2**31 - 1
_INT64_MAX = 2**63 - 1


class ServerConfigError(ValueError):
    """Raised when a launch option cannot be honored by Edge-LLM."""


@dataclass(frozen=True)
class SpeculativeConfig:
    """Speculative model pair and runtime proposal length."""

    method: str
    num_speculative_tokens: Optional[int] = None
    draft_model: str = ""

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

        supported_keys = {"method", "num_speculative_tokens", "model"}
        unsupported = sorted(set(raw) - supported_keys)
        if unsupported:
            raise ServerConfigError("unsupported --speculative-config keys: " +
                                    ", ".join(unsupported))

        method = raw.get("method")
        if method not in {"eagle3", "mtp", "dflash", "jetspec", "dspark"}:
            raise ServerConfigError(
                "--speculative-config.method must be one of eagle3, mtp, "
                "dflash, jetspec, or dspark")
        num_tokens = raw.get("num_speculative_tokens")
        if num_tokens is not None and (isinstance(num_tokens, bool)
                                       or not isinstance(num_tokens, int)
                                       or num_tokens < 1):
            raise ServerConfigError(
                "--speculative-config.num_speculative_tokens must be a "
                "positive integer")
        draft_model = raw.get("model")
        if draft_model is not None and (not isinstance(draft_model, str)
                                        or not draft_model.strip()):
            raise ServerConfigError(
                "--speculative-config.model must be a checkpoint path or "
                "Hugging Face model ID")
        if method in {"eagle3", "dflash", "jetspec", "dspark"
                      } and not draft_model:
            raise ServerConfigError(
                f"--speculative-config.method {method!r} requires "
                "speculative-config.model")
        return cls(method=method,
                   num_speculative_tokens=num_tokens,
                   draft_model=draft_model or "")


@dataclass(frozen=True)
class ContextCacheConfig:
    """Deployment-scoped KV-cache reuse configuration."""

    enabled: bool = False
    max_records: int = 1024
    recurrent_snapshot_pool_bytes: int = 0
    partial_kv_snapshot_pool_bytes: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ServerConfigError("context_cache.enabled must be boolean")
        for name in (
                "max_records",
                "recurrent_snapshot_pool_bytes",
                "partial_kv_snapshot_pool_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value,
                          bool) or not isinstance(value, int) or value < 0:
                raise ServerConfigError(
                    f"context_cache.{name} must be a non-negative integer")
        if self.max_records > _INT32_MAX:
            raise ServerConfigError(
                "context_cache.max_records must fit a signed 32-bit integer")
        for name in ("recurrent_snapshot_pool_bytes",
                     "partial_kv_snapshot_pool_bytes"):
            if getattr(self, name) > _INT64_MAX:
                raise ServerConfigError(
                    f"context_cache.{name} must fit a signed 64-bit integer")
        if (not self.enabled and
            (self.max_records != 1024 or self.recurrent_snapshot_pool_bytes
             or self.partial_kv_snapshot_pool_bytes)):
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
        supported = {
            "enabled",
            "max_records",
            "recurrent_snapshot_pool_bytes",
            "partial_kv_snapshot_pool_bytes",
        }
        unsupported = sorted(set(raw) - supported)
        if unsupported:
            raise ServerConfigError("unsupported context-cache keys: " +
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
    draft_top_k: Optional[int] = None
    draft_step: Optional[int] = None
    verify_tree_size: Optional[int] = None
    speculative_config: Optional[SpeculativeConfig] = None
    context_cache_config: ContextCacheConfig = field(
        default_factory=ContextCacheConfig)

    def __post_init__(self) -> None:
        if (isinstance(self.engine_cache_max_size_gb, bool)
                or not math.isfinite(self.engine_cache_max_size_gb)
                or self.engine_cache_max_size_gb <= 0):
            raise ServerConfigError(
                "engine_cache_max_size_gb must be positive")

    def llm_kwargs(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "cache_dir": self.cache_dir,
            "engine_cache_max_size_gb": self.engine_cache_max_size_gb,
            "clear_engine_cache": self.clear_engine_cache,
            "max_input_len": self.max_input_len,
            "max_batch_size": self.max_batch_size,
            "max_kv_cache_capacity": self.max_kv_cache_capacity,
            "draft_top_k": self.draft_top_k,
            "draft_step": self.draft_step,
            "verify_tree_size": self.verify_tree_size,
            "speculative_config": self.speculative_config,
            "context_cache_config": self.context_cache_config,
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
    max_queued_requests: int = 16
    queue_timeout: float = 600.0
    allowed_local_media_path: str = ""
    log_level: str = "info"

    def __post_init__(self) -> None:
        if self.max_queued_requests < 0:
            raise ServerConfigError("max_queued_requests must be non-negative")
        if self.queue_timeout <= 0:
            raise ServerConfigError("queue_timeout must be positive")


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


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorrt-edgellm-serve",
        description="TensorRT Edge-LLM OpenAI-compatible server",
    )
    parser.add_argument(
        "model",
        help="Hugging Face model ID or local checkpoint directory",
    )

    api = parser.add_argument_group("HTTP server")
    api.add_argument("--host", default="0.0.0.0")
    api.add_argument("--port", type=_bounded_port, default=8000)
    api.add_argument("--served-model-name", default="")
    api.add_argument("--api-key", default="")
    api.add_argument(
        "--reasoning-parser",
        choices=("auto", "none", "qwen3", "deepseek_r1", "nemotron"),
        default="auto",
    )
    api.add_argument(
        "--tool-call-parser",
        choices=("auto", "generic", "hermes", "qwen3_xml", "nemotron",
                 "openai"),
        default="auto",
    )
    api.add_argument("--enable-auto-tool-choice", action="store_true")
    api.add_argument("--max-queued-requests", type=int, default=16)
    api.add_argument("--queue-timeout", type=float, default=600.0)
    api.add_argument("--allowed-local-media-path", default="")
    api.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default="info",
    )

    model = parser.add_argument_group("Model build and runtime")
    model.add_argument(
        "--cache-dir",
        default="",
        help="Root for downloaded checkpoints and complete engine bundles. "
        "A matching bundle is reused; a cache miss builds all components.",
    )
    model.add_argument(
        "--engine-cache-max-size-gb",
        type=_positive_float,
        default=50.0,
        help="Maximum disk space for compiled engine bundles. Older inactive "
        "profiles are evicted least-recently-used.",
    )
    model.add_argument(
        "--clear-engine-cache",
        action="store_true",
        help="Remove compiled engine bundles before preparing this model. "
        "Downloaded checkpoints are preserved.",
    )
    model.add_argument("--max-input-len", type=_positive_int, default=4096)
    model.add_argument("--max-batch-size", type=_positive_int, default=1)
    model.add_argument("--max-kv-cache-capacity",
                       type=_positive_int,
                       default=8192)
    model.add_argument("--draft-top-k", type=_positive_int)
    model.add_argument("--draft-step", type=_positive_int)
    model.add_argument("--verify-tree-size", type=_positive_int)
    model.add_argument("--speculative-config", default="")
    model.add_argument(
        "--enable-context-reuse",
        action="store_true",
        help="Reuse matching text prefixes across requests in this trusted "
        "server instance.",
    )
    model.add_argument("--context-cache-max-records",
                       type=_non_negative_int,
                       default=1024)
    model.add_argument("--context-cache-recurrent-snapshot-pool-bytes",
                       type=_non_negative_int,
                       default=0)
    model.add_argument("--context-cache-partial-kv-snapshot-pool-bytes",
                       type=_non_negative_int,
                       default=0)
    return parser


def parse_server_config(argv: Optional[Sequence[str]] = None) -> ServerConfig:
    parser = create_argument_parser()
    args = parser.parse_args(argv)

    if args.max_queued_requests < 0:
        raise ServerConfigError("--max-queued-requests must be non-negative")
    if args.queue_timeout <= 0:
        raise ServerConfigError("--queue-timeout must be positive")

    speculative = SpeculativeConfig.parse(args.speculative_config)
    draft_step = args.draft_step
    if (speculative and speculative.method in {"eagle3", "mtp"}
            and speculative.num_speculative_tokens is not None):
        draft_step = speculative.num_speculative_tokens
    context_cache = ContextCacheConfig(
        enabled=args.enable_context_reuse,
        max_records=args.context_cache_max_records,
        recurrent_snapshot_pool_bytes=(
            args.context_cache_recurrent_snapshot_pool_bytes),
        partial_kv_snapshot_pool_bytes=(
            args.context_cache_partial_kv_snapshot_pool_bytes),
    )
    model = ModelConfig(
        model=args.model,
        cache_dir=args.cache_dir,
        engine_cache_max_size_gb=args.engine_cache_max_size_gb,
        clear_engine_cache=args.clear_engine_cache,
        max_input_len=args.max_input_len,
        max_batch_size=args.max_batch_size,
        max_kv_cache_capacity=args.max_kv_cache_capacity,
        draft_top_k=args.draft_top_k,
        draft_step=draft_step,
        verify_tree_size=args.verify_tree_size,
        speculative_config=speculative,
        context_cache_config=context_cache,
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
    )
    return ServerConfig(model=model, api=api)
