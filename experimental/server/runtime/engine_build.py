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
"""Checkpoint resolution and complete-model engine caching."""

import fcntl
import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Iterator, List, Optional

from . import engine_layout

_CACHE_LAYOUT_VERSION = 2
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "tensorrt-edgellm"
_DEFAULT_ENGINE_CACHE_MAX_BYTES = 50 * (1 << 30)
_CHECKPOINT_SUFFIXES = (".bin", ".json", ".model", ".pt", ".pth",
                        ".safetensors")
_CONTENT_SAMPLE_BYTES = 64

logger = logging.getLogger("edgellm.server")


@dataclass(frozen=True)
class BuildOptions:
    """Options forwarded to one all-component builder invocation."""

    spec_type: str = "none"
    draft_model_dir: str = ""
    max_input_len: Optional[int] = None
    max_kv_cache_capacity: Optional[int] = None
    max_batch_size: Optional[int] = None
    max_image_tokens: Optional[int] = None
    max_image_tokens_per_image: Optional[int] = None
    tp_size: Optional[int] = None
    plugin_path: str = ""
    tree_base: bool = False

    @property
    def builder_spec_type(self) -> str:
        # A separate MTP checkpoint uses the Gemma assistant contract. Native
        # MTP layers remain part of the base checkpoint.
        if self.spec_type == "mtp" and self.draft_model_dir:
            return "gemma4_mtp"
        return self.spec_type

    def to_argv(self, model_dir: str, bundle_dir: str) -> List[str]:
        argv = [
            "--model-dir",
            model_dir,
            "--engine-dir",
            bundle_dir,
            "--components",
            "all",
            "--externalize-weights",
            "all",
        ]
        if self.builder_spec_type != "none":
            argv += ["--spec-type", self.builder_spec_type]
        if self.draft_model_dir:
            argv += ["--draft-model-dir", self.draft_model_dir]
        if self.plugin_path:
            argv += ["--plugin-path", self.plugin_path]
        if self.tree_base:
            argv.append("--tree-base")
        for flag, value in (
            ("--max-input-len", self.max_input_len),
            ("--max-kv-cache-capacity", self.max_kv_cache_capacity),
            ("--max-batch-size", self.max_batch_size),
            ("--max-image-tokens", self.max_image_tokens),
            ("--max-image-tokens-per-image", self.max_image_tokens_per_image),
            ("--tp-size", self.tp_size),
        ):
            if value is not None:
                argv += [flag, str(value)]
        return argv


@dataclass(frozen=True)
class PreparedModel:
    """Resolved checkpoints and their profile-specific engine bundle."""

    bundle_dir: str
    model_dir: str
    draft_model_dir: str = ""
    built: bool = False


def cache_root(cache_dir: str = "") -> str:
    """Return the one user-controlled artifact root."""
    root = Path(cache_dir).expanduser() if cache_dir else _DEFAULT_CACHE_DIR
    return str(root.resolve())


def resolve_model_dir(model: str, cache_dir: str = "") -> str:
    """Resolve a local checkpoint or Hugging Face model ID."""
    if os.path.isdir(model):
        path = os.path.abspath(model)
        source_type = engine_layout.classify_model_source(path)
        if source_type == "onnx_dir":
            raise ValueError(
                "ONNX model directories are not supported; pass the original "
                "checkpoint")
        if source_type == "engine_dir":
            raise ValueError(
                "engine bundles are cache artifacts; pass the checkpoint or "
                "Hugging Face model ID used to build the model")
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for Hugging Face model IDs; install "
            "the TensorRT Edge-LLM Python package with server dependencies"
        ) from exc
    checkpoint_cache = os.path.join(cache_root(cache_dir), "checkpoints")
    return snapshot_download(model, cache_dir=checkpoint_cache)


def _checkpoint_digest(model_dir: str) -> str:
    """Fingerprint a checkpoint independently of its path and mtimes.

    Small metadata files are hashed completely. Large checkpoint shards use
    their safetensors header or file size plus bounded content samples. The
    samples keep structurally identical checkpoints with different weights in
    separate cache entries without reading full tensor payloads.
    """
    digest = hashlib.sha256()
    for root, dirs, files in os.walk(model_dir):
        dirs[:] = sorted(directory for directory in dirs
                         if directory != ".edgellm")
        for filename in sorted(files):
            if not filename.endswith(_CHECKPOINT_SUFFIXES):
                continue
            path = os.path.join(root, filename)
            digest.update(os.path.relpath(path, model_dir).encode("utf-8"))
            _update_file_digest(path, digest)
    return digest.hexdigest()


def _update_file_digest(path: str, digest) -> None:
    """Hash a checkpoint file with bounded reads for weight payloads."""
    size = os.path.getsize(path)
    digest.update(size.to_bytes(8, "little"))
    with open(path, "rb") as checkpoint_file:
        if path.endswith(".safetensors") and size >= 8:
            prefix = checkpoint_file.read(8)
            header_size = int.from_bytes(prefix, "little")
            if header_size <= size - 8:
                digest.update(prefix)
                digest.update(checkpoint_file.read(header_size))
                _update_content_samples(checkpoint_file, digest,
                                        8 + header_size,
                                        size - 8 - header_size)
                return
            checkpoint_file.seek(0)
        if path.endswith((".json", ".model")):
            for chunk in iter(lambda: checkpoint_file.read(1 << 20), b""):
                digest.update(chunk)
            return
        _update_content_samples(checkpoint_file, digest, 0, size)


def _update_content_samples(checkpoint_file, digest, start: int,
                            size: int) -> None:
    """Hash fixed start, middle, and end ranges of a weight payload."""
    if size <= 0:
        return
    width = min(_CONTENT_SAMPLE_BYTES, size)
    offsets = sorted(set((0, (size - width) // 2, size - width)))
    for offset in offsets:
        checkpoint_file.seek(start + offset)
        digest.update(offset.to_bytes(8, "little"))
        digest.update(checkpoint_file.read(width))


def bundle_cache_path(model_dir: str, cache_dir: str,
                      options: BuildOptions) -> str:
    """Return the cache path for one checkpoint and complete build profile."""
    model_digest = _checkpoint_digest(model_dir)
    draft_digest = (_checkpoint_digest(options.draft_model_dir)
                    if options.draft_model_dir else "")
    profile = {
        "cache_layout_version": _CACHE_LAYOUT_VERSION,
        "model": model_digest,
        "draft_model": draft_digest,
        **{
            field.name: getattr(options, field.name)
            for field in fields(options) if field.name not in {
                "draft_model_dir", "plugin_path"
            }
        },
        "builder_spec_type": options.builder_spec_type,
    }
    profile_digest = hashlib.sha256(
        json.dumps(profile, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    profile_name = (f"i{options.max_input_len or 'default'}-"
                    f"b{options.max_batch_size or 'default'}-"
                    f"kv{options.max_kv_cache_capacity or 'default'}-"
                    f"{options.builder_spec_type}-{profile_digest}")
    return os.path.join(cache_root(cache_dir), "engines",
                        f"model-{model_digest[:16]}", profile_name)


def _is_ready(model_dir: str, bundle_dir: str, options: BuildOptions) -> bool:
    """Verify every component and the requested runtime profile."""
    from experimental.builder.core import contracts
    from experimental.builder.core.bundle import BundleConfig

    try:
        bundle = BundleConfig.from_pretrained(model_dir)
        components = contracts.resolve_components(bundle.root_model_type,
                                                  ("all", ),
                                                  available=bundle.components)
    except (OSError, TypeError, ValueError):
        return False

    config_paths = []
    built_components = set()
    for component in components:
        spec = contracts.component_spec(component)
        if component == contracts.Component.LLM and options.spec_type != "none":
            for role in (contracts.SpecRole.BASE, contracts.SpecRole.DRAFT):
                if not (os.path.isfile(spec.output_path(bundle_dir, role)) and
                        os.path.isfile(spec.config_path(bundle_dir, role))):
                    return False
                config_paths.append(spec.config_path(bundle_dir, role))
            built_components.add(component)
            continue
        if not (os.path.isfile(spec.output_path(bundle_dir))
                and os.path.isfile(spec.config_path(bundle_dir))):
            return False
        config_paths.append(spec.config_path(bundle_dir))
        built_components.add(component)

    has_text_runtime = bool(
        built_components
        & {contracts.Component.LLM, contracts.Component.DLLM})
    has_tts_runtime = {
        contracts.Component.TALKER,
        contracts.Component.CODE_PREDICTOR,
        contracts.Component.CODE2WAV,
    }.issubset(built_components)
    if not (has_text_runtime or has_tts_runtime):
        return False

    try:
        configs = []
        for path in config_paths:
            with open(path, encoding="utf-8") as file:
                configs.append(json.load(file))
    except (OSError, ValueError):
        return False

    base = next((config for config in configs
                 if config.get("engine_role") in ("llm", "dllm", "base")
                 and "max_input_len" in config.get("builder_config", {})), {})
    builder = base.get("builder_config", {})
    expected = {
        "max_input_len": options.max_input_len,
        "max_kv_cache_capacity": options.max_kv_cache_capacity,
        "max_batch_size": options.max_batch_size,
    }
    if any(value is not None and builder.get(name) != value
           for name, value in expected.items()):
        return False
    if (options.spec_type != "none"
            and base.get("spec_decode_type") != options.builder_spec_type):
        return False
    return any("checkpoint_identity" in config for config in configs)


def _resolve_plugin_path(explicit: str) -> str:
    if explicit:
        return explicit
    from .engine import _ensure_plugin_path

    _ensure_plugin_path()
    return os.environ.get("EDGELLM_PLUGIN_PATH", "")


@contextmanager
def _bundle_lock(bundle_dir: str) -> Iterator[None]:
    lock_path = bundle_dir + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


@contextmanager
def _cache_lock(cache_dir: str) -> Iterator[None]:
    """Serialize cache eviction and explicit clear operations."""
    root = cache_root(cache_dir)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, ".engine-cache.lock"), "w",
              encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _bundle_directories(cache_dir: str) -> List[Path]:
    """List published engine profiles while excluding staging directories."""
    engine_root = Path(cache_root(cache_dir)) / "engines"
    if not engine_root.is_dir():
        return []
    return [
        profile for model in engine_root.iterdir()
        if model.is_dir() and not model.name.startswith(".")
        for profile in model.iterdir()
        if profile.is_dir() and not profile.name.startswith(".")
    ]


def _directory_size(path: Path) -> int:
    """Return logical file bytes below one published bundle."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            try:
                total += os.path.getsize(os.path.join(root, filename))
            except FileNotFoundError:
                continue
    return total


def _remove_bundle(bundle: Path) -> Optional[int]:
    """Remove an idle bundle, or return None when another process owns it."""
    lock_path = str(bundle) + ".lock"
    with open(lock_path, "w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        if not bundle.is_dir():
            return 0
        size = _directory_size(bundle)
        shutil.rmtree(bundle)
        return size


def clear_engine_cache(cache_dir: str = "") -> int:
    """Remove published engine bundles without deleting checkpoints."""
    removed = 0
    with _cache_lock(cache_dir):
        for bundle in _bundle_directories(cache_dir):
            if _remove_bundle(bundle) is not None:
                removed += 1
    logger.info("Cleared %d engine cache bundle(s)", removed)
    return removed


def prune_engine_cache(cache_dir: str = "",
                       max_size_bytes: int = _DEFAULT_ENGINE_CACHE_MAX_BYTES,
                       keep: str = "") -> int:
    """Evict least-recently-used bundles until the cache is within its cap."""
    if max_size_bytes < 1:
        raise ValueError("max_size_bytes must be positive")
    keep_path = os.path.realpath(keep) if keep else ""
    removed = 0
    with _cache_lock(cache_dir):
        entries = []
        for bundle in _bundle_directories(cache_dir):
            try:
                entries.append((bundle.stat().st_mtime_ns,
                                _directory_size(bundle), bundle))
            except FileNotFoundError:
                continue
        total = sum(size for _mtime, size, _bundle in entries)
        for _mtime, size, bundle in sorted(entries):
            if total <= max_size_bytes:
                break
            if keep_path and os.path.realpath(bundle) == keep_path:
                continue
            removed_size = _remove_bundle(bundle)
            if removed_size is None:
                continue
            total -= size
            removed += 1
    if removed:
        logger.info("Evicted %d engine cache bundle(s)", removed)
    if total > max_size_bytes:
        logger.warning(
            "Engine cache remains above its %.2f GiB limit because the active "
            "or locked bundle cannot be evicted", max_size_bytes / (1 << 30))
    return removed


def prepare_model(model: str,
                  cache_dir: str = "",
                  options: Optional[BuildOptions] = None,
                  *,
                  max_cache_size_bytes: int = _DEFAULT_ENGINE_CACHE_MAX_BYTES,
                  clear_cache: bool = False) -> PreparedModel:
    """Resolve a model and build every component on a cache miss."""
    options = options or BuildOptions()
    model_dir = resolve_model_dir(model, cache_dir)
    draft_model_dir = ""
    if options.draft_model_dir:
        draft_model_dir = resolve_model_dir(options.draft_model_dir, cache_dir)
        options = replace(options, draft_model_dir=draft_model_dir)

    if clear_cache:
        clear_engine_cache(cache_dir)
    bundle_dir = bundle_cache_path(model_dir, cache_dir, options)
    built = False
    with _bundle_lock(bundle_dir):
        if _is_ready(model_dir, bundle_dir, options):
            os.utime(bundle_dir)
        else:
            parent = os.path.dirname(bundle_dir)
            os.makedirs(parent, exist_ok=True)
            staging_dir = tempfile.mkdtemp(prefix=".building-", dir=parent)
            try:
                build_options = replace(
                    options,
                    plugin_path=_resolve_plugin_path(options.plugin_path),
                )
                from experimental.builder.cli import main as builder_main

                builder_main(build_options.to_argv(model_dir, staging_dir))
                if not _is_ready(model_dir, staging_dir, build_options):
                    raise RuntimeError(
                        "the checkpoint-native builder did not produce a "
                        f"complete runtime bundle for {model!r}")
                if os.path.exists(bundle_dir):
                    shutil.rmtree(bundle_dir)
                os.chmod(staging_dir, stat.S_IMODE(os.stat(parent).st_mode))
                os.replace(staging_dir, bundle_dir)
                built = True
            finally:
                shutil.rmtree(staging_dir, ignore_errors=True)
    prune_engine_cache(cache_dir, max_cache_size_bytes, keep=bundle_dir)
    return PreparedModel(bundle_dir, model_dir, draft_model_dir, built)
