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
"""Opt-in CI adapter for the checkpoint-native server's existing cache."""

import dataclasses
import functools
import importlib.metadata
import os
import pathlib
import subprocess
import time

from . import ci_engine_cache as cache


def checkpoint(path):
    root = pathlib.Path(path).resolve(strict=True)
    files = []
    for directory, names, filenames in os.walk(root):
        names[:] = sorted(name for name in names
                          if name not in {".cache", ".edgellm", ".git"})
        if any(
            (pathlib.Path(directory) / name).is_symlink() for name in names):
            raise ValueError("symlinked-checkpoint-directory")
        for name in sorted(filenames):
            file = pathlib.Path(directory) / name
            # HF snapshots commonly contain symlinked shards; hash resolved bytes.
            files.append({
                "name": file.relative_to(root).as_posix(),
                "sha256": cache._sha256(file.resolve(strict=True))
            })
    if not files:
        raise ValueError("empty-checkpoint")
    return files


def source_identity(root):
    names = subprocess.check_output([
        "git", "-C",
        str(root), "ls-files", "-z", "--", "tensorrt_edgellm/builder",
        "tensorrt_edgellm/server/runtime", "tensorrt_edgellm", "cpp", "cmake",
        "CMakeLists.txt", "pyproject.toml"
    ],
                                    timeout=30).decode().split("\0")
    return {name: cache._sha256(root / name) for name in names if name}


def coordinator(model_dir, root, options, local_root):
    platform = cache._platform_descriptor()
    excluded = os.environ.get("EDGE_LLM_ENGINE_CACHE_EXCLUDED_GPUS",
                              "").split(",")
    if platform["gpu"]["name"] in excluded:
        raise ValueError("restore-cost-gate")
    plugin = pathlib.Path(options.plugin_path).resolve(strict=True)
    builders, _ = cache._builder_descriptor([([str(plugin)], 0)])
    profile = dataclasses.asdict(options)
    profile.pop("plugin_path")
    draft = profile.pop("draft_model_dir", "")
    descriptor = {
        "schema_version":
        "edgellm-ci-server-descriptor/v1",
        "checkpoint":
        checkpoint(model_dir),
        "draft_checkpoint":
        checkpoint(draft) if draft else [],
        "source":
        source_identity(root),
        "options":
        profile,
        "platform":
        platform,
        "plugins":
        builders,
        "dependencies":
        sorted((item.metadata["Name"], item.version)
               for item in importlib.metadata.distributions()),
        "epoch":
        os.environ.get("EDGE_LLM_ENGINE_CACHE_EPOCH", "1"),
    }
    from tensorrt_edgellm.server.runtime import engine_build
    bundle = pathlib.Path(
        engine_build.bundle_cache_path(model_dir, str(local_root), options))
    return cache.EngineBundleCache.from_descriptor(
        local_root, [pathlib.PurePosixPath(bundle.relative_to(local_root))],
        descriptor, {
            "options": profile,
            "model": pathlib.Path(model_dir).name,
            "platform": platform
        }, [plugin])


def install():
    """Observe prepare_model only inside an explicitly opted-in CI process."""
    if not os.environ.get("CI_JOB_ID"):
        return
    if os.environ.get("CI_SERVER_CACHE_ENABLED") != "1":
        cache._cache_event(None,
                           "server",
                           "lookup",
                           time.monotonic(),
                           "policy-disabled",
                           detail="server-validation-pending")
        return
    from tensorrt_edgellm.server.runtime import engine_build
    original = engine_build.prepare_model
    if getattr(original, "_ci_observer", False):
        return

    @functools.wraps(original)
    def prepare(model, cache_dir="", options=None, **kwargs):
        cache_dir = engine_build.cache_root(cache_dir)
        started = time.monotonic()
        coordinator_instance = None
        if cache._cache_mode() is not cache.CacheMode.OFF:
            try:
                model_dir = engine_build.resolve_model_dir(model, cache_dir)
                options = options or engine_build.BuildOptions()
                coordinator_instance = coordinator(
                    model_dir,
                    pathlib.Path(__file__).resolve().parents[3], options,
                    pathlib.Path(cache_dir))
                cache._cache_event(None, "server", "key", started, "ready",
                                   coordinator_instance.key)
                if kwargs.get("clear_cache"):
                    cache._cache_event(None,
                                       "server",
                                       "lookup",
                                       time.monotonic(),
                                       "forced-cold",
                                       detail="explicit-clear")
                    coordinator_instance.prepare_build()
                elif not coordinator_instance.restore_bundle():
                    coordinator_instance.prepare_build()
            except (OSError, ValueError, RuntimeError,
                    subprocess.SubprocessError) as error:
                cache._cache_event(None,
                                   "server",
                                   "lookup",
                                   started,
                                   "unrepresentable-input",
                                   detail="unrepresentable-input")
                print(f"WARNING: server cache unavailable: {error}")
                coordinator_instance = None
        else:
            cache._cache_event(None,
                               "server",
                               "lookup",
                               started,
                               "policy-disabled",
                               detail="policy-disabled")
        build_started = time.monotonic()
        try:
            result = original(model, cache_dir, options, **kwargs)
        except Exception:
            cache._cache_event(None,
                               "server",
                               "build",
                               build_started,
                               "error",
                               detail="build-failed")
            raise
        cache._cache_event(None, "server", "build", build_started,
                           "built" if result.built else "skipped")
        if coordinator_instance and result.built:
            coordinator_instance.publish_bundle()
        return result

    prepare._ci_observer = True
    engine_build.prepare_model = prepare
