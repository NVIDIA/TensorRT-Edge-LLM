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
"""Bundle identity: every component config and ``policy.json`` of one build carry the same
``export_id``, so a directory assembled from two builds or two build routes is refused."""

from __future__ import annotations

import functools
import glob
import hashlib
import json
import os
import struct

from ..._version import __version__
from .contracts import CONTRACT_VERSION
from .geometry import Geometry
from .policy_contract import POLICY_CONTRACT_FILENAME

COMPONENTS = ("visual", "vl_backbone", "action")
FRONTENDS = ("onnx", "direct")
IDENTITY_KEYS = ("checkpoint_fingerprint", "export_id", "frontend")

# Bytes per window, three windows per tensor: the same sampling as pi0.5's
# checkpoint_fingerprint, kept here because importing that package pulls in torch,
# which the direct builder's environment does not have.
FINGERPRINT_WINDOW = 4096


@functools.lru_cache(maxsize=4)
def _fingerprint(checkpoint: str) -> str:
    """Hash each shard's safetensors header plus the head, middle and tail window of
    every tensor, in file-offset order."""
    shards = sorted(glob.glob(os.path.join(checkpoint, "*.safetensors")))
    if not shards:
        raise FileNotFoundError(
            f"{checkpoint} holds no *.safetensors; a GR00T "
            "bundle cannot be identified without its weights")
    digest = hashlib.sha256()
    for shard in shards:
        with open(shard, "rb") as stream:
            length = struct.unpack("<Q", stream.read(8))[0]
            header = stream.read(length)
        entries = json.loads(header)
        entries.pop("__metadata__", None)
        base = 8 + length
        digest.update(os.path.basename(shard).encode())
        digest.update(header)
        ordered = sorted(entries.items(),
                         key=lambda kv: kv[1]["data_offsets"][0])
        with open(shard, "rb") as stream:
            for name, entry in ordered:
                start, end = entry["data_offsets"]
                size = end - start
                window = min(FINGERPRINT_WINDOW, size)
                digest.update(name.encode())
                for offset in sorted({0, (size - window) // 2, size - window}):
                    stream.seek(base + start + offset)
                    digest.update(stream.read(window))
    return digest.hexdigest()


def bundle_identity(checkpoint: str, geometry: Geometry,
                    frontend: str) -> dict:
    """Derived, not random: rebuilding one component from the same checkpoint, options and
    route reproduces it, which is what makes a partial rebuild checkable."""
    if frontend not in FRONTENDS:
        raise ValueError(f"unknown GR00T build frontend {frontend!r}")
    fingerprint = _fingerprint(os.path.abspath(checkpoint))
    payload = json.dumps(
        {
            "contract_version": CONTRACT_VERSION,
            "edgellm_version": __version__,
            "checkpoint_fingerprint": fingerprint,
            "frontend": frontend,
            "embodiment_id": geometry.embodiment_id,
            "max_vl_len": geometry.max_vl_len,
            "max_image_len": geometry.max_image_len,
        },
        sort_keys=True)
    return {
        "checkpoint_fingerprint": fingerprint,
        "export_id": hashlib.sha256(payload.encode()).hexdigest()[:32],
        "frontend": frontend,
    }


def stamp(config: dict, identity: dict) -> dict:
    stamped = dict(config)
    for key in IDENTITY_KEYS:
        stamped[key] = identity[key]
    return stamped


def stamp_contract(contract: dict, identity: dict) -> dict:
    stamped = stamp(contract, identity)
    stamped["components"] = list(COMPONENTS)
    return stamped


def _load(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as stream:
        return json.load(stream)


def validate_existing(root: str, identity: dict,
                      rebuilding: "list[str] | tuple[str, ...]") -> None:
    """Refuse to write beside components of another build. Components in ``rebuilding``
    are overwritten by this run and exempt; anything else present must match, and a
    component without an identity predates this check and must be rebuilt."""
    conflicts = []
    tagged = [(os.path.join(root, c, "config.json"), f"{c}/config.json")
              for c in COMPONENTS if c not in rebuilding]
    if "action" not in rebuilding:
        tagged.append((os.path.join(root, "action", POLICY_CONTRACT_FILENAME),
                       f"action/{POLICY_CONTRACT_FILENAME}"))
    for path, label in tagged:
        found = _load(path)
        if found is None:
            continue
        if not found.get("export_id"):
            conflicts.append(f"{label} carries no export_id (built before the "
                             "identity check)")
            continue
        for key in ("export_id", "frontend"):
            if found.get(key) != identity[key]:
                conflicts.append(
                    f"{label} has {key}={found.get(key)!r}, this build has "
                    f"{identity[key]!r}")
    if conflicts:
        raise ValueError(
            f"{root} already holds a different GR00T build: "
            f"{'; '.join(conflicts)}. Build into an empty directory, or "
            "rebuild every component from the same checkpoint, options and "
            "route.")
