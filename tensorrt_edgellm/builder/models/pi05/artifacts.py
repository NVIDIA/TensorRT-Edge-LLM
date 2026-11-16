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
"""Runtime sidecars for checkpoint-direct pi0.5 engine bundles."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import List, Optional, Tuple

from tensorrt_edgellm._version import __version__
from tensorrt_edgellm.models.pi05 import policy_assets
from tensorrt_edgellm.models.pi05.component_contracts import CONTRACT_VERSION

from ...core import contracts
from ...core.artifacts.tensors import save_safetensors
from ...core.weights import Weights
from . import configuration, runtime_config, weights
from .configuration import Pi05Geometry

logger = logging.getLogger(__name__)

_COMPONENTS = ("visual", "prefix", "action", "cond")


def _export_identity(bundle, geometry: Pi05Geometry) -> Tuple[str, str]:
    fingerprint = policy_assets.checkpoint_fingerprint(bundle.model_dir)
    options = {
        "frontend": "direct",
        "dtype": "float16",
        "max_batch_size": geometry.max_batch_size,
        "max_prefix_len": geometry.max_prefix_len,
        "num_denoise_steps": geometry.num_denoise_steps,
        "policy_config": geometry.policy_config,
        "action_dim": geometry.action_dim,
        "action_horizon": geometry.action_horizon,
        "paligemma_variant": bundle.root["paligemma_variant"],
        "expert_variant": bundle.root["action_expert_variant"],
    }
    payload = json.dumps(
        {
            "contract_version": CONTRACT_VERSION,
            "edgellm_version": __version__,
            "checkpoint_fingerprint": fingerprint,
            "options": options,
        },
        sort_keys=True)
    return fingerprint, hashlib.sha256(payload.encode()).hexdigest()[:32]


def _component_names(engine_dir: str) -> List[str]:
    return [
        name for name in _COMPONENTS
        if os.path.isfile(os.path.join(engine_dir, name, "config.json"))
    ]


def _write_config(bundle, args, engine_dir: str, export_id: str) -> None:
    component = args.resolved_component
    payload = runtime_config.component_runtime_config(bundle, component, args)
    if payload is None:
        raise ValueError(f"pi0.5 has no config for {component.value}")
    payload["contract_version"] = CONTRACT_VERSION
    payload["edgellm_version"] = __version__
    payload["export_id"] = export_id
    output_dir = contracts.component_spec(component).output_dir(engine_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as config_file:
        json.dump(payload, config_file, indent=2)


def _write_embedding(bundle, engine_dir: str) -> None:
    checkpoint = Weights(bundle.model_dir,
                         component="prefix",
                         conversion=weights)
    try:
        table = checkpoint.f16("embed_tokens.weight")
    finally:
        checkpoint.close()
    path = os.path.join(engine_dir, "embed_tokens.safetensors")
    save_safetensors(path, {"embed_tokens.weight": table})
    config_path = os.path.join(engine_dir, "prefix", "config.json")
    with open(config_path) as config_file:
        config = json.load(config_file)
    config["artifacts"] = {
        "embed_tokens.safetensors": {
            "sha256": policy_assets.sha256_file(path),
            "dtype": "float16",
            "shape": list(table.shape),
        }
    }
    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=2)


def _stage_assets(bundle, engine_dir: str) -> None:
    source = os.path.join(bundle.model_dir, "assets")
    if not any("norm_stats.json" in filenames
               for _, _, filenames in os.walk(source)):
        logger.warning(
            "pi0.5 checkpoint has no norm_stats.json; stage it under %s "
            "before running the policy -- the runtime refuses to load "
            "without it", os.path.join(engine_dir, "assets"))
        return
    policy_assets.copy_asset_tree(source, os.path.join(engine_dir, "assets"))
    logger.info("Staged pi0.5 policy assets from %s", source)


def _tokenizer_contract(engine_dir: str, existing: Optional[dict]) -> dict:
    tokenizer_dir = os.path.join(engine_dir,
                                 policy_assets.TEXT_TOKENIZER_DIRNAME)
    if os.path.isdir(tokenizer_dir):
        if existing is None or not isinstance(existing.get("tokenizer"), dict):
            raise ValueError(
                f"{tokenizer_dir} exists without a verifiable policy.json")
        return dict(existing["tokenizer"])
    return policy_assets.stage_text_tokenizer(engine_dir)


def _validate_existing(engine_dir: str,
                       fingerprint: str,
                       export_id: str,
                       reject_unidentified: bool = False) -> None:
    manifest = policy_assets.load_json_if_present(
        os.path.join(engine_dir, policy_assets.POLICY_CONTRACT_FILENAME))
    candidates = []
    if manifest is not None:
        candidates.append((policy_assets.POLICY_CONTRACT_FILENAME, manifest))
    for component in _COMPONENTS:
        path = os.path.join(engine_dir, component, "config.json")
        config = policy_assets.load_json_if_present(path)
        if config is not None:
            candidates.append((f"{component}/config.json", config))
    if (reject_unidentified and not candidates and os.path.isdir(engine_dir)
            and os.listdir(engine_dir)):
        raise ValueError(
            f"{engine_dir} is not empty but holds no identifiable pi0.5 "
            "bundle; use an empty engine directory")
    for path, payload in candidates:
        if (payload.get("export_id") != export_id
                or (path == policy_assets.POLICY_CONTRACT_FILENAME
                    and payload.get("checkpoint_fingerprint") != fingerprint)):
            raise ValueError(
                f"{engine_dir} already contains a different pi0.5 build in "
                f"{path}; use an empty engine directory")


def _policy_contract(bundle, engine_dir: str, tokenizer: dict,
                     fingerprint: str, export_id: str,
                     geometry: Pi05Geometry) -> dict:
    return policy_assets.build_policy_contract(
        bundle.model_dir, tokenizer, {
            "contract_version": CONTRACT_VERSION,
            "checkpoint_fingerprint": fingerprint,
            "export_id": export_id,
            "components": _component_names(engine_dir),
        }, geometry.policy_config)


def _validate_policy_semantics(engine_dir: str, existing: Optional[dict],
                               contract: dict) -> None:
    if (existing is not None and policy_assets.policy_semantics(existing)
            != policy_assets.policy_semantics(contract)):
        raise ValueError(
            f"{engine_dir} already describes different pi0.5 policy semantics")


def _write_policy(bundle, engine_dir: str, fingerprint: str, export_id: str,
                  geometry: Pi05Geometry) -> None:
    manifest_path = os.path.join(engine_dir,
                                 policy_assets.POLICY_CONTRACT_FILENAME)
    existing = policy_assets.load_json_if_present(manifest_path)
    if existing is not None:
        for field, expected in (("checkpoint_fingerprint", fingerprint),
                                ("export_id", export_id)):
            if existing.get(field) != expected:
                raise ValueError(
                    f"{engine_dir} already contains a different pi0.5 build: "
                    f"{field}={existing.get(field)!r}, expected {expected!r}")
    tokenizer = _tokenizer_contract(engine_dir, existing)
    contract = _policy_contract(bundle, engine_dir, tokenizer, fingerprint,
                                export_id, geometry)
    _validate_policy_semantics(engine_dir, existing, contract)
    policy_assets.write_policy_contract(engine_dir, contract)


def preflight_bundle(bundle, args, engine_dir: str) -> None:
    """Validate identity and precision before TensorRT or output writes."""
    configuration.validate_checkpoint_precision(bundle, args)
    geometry = Pi05Geometry.from_bundle(bundle, args)
    fingerprint, export_id = _export_identity(bundle, geometry)
    _validate_existing(engine_dir,
                       fingerprint,
                       export_id,
                       reject_unidentified=True)
    manifest_path = os.path.join(engine_dir,
                                 policy_assets.POLICY_CONTRACT_FILENAME)
    existing = policy_assets.load_json_if_present(manifest_path)
    if existing is not None:
        tokenizer = existing.get("tokenizer")
        if not isinstance(tokenizer, dict):
            raise ValueError(
                f"{manifest_path} has no verifiable tokenizer contract")
        contract = _policy_contract(bundle, engine_dir, tokenizer, fingerprint,
                                    export_id, geometry)
        _validate_policy_semantics(engine_dir, existing, contract)


def write_artifacts(bundle, config, args, engine_dir: str) -> None:
    """Write the sidecars expected by the existing pi0.5 C++ runtime."""
    if config is not None:
        raise ValueError("pi0.5 components do not use DeviceConfig artifacts")
    geometry = Pi05Geometry.from_bundle(bundle, args)
    fingerprint, export_id = _export_identity(bundle, geometry)
    _validate_existing(engine_dir, fingerprint, export_id)
    _write_config(bundle, args, engine_dir, export_id)
    if args.resolved_component == contracts.Component.PREFIX:
        _write_embedding(bundle, engine_dir)
    _stage_assets(bundle, engine_dir)
    _write_policy(bundle, engine_dir, fingerprint, export_id, geometry)
