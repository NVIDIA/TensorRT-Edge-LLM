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
"""pi0.5 policy component export (library; no standalone CLI).

Invoked through the unified ``tensorrt-edgellm-export`` entry point, which
routes pi0.5 checkpoints here. Exports the policy components -- ``visual``,
``prefix``, ``action`` and, unless hoisting is turned off, ``cond`` -- plus the
runtime sidecars from a converted openpi PyTorch checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from typing import Dict

import torch

from ..._version import __version__
from ...onnx.export import _strip_attention_plugin_optional_inputs
from ...onnx.export_encoder import _run_dynamo_export
from . import component_contracts
from .modeling_pi05_action import (Pi05Action, Pi05ActionConfig,
                                   build_pi05_action, build_pi05_cond)
from .modeling_pi05_prefix import (Pi05Prefix, Pi05PrefixConfig,
                                   build_pi05_prefix)
from .modeling_pi05_visual import (Pi05Visual, Pi05VisualConfig,
                                   build_pi05_visual)
from .policy_assets import (MAX_CAMERA_SLOTS, OPENPI_POLICY_CONTRACTS,
                            POLICY_CONTRACT_FILENAME, TEXT_TOKENIZER_DIRNAME,
                            build_policy_contract, checkpoint_fingerprint,
                            copy_asset_tree, load_json_if_present,
                            policy_config_name, policy_semantics, sha256_file,
                            stage_text_tokenizer, write_policy_contract)
from .weights import (is_pi05_weights, load_checkpoint_weights,
                      load_pi05_config, split_pi05_weights)

logger = logging.getLogger(__name__)

PI05_COMPONENTS = ("visual", "prefix", "action")
# ``cond`` is emitted by the action export under hoisting, not selected directly.
ALL_COMPONENTS = PI05_COMPONENTS + ("cond", )


def _write_component_config(out_dir: str, config: dict,
                            export_id: str) -> None:
    """Stamp contract/package versions and the export identity, and write it."""
    config = dict(config)
    config["contract_version"] = component_contracts.CONTRACT_VERSION
    config["edgellm_version"] = __version__
    config["export_id"] = export_id
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


def _export_id(fingerprint: str, options: dict) -> str:
    """Identity stamped on every component of one export.

    Derived rather than random: re-exporting a single component from the same
    checkpoint and options must reproduce it, which is what makes a partial
    re-export into an existing directory provable rather than merely plausible.
    """
    payload = json.dumps(
        {
            "contract_version": component_contracts.CONTRACT_VERSION,
            "edgellm_version": __version__,
            "checkpoint_fingerprint": fingerprint,
            "options": options,
        },
        sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def _present_components(output_dir: str) -> "list[str]":
    return [
        c for c in ALL_COMPONENTS
        if os.path.isfile(os.path.join(output_dir, c, "config.json"))
    ]


def _validate_existing_export(output_dir: str, fingerprint: str,
                              export_id: str) -> None:
    """Refuse to write ONNX components beside ones from a different export.

    A partial export reuses whatever is already in ``output_dir``, so without this
    a directory can end up holding components from two checkpoints or two sets of
    export options. Identity is the ``export_id``, which hashes the checkpoint
    fingerprint together with the options. Runs before anything is written, so a
    rejected export leaves the directory exactly as it found it.
    """
    conflicts = []
    manifest = load_json_if_present(
        os.path.join(output_dir, POLICY_CONTRACT_FILENAME))
    if manifest is not None:
        for field, expected in (("checkpoint_fingerprint", fingerprint),
                                ("export_id", export_id)):
            if manifest.get(field) != expected:
                conflicts.append((POLICY_CONTRACT_FILENAME, field,
                                  manifest.get(field), expected))
    for component in _present_components(output_dir):
        config = load_json_if_present(
            os.path.join(output_dir, component, "config.json")) or {}
        if config.get("export_id") != export_id:
            conflicts.append((f"{component}/config.json", "export_id",
                              config.get("export_id"), export_id))
    if not conflicts:
        # assets/ and text_tokenizer/ carry no export identity, so a directory
        # holding only those cannot be told apart from an earlier export's
        # leftovers and must not be merged into.
        if (manifest is None and not _present_components(output_dir)
                and os.path.isdir(output_dir) and os.listdir(output_dir)):
            raise ValueError(
                f"{output_dir} is not empty but holds no pi0.5 export: "
                f"{sorted(os.listdir(output_dir))}. Export into an empty "
                "directory, so no untagged leftover of an earlier export "
                "survives beside it.")
        return
    detail = "; ".join(f"{path} has {field}={found!r}, this export has "
                       f"{expected!r}"
                       for path, field, found, expected in conflicts)
    raise ValueError(
        f"{output_dir} already holds a different pi0.5 export: {detail}. "
        "Export into an empty directory, or re-export every component from the "
        "same checkpoint and options.")


def _validate_policy_semantics(manifest_path: str, contract: dict) -> None:
    """Refuse a partial re-export that would redefine the policy in place.

    ``export_id`` covers the weights and the export options, not the policy
    semantics, the feature contract or the tokenizer the manifest is derived
    from. A re-export rewrites the manifest but leaves the components it was not
    asked for, so without this those declarations can move under components built
    against the old ones. Runs before any output artifact is mutated and before the
    checkpoint weights are loaded.
    """
    previous = load_json_if_present(manifest_path)
    if previous is None:
        return
    was, now = policy_semantics(previous), policy_semantics(contract)
    changed = sorted(k for k in now if was.get(k) != now[k])
    if not changed:
        return
    detail = "; ".join(f"{k}: {was.get(k)!r} -> {now[k]!r}" for k in changed)
    raise ValueError(
        f"{manifest_path} describes a different policy than this export would: "
        f"{detail}. Re-export every component, into an empty directory.")


def _discard(paths: "list[str]") -> None:
    """Drop what a failed run wrote, so no half-updated export looks complete."""
    for path in paths:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.isfile(path):
            os.remove(path)
        else:
            continue
        logger.error("Removed %s left behind by the failed pi0.5 export", path)


# ---------------------------------------------------------------------------
# Sidecars
# ---------------------------------------------------------------------------


def _save_embedding_sidecar(out_dir: str, embedding: Dict[str, torch.Tensor],
                            dtype: torch.dtype) -> dict:
    """Write the token embedding table the host gathers on the prefix path."""
    import safetensors.torch as st

    if "embed_tokens.weight" not in embedding:
        raise KeyError("pi0.5 checkpoint has no language_model.embed_tokens")
    tensor = embedding["embed_tokens.weight"].to(dtype).contiguous()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "embed_tokens.safetensors")
    st.save_file({"embed_tokens.weight": tensor}, path)
    logger.info("Saved embed_tokens.safetensors %s", tuple(tensor.shape))
    return {
        "embed_tokens.safetensors": {
            "sha256": sha256_file(path),
            "dtype": str(dtype).replace("torch.", ""),
            "shape": list(tensor.shape),
        }
    }


def _has_norm_stats(assets_dir: str) -> bool:
    return any("norm_stats.json" in files
               for _, _, files in os.walk(assets_dir))


def _stage_policy_assets(checkpoint: str, output_dir: str) -> None:
    """Copy the normalization statistics the policy layer unnormalizes with.

    The supported HF weight conversion does not include openpi's normalization
    assets, so their absence cannot be an export error; the documented workflow
    stages the matching ``norm_stats.json`` after export. An ``assets/`` carrying
    none is an error: it would pass a staging check while normalizing nothing.
    """
    src = os.path.join(checkpoint, "assets")
    dst = os.path.join(output_dir, "assets")
    if not os.path.isdir(src):
        logger.warning(
            "pi0.5 checkpoint has no assets/; stage norm_stats.json under %s "
            "before running the policy -- the runtime refuses to load without "
            "it", dst)
        return
    if not _has_norm_stats(src):
        raise FileNotFoundError(
            f"{src} carries no norm_stats.json; the policy layer cannot "
            "normalize the state or convert actions to robot units")
    copy_asset_tree(src, dst)
    logger.info("Staged policy assets: %s", dst)


# ---------------------------------------------------------------------------
# Per-component export
# ---------------------------------------------------------------------------


def export_visual(weights: Dict[str, torch.Tensor], out_dir: str,
                  dtype: torch.dtype, num_views: int, max_views: int,
                  export_id: str) -> None:
    cfg = Pi05VisualConfig()
    model: Pi05Visual = build_pi05_visual(cfg, weights, dtype).to("cpu")
    args, input_names, output_names, dynamic_shapes = (
        model.get_onnx_export_args("cpu"))
    os.makedirs(out_dir, exist_ok=True)
    _run_dynamo_export(model, args, os.path.join(out_dir, "model.onnx"),
                       input_names, output_names, dynamic_shapes)
    _write_component_config(
        out_dir,
        component_contracts.make_visual_config(cfg, num_views, max_views),
        export_id)
    logger.info("pi0.5 visual export complete: %s", out_dir)


def export_prefix(weights: Dict[str, torch.Tensor],
                  embedding: Dict[str, torch.Tensor], out_dir: str,
                  dtype: torch.dtype, variant: str, max_prefix_len: int,
                  max_batch_size: int, export_id: str) -> None:
    cfg = Pi05PrefixConfig(**component_contracts.gemma_variant(variant))
    model: Pi05Prefix = build_pi05_prefix(cfg, weights, dtype).to("cpu")
    args, input_names, output_names, dynamic_shapes = (
        model.get_onnx_export_args("cpu"))
    os.makedirs(out_dir, exist_ok=True)
    _run_dynamo_export(model, args, os.path.join(out_dir, "model.onnx"),
                       input_names, output_names, dynamic_shapes)
    artifacts = _save_embedding_sidecar(out_dir, embedding, dtype)
    config = component_contracts.make_prefix_config(cfg, max_prefix_len,
                                                    max_batch_size)
    config["artifacts"] = artifacts
    _write_component_config(out_dir, config, export_id)
    logger.info("pi0.5 prefix export complete: %s", out_dir)


def export_action(weights: Dict[str, torch.Tensor], out_dir: str,
                  dtype: torch.dtype, variant: str, action_dim: int,
                  action_horizon: int, max_prefix_len: int,
                  max_batch_size: int, num_denoise_steps: int,
                  hoist_cond: bool, cond_dir: str, export_id: str) -> None:
    cfg = Pi05ActionConfig(**component_contracts.gemma_variant(variant),
                           action_dim=action_dim,
                           action_horizon=action_horizon)
    model: Pi05Action = build_pi05_action(cfg, weights, dtype,
                                          hoist_cond).to("cpu")
    args, input_names, output_names, dynamic_shapes = (
        model.get_onnx_export_args(max_prefix_len, "cpu"))
    os.makedirs(out_dir, exist_ok=True)
    onnx_path = os.path.join(out_dir, "model.onnx")
    _run_dynamo_export(model, args, onnx_path, input_names, output_names,
                       dynamic_shapes)
    # The translation emits every optional AttentionPlugin input; the C++ contract
    # counts only the enabled groups.
    _strip_attention_plugin_optional_inputs(onnx_path)
    _write_component_config(
        out_dir,
        component_contracts.make_action_config(cfg, max_prefix_len,
                                               max_batch_size,
                                               num_denoise_steps, hoist_cond),
        export_id)
    logger.info("pi0.5 action export complete: %s", out_dir)

    if hoist_cond:
        # Exported from the same loaded model: the cond graph's row order and
        # the action graph's slice indices come from one site enumeration.
        export_cond(model, cfg, cond_dir, num_denoise_steps, export_id)


def export_cond(action: Pi05Action, cfg: Pi05ActionConfig, out_dir: str,
                num_denoise_steps: int, export_id: str) -> None:
    model = build_pi05_cond(action)
    args, input_names, output_names, dynamic_shapes = (
        model.get_onnx_export_args(num_denoise_steps, "cpu"))
    os.makedirs(out_dir, exist_ok=True)
    _run_dynamo_export(model, args, os.path.join(out_dir, "model.onnx"),
                       input_names, output_names, dynamic_shapes)
    _write_component_config(
        out_dir, component_contracts.make_cond_config(cfg, num_denoise_steps),
        export_id)
    logger.info("pi0.5 cond export complete: %s", out_dir)


def export_pi05_components(
    checkpoint: str,
    output_dir: str,
    components: "list[str] | None" = None,
    dtype: torch.dtype = torch.float16,
    policy_config: "str | None" = None,
    num_views: "int | None" = None,
    max_views: int = MAX_CAMERA_SLOTS,
    max_token_len: int = component_contracts.DEFAULT_MAX_TOKEN_LEN,
    max_prefix_len: "int | None" = None,
    max_batch_size: int = 1,
    num_denoise_steps: "int | None" = None,
    hoist_adarms_cond: bool = True,
) -> None:
    """Export pi0.5 policy components from a converted openpi checkpoint.

    ``hoist_adarms_cond`` (default) moves the time embedder and the per-site
    AdaRMS Denses out of the per-step action graph into a one-shot ``cond``
    component. Turning it off exports no ``cond`` and keeps them per-step.
    """
    if dtype != torch.float16:
        # Rejected here rather than at load: the attention plugin and the component
        # contracts are fp16-only, so another dtype would export and build cleanly and
        # only then bind wrong shapes.
        raise ValueError(f"pi0.5 export supports float16 only, got {dtype}")

    components = list(components or PI05_COMPONENTS)
    unknown = [c for c in components if c not in PI05_COMPONENTS]
    if unknown:
        raise ValueError(f"Unsupported pi0.5 components: {unknown}; "
                         f"expected a subset of {list(PI05_COMPONENTS)}")

    config = load_pi05_config(checkpoint)
    action_dim = int(
        component_contracts.config_field(config,
                                         ("action_dim", "max_action_dim"),
                                         "action dim"))
    # From the openpi configuration, not the checkpoint: a Hugging Face mirror ships
    # LeRobot's own horizon (chunk_size 50, n_action_steps 10) and the graph must be
    # built at the same H the manifest declares.
    config_name = policy_config_name(checkpoint, config, policy_config)
    policy = OPENPI_POLICY_CONTRACTS[config_name]
    action_horizon = int(policy["action_horizon"])
    if num_views is None:
        num_views = int(policy["opt_views"])
    num_denoise_steps = component_contracts.resolve_num_denoise_steps(
        config, num_denoise_steps)
    paligemma_variant = component_contracts.config_field(
        config, ("paligemma_variant", ), "paligemma variant")
    expert_variant = component_contracts.config_field(
        config, ("action_expert_variant", ), "action expert variant")

    if max_prefix_len is None:
        max_prefix_len = component_contracts.default_max_prefix_len(
            max_token_len)
    component_contracts.validate_max_prefix_len(config_name, policy,
                                                max_prefix_len)

    options = {
        "frontend": "onnx",
        "dtype": "float16",
        "num_views": num_views,
        "max_views": max_views,
        "max_token_len": max_token_len,
        "max_prefix_len": max_prefix_len,
        "max_batch_size": max_batch_size,
        "num_denoise_steps": num_denoise_steps,
        "hoist_adarms_cond": hoist_adarms_cond,
        "policy_config": config_name,
        # Shapes and variants the weights do not pin: the same checkpoint under a
        # changed config must not reuse a partial export built from the old one.
        "action_dim": action_dim,
        "action_horizon": action_horizon,
        "paligemma_variant": paligemma_variant,
        "expert_variant": expert_variant,
    }
    fingerprint = checkpoint_fingerprint(checkpoint)
    export_id = _export_id(fingerprint, options)
    _validate_existing_export(output_dir, fingerprint, export_id)
    logger.info(
        "pi0.5 export %s (%s): action_dim=%d action_horizon=%d "
        "max_prefix_len=%d num_denoise_steps=%d", export_id, config_name,
        action_dim, action_horizon, max_prefix_len, num_denoise_steps)

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, POLICY_CONTRACT_FILENAME)
    # Written last, and discarded on failure only when this run created it: it is the file
    # that claims the set is complete. A re-export into an existing directory keeps the
    # previous manifest and components, so a failure partway can still leave a directory
    # that looks complete and carries the same export id. Detecting that needs a
    # transactional export, which this is not.
    staged = [] if os.path.exists(manifest_path) else [manifest_path]
    try:
        # Only what this run creates is discardable. A partial re-export reuses whatever
        # is already here, and removing that on failure would destroy a valid export.
        def stage_if_new(path: str) -> None:
            if not os.path.exists(path):
                staged.append(path)

        assets_dir = os.path.join(output_dir, "assets")
        # Nothing in output_dir may be touched before the contract is accepted: a
        # rejected export must leave the bundle it refused exactly as it found it.
        with tempfile.TemporaryDirectory() as scratch:
            tokenizer = stage_text_tokenizer(scratch)
            contract = build_policy_contract(
                checkpoint, tokenizer, {
                    "contract_version": component_contracts.CONTRACT_VERSION,
                    "checkpoint_fingerprint": fingerprint,
                    "export_id": export_id,
                    "components": [],
                }, config_name)
            _validate_policy_semantics(manifest_path, contract)

            assets_existed = os.path.isdir(assets_dir)
            _stage_policy_assets(checkpoint, output_dir)
            if os.path.isdir(assets_dir) and not assets_existed:
                staged.append(assets_dir)
            tokenizer_dir = os.path.join(output_dir, TEXT_TOKENIZER_DIRNAME)
            stage_if_new(tokenizer_dir)
            shutil.rmtree(tokenizer_dir, ignore_errors=True)
            shutil.move(os.path.join(scratch, TEXT_TOKENIZER_DIRNAME),
                        tokenizer_dir)

        raw = load_checkpoint_weights(checkpoint)
        if not is_pi05_weights(raw):
            raise ValueError(
                f"{checkpoint} is not a pi0.5 checkpoint (no time_mlp/adaRMS "
                "signature); pi0 is not supported by this exporter")
        visual_w, prefix_w, embedding_w, action_w = split_pi05_weights(raw)

        for component in components:
            out_dir = os.path.join(output_dir, component)
            stage_if_new(out_dir)
            if component == "visual":
                export_visual(visual_w, out_dir, dtype, num_views, max_views,
                              export_id)
            elif component == "prefix":
                export_prefix(prefix_w, embedding_w, out_dir, dtype,
                              paligemma_variant, max_prefix_len,
                              max_batch_size, export_id)
            elif component == "action":
                if hoist_adarms_cond:
                    stage_if_new(os.path.join(output_dir, "cond"))
                export_action(action_w, out_dir, dtype, expert_variant,
                              action_dim, action_horizon, max_prefix_len,
                              max_batch_size, num_denoise_steps,
                              hoist_adarms_cond,
                              os.path.join(output_dir, "cond"), export_id)

        contract["components"] = _present_components(output_dir)
        write_policy_contract(output_dir, contract)
    except BaseException:
        _discard(staged)
        raise
