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
"""pi0.5 policy configuration and optimization profiles."""

from __future__ import annotations

from dataclasses import dataclass

from tensorrt_edgellm.models.pi05 import policy_assets
from tensorrt_edgellm.models.pi05.component_contracts import (
    MAX_DENOISE_STEPS, config_field, default_max_prefix_len, gemma_variant,
    kv_pages_per_seq, num_adarms_sites, resolve_num_denoise_steps,
    validate_max_prefix_len, xqa_cache_capacity)

from ...core import contracts


def component_config(root: dict, component: contracts.Component) -> dict:
    """Return the raw policy checkpoint config for every pi0.5 component."""
    del component
    return root


def validate_build(args, components) -> None:
    """Validate pi0.5-only direct-builder options."""
    del components
    if args.spec_type != "none" or args.spec_role != "none":
        raise ValueError("pi0.5 does not support speculative decoding")
    dense = getattr(args, "dense", getattr(args, "dense_quant", "auto"))
    if dense not in ("auto", "fp16"):
        raise ValueError(
            "pi0.5 direct builds support FP16 dense precision only")
    if getattr(args, "externalize_weights", None):
        raise ValueError("pi0.5 direct builds do not support external weights")
    if getattr(args, "fp8_embedding", False):
        raise ValueError("pi0.5 direct builds require FP16 embeddings")
    if args.pi05_max_prefix_len < 0:
        raise ValueError("--pi05-max-prefix-len must be non-negative")
    if args.pi05_denoise_steps < 0:
        raise ValueError("--pi05-denoise-steps must be non-negative")


def validate_checkpoint_precision(bundle, args) -> None:
    """Reject quantized checkpoints before TensorRT network creation."""
    from ...core import quantization
    from . import weights

    quant = quantization.parse_quantization(bundle.model_dir, bundle.root,
                                            bundle.root, weights)
    if quant.is_quantized or quant.kv_cache_quant is not None:
        raise ValueError(
            "pi0.5 direct builds support unquantized checkpoints only; "
            f"resolved checkpoint precision is {quant.quant_type}")
    if args.dense_quant not in ("auto", "fp16"):
        raise ValueError(
            "pi0.5 direct builds support FP16 dense precision only")


def component_weight_policy(args, policy):
    """Keep weights in plans because the experimental runtime binds no weights."""
    return policy.without(policy.kinds, strict=bool(args.externalize_weights))


@dataclass(frozen=True)
class Pi05Geometry:
    """Model and policy dimensions shared by all component graphs."""

    policy_config: str
    prefix: dict
    action: dict
    action_dim: int
    action_horizon: int
    max_prefix_len: int
    num_denoise_steps: int
    opt_views: int
    max_views: int
    max_batch_size: int

    @property
    def num_adarms_sites(self) -> int:
        return num_adarms_sites(int(self.action["num_hidden_layers"]))

    @property
    def modulation_dim(self) -> int:
        return 3 * int(self.action["hidden_size"])

    @property
    def cache_capacity(self) -> int:
        return xqa_cache_capacity(self.max_prefix_len, self.action_horizon)

    @property
    def pages_per_seq(self) -> int:
        return kv_pages_per_seq(self.max_prefix_len, self.action_horizon)

    @classmethod
    def from_bundle(cls, bundle, args) -> "Pi05Geometry":
        root = bundle.root
        requested = args.pi05_policy_config or None
        policy_name = policy_assets.policy_config_name(bundle.model_dir, root,
                                                       requested)
        policy = policy_assets.OPENPI_POLICY_CONTRACTS[policy_name]
        prefix = gemma_variant(
            str(
                config_field(root, ("paligemma_variant", ),
                             "paligemma_variant")))
        action = gemma_variant(
            str(
                config_field(root, ("action_expert_variant", ),
                             "action_expert_variant")))
        action_dim = int(
            config_field(root, ("action_dim", "max_action_dim"), "action dim"))
        max_prefix_len = int(args.pi05_max_prefix_len
                             or default_max_prefix_len())
        steps = resolve_num_denoise_steps(root, args.pi05_denoise_steps
                                          or None)
        geometry = cls(
            policy_config=policy_name,
            prefix=prefix,
            action=action,
            action_dim=action_dim,
            action_horizon=int(policy["action_horizon"]),
            max_prefix_len=max_prefix_len,
            num_denoise_steps=steps,
            opt_views=int(policy["opt_views"]),
            max_views=policy_assets.MAX_CAMERA_SLOTS,
            max_batch_size=int(args.max_batch_size),
        )
        geometry.validate()
        return geometry

    def validate(self) -> None:
        validate_max_prefix_len(
            self.policy_config,
            policy_assets.OPENPI_POLICY_CONTRACTS[self.policy_config],
            self.max_prefix_len)
        if not 1 <= self.num_denoise_steps <= MAX_DENOISE_STEPS:
            raise ValueError(
                "pi0.5 denoise steps must be in "
                f"[1, {MAX_DENOISE_STEPS}], got {self.num_denoise_steps}")
        if self.max_batch_size <= 0:
            raise ValueError("pi0.5 max batch size must be positive")
        if self.action_horizon <= 0 or self.action_dim <= 0:
            raise ValueError("pi0.5 action dimensions must be positive")
        if int(self.prefix["num_hidden_layers"]) != int(
                self.action["num_hidden_layers"]):
            raise ValueError("pi0.5 prefix and action layer counts must match")
        for field in ("head_dim", "num_key_value_heads"):
            if int(self.prefix[field]) != int(self.action[field]):
                raise ValueError(
                    f"pi0.5 prefix and action {field} values must match")


def setup_profiles(builder, builder_config, network, args, bundle) -> bool:
    """Install the optimization profile declared by the shared ABI."""
    del network
    from . import runtime_config

    config = runtime_config.component_runtime_config(bundle,
                                                     args.resolved_component,
                                                     args)
    if config is None:
        return False
    profile = builder.create_optimization_profile()
    for name, bounds in config["optimization_profile"].items():
        profile.set_shape(name, tuple(bounds["min"]), tuple(bounds["opt"]),
                          tuple(bounds["max"]))
    builder_config.add_optimization_profile(profile)
    return True
