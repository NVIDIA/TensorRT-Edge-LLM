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
"""Nemotron-3.5-ASR component and profile configuration."""

from ...core import contracts, weight_policy


def component_config(root: dict, component: contracts.Component) -> dict:
    if component == contracts.Component.AUDIO:
        return root["encoder_config"]
    if component == contracts.Component.RNNT:
        return root
    raise ValueError(
        f"Nemotron-3.5-ASR has no {component.value} configuration")


def component_weight_policy(args, policy):
    """Keep both standalone engines self-contained for their C++ runtime."""
    del args
    return policy.without(weight_policy.EXTERNAL_WEIGHT_KINDS)


def setup_profiles(builder, builder_config, network, args, bundle) -> bool:
    del network
    if args.resolved_component == contracts.Component.RNNT:
        return True
    if args.resolved_component != contracts.Component.AUDIO:
        return False
    optimum = max(args.min_time_steps, args.max_time_steps // 2)
    mel_bins = int(bundle.root["encoder_config"].get("num_mel_bins", 128))
    profile = builder.create_optimization_profile()
    profile.set_shape("input_features", (1, args.min_time_steps, mel_bins),
                      (1, optimum, mel_bins),
                      (1, args.max_time_steps, mel_bins))
    profile.set_shape("prompt_ids", (1, ), (1, ), (1, ))
    builder_config.add_optimization_profile(profile)
    return True
