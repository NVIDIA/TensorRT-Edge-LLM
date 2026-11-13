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
"""MiniMax-H3 runtime artifacts written next to each engine."""

from __future__ import annotations

import json
import math
import os
import shutil

import numpy as np

from ...core import contracts
from . import configuration
from .modeling_minimax_h3_vae import TILE_LATENT_FRAMES, TILE_LATENT_SIZE

Component = contracts.Component


def _euler_schedule(num_steps: int, shift: float) -> list:
    """Return per-step ``[sigma_from_timestep, sigma_next / sigma]``.

    The model is conditioned on ``t = 1 - sigma`` and the scheduler recovers
    sigma from that float32 timestep, so both values are kept as float32.
    """
    sigmas = configuration.flow_sigmas(num_steps, shift)
    timestep_sigmas = np.float32(1.0) - (np.float32(1.0) - sigmas[:-1])
    ratios = sigmas[1:] / sigmas[:-1]
    return [[float(sigma), float(ratio)]
            for sigma, ratio in zip(timestep_sigmas, ratios)]


def _video_layout(bundle) -> dict:
    """Return how the video VAE splits a request into clips and tiles."""
    config = configuration.component_config(bundle.root, Component.VAE_DECODER)
    clip_length = int(config["clip_length"])
    temporal = math.prod(config["temporal_downsample_factors"])
    chunk = math.ceil(clip_length / temporal)
    padding = -clip_length % temporal
    overlap = max(-int(config["token_drop"]) % chunk * temporal - padding, 0)
    return {
        "spatial_ratio": math.prod(config["spatial_downsample_factors"]),
        "temporal_ratio": temporal,
        "tile_latent_frames": TILE_LATENT_FRAMES,
        "tile_latent_size": TILE_LATENT_SIZE,
        "tile_min_overlap": 64,
        "chunk_latent_frames": chunk,
        "chunk_frame_padding": padding,
        "chunk_frame_overlap": overlap,
    }


def _runtime_config(bundle, component: Component, args) -> dict:
    config = configuration.component_config(bundle.root, component)
    gen = configuration.component_config(bundle.root, Component.GEN)
    if component == Component.TEXT_ENCODER:
        return {
            "max_text_tokens": configuration.MAX_TEXT_TOKENS,
            "hidden_size": int(config["text_config"]["hidden_size"]),
        }
    if component == Component.GEN:
        video_shift, audio_shift = bundle.root["_flow_shifts"]
        steps = args.minimax_h3_num_steps
        return {
            "max_text_tokens": configuration.MAX_TEXT_TOKENS,
            "max_audio_tokens": configuration.MAX_AUDIO_TOKENS,
            "max_video_tokens": args.minimax_h3_max_video_tokens,
            "latent_channels": int(config["in_channels"]),
            "audio_channels": int(config["audio_in_channels"]),
            "num_steps": steps,
            "video_schedule": _euler_schedule(steps, video_shift),
            "audio_schedule": _euler_schedule(steps, audio_shift),
            "vae": _video_layout(bundle),
        }
    if component == Component.VAE_DECODER:
        return {
            "latent_channels": int(config["latent_channels"]),
            **_video_layout(bundle),
        }
    return {
        "latent_channels": int(gen["audio_in_channels"]),
        "max_frames": configuration.MAX_AUDIO_TOKENS // 2,
        "hop_length": math.prod(config["encoder_rates"]),
        "sampling_rate": int(config["sampling_rate"]),
    }


def write_artifacts(bundle, config, args, engine_dir: str) -> None:
    """Write the component's runtime config and, next to the text encoder, the tokenizer."""
    del config
    component = Component(args.resolved_component)
    spec = contracts.component_spec(component)
    output_dir = spec.output_dir(engine_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as file:
        json.dump(_runtime_config(bundle, component, args), file, indent=2)
    if component == Component.TEXT_ENCODER:
        shutil.copytree(os.path.join(bundle.model_dir, "tokenizer"),
                        output_dir,
                        dirs_exist_ok=True,
                        copy_function=shutil.copyfile)
