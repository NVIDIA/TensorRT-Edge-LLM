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
"""MiniMax-H3 Diffusers checkpoint components and engine bounds."""

from __future__ import annotations

import json
import logging
import os

import numpy as np

from ...core import contracts, weight_policy

Component = contracts.Component

logger = logging.getLogger(__name__)

#: GEN is conditioned on the unnormalized hidden state after this Qwen3-VL layer.
TEXT_ENCODER_LAYERS = 50
MAX_TEXT_TOKENS = 4096
#: Stereo audio rows: two channels of 40 Hz latents, enough for 25 seconds.
MAX_AUDIO_TOKENS = 2048
#: Optimization-profile hint: the default 1344x768, 124-frame request.
OPT_TEXT_TOKENS = 128
OPT_VIDEO_TOKENS = 37296
OPT_AUDIO_TOKENS = 414

_SUBFOLDERS = {
    Component.TEXT_ENCODER: "text_encoder",
    Component.GEN: "transformer",
    Component.VAE_DECODER: "vae",
    Component.AUDIO_DECODER: "audio_vae",
}


def subfolder(component) -> str:
    """Return the Diffusers checkpoint folder of one component."""
    return _SUBFOLDERS[Component(component)]


def _load_json(path: str):
    with open(path) as file:
        return json.load(file)


def prepare_root(model_dir: str, root: dict) -> dict:
    """Attach the component and scheduler configs of the pipeline."""
    prepared = dict(root)
    configs = {}
    for component, folder in _SUBFOLDERS.items():
        path = os.path.join(model_dir, folder, "config.json")
        if os.path.isfile(path):
            configs[component.value] = _load_json(path)
    prepared["_component_configs"] = configs
    prepared["_flow_shifts"] = tuple(
        float(
            _load_json(os.path.join(model_dir, folder,
                                    "scheduler_config.json"))["shift"])
        for folder in ("scheduler", "audio_scheduler"))
    return prepared


def available_components(root: dict, registered):
    return frozenset(
        Component(name)
        for name in root["_component_configs"]).intersection(registered)


def component_config(root: dict, component) -> dict:
    return dict(root["_component_configs"][Component(component).value])


def flow_sigmas(num_steps: int, shift: float) -> np.ndarray:
    """Return the ``num_steps + 1`` shifted rectified-flow sigmas.

    Matches ``MiniMaxH3Scheduler.set_timesteps(num_steps + 1)``: a float32
    ``linspace(1, 0)`` grid pushed through ``s * x / (1 + (s - 1) * x)``.
    """
    base = np.linspace(1.0, 0.0, num_steps + 1, dtype=np.float32)
    shift = np.float32(shift)
    return shift * base / (np.float32(1.0) + (shift - np.float32(1.0)) * base)


def component_weight_policy(args, policy):
    """Bind the large FP16 weights from the checkpoint when the engines load.

    Baking the 50 GB text encoder or the GEN transformer into an engine needs
    several copies of the weights at build time, more than Thor has. The other
    components keep their weights in the engine.
    """
    if Component(args.resolved_component) in (Component.TEXT_ENCODER,
                                              Component.GEN):
        return weight_policy.WeightPolicy(
            kinds=(weight_policy.EXTERNAL_WEIGHT_FP16,
                   weight_policy.EXTERNAL_WEIGHT_EMBEDDING))
    return policy


def validate_build(args, components) -> None:
    del components
    if args.spec_type != "none" or int(args.max_lora_rank) != 0:
        raise ValueError(
            "MiniMax-H3 does not support speculative decoding or LoRA")
    if int(args.max_batch_size) != 1:
        raise ValueError("MiniMax-H3 engines run one request; use "
                         "--max-batch-size 1")
    if args.minimax_h3_num_steps < 1 or args.minimax_h3_max_video_tokens < 1:
        raise ValueError("MiniMax-H3 step count and video capacity must be "
                         "positive")


def setup_profiles(builder, builder_config, network, args, bundle) -> bool:
    """Bound the dynamic text, audio, and video rows of every component."""
    del bundle
    video = (1, min(OPT_VIDEO_TOKENS, args.minimax_h3_max_video_tokens),
             args.minimax_h3_max_video_tokens)
    text = (1, OPT_TEXT_TOKENS, MAX_TEXT_TOKENS)
    audio = (2, OPT_AUDIO_TOKENS, MAX_AUDIO_TOKENS)
    rows = {
        "input_ids": text,
        "text_hidden": text,
        "audio_rows": audio,
        "video_rows": video,
        "position_ids": tuple(map(sum, zip(text, audio, video))),
        "audio_latents": tuple(count // 2 for count in audio),
    }
    dynamic = [
        tensor for tensor in map(network.get_input, range(network.num_inputs))
        if -1 in tuple(tensor.shape)
    ]
    if dynamic:
        profile = builder.create_optimization_profile()
        for tensor in dynamic:
            profile.set_shape(
                tensor.name,
                *(tuple(count if dim == -1 else dim for dim in tensor.shape)
                  for count in rows[tensor.name]))
        builder_config.add_optimization_profile(profile)
    return True


def _edge(name: str, source: str, on_missing: str = "fail") -> dict:
    return {"name": name, "from": source, "on_missing": on_missing}


def _stage(stage_id: str, runner: str, component, inputs: list,
           emits: list) -> dict:
    from ...core.artifacts.multistage import artifact_of

    return {
        "id": stage_id,
        "runner": f"minimax_h3.{runner}",
        "artifact": artifact_of(component),
        "inputs": inputs,
        "emits": emits,
    }


def multistage_topology(bundle, components, args) -> dict | None:
    """Prompt to stereo audio and video frames.

    The text encoder conditions the denoiser, whose final latents feed the
    audio and video decoders. A request that attaches only one output runs
    only the decoder behind it.
    """
    del bundle, args
    present = frozenset(components)
    if not {Component.TEXT_ENCODER, Component.GEN}.issubset(present):
        logger.info("No multistage_config.json: MiniMax-H3 needs the text "
                    "encoder and the denoiser")
        return None
    latent = {"kind": "embeds", "transport": "device_tensor"}
    stages = [
        _stage("text_encoder", "text_encoder", Component.TEXT_ENCODER,
               [_edge("prompt", "request:messages")], [{
                   "id": "text_hidden",
                   **latent
               }]),
        _stage("denoiser", "denoiser", Component.GEN, [
            _edge("text_hidden", "text_encoder:text_hidden"),
            _edge("style", "request:style", "skip"),
            _edge("sampling", "request:sampling", "skip"),
        ], [{
            "id": "audio_latents",
            **latent
        }, {
            "id": "video_latents",
            **latent
        }]),
    ]
    decoders = ((Component.AUDIO_DECODER, "audio_decoder", "audio_latents",
                 "audio", "audio"), (Component.VAE_DECODER, "video_decoder",
                                     "video_latents", "video", "image"))
    terminals = []
    for component, stage_id, latents, port, kind in decoders:
        if component not in present:
            continue
        stages.append(
            _stage(stage_id, stage_id, component,
                   [_edge("latents", f"denoiser:{latents}")], [{
                       "id": port,
                       "kind": kind,
                       "transport": "chunk_stream",
                       "terminal": True
                   }]))
        terminals.append(f"{stage_id}:{port}")
    if not terminals:
        logger.info("No multistage_config.json: MiniMax-H3 needs the audio "
                    "or the video decoder")
        return None
    return {"terminals": terminals, "stages": stages}
