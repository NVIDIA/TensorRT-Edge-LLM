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
"""Convert NVIDIA PhysicalAI-AV clips into Alpamayo-R1 action_inference requests.

Each request has 16 full-resolution images (4 cameras x 4 timesteps,
camera-major) and the 16-point egomotion history in the ego frame at t0. The
runtime resizes the images. Only the requested clips are streamed from Hugging
Face.

Dataset: https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
"""

import json
import os

import numpy as np
from PIL import Image

# Camera order expected by Alpamayo-R1.
CAMERAS = ("camera_cross_left_120fov", "camera_front_wide_120fov",
           "camera_cross_right_120fov", "camera_front_tele_30fov")
NUM_FRAMES = 4
NUM_HISTORY = 16
STEP_US = 100_000  # 10 Hz
T0_US = 5_100_000
SYSTEM_PROMPT = (
    "You are a driving assistant that generates safe and accurate "
    "actions.")
USER_PROMPT = ("output the chain-of-thought reasoning of the driving process, "
               "then output the future trajectory.")


def _build_request(dataset, clip_id: str, image_dir: str) -> dict:
    egomotion = dataset.get_clip_feature(clip_id,
                                         "egomotion",
                                         maybe_stream=True)
    history_us = T0_US + STEP_US * np.arange(-(NUM_HISTORY - 1), 1)
    pose = egomotion(history_us).pose
    # Ego frame at t0: x forward, last point is the origin.
    history = pose.rotation[-1].inv().apply(pose.translation -
                                            pose.translation[-1])

    frame_us = T0_US + STEP_US * np.arange(-(NUM_FRAMES - 1), 1)
    os.makedirs(image_dir, exist_ok=True)
    content = []
    for camera in CAMERAS:
        reader = dataset.get_clip_feature(clip_id, camera, maybe_stream=True)
        frames, _ = reader.decode_images_from_timestamps(frame_us)
        reader.close()
        for index, frame in enumerate(frames):
            name = f"{camera.removeprefix('camera_')}_t{index}.png"
            path = os.path.abspath(os.path.join(image_dir, name))
            Image.fromarray(frame).save(path)
            content.append({"type": "image", "image": path})
    content.append({"type": "trajectory", "trajectory": history.tolist()})
    content.append({"type": "text", "text": USER_PROMPT})
    return {
        "messages": [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": SYSTEM_PROMPT
            }]
        }, {
            "role": "user",
            "content": content
        }]
    }


def convert_physical_ai_av_dataset(config, dataset_name_or_dir: str,
                                   output_dir: str) -> None:
    """Write ``physical_ai_av_dataset.json`` and PNG frames to ``output_dir``.

    ``config.max_samples`` sets the number of clips (default 1).
    """
    # Optional dependency, see requirements-physical-ai-av.txt.
    import physical_ai_av

    if os.path.isdir(dataset_name_or_dir):
        dataset = physical_ai_av.PhysicalAIAVDatasetInterface(
            local_dir=dataset_name_or_dir)
    else:
        dataset = physical_ai_av.PhysicalAIAVDatasetInterface(
            repo_id=dataset_name_or_dir)

    requests = []
    for clip_id in dataset.clip_index.index[:config.max_samples or 1]:
        image_dir = os.path.join(output_dir, "images", clip_id)
        requests.append(_build_request(dataset, clip_id, image_dir))

    output_file = os.path.join(output_dir, "physical_ai_av_dataset.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "batch_size": config.batch_size,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "max_generate_length": config.max_generate_length,
                "requests": requests,
            },
            f,
            indent=2)
    print(f"Wrote {len(requests)} request(s) to {output_file}")
