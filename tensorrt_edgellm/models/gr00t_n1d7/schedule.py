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
"""GR00T N1.7 timestep-only constants. The schedule is fixed by the checkpoint, so every
timestep-only term is evaluated here and bound per step at run time. ``read`` maps a
checkpoint tensor name (``action_head.`` stripped) onto a float32 array.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .geometry import Geometry

Reader = Callable[[str], np.ndarray]

MODULATION_FILENAME = "adanorm_modulation.safetensors"
TIME_BIAS_FILENAME = "action_time_bias.safetensors"


def timestep_buckets(geometry: Geometry) -> np.ndarray:
    """The reference's discretized schedule: ``int(t / N * buckets)``."""
    steps = geometry.num_denoise_steps
    return np.array([
        int((step / float(steps)) * geometry.num_timestep_buckets)
        for step in range(steps)
    ],
                    dtype=np.float32)


def _sinusoidal_timestep_embedding(
        timesteps: np.ndarray,
        dim: int,
        max_period: float = 10000.0,
        flip_sin_to_cos: bool = True,
        downscale_freq_shift: float = 1.0) -> np.ndarray:
    """``diffusers.get_timestep_embedding`` in numpy, in float64 so the fp16
    tables round identically on every host's libm."""
    half = dim // 2
    exponent = -math.log(max_period) * np.arange(half, dtype=np.float64)
    exponent = exponent / (half - downscale_freq_shift)
    frequency = np.exp(exponent)
    emb = timesteps.astype(np.float64)[:, None] * frequency[None, :]
    emb = np.concatenate([np.sin(emb), np.cos(emb)], axis=-1)
    if flip_sin_to_cos:
        emb = np.concatenate([emb[:, half:], emb[:, :half]], axis=-1)
    return emb


def _sinusoidal_positional_encoding(timesteps: np.ndarray,
                                    dim: int) -> np.ndarray:
    """``SinusoidalPositionalEncoding`` from the action encoder, in numpy."""
    half = dim // 2
    exponent = -np.arange(half, dtype=np.float64) * (math.log(10000.0) / half)
    frequency = timesteps.astype(np.float64)[:, None] * np.exp(exponent)[None]
    return np.concatenate([np.sin(frequency), np.cos(frequency)], axis=-1)


def _silu(values: np.ndarray) -> np.ndarray:
    return values / (1.0 + np.exp(-values))


def _dense(read: Reader, prefix: str, values: np.ndarray) -> np.ndarray:
    return (values.astype(np.float64) @ read(f"{prefix}.weight").astype(
        np.float64).T + read(f"{prefix}.bias").astype(np.float64))


def modulation_table(read: Reader, geometry: Geometry,
                     buckets: np.ndarray) -> np.ndarray:
    """Every AdaLayerNorm site for every denoise step: row ``i`` is block ``i``'s
    ``norm1.linear``, the last row ``proj_out_1``; the graph owns the split."""
    projected = _sinusoidal_timestep_embedding(buckets, 256)
    temb = _dense(read, "model.timestep_encoder.timestep_embedder.linear_1",
                  projected)
    temb = _dense(read, "model.timestep_encoder.timestep_embedder.linear_2",
                  _silu(temb))
    conditioning = _silu(temb)
    sites = [
        f"model.transformer_blocks.{index}.norm1.linear"
        for index in range(geometry.num_layers)
    ]
    sites.append("model.proj_out_1")
    table = np.stack([_dense(read, site, conditioning) for site in sites],
                     axis=1)
    expected = (geometry.num_denoise_steps, geometry.num_modulation_sites,
                geometry.modulation_dim)
    if table.shape != expected:
        raise ValueError(f"GR00T modulation table has shape {table.shape}, "
                         f"expected {expected}")
    return table


def action_time_bias(read: Reader, geometry: Geometry,
                     buckets: np.ndarray) -> np.ndarray:
    """Fold the action encoder's timestep half of ``W2`` into a constant."""
    width = geometry.inner_dim
    stacked = read("action_encoder.W2.W")[geometry.embodiment_id]
    bias = read("action_encoder.W2.b")[geometry.embodiment_id]
    tau = _sinusoidal_positional_encoding(buckets, width).astype(np.float64)
    return tau @ stacked[width:].astype(np.float64) + bias.astype(np.float64)


def action_constants(read: Reader, geometry: Geometry) -> dict:
    """Both fp16 tables, keyed by the tensor name the runtime loads."""
    buckets = timestep_buckets(geometry)
    return {
        "adanorm_modulation":
        modulation_table(read, geometry, buckets).astype(np.float16),
        "action_time_bias":
        action_time_bias(read, geometry, buckets).astype(np.float16),
    }
