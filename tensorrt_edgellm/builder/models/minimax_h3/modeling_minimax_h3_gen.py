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
"""Checkpoint-direct MiniMax-H3 generation transformer.

One forward is one rectified-flow velocity evaluation over the packed
``[text | audio | video]`` sequence of a text-to-video-and-audio request.
Text rows share the video timestep. The timestep-only AdaLN branch
(``time_embedder``, every ``adaln_proj``, and ``norm_out.linear``) depends
only on the step schedule, so it is evaluated once at build time and baked as
per-step modulation tables that the ``step`` input selects.
"""

from __future__ import annotations

import logging

import numpy as np
import tensorrt as trt

from ...ops import Linear, Module, NetworkModule, RMSNorm
from ...ops import functional as F
from . import configuration
from .layers import Attention, FeedForward

LOGGER = logging.getLogger("builder.minimax_h3")

#: AdaLN modality rows of the ``[text, audio, video]`` segments.
_SEGMENT_MODALITIES = (1, 2, 0)
#: ``[text, audio, video]`` segments use the video, audio, video timestep.
_SEGMENT_TIMESTEPS = (0, 1, 0)


def _silu(values: np.ndarray) -> np.ndarray:
    return values / (1.0 + np.exp(-values))


class _AdaLNTables:
    """Evaluate the AdaLN branch for every step of the schedule.

    ``blocks[layer]`` is ``[steps, segment, 6, hidden]`` in the Diffusers
    ``shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`` order;
    ``output`` is ``[steps, segment, 2, hidden]`` as ``shift, scale``. Scales
    are stored as ``1 + scale``.
    """

    def __init__(self, weights, config: dict, num_layers: int,
                 timesteps: np.ndarray) -> None:
        hidden = int(config["hidden_size"])
        half = int(config["freq_dim"]) // 2
        exponent = np.exp(-np.log(10000.0) *
                          np.arange(half, dtype=np.float32) / half)
        angles = timesteps.reshape(-1, 1) * exponent
        embedding = np.concatenate((np.cos(angles), np.sin(angles)), -1)
        embedding = _silu(embedding @ weights.f32("time_embedder.linear_1."
                                                  "weight").T +
                          weights.f32("time_embedder.linear_1.bias"))
        embedding = (
            embedding @ weights.f32("time_embedder.linear_2.weight").T +
            weights.f32("time_embedder.linear_2.bias"))
        # Rows of ``timesteps`` are (video, audio) per step.
        self._activated = _silu(embedding).reshape(-1, 2, embedding.shape[-1])
        self._hidden = hidden
        self.blocks = []
        for layer in range(num_layers):
            prefix = f"transformer_blocks.{layer}.adaln_proj.linear."
            values = self._project(weights, prefix, 3 * 6)
            # [steps, timestep, modality, 6, hidden] -> per segment.
            values = values.reshape(-1, 2, 3, 6, hidden)
            self.blocks.append(self._segments(values, scale_rows=(1, 4)))
        values = self._project(weights, "norm_out.linear.", 2)
        self.output = self._segments(values.reshape(-1, 2, 1, 2, hidden),
                                     scale_rows=(1, ),
                                     per_modality=False)

    def _project(self, weights, prefix: str, rows: int) -> np.ndarray:
        weight = weights.f32(prefix + "weight")
        if weight.shape[0] != rows * self._hidden:
            raise ValueError(f"{prefix}weight has unexpected shape "
                             f"{weight.shape}")
        return self._activated @ weight.T + weights.f32(prefix + "bias")

    @staticmethod
    def _segments(values: np.ndarray,
                  scale_rows,
                  per_modality: bool = True) -> np.ndarray:
        segments = [
            values[:, timestep, modality if per_modality else 0] for timestep,
            modality in zip(_SEGMENT_TIMESTEPS, _SEGMENT_MODALITIES)
        ]
        table = np.stack(segments, axis=1)
        table[:, :, scale_rows] += 1.0
        return np.ascontiguousarray(table, dtype=np.float16)


class _Segments:
    """Split and join the packed ``[text | audio | video]`` row segments."""

    def __init__(self, lengths, hidden_size: int) -> None:
        self.lengths = lengths
        self.hidden_size = hidden_size

    def split(self, hidden):
        text, audio, video = self.lengths
        return [
            F.dynamic_slice(hidden, (0, offset, 0),
                            (1, length, self.hidden_size))
            for offset, length in ((0, text), (text, audio), (text + audio,
                                                              video))
        ]

    @staticmethod
    def join(parts):
        return F.concatenate(parts, 1)


def _modulate(hidden, scale_plus_one, shift):
    return (hidden.cast(trt.float32) * scale_plus_one + shift).cast(
        trt.float16)


class MiniMaxH3TransformerBlock(Module):
    """Pre-norm block whose AdaLN rows are selected per segment."""

    def __init__(self, ctx, prefix: str, config: dict,
                 table: np.ndarray) -> None:
        super().__init__(ctx, prefix)
        eps = float(config["norm_eps"])
        self.norm1 = RMSNorm(ctx, self.key("norm1"), eps)
        self.attn = Attention(ctx, self.key("attn"),
                              int(config["num_attention_heads"]),
                              int(config["attention_head_dim"]),
                              float(config["qk_norm_eps"]))
        self.norm2 = RMSNorm(ctx, self.key("norm2"), eps)
        self.ff = FeedForward(ctx, self.key("ff"), int(config["ffn_dim"]))
        self.table = table

    def _modulation(self, step):
        """Return ``[segment][row]`` FP32 ``[1, 1, hidden]`` rows."""
        hidden = self.table.shape[-1]
        table = F.constant(self.table,
                           self.key("adaln_table")).gather(step,
                                                           0).cast(trt.float32)
        return [[
            table[:, segment, row:row + 1].reshape((1, 1, hidden))
            for row in range(6)
        ] for segment in range(len(_SEGMENT_MODALITIES))]

    def forward(self, hidden, segments: _Segments, step, rope, suffix_rows):
        mods = self._modulation(step)
        normed = segments.split(self.norm1(hidden))
        attended = self.attn(
            segments.join([
                _modulate(part, mod[1], mod[0])
                for part, mod in zip(normed, mods)
            ]), rope, suffix_rows)
        hidden = segments.join([
            part + mod[2] * update for part, mod, update in zip(
                segments.split(hidden), mods, segments.split(attended))
        ])
        normed = segments.split(self.norm2(hidden))
        fed = self.ff(
            segments.join([
                _modulate(part, mod[4], mod[3])
                for part, mod in zip(normed, mods)
            ]))
        return segments.join([
            part + mod[5] * update for part, mod, update in zip(
                segments.split(hidden), mods, segments.split(fed))
        ])


class MiniMaxH3TokenRefinerBlock(Module):
    """Unmodulated text block applied before the text rows are packed."""

    def __init__(self, ctx, prefix: str, config: dict) -> None:
        super().__init__(ctx, prefix)
        eps = float(config["norm_eps"])
        self.norm1 = RMSNorm(ctx, self.key("norm1"), eps)
        self.attn = Attention(ctx, self.key("attn"),
                              int(config["num_attention_heads"]),
                              int(config["attention_head_dim"]),
                              float(config["qk_norm_eps"]))
        self.norm2 = RMSNorm(ctx, self.key("norm2"), eps)
        self.ff = FeedForward(ctx, self.key("ff"), int(config["ffn_dim"]))

    def forward(self, hidden):
        hidden = hidden + self.attn(self.norm1(hidden))
        return hidden + self.ff(self.norm2(hidden))


class MiniMaxH3GenModel(NetworkModule):
    """One MiniMax-H3 joint video and stereo-audio velocity evaluation."""

    @classmethod
    def from_config(cls, ctx):
        return cls(ctx, ctx.bundle)

    def __init__(self, ctx, bundle) -> None:
        super().__init__(ctx)
        config = configuration.component_config(bundle.root,
                                                configuration.Component.GEN)
        self.config = config
        self.hidden_size = int(config["hidden_size"])
        self.rope_freq_dim = int(config["rope_freq_dim"])
        self.rope_theta = float(config["rope_theta"])
        self.video_patch_dim = int(config["in_channels"]) * int(
            np.prod(config["patch_size"]))
        self.audio_channels = int(config["audio_in_channels"])
        num_steps = ctx.args.minimax_h3_num_steps
        # Per step, the (video, audio) timesteps ``1 - sigma``.
        sigmas = [
            configuration.flow_sigmas(num_steps, shift)[:-1]
            for shift in bundle.root["_flow_shifts"]
        ]
        timesteps = np.float32(1.0) - np.stack(sigmas, axis=1)
        tables = _AdaLNTables(self.weights, config, int(config["num_layers"]),
                              timesteps)

        self.context_embedder = Linear(ctx,
                                       "context_embedder",
                                       tensor_parallel=False)
        self.token_refiner = [
            MiniMaxH3TokenRefinerBlock(
                ctx, f"token_refiner.refiner_blocks.{index}", config)
            for index in range(int(config["num_refiner_layers"]))
        ]
        self.refiner_norm = RMSNorm(ctx, "token_refiner.final_norm",
                                    float(config["final_norm_eps"]))
        self.proj_in = Linear(ctx, "proj_in", tensor_parallel=False)
        self.audio_proj_in = Linear(ctx,
                                    "audio_proj_in",
                                    tensor_parallel=False)
        self.blocks = [
            MiniMaxH3TransformerBlock(ctx, f"transformer_blocks.{index}",
                                      config, table)
            for index, table in enumerate(tables.blocks)
        ]
        self.norm_out = RMSNorm(ctx, "norm_out.norm",
                                float(config["final_norm_eps"]))
        self.output_table = tables.output
        self.proj_out = Linear(ctx, "proj_out", tensor_parallel=False)
        self.audio_proj_out = Linear(ctx,
                                     "audio_proj_out",
                                     tensor_parallel=False)

    def input_tensors(self):
        return {
            "text_hidden":
            self.add_input("text_hidden", trt.float16,
                           (1, -1, int(self.config["text_dim"]))),
            "audio_rows":
            self.add_input("audio_rows", trt.float32,
                           (1, -1, self.audio_channels)),
            "video_rows":
            self.add_input("video_rows", trt.float32,
                           (1, -1, self.video_patch_dim)),
            "position_ids":
            self.add_input("position_ids", trt.float32, (-1, 3)),
            "step":
            self.add_input("step", trt.int32, (1, )),
        }

    def _rope(self, position_ids):
        """Return ``[1, S, 1, 6 * rope_freq_dim]`` cos/sin of ``(t, h, w)``."""
        dim = 2 * self.rope_freq_dim
        inverse = 1.0 / self.rope_theta**(
            np.arange(0, dim, 2, dtype=np.float32) / dim)
        angles = position_ids.reshape((-1, 3, 1)) * F.constant(
            inverse.reshape(1, 1, -1).astype(np.float32), "rope_inv_freq")
        angles = angles.reshape((-1, 3 * self.rope_freq_dim))
        angles = F.concatenate((angles, angles), 1).reshape(
            (1, -1, 1, 6 * self.rope_freq_dim))
        return angles.cos(), angles.sin(), 6 * self.rope_freq_dim

    def forward(self, text_hidden, audio_rows, video_rows, position_ids, step):
        # Massive activations of the text hidden state overflow an FP16 output.
        text = self.context_embedder.forward_f32(text_hidden)
        for block in self.token_refiner:
            text = block(text)
        text = self.refiner_norm(text).cast(trt.float32)
        audio = self.audio_proj_in.forward_f32(audio_rows)
        video = self.proj_in.forward_f32(video_rows)
        segments = _Segments([
            F.shape_of(rows)[1:2]
            for rows in (text_hidden, audio_rows, video_rows)
        ], self.hidden_size)

        rope = self._rope(position_ids)
        # The video rows are the block-sparse suffix of the packed sequence.
        suffix_rows = video_rows.reshape((-1, self.video_patch_dim))
        if not F.sparse_attention_available():
            LOGGER.warning("SparseAttentionPlugin is unavailable; MiniMax-H3 "
                           "GEN falls back to dense attention")
            suffix_rows = None
        hidden = segments.join([text, audio, video])
        for block in self.blocks:
            hidden = block(hidden, segments, step, rope, suffix_rows)

        table = F.constant(self.output_table,
                           "norm_out_table").gather(step, 0).cast(trt.float32)

        def modulate_output(part, segment):
            shift, scale_plus_one = (table[:, segment, row:row + 1].reshape(
                (1, 1, self.hidden_size)) for row in range(2))
            return _modulate(part, scale_plus_one, shift)

        _, audio, video = segments.split(self.norm_out(hidden))
        audio, video = modulate_output(audio, 1), modulate_output(video, 2)
        return {
            "audio_pred": self.audio_proj_out.forward_f32(audio),
            "video_pred": self.proj_out.forward_f32(video),
        }
