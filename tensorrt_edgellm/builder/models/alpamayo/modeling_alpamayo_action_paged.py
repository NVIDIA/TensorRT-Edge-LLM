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
"""TP=1 action graph consuming an immutable prefix and private action pages."""

import tensorrt as trt

from ...core import ragged as ragged_shapes
from ...ops import functional as F
from ...ops.ragged import RaggedDecoderInputs
from .configuration import prefix_kv_scales
from .modeling_alpamayo_action import (AlpamayoActionDecoderLayer,
                                       AlpamayoActionModel)

# Matches rt::ExecutionPhase::kDiffusionDenoise.
_DENOISE_PHASE = 6


class PagedAlpamayoActionDecoderLayer(AlpamayoActionDecoderLayer):
    """Non-causal attention over separate prefix and action pools."""

    def __init__(self, *args):
        super().__init__(*args)
        self.kv_scales = None

    def attention(self, query, key, value, prefix, action, rope, ragged,
                  prefix_page_table, context_mask_selector):
        query_width = self.num_heads * self.head_dim
        kv_width = self.num_kv_heads * self.head_dim
        qkv = F.concatenate((query.reshape(
            (-1, query_width)), key.reshape(
                (-1, kv_width)), value.reshape((-1, kv_width))), 1)
        attended, present = F.attention(
            qkv,
            action,
            rope,
            ragged,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            attention_scale=self.head_dim**-0.5,
            context_mask_selector=context_mask_selector,
            prefix_key_value=prefix,
            prefix_page_table=prefix_page_table,
            fp8_kv_prefix=self.kv_scales is not None,
            qkv_scales=(1.0,
                        *self.kv_scales) if self.kv_scales is not None else
            (1.0, 1.0, 1.0))
        return attended.reshape(
            (-1, int(query.shape[1]), query_width)), present


class PagedAlpamayoActionModel(AlpamayoActionModel):
    """One denoising step using the runner's two-pool paged KV contract."""

    layer_class = PagedAlpamayoActionDecoderLayer

    def __init__(self, ctx, bundle):
        super().__init__(ctx, bundle)
        self.kv_scales = prefix_kv_scales(bundle, self.num_layers)
        if self.kv_scales is not None:
            for layer, scales in zip(self.layers, self.kv_scales):
                layer.kv_scales = scales

    def input_tensors(self):
        prefix_pages, _ = ragged_shapes.checked_kv_pool_pages(
            self.ctx.args.max_batch_size, self.ctx.args.max_kv_cache_capacity)
        action_pages, _ = ragged_shapes.checked_kv_pool_pages(
            self.ctx.args.max_batch_size, self.diffusion_tokens)
        pool_shape = (2, -1, F.KV_PAGE_SIZE, self.num_kv_heads, self.head_dim)
        prefix_dtype = trt.float16 if self.kv_scales is None else trt.DataType.FP8
        inputs = {
            "noise":
            self.add_input("noise_trajectory", trt.float32,
                           (-1, self.diffusion_tokens, 2)),
            "time0":
            self.add_input("time_steps_t0", trt.float32, (1, )),
            "time1":
            self.add_input("time_steps_t1", trt.float32, (1, )),
            "rope":
            self.add_input("rope_rotary_cos_sin", trt.float32,
                           (-1, self.diffusion_tokens, self.head_dim)),
            "kv_page_table":
            self.add_input("kv_page_table", trt.int32, (-1, 2, prefix_pages)),
            "action_page_table":
            self.add_input("action_page_table", trt.int32,
                           (-1, 2, action_pages)),
            "execution_phase_marker":
            self.add_input("execution_phase_marker", trt.int32,
                           (_DENOISE_PHASE, )),
            "context_sequence_count_carrier":
            self.add_input("context_sequence_count_carrier", trt.int32, (0, )),
            "prefix_pools": [
                self.add_input(f"past_key_values_{index}", prefix_dtype,
                               pool_shape) for index in range(self.num_layers)
            ],
            "action_pools": [
                self.add_input(f"action_key_values_{index}", trt.float16,
                               pool_shape) for index in range(self.num_layers)
            ],
        }
        for name in ("query_lengths", "past_lengths",
                     "attention_sequence_lengths", "query_start_offsets",
                     "context_mask_selector"):
            inputs[name] = self.add_input(name, trt.int32, (-1, ))
        return inputs

    def forward(self, noise, time0, time1, rope, kv_page_table,
                action_page_table, execution_phase_marker,
                context_sequence_count_carrier, prefix_pools, action_pools,
                query_lengths, past_lengths, attention_sequence_lengths,
                query_start_offsets, context_mask_selector):
        ragged = RaggedDecoderInputs(
            positions=None,
            query_start_offsets=query_start_offsets,
            query_lengths=query_lengths,
            past_lengths=past_lengths,
            attention_sequence_lengths=attention_sequence_lengths,
            state_indices=None,
            logits_indices=None,
            execution_phase_marker=execution_phase_marker,
            context_sequence_count_carrier=context_sequence_count_carrier,
            kv_page_table=action_page_table)
        hidden = self.input_projection(
            noise.cast(trt.float16),
            time0.reshape((-1, 1)).cast(trt.float32))
        rope = rope.reshape((-1, self.head_dim))
        outputs = {}
        for index, layer in enumerate(self.layers):
            hidden, present = layer(hidden, prefix_pools[index],
                                    action_pools[index], rope, ragged,
                                    kv_page_table, context_mask_selector)
            outputs[f"present_key_values_{index}"] = present
        prediction = self.action_out_proj(self.norm(hidden)).cast(trt.float32)
        delta = (time1 - time0).reshape((0, 1, 1)).cast(trt.float32)
        outputs["denoised_trajectory"] = noise + delta * prediction
        return outputs
