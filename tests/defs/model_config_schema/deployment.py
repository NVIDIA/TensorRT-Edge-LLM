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
"""Export and engine settings supplied by model-test YAML."""

from typing import Annotated, Literal, Optional

import pydantic

from .common import ConfigModel, NonnegativeInt, PositiveInt, reject_null


class ExportConfig(ConfigModel):
    target_sm: Annotated[Optional[PositiveInt],
                         pydantic.BeforeValidator(reject_null)] = None
    int4_gemm_plugin_version: Annotated[
        Optional[int],
        pydantic.Field(ge=1, le=2),
        pydantic.BeforeValidator(reject_null)] = None
    externalize_weights: Optional[str] = None
    lora: Optional[bool] = None
    merge_lora: Optional[bool] = None
    nvfp4_moe_target: Optional[str] = None
    reduced_vocab_size: Optional[int] = None
    trt_native_attention: Optional[bool] = None
    vocab_reduction_max_samples: Optional[int] = None
    vocab_reduction_method: Optional[str] = None


class EngineConfig(ConfigModel):
    max_batch_size: Optional[PositiveInt] = None
    max_input_len: Optional[PositiveInt] = None
    max_kv_cache_capacity: Optional[PositiveInt] = None
    tp_size: Optional[PositiveInt] = None
    min_image_tokens: Optional[PositiveInt] = None
    max_image_tokens: Optional[PositiveInt] = None
    max_image_tokens_per_image: Optional[PositiveInt] = None
    min_time_steps: Optional[PositiveInt] = None
    max_time_steps: Optional[PositiveInt] = None
    max_spec_verify_tokens: Optional[PositiveInt] = None
    spec_verify_tokens: Optional[PositiveInt] = None
    max_spec_draft_tokens: Optional[PositiveInt] = None
    eagle_draft_top_k: Optional[PositiveInt] = None
    eagle_draft_step: Optional[PositiveInt] = None
    max_lora_rank: Optional[NonnegativeInt] = None
    max_kv_pool_pages: Optional[PositiveInt] = None
    context_cache_max_snapshots: Optional[PositiveInt] = None
    context_reuse: Optional[bool] = None
    in_flight_batching: bool = False
    md_launch_mode: Optional[Literal["thread", "mpi"]] = None
    allreduce_backend: Optional[Literal["shm", "nccl"]] = None

    @pydantic.model_validator(mode="after")
    def compatible_settings(self):
        for minimum, maximum in (("min_image_tokens", "max_image_tokens"),
                                 ("min_time_steps", "max_time_steps")):
            low, high = getattr(self, minimum), getattr(self, maximum)
            if low is not None and high is not None and low > high:
                raise ValueError(f"{minimum} must not exceed {maximum}")
        if (self.context_cache_max_snapshots is not None
                and not self.context_reuse):
            raise ValueError(
                "context_cache_max_snapshots requires context_reuse: true")
        if self.spec_verify_tokens is not None and (
                self.max_spec_verify_tokens is None
                or self.spec_verify_tokens > self.max_spec_verify_tokens):
            raise ValueError(
                "spec_verify_tokens must be in [1, max_spec_verify_tokens]")
        return self


class Requirements(ConfigModel):
    min_sm: PositiveInt
    min_gpu_memory_gb: PositiveInt
