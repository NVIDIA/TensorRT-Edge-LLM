/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "runtime/pi05Policy.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace pi05
{

//! \brief The canonical tensors a pi0.5 request can be recorded and replayed as.
//!
//! Every reader takes a raw dump of exactly the tensor's bytes, which is what the openpi
//! reference dumps, so a numerical comparison can feed both sides byte-identical inputs.

//! \brief Report per-stage, engine-chain and policy end-to-end latency over \p rounds.
//! Harness output, not part of the policy API: the stage names are what the latency
//! scripts scrape.
void logLatency(std::vector<Pi05Timings> const& rounds, int32_t warmup);

//! \brief Language token ids from a comma-separated list. An empty \p spec yields a short
//! fixed prompt, for a smoke run with no inputs staged.
std::vector<int32_t> readTokenIds(std::string const& spec);

//! \brief Device FLOAT16 [views, 3, S, S] pixel values, uploaded from \p path.
//! An empty \p path synthesizes N(0,1) views from \p seed, which profiles the towers
//! without staging a reference tensor.
rt::Tensor readPixelValues(
    std::string const& path, std::vector<int64_t> const& shape, int32_t seed, cudaStream_t stream);

//! \brief Initial x_0 as [batch, chunkElems] floats. A dump holding one request's
//! trajectory is replicated so every batch entry denoises the same one.
std::vector<float> readNoise(std::string const& path, size_t chunkElems, int32_t batch);

//! \brief Read an observation from a JSON request: ``task``, a ``cameras`` object keyed by
//! contract name, and the state as either ``state`` or DROID's
//! ``joint_position`` / ``gripper_position`` pair.
//! \throws std::runtime_error On a missing file, malformed JSON, a missing field, or a
//!         request that spells the state both ways.
Pi05Observation readObservation(std::string const& path);

//! \brief True when \p path names the safetensors form writeActionChunk() writes.
//! A batched run has no other output form, so a caller can reject one before it runs.
bool isSafetensorsPath(std::string const& path);

//! \brief Write \p chunk to \p path: a ``.safetensors`` suffix writes the raw normalized
//! [batch, horizon, modelActionDim] tensor, any other writes the JSON response, and an
//! empty path writes the JSON to stdout.
//! \throws std::runtime_error On a JSON path when \p chunk holds more than one request;
//!         the response records a single [horizon, dim] chunk.
void writeActionChunk(std::string const& path, Pi05ActionChunk const& chunk, int32_t robotDim, cudaStream_t stream);

} // namespace pi05
} // namespace trt_edgellm
