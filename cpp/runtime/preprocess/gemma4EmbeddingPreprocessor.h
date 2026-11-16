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

#include "common/tensor.h"
#include "runtime/config/llmEngineConfig.h"
#include "runtime/config/pleRuntimeConfig.h"
#include "runtime/exec/tensorMap.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <memory>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class PleIoBudget;

//! Runtime preprocessor for Gemma4 E-model per-layer embeddings (PLE).
//!
//! Owns a resident/cached pool and a token-to-slot map shared by all PLE layers.
//! The supplied stream outlives the preprocessor and is used by all engine consumers.
class Gemma4EmbeddingPreprocessor
{
public:
    Gemma4EmbeddingPreprocessor(LLMEngineConfig const& config, int32_t maxBatchSize, int32_t maxSeqLen,
        TensorMap& tensorMap, cudaStream_t stream, PleRuntimeConfig const& pleConfig = {},
        std::filesystem::path const& checkpointDir = {}, std::shared_ptr<PleIoBudget> pleIoBudget = nullptr);
    ~Gemma4EmbeddingPreprocessor() noexcept;

    //! Gather PLE tensors for the current token-id tensor shape.
    //! Device query lengths mask entry padding and must remain valid through queued work.
    //! Cached residency requires matching host lengths and a CPU INT32 token snapshot in physical row order.
    //! The host snapshot is read synchronously; resident mode can use GPU-only inputs.
    void embed(Tensor const& tokenIds, cudaStream_t stream, std::vector<int32_t> const& queryLengths = {},
        Tensor const* deviceQueryLengths = nullptr, Tensor const* hostTokenIds = nullptr);

    //! Gather PLE tensors for a packed token-major step whose every token ID is a real token (no row padding).
    //! The token count may exceed one request's maximum sequence length, up to the allocated map capacity.
    //! Cached residency requires a CPU INT32 snapshot of the same token IDs.
    void embedPacked(Tensor const& tokenIds, cudaStream_t stream, Tensor const* hostTokenIds);

    //! Reshape already-bound token-major outputs for an execution or CUDA-graph capture shape.
    void reshapeOutputsTokenMajor(int64_t physicalTokens);

    //! Whether embed needs host token IDs, which only cached residency does.
    bool needsHostTokens() const noexcept;

    //! The most distinct table rows one gather can map: every table row under resident storage, the
    //! cache slots less the reserved zero slot under cached storage.
    int64_t distinctRowCapacity() const noexcept
    {
        return mDistinctRowCapacity;
    }

private:
    struct PooledResources;
    bool drainPooled() noexcept;
    void releasePooledLeases();
    void embedPooled(Tensor const& tokenIds, Coords const& shape, std::vector<int32_t> const& queryLengths,
        Tensor const* deviceQueryLengths, Tensor const* hostTokenIds);

    LLMEngineConfig mConfig{};
    std::unique_ptr<PooledResources> mPooled;
    cudaStream_t mStream{};
    int32_t mMaxBatchSize{};
    int32_t mMaxSeqLen{};
    int64_t mDistinctRowCapacity{};
    bool mPoisoned{false};
};

} // namespace rt
} // namespace trt_edgellm
