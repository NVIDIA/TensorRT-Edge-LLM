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
#include "runtime/exec/tensorMap.h"

#include <cstdint>
#include <cuda_runtime.h>
#include <filesystem>
#include <optional>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

//! Runtime preprocessor for Gemma4 E-model per-layer embeddings (PLE).
//!
//! Loads the token-identity PLE table from ple_embedding.safetensors, gathers
//! one [physical_tokens, ple_hidden_size] tensor per decoder layer from token
//! IDs, and binds those tensors as ple_token_embeds_{layer_idx} engine inputs.
class Gemma4EmbeddingPreprocessor
{
public:
    Gemma4EmbeddingPreprocessor(std::filesystem::path const& engineDir, LLMEngineConfig const& config,
        int32_t maxBatchSize, int32_t maxSeqLen, TensorMap& tensorMap, cudaStream_t stream,
        std::optional<Tensor> checkpointTable = std::nullopt);

    //! Gather PLE tensors for the current token-id tensor shape.
    void embed(Tensor const& tokenIds, cudaStream_t stream);

    //! Reshape already-bound token-major outputs for an execution or CUDA-graph capture shape.
    void reshapeOutputsTokenMajor(int64_t physicalTokens);

private:
    LLMEngineConfig mConfig{};
    Tensor mPleTable{};
    Tensor mPleOutputBuffer{}; //!< Unified owned backing buffer for all PLE layer outputs.
    //! Non-owned tensor views into mPleOutputBuffer. TensorMap stores pointers to these stable objects.
    std::vector<Tensor> mPleOutputViews{};

    Tensor makeTokenMajorOutputViewForLayer(int32_t layerIdx, int64_t physicalTokens);
};

} // namespace rt
} // namespace trt_edgellm
