/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "runtime/config/deploymentConfig.h"
#include "runtime/exec/engineExecutor.h"
#include "runtime/exec/tensorMap.h"
#include "runtime/features/deepstackBinding.h"
#include "runtime/hybridCacheManager.h"
#include "runtime/llmRuntimeUtils.h"
#include "runtime/preprocess/embeddingPreprocessor.h"
#include "runtime/preprocess/gemma4EmbeddingPreprocessor.h"
#include "runtime/preprocess/stepPreparer.h"
#include "runtime/state/decodingInferenceContext.h"
#include "runtime/state/pipelineIO.h"
#include "runtime/state/sharedResources.h"
#include "runtime/state/systemPromptKVCache.h"
#include "sampler/sampling.h"
#include "tokenizer/tokenizer.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

struct LogitBias;

enum class DecodingStrategyKind : int32_t
{
    kVanilla,
    kEAGLE,
    kMTP,
    kDFlash,
    kGemma4MTP,
};

//! Return whether this request should use the stable Hybrid-MTP endpoint path.
constexpr bool shouldUseHybridMtpEndpointReuse(DecodingStrategyKind selectedStrategy, bool hybridBase,
    bool contextCacheLookupEnabled, bool contextCachePublicationEnabled) noexcept
{
    return selectedStrategy == DecodingStrategyKind::kMTP && hybridBase
        && (contextCacheLookupEnabled || contextCachePublicationEnabled);
}

struct SamplingBuffers
{
    Tensor& workspace;
    Tensor& indices;
    Tensor& scores;
    Tensor& baseVocabMappingTable;
    Tensor& hostPackedTokenIds;
    Tensor& hostSelectedTokenIds;
};

/*!
 * @brief GPU/CPU buffers for logprobs computation (owned by LLMInferenceRuntime).
 * Referenced here so decoders can call log-softmax + top-K on the output logits.
 */
struct LogprobsBuffers
{
    Tensor& deviceLogprobsValues;  //!< GPU [logprobsMaxBatch, kMaxLogprobsK] top-K log-prob values
    Tensor& deviceLogprobsIndices; //!< GPU [logprobsMaxBatch, kMaxLogprobsK] top-K token indices
    Tensor& hostLogprobsValues;    //!< CPU pinned [logprobsMaxBatch, kMaxLogprobsK]
    Tensor& hostLogprobsIndices;   //!< CPU pinned [logprobsMaxBatch, kMaxLogprobsK]
    //! GPU [maxBatch * maxAcceptDepth, vocab] accepted verify rows gathered before extraction.
    //! Used by the spec-decode verify paths whose accepted rows are non-contiguous in the
    //! output logits (EAGLE / MTP / DFlash); Gemma4 MTP's sequential chain reads logits directly.
    Tensor& gatheredLogits;
};

//! Base-engine execution infrastructure: executor, tensor map, KV cache,
//! pipeline I/O, shared resources, and CUDA-graph capture callback.
struct BaseEngineResources
{
    EngineExecutor& executor;
    TensorMap& tensorMap;
    SharedResources& sharedResources;
    HybridCacheManager& cacheManager;
    PipelineIO& pipelineIO;
    std::function<bool(InferenceDims const&, cudaStream_t)> captureGraph;
    //! Execute the base engine and publish logits through PipelineIO's common
    //! FP32 contract. The callback is a no-op for FLOAT-output engines and
    //! upcasts a dedicated engine-output buffer for supported HALF engines.
    std::function<void(cudaStream_t)> publishLogits;

    bool execute(cudaStream_t stream)
    {
        bool const success = executor.execute(stream);
        if (success)
        {
            publishLogits(stream);
        }
        return success;
    }
};

//! Preprocessing resources: embedding lookup, step preparation, deepstack.
struct PreprocessResources
{
    StepPreparer& stepPreparer;
    EmbeddingPreprocessor& embeddingPreprocessor;
    EmbeddingData& embedding;
    Tensor& idsInput;
    DeepstackBinding* deepstack;
    Gemma4EmbeddingPreprocessor* gemma4Ple;
};

struct DecodingRuntimeContext
{
    DeploymentConfig& deployment;
    int32_t maxRuntimeBatchSize;

    BaseEngineResources base;
    PreprocessResources preprocess;
    tokenizer::Tokenizer& tokenizer;
    LogitBias& logitBias;
    SamplingBuffers sampling;
    LogprobsBuffers logprobs;
};

class DecodingStrategy
{
public:
    virtual ~DecodingStrategy() noexcept = default;

    virtual DecodingStrategyKind kind() const noexcept = 0;
    virtual char const* name() const noexcept = 0;
    virtual bool isSpeculative() const noexcept = 0;

    virtual bool decodeStep(DecodingInferenceContext& context) = 0;
    //! Prepare strategy-owned state needed before round zero. The runtime may call this early when a coherent
    //! prefill checkpoint must be published before verification mutates base state.
    virtual bool prepareFirstDecodeStep(DecodingInferenceContext&)
    {
        return true;
    }
    virtual bool captureCudaGraphs(cudaStream_t stream) = 0;

    virtual int64_t getRequiredContextMemorySize() const noexcept = 0;
    virtual void setContextMemory(Tensor&) = 0;

    // TODO(follow-up): Expose a prefill entry point on DecodingStrategy and remove
    // these system-prompt-specific interfaces. The runtime should call a generic
    // prefill method; system prompt caching becomes an internal optimisation.
    virtual bool hasSystemPromptKVCache(SystemPromptCacheKey const&) const = 0;
    virtual void restoreSystemPromptKVCache(SystemPromptCacheKey const&, int32_t, cudaStream_t) = 0;
    virtual bool runSystemPromptPrefill(DecodingInferenceContext&) = 0;
    virtual void saveSystemPromptKVCache(SystemPromptCacheKey const&, std::string const&,
        std::vector<tokenizer::Rank> const&, int32_t, cudaStream_t) = 0;

    virtual void resetForNewSequences(Tensor&, cudaStream_t) = 0;
    virtual void onBatchEvict(std::vector<int32_t> const&, int32_t, int32_t, Tensor&, cudaStream_t) = 0;
};

} // namespace rt
} // namespace trt_edgellm
