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
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace trt_edgellm
{
namespace rt
{

class GuidedDecoder;

struct LogitBias;

enum class DecodingStrategyKind : int32_t
{
    kVanilla,
    kEAGLE,
    kMTP,
    kDFlash,
    kGemma4MTP,
    kDSpark,
    kBlockDiffusion,
};

//! Return whether this request should use the stable Hybrid-MTP endpoint path. Mirrors the reference
//! decodingStrategy.h::shouldUseHybridMtpEndpointReuse: true iff the selected strategy is MTP, the deployment has a
//! hybrid base (numLinearAttnLayers > 0), and the context cache will either look up or publish state for the request.
constexpr bool shouldUseHybridMtpEndpointReuse(DecodingStrategyKind selectedStrategy, bool hybridBase,
    bool contextCacheLookupEnabled, bool contextCachePublicationEnabled) noexcept
{
    return selectedStrategy == DecodingStrategyKind::kMTP && hybridBase
        && (contextCacheLookupEnabled || contextCachePublicationEnabled);
}

struct DecodingStrategyCapabilities
{
    bool ownsBaseVerificationCudaGraphs{false};
    bool supportsLosslessSampling{false};
    int32_t maxSamplingSupport{0}; //!< 0 when the decoder does not require a bounded sampling support.
    //! Preserve request sampling semantics instead of coercing a greedy-only strategy.
    bool fallbackToVanillaForNonGreedySampling{false};
    //! Capture vanilla graphs when request routing may select the default decoder.
    bool requiresDefaultDecoderCudaGraphs{false};
};

struct SamplingBuffers
{
    Tensor& workspace;
    Tensor& indices;
    Tensor& scores;
    Tensor& baseVocabMappingTable;
    Tensor& hostPackedTokenIds;
    Tensor& hostSelectedTokenIds;
    //! Sampled indices captured *before* mapReducedVocabToFullVocab, i.e. still in the
    //! engine's output vocabulary. Grammar matchers live in that space (see GuidedDecoder),
    //! so they must be advanced with these rather than the remapped full IDs.
    Tensor& hostOutputSpaceIds;
    Tensor& uniforms;
    Tensor& hostUniforms;
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
    //! output logits (EAGLE / MTP / DFlash / JetSpec); Gemma4 MTP's sequential chain reads logits directly.
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
    std::filesystem::path const& checkpointDir;
    std::filesystem::path const& draftCheckpointDir;

    BaseEngineResources base;
    PreprocessResources preprocess;
    tokenizer::Tokenizer& tokenizer;
    LogitBias& logitBias;
    GuidedDecoder& guidedDecoder;
    SamplingBuffers sampling;
    LogprobsBuffers logprobs;

    //! Optional sampled-token synchronization callback. Rank 0 sends; peer ranks receive.
    std::function<bool(void* buffer, int32_t count, cudaStream_t stream)> tokenBroadcast{};
    int32_t parallelRank{-1};
};

class DecodingStrategy
{
public:
    virtual ~DecodingStrategy() noexcept = default;

    virtual DecodingStrategyKind kind() const noexcept = 0;
    virtual char const* name() const noexcept = 0;
    virtual bool isSpeculative() const noexcept = 0;
    virtual DecodingStrategyCapabilities capabilities() const noexcept
    {
        return {};
    }

    virtual DecodingKvHeadroom requiredKvHeadroom() const
    {
        return {/*.baseExtraTokens=*/1, /*.draftExtraTokens=*/0};
    }

    virtual DecodingTokenStateContract tokenStateContract() const noexcept
    {
        return DecodingTokenStateContract::kCommittedPlusLookahead;
    }

    virtual bool decodeStep(DecodingInferenceContext& context) = 0;
    virtual bool captureCudaGraphs(cudaStream_t stream) = 0;

    //! Initialize decoder-private generation state after base prefill. Non-speculative strategies are no-ops.
    virtual bool initializeForGeneration(DecodingInferenceContext&)
    {
        return true;
    }

    //! Greatest per-slot logical prefix whose continuation state is materialized by every model in this strategy.
    //! Physical model-state tails may extend beyond this boundary. This reports decoding progress only;
    //! context-cache policy decides whether that prefix can be published.
    virtual std::vector<int32_t> const& commonMaterializedStateLengths() const noexcept
    {
        static std::vector<int32_t> const kEMPTY;
        return kEMPTY;
    }

    virtual int64_t getRequiredContextMemorySize() const noexcept = 0;
    virtual void setContextMemory(Tensor&) = 0;

    // TODO(follow-up): Expose a prefill entry point on DecodingStrategy and remove
    // these system-prompt-specific interfaces. The runtime should call a generic
    // prefill method; system prompt caching becomes an internal optimisation.
    virtual bool hasSystemPromptKVCache(SystemPromptCacheKey const&) const = 0;
    virtual void restoreSystemPromptKVCache(SystemPromptCacheKey const&, int32_t residentSlot, cudaStream_t) = 0;
    virtual bool runSystemPromptPrefill(DecodingInferenceContext&) = 0;
    virtual void saveSystemPromptKVCache(SystemPromptCacheKey const&, std::string const&,
        std::vector<tokenizer::Rank> const&, int32_t, cudaStream_t) = 0;

    virtual void resetForNewSequences(Tensor&, cudaStream_t) = 0;
    virtual void onBatchEvict(std::vector<int32_t> const&, int32_t, int32_t, Tensor&, cudaStream_t) = 0;
};

} // namespace rt
} // namespace trt_edgellm
