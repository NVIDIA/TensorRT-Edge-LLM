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

#include "runtime/state/decodingInferenceContext.h"

#include "runtime/debug/layerDebugger.h"

namespace trt_edgellm
{
namespace rt
{

// Out-of-line so the unique_ptr<LayerDebugger> member can hold an incomplete type
// in the header; LayerDebugger is complete here. Move ops are defined too: a
// user-declared destructor suppresses the implicit move, but the context is
// returned by value (e.g. tests' makeContext), so it must stay movable. It
// remains non-copyable through the unique_ptr member.
DecodingInferenceContext::DecodingInferenceContext() = default;
DecodingInferenceContext::DecodingInferenceContext(DecodingInferenceContext&&) noexcept = default;
DecodingInferenceContext& DecodingInferenceContext::operator=(DecodingInferenceContext&&) noexcept = default;
DecodingInferenceContext::~DecodingInferenceContext() = default;

void DecodingInferenceContext::initialize(int32_t batchSize, int32_t maxGenLength,
    rt::OptionalInputTensor const& visual, rt::OptionalInputTensors const& deepstack, std::string const& loraName,
    cudaStream_t cudaStream)
{
    systemPrompts.resize(batchSize);
    rawBatchedInputIds.reserve(batchSize);
    tokenIds.resize(batchSize);
    currentGenerateLengths.resize(batchSize, 0);
    effectivePrefillLengths.resize(batchSize, 0);
    finishedStates.resize(batchSize, 0);
    sequenceCacheStates.clear();
    sequenceCacheStates.resize(batchSize);
    contextReuseEnabled = false;
    contextCacheLookupEnabled = false;
    contextCachePublicationEnabled = false;
    contextCacheIsolationKey.clear();
    contextCacheCommitPolicy = CommitPolicy::kIncludingGeneratedTokens;
    recurrentCaptureInterval = 0;
    contextCacheReplayTailLength = 0;
    speculativeDraftPrefillComplete = false;
    mediaIdentityComplete = true;
    hasVisionInput = false;
    hasAudioInput = false;
    mediaArtifactReuseEnabled = false;
    mediaSpans.clear();
    mediaSpans.resize(batchSize);
    visionEmbeddingOffsets.assign(batchSize, 0);
    audioEmbeddingOffsets.assign(batchSize, 0);
    mediaArtifactLeases.clear();
    slotStreams.clear();
    slotStreams.resize(batchSize);
    stopStringsPerSlot.clear();
    stopStringsPerSlot.resize(batchSize);
    logitBiasPerSlot.clear();
    logitBiasPerSlot.resize(batchSize);
    hasLogitBias = false;
    logitBiasGpuDirty = false;
    shouldStopAfterAcceptedToken = {};

    batchIndexMapping.resize(batchSize);
    for (int32_t i = 0; i < batchSize; ++i)
    {
        batchIndexMapping[i] = i;
    }

    completedBatches.clear();

    // Initialize per-batch logprobs accumulator (populated only when numLogprobs > 0)
    stepLogprobs.clear();
    stepLogprobs.resize(batchSize);

    visualEmbeddings = visual;
    deepstackFeatures = deepstack;
    generationRound = 0;
    maxGenerateLength = maxGenLength;
    activeBatchSize = batchSize;
    loraWeightsName = loraName;
    stream = cudaStream;
}

} // namespace rt
} // namespace trt_edgellm
