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

#include "common/executionPhase.h"
#include "common/tensor.h"

#include <optional>

namespace trt_edgellm::rt
{
struct SpecDecodeConfig;
struct DecodingRuntimeContext;
struct DecodingInferenceContext;

//! Borrowed root-inclusive candidates and metadata for one target verification.
struct SpecVerifyInputs
{
    Tensor const& tokenIds;
    Tensor const* executionValidTokenCounts;
    Tensor const* parentIds;
    Tensor const* depths;
    int32_t requestCount;
    int32_t tokensPerRequest;
    int32_t maxPathLength;
    SpecVerifyLayout layout;
    bool metadataReady{false};
    //! INT8 flags at the verify stride, 1 on a grafted forced token; null when the round grafted none.
    Tensor const* forcedFlags{nullptr};
};

//! Borrowed accepted tokens; row indices address the verification stride.
struct SpecAcceptResult
{
    Tensor const& tokenIds;
    Tensor const& emittedTokenCounts;
    Tensor const* acceptedRowIndices;
    int32_t outputStride;
};

struct GreedyAcceptInputs
{
    SpecVerifyInputs verify;
    Tensor const& targetTokenIds;
    Tensor const* acceptValidTokenCounts;
    //! Per-candidate INT8 flags at the verify stride; a flagged child is accepted in place of the prediction.
    Tensor const* forcedFlags{nullptr};
};

struct DenseProbabilityAcceptInputs
{
    Tensor const& targetProbabilities;
    Tensor const& draftProbabilities;
    Tensor const& proposalTokenIds;
    Tensor const& proposalLengths;
    Tensor const& uniforms;
    int32_t batchSize;
    int32_t draftStride;
    int32_t verifyProposalLen;
    int32_t vocabSize;
};
struct SparseProbabilityAcceptInputs
{
    Tensor const& targetProbabilities;
    Tensor const& targetSupportIds;
    Tensor const& draftProbabilities;
    Tensor const& draftSupportIds;
    Tensor const& proposalTokenIds;
    Tensor const& proposalLengths;
    Tensor const& uniforms;
    Tensor const* maxAcceptLengths;
};
struct DenseTargetSparseDraftAcceptInputs
{
    Tensor const& targetProbabilities;
    Tensor const& draftProbabilities;
    Tensor const& draftSupportIds;
    Tensor const& proposalTokenIds;
    Tensor const& proposalLengths;
    Tensor const& uniforms;
    Tensor const* maxAcceptLengths;
};
namespace spec_decode
{
void accept(GreedyAcceptInputs const& inputs, Tensor& ids, Tensor& counts, Tensor* rows, cudaStream_t stream,
    int32_t forcedAcceptanceLength = -1);
void accept(DenseProbabilityAcceptInputs const& inputs, Tensor& ids, Tensor& counts, cudaStream_t stream,
    int32_t forcedAcceptanceLength = -1);
void accept(SparseProbabilityAcceptInputs const& inputs, Tensor& ids, Tensor& counts, Tensor* rows, cudaStream_t stream,
    int32_t forcedAcceptanceLength = -1);
void accept(DenseTargetSparseDraftAcceptInputs const& inputs, Tensor& ids, Tensor& counts, Tensor* rows,
    cudaStream_t stream, int32_t forcedAcceptanceLength = -1);

//! Commit a clamped prefix or indexed path according to the resolved verification layout.
//! Hidden rows retain verifyTokenCount stride for chains and are packed to outputStride for trees.
//! @param layout The layout the path was accepted in; the deployment's when absent. A round that grafted forced
//!               tokens is indexed even on a chain deployment.
void commitAcceptedTargetPath(DecodingRuntimeContext& runtime, DecodingInferenceContext const& context,
    SpecAcceptResult const& accepted, int32_t verifyTokenCount, std::optional<SpecVerifyLayout> layout = std::nullopt);

//! Validate host-visible shape, device, dtype and configured capacity before enqueue.
void validateSpecVerifyInputs(SpecVerifyInputs const& inputs, SpecDecodeConfig const& config);

//! Prepare existing target bindings without executing the target or committing state.
void prepareTargetVerifyInputs(SpecVerifyInputs const& inputs, DecodingRuntimeContext& runtime,
    DecodingInferenceContext const& context, int32_t targetKvCacheIndex);
} // namespace spec_decode
} // namespace trt_edgellm::rt
