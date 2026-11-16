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

#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace trt_edgellm::rt
{
struct DecodingRuntimeContext;
struct DecodingInferenceContext;

//! Host staging for the PLE preprocessors that read speculative verify rows on the host
//! (Gemma4 host-token lookups, the Qwen4-Exp n-gram injection).
class SpecPleStaging
{
public:
    SpecPleStaging() = default;

    //! Call once the base engine and its PipelineIO exist.
    SpecPleStaging(DecodingRuntimeContext const& runtime, int32_t maxVerifyTokens, SpecVerifyLayout layout);

    //! Feed the verify rows to the PLE preprocessors; call after the ragged bindings, which fill
    //! the query lengths that mask padding.
    void embed(DecodingRuntimeContext& runtime, DecodingInferenceContext const& context, Tensor const& tokenIds,
        int32_t requestCount, int32_t tokensPerRequest, cudaStream_t stream);

private:
    Tensor mHostTokenIds;               //!< [maxBatch, maxVerifyTokens] pinned
    Tensor mHostQueryLengths;           //!< [maxBatch] pinned
    Tensor mHostTreeParents;            //!< [maxBatch, maxVerifyTokens] pinned; n-gram tree verify only
    std::vector<int32_t> mQueryLengths; //!< Gemma's embed takes host lengths as a vector
    std::vector<int32_t> mSlots;        //!< Resident slot per execution row
    std::vector<int32_t> mTail;         //!< One sequence's consumed-token tail
};
} // namespace trt_edgellm::rt
