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

#include "runtime/decoding/specPleStaging.h"
#include "common/checkMacros.h"
#include "common/cudaUtils.h"
#include "runtime/decoding/decoderUtils.h"
#include "runtime/decoding/decodingStrategy.h"

namespace trt_edgellm::rt
{
SpecPleStaging::SpecPleStaging(DecodingRuntimeContext const& runtime, int32_t maxVerifyTokens, SpecVerifyLayout layout)
{
    int32_t const maxBatch = runtime.maxRuntimeBatchSize;
    auto const* const gemma = runtime.preprocess.gemma4Ple;
    auto const* const ngram = runtime.preprocess.qwen4ExpNgram;
    if (ngram != nullptr || (gemma != nullptr && gemma->needsHostTokens()))
    {
        mHostTokenIds = Tensor({maxBatch, maxVerifyTokens}, DeviceType::kCPU, trt_edgellm::DataType::kINT32,
            "SpecPleStaging::hostTokenIds");
        mHostQueryLengths
            = Tensor({maxBatch}, DeviceType::kCPU, trt_edgellm::DataType::kINT32, "SpecPleStaging::hostQueryLengths");
        mQueryLengths.reserve(static_cast<size_t>(maxBatch));
        mSlots.resize(static_cast<size_t>(maxBatch));
        if (ngram != nullptr && layout == SpecVerifyLayout::kTree)
        {
            mHostTreeParents = Tensor({maxBatch, maxVerifyTokens}, DeviceType::kCPU, trt_edgellm::DataType::kINT32,
                "SpecPleStaging::hostTreeParents");
        }
    }
}

void SpecPleStaging::embed(DecodingRuntimeContext& runtime, DecodingInferenceContext const& context,
    Tensor const& tokenIds, int32_t requestCount, int32_t tokensPerRequest, cudaStream_t stream)
{
    auto* const gemma = runtime.preprocess.gemma4Ple;
    auto* const ngram = runtime.preprocess.qwen4ExpNgram;
    Tensor& deviceQueryLengths = runtime.base.pipelineIO.queryLengths;
    bool const needsHost = ngram != nullptr || (gemma != nullptr && gemma->needsHostTokens());
    if (!needsHost)
    {
        if (gemma != nullptr)
        {
            gemma->embed(tokenIds, stream, {}, &deviceQueryLengths);
        }
        return;
    }
    // Graph-capture contexts carry no request tokens; the capture shapes the PLE bindings itself.
    if (context.tokenIds.size() < static_cast<size_t>(requestCount))
    {
        return;
    }
    ELLM_CHECK(!mHostTokenIds.isEmpty(), "Speculative PLE staging was not set up for this model");
    int32_t const batch = requestCount;
    int32_t const width = tokensPerRequest;
    check::check(mHostTokenIds.reshape({batch, width}) && mHostQueryLengths.reshape({batch}),
        "Speculative PLE staging exceeds its allocation");
    CUDA_CHECK(cudaMemcpyAsync(mHostTokenIds.rawPointer(), tokenIds.rawPointer(),
        static_cast<size_t>(batch) * width * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(mHostQueryLengths.rawPointer(), deviceQueryLengths.rawPointer(),
        static_cast<size_t>(batch) * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    if (!mHostTreeParents.isEmpty())
    {
        Tensor const& parents = runtime.base.pipelineIO.specTreeParentIds;
        ELLM_CHECK(parents.getShape().volume() == static_cast<int64_t>(batch) * width,
            "Tree verification needs one parent per verify token");
        check::check(mHostTreeParents.reshape({batch, width}), "Speculative PLE staging exceeds its allocation");
        CUDA_CHECK(cudaMemcpyAsync(mHostTreeParents.rawPointer(), parents.rawPointer(),
            static_cast<size_t>(batch) * width * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    int32_t const* lengths = mHostQueryLengths.dataPointer<int32_t>();
    mQueryLengths.assign(lengths, lengths + batch);

    if (gemma != nullptr)
    {
        if (gemma->needsHostTokens())
        {
            gemma->embed(tokenIds, stream, mQueryLengths, &deviceQueryLengths, &mHostTokenIds);
        }
        else
        {
            gemma->embed(tokenIds, stream, {}, &deviceQueryLengths);
        }
    }
    if (ngram != nullptr)
    {
        // The history ends at the last committed token, which a rejected draft may have run past.
        int32_t const contextLength = runtime.deployment.base.ngramContextLength();
        for (int32_t i = 0; i < batch; ++i)
        {
            auto const row = static_cast<size_t>(i);
            mSlots[row] = context.residentRefs[row].slot;
            decoder_utils::consumedTokenTail(context.tokenIds[row], context.rawBatchedInputIds[row],
                context.committedLengths[row], contextLength, mTail);
            ngram->seedHistory(mSlots[row], mTail.data(), static_cast<int32_t>(mTail.size()));
        }
        ngram->embed(mHostTokenIds.dataPointer<int32_t>(), mHostQueryLengths.dataPointer<int32_t>(), mSlots.data(),
            batch, width, stream, mHostTreeParents.isEmpty() ? nullptr : mHostTreeParents.dataPointer<int32_t>());
    }
}

} // namespace trt_edgellm::rt
