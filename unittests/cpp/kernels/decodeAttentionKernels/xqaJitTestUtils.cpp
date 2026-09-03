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

#include "xqaJitTestUtils.h"

#include "kernels/decodeAttentionKernels/decoderXQAJitCompiler.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"

#include <mutex>
#include <set>
#include <tuple>

namespace trt_edgellm
{
namespace
{

using CacheKey = std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t, bool, bool>;

CacheKey makeCacheKey(XQAJitKey const& key)
{
    return {key.sm, static_cast<int32_t>(key.dataType), static_cast<int32_t>(key.kvDataType), key.headSize,
        key.qHeadsPerKv, key.tokensPerPage, key.slidingWindow, key.specDecode};
}

} // namespace

bool loadXQAJitKernelForTest(int32_t smVersion, nvinfer1::DataType dataType, nvinfer1::DataType kvDataType,
    int32_t headSize, int32_t numQHeads, int32_t numKVHeads, bool slidingWindow, bool specDecode, int32_t tokensPerPage)
{
    if (numKVHeads == 0 || numQHeads % numKVHeads != 0)
    {
        return false;
    }

    XQAJitKey const key{
        smVersion, dataType, kvDataType, headSize, numQHeads / numKVHeads, tokensPerPage, slidingWindow, specDecode};
    CacheKey const cacheKey = makeCacheKey(key);

    static std::mutex sMutex;
    static std::set<CacheKey> sLoadedKeys;

    std::lock_guard<std::mutex> const lock{sMutex};
    if (sLoadedKeys.count(cacheKey) > 0)
    {
        return true;
    }

    XQAJitResult const jitResult = compileXQAKernel(key);
    bool const loaded
        = DecoderXQARunner::loadDecodeXQAKernelFromCubin(key, jitResult.cubin.data(), jitResult.cubin.size());
    if (loaded)
    {
        sLoadedKeys.insert(cacheKey);
    }
    return loaded;
}

} // namespace trt_edgellm
