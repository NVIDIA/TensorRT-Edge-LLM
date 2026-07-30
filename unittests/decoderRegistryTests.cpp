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

#include "runtime/decoding/decoderRegistry.h"

#include <gtest/gtest.h>

using namespace trt_edgellm::rt;

TEST(DecoderRegistryPolicyTests, EagleFallsBackToVanillaForUnsupportedRequestFeatures)
{
    LLMGenerationRequest request{};
    request.temperature = 0.0F;
    request.topK = 1;
    request.topP = 1.0F;
    request.requests.resize(1);

    EXPECT_FALSE(eagleRequiresVanillaFallback(request, /*visionNeedsFallback=*/false));
    request.disableContextReuse = true;
    EXPECT_FALSE(eagleRequiresVanillaFallback(request, /*visionNeedsFallback=*/false));
    request.disableContextReuse = false;

    request.topK = 2;
    EXPECT_TRUE(eagleRequiresVanillaFallback(request, /*visionNeedsFallback=*/false));
    request.topK = 1;

    request.requests[0].audioBuffers.resize(1);
    EXPECT_TRUE(eagleRequiresVanillaFallback(request, /*visionNeedsFallback=*/false));
    request.requests[0].audioBuffers.clear();

    request.requests[0].imageBuffers.resize(1);
    EXPECT_FALSE(eagleRequiresVanillaFallback(request, /*visionNeedsFallback=*/false));
    EXPECT_TRUE(eagleRequiresVanillaFallback(request, /*visionNeedsFallback=*/true));
}

TEST(DecoderRegistryPolicyTests, HybridMtpEndpointReuseRequiresSelectedMtpAndLiveCachePolicy)
{
    EXPECT_TRUE(shouldUseHybridMtpEndpointReuse(DecodingStrategyKind::kMTP, /*hybridBase=*/true,
        /*contextCacheLookupEnabled=*/true, /*contextCachePublicationEnabled=*/false));
    EXPECT_TRUE(shouldUseHybridMtpEndpointReuse(DecodingStrategyKind::kMTP, /*hybridBase=*/true,
        /*contextCacheLookupEnabled=*/false, /*contextCachePublicationEnabled=*/true));
    EXPECT_FALSE(shouldUseHybridMtpEndpointReuse(DecodingStrategyKind::kMTP, /*hybridBase=*/true,
        /*contextCacheLookupEnabled=*/false, /*contextCachePublicationEnabled=*/false));
    EXPECT_FALSE(shouldUseHybridMtpEndpointReuse(DecodingStrategyKind::kVanilla, /*hybridBase=*/true,
        /*contextCacheLookupEnabled=*/true, /*contextCachePublicationEnabled=*/true));
    EXPECT_FALSE(shouldUseHybridMtpEndpointReuse(DecodingStrategyKind::kMTP, /*hybridBase=*/false,
        /*contextCacheLookupEnabled=*/true, /*contextCachePublicationEnabled=*/true));
}
