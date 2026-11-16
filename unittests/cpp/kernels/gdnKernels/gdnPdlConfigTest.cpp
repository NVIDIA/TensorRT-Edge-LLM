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

#include "kernels/gdnKernels/gdnPdlConfig.h"
#include "common/cudaMacros.h"

#include <gtest/gtest.h>

using namespace trt_edgellm;

TEST(GdnPdlConfigTest, EnablesOnlyRequestedSm12xPrefill)
{
    EXPECT_TRUE(useGdnPdl(/*requested=*/true, /*seqLen=*/2, /*smVersion=*/120,
        /*toolchainSupportsPdl=*/true));
    EXPECT_TRUE(useGdnPdl(/*requested=*/true, /*seqLen=*/2048, /*smVersion=*/121,
        /*toolchainSupportsPdl=*/true));
}

TEST(GdnPdlConfigTest, PreservesExplicitOffSwitch)
{
    for (int32_t const sm : {120, 121})
    {
        EXPECT_FALSE(useGdnPdl(/*requested=*/false, /*seqLen=*/2048, sm,
            /*toolchainSupportsPdl=*/true));
    }
}

TEST(GdnPdlConfigTest, RejectsUnsupportedToolchain)
{
    EXPECT_FALSE(useGdnPdl(/*requested=*/true, /*seqLen=*/2048, /*smVersion=*/121,
        /*toolchainSupportsPdl=*/false));
}

TEST(GdnPdlConfigTest, RejectsNonSm12xDevices)
{
    for (int32_t const sm : {90, 100, 110, 119, 122})
    {
        EXPECT_FALSE(useGdnPdl(/*requested=*/true, /*seqLen=*/2048, sm,
            /*toolchainSupportsPdl=*/true))
            << "sm=" << sm;
    }
}

TEST(GdnPdlConfigTest, RejectsDecodeAndInvalidSequenceLengths)
{
    for (int32_t const seqLen : {-1, 0, 1})
    {
        EXPECT_FALSE(useGdnPdl(/*requested=*/true, seqLen, /*smVersion=*/121,
            /*toolchainSupportsPdl=*/true))
            << "seqLen=" << seqLen;
    }
}

TEST(GdnPdlConfigTest, UsesCompileTimeToolchainGateByDefault)
{
    EXPECT_EQ(
        useGdnPdl(/*requested=*/true, /*seqLen=*/2048, /*smVersion=*/121), SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH != 0);
}
