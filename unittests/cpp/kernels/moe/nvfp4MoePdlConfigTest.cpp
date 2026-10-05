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

#include "kernels/moe/nvfp4_cutedsl/nvfp4MoePdlConfig.h"
#include "common/cudaMacros.h"

#include <gtest/gtest.h>

using namespace trt_edgellm;

TEST(Nvfp4MoePdlConfigTest, EnablesRequestedPdlOnSupportedSms)
{
    for (int32_t const sm : {90, 100, 101, 110})
    {
        EXPECT_TRUE(useNvfp4MoePdl(/*requested=*/true, sm, /*toolchainSupportsPdl=*/true)) << "sm=" << sm;
    }
}

TEST(Nvfp4MoePdlConfigTest, PreservesExplicitOffSwitch)
{
    for (int32_t const sm : {90, 100, 101, 110})
    {
        EXPECT_FALSE(useNvfp4MoePdl(/*requested=*/false, sm, /*toolchainSupportsPdl=*/true)) << "sm=" << sm;
    }
}

TEST(Nvfp4MoePdlConfigTest, RejectsPreSm90)
{
    EXPECT_FALSE(useNvfp4MoePdl(/*requested=*/true, /*smVersion=*/89, /*toolchainSupportsPdl=*/true));
}

TEST(Nvfp4MoePdlConfigTest, RejectsUnsupportedToolchain)
{
    EXPECT_FALSE(useNvfp4MoePdl(/*requested=*/true, /*smVersion=*/110, /*toolchainSupportsPdl=*/false));
}

TEST(Nvfp4MoePdlConfigTest, UsesCompileTimeToolchainGateByDefault)
{
    EXPECT_EQ(useNvfp4MoePdl(/*requested=*/true, /*smVersion=*/110), SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH != 0);
}
