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

#include "kernels/nvfp4A16BlackwellGemm/nvfp4A16BlackwellGemmRunner.h"
#include "kernels/nvfp4A16BlackwellSupport.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

namespace trt_edgellm::kernels
{
namespace
{

TEST(Nvfp4A16BlackwellGemmRunnerTest, SelectsSmallestCoveringTokenTile)
{
    EXPECT_EQ(Nvfp4A16BlackwellGemmRunner::selectTokenTile(1), Nvfp4A16BlackwellTokenTile::kM8);
    EXPECT_EQ(Nvfp4A16BlackwellGemmRunner::selectTokenTile(8), Nvfp4A16BlackwellTokenTile::kM8);
    EXPECT_EQ(Nvfp4A16BlackwellGemmRunner::selectTokenTile(9), Nvfp4A16BlackwellTokenTile::kM16);
    EXPECT_EQ(Nvfp4A16BlackwellGemmRunner::selectTokenTile(17), Nvfp4A16BlackwellTokenTile::kM32);
    EXPECT_EQ(Nvfp4A16BlackwellGemmRunner::selectTokenTile(33), Nvfp4A16BlackwellTokenTile::kM64);
    EXPECT_EQ(Nvfp4A16BlackwellGemmRunner::selectTokenTile(65), Nvfp4A16BlackwellTokenTile::kM128);
    EXPECT_EQ(Nvfp4A16BlackwellGemmRunner::selectTokenTile(129), Nvfp4A16BlackwellTokenTile::kM256);
    EXPECT_EQ(Nvfp4A16BlackwellGemmRunner::selectTokenTile(4096), Nvfp4A16BlackwellTokenTile::kM256);
}

TEST(Nvfp4A16BlackwellGemmRunnerTest, SelectsTailTokenTileClasses)
{
    struct TailCase
    {
        int32_t numTokens;
        Nvfp4A16BlackwellTokenTile expected;
    };
    constexpr TailCase cases[]{{9, Nvfp4A16BlackwellTokenTile::kM16}, {17, Nvfp4A16BlackwellTokenTile::kM32},
        {33, Nvfp4A16BlackwellTokenTile::kM64}, {65, Nvfp4A16BlackwellTokenTile::kM128},
        {129, Nvfp4A16BlackwellTokenTile::kM256}, {257, Nvfp4A16BlackwellTokenTile::kM256},
        {1023, Nvfp4A16BlackwellTokenTile::kM256}, {1025, Nvfp4A16BlackwellTokenTile::kM256},
        {2049, Nvfp4A16BlackwellTokenTile::kM256}, {4096, Nvfp4A16BlackwellTokenTile::kM256}};

    for (TailCase const& testCase : cases)
    {
        EXPECT_EQ(Nvfp4A16BlackwellGemmRunner::selectTokenTile(testCase.numTokens), testCase.expected)
            << "numTokens=" << testCase.numTokens;
    }
}

TEST(Nvfp4A16BlackwellGemmRunnerTest, RejectsInvalidOrNonSm110Shapes)
{
    EXPECT_FALSE(Nvfp4A16BlackwellGemmRunner::isSupported(100, Nvfp4A16BlackwellDtype::kFp16, 8, 3712, 2688));
    EXPECT_FALSE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kFp16, 0, 3712, 2688));
    EXPECT_FALSE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kFp16, 8, 129, 64));
    EXPECT_FALSE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kFp16, 8, 128, 65));
    EXPECT_FALSE(Nvfp4A16BlackwellGemmRunner::isSupported(
        110, Nvfp4A16BlackwellDtype::kFp16, std::numeric_limits<int32_t>::max(), 128, 512));
    EXPECT_FALSE(Nvfp4A16BlackwellGemmRunner::isSupported(110, static_cast<Nvfp4A16BlackwellDtype>(-1), 8, 3712, 2688));
}

TEST(Nvfp4A16BlackwellGemmRunnerTest, GenericShapeAndTmaPredicatesAreChecked)
{
    using namespace nvfp4_a16_blackwell;
    EXPECT_TRUE(isSupportedProblemShape(128, 64));
    EXPECT_TRUE(isSupportedProblemShape(256, 128));
    EXPECT_TRUE(isSupportedProblemShape(384, 192));
    EXPECT_TRUE(isSupportedProblemShape(3712, 2688));
    EXPECT_FALSE(isSupportedProblemShape(0, 64));
    EXPECT_FALSE(isSupportedProblemShape(128, -64));
    EXPECT_FALSE(isSupportedProblemShape(129, 64));
    EXPECT_FALSE(isSupportedProblemShape(128, 65));
    EXPECT_FALSE(isSupportedProblemShape(static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1, 64));

    EXPECT_TRUE(isTmaRepresentableProblem(4096, 384, 192));
    EXPECT_FALSE(isTmaRepresentableProblem(std::numeric_limits<int32_t>::max(), 128, 512));
    EXPECT_FALSE(isTmaRepresentableProblem(1, 2147483520LL, 1088));
}

TEST(Nvfp4A16BlackwellGemmRunnerTest, RequiresNoWorkspace)
{
    Nvfp4A16BlackwellGemmParams params{};
    EXPECT_EQ(Nvfp4A16BlackwellGemmRunner::getWorkspaceSize(params), 0U);
}

TEST(Nvfp4A16BlackwellGemmRunnerTest, RuntimeShapesShareCompiledTileVariants)
{
#if defined(CUTE_DSL_NVFP4_A16_BLACKWELL_GEMM_ENABLED)
    EXPECT_TRUE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kFp16, 8, 3712, 2688));
    EXPECT_TRUE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kFp16, 8, 2688, 3712));
    EXPECT_TRUE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kFp16, 8, 131072, 2688));
    EXPECT_TRUE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kFp16, 2049, 128, 64));
    EXPECT_TRUE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kFp16, 4096, 384, 192));
#else
    EXPECT_FALSE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kFp16, 8, 3712, 2688));
    EXPECT_FALSE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kFp16, 8, 2688, 3712));
    EXPECT_FALSE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kFp16, 8, 131072, 2688));
    EXPECT_FALSE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kFp16, 2049, 128, 64));
    EXPECT_FALSE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kFp16, 4096, 384, 192));
#endif
#if defined(CUTE_DSL_NVFP4_A16_BLACKWELL_GEMM_ENABLED)
    EXPECT_TRUE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kBf16, 8, 2688, 3712));
#else
    EXPECT_FALSE(Nvfp4A16BlackwellGemmRunner::isSupported(110, Nvfp4A16BlackwellDtype::kBf16, 8, 2688, 3712));
#endif
}

} // namespace
} // namespace trt_edgellm::kernels
