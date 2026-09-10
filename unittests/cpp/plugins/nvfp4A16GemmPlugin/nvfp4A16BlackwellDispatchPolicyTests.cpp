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

#include "plugins/nvfp4A16GemmPlugin/nvfp4A16BlackwellDispatchPolicy.h"

#include <array>
#include <cstdint>

#include <gtest/gtest.h>

namespace trt_edgellm::plugins
{
namespace
{

constexpr std::array<int32_t, 5> kGemvM{1, 2, 4, 8, 16};
constexpr std::array<int32_t, 12> kRequiredM{1, 2, 4, 8, 16, 32, 64, 128, 1024, 2048, 2049, 4096};

struct ShapePolicyCase
{
    int32_t n;
    int32_t k;
    std::array<int32_t, 5> fp16SplitK;
    std::array<int32_t, 5> bf16SplitK;
};

constexpr std::array<ShapePolicyCase, 3> kShapePolicies{{
    {3712, 2688, {2, 2, 2, 2, 4}, {2, 2, 2, 2, 4}},
    {2688, 3712, {4, 4, 4, 4, 4}, {4, 4, 4, 4, 4}},
    {131072, 2688, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
}};

constexpr std::array<Nvfp4A16BlackwellDispatchDtype, 2> kDtypes{
    Nvfp4A16BlackwellDispatchDtype::kFp16, Nvfp4A16BlackwellDispatchDtype::kBf16};

TEST(Nvfp4A16BlackwellDispatchPolicyTest, LocksEveryForcedGemvSplitKEntry)
{
    for (auto const& shape : kShapePolicies)
    {
        for (auto const dtype : kDtypes)
        {
            auto const& expectedSplitK
                = dtype == Nvfp4A16BlackwellDispatchDtype::kFp16 ? shape.fp16SplitK : shape.bf16SplitK;
            for (std::size_t i = 0; i < kGemvM.size(); ++i)
            {
                auto const dispatch = getNvfp4A16BlackwellDispatch(
                    Nvfp4A16BlackwellDispatchBackend::kGemv, kGemvM[i], shape.n, shape.k, dtype);
                EXPECT_EQ(dispatch.kernel, Nvfp4A16BlackwellKernel::kGemv)
                    << "N=" << shape.n << " K=" << shape.k << " M=" << kGemvM[i]
                    << " dtype=" << static_cast<int32_t>(dtype);
                EXPECT_EQ(dispatch.splitK, expectedSplitK[i])
                    << "N=" << shape.n << " K=" << shape.k << " M=" << kGemvM[i]
                    << " dtype=" << static_cast<int32_t>(dtype);
            }
        }
    }
}

TEST(Nvfp4A16BlackwellDispatchPolicyTest, LocksFixedDefaultDispatchForEveryRequiredCase)
{
    for (auto const& shape : kShapePolicies)
    {
        for (auto const dtype : kDtypes)
        {
            auto const& expectedSplitK
                = dtype == Nvfp4A16BlackwellDispatchDtype::kFp16 ? shape.fp16SplitK : shape.bf16SplitK;
            for (int32_t const m : kRequiredM)
            {
                auto const dispatch = getNvfp4A16BlackwellDispatch(
                    Nvfp4A16BlackwellDispatchBackend::kDefault, m, shape.n, shape.k, dtype);
                bool const expectedGemv = m == 1;
                EXPECT_EQ(
                    dispatch.kernel, expectedGemv ? Nvfp4A16BlackwellKernel::kGemv : Nvfp4A16BlackwellKernel::kTcgen05)
                    << "N=" << shape.n << " K=" << shape.k << " M=" << m << " dtype=" << static_cast<int32_t>(dtype);
                int32_t const expected = expectedGemv ? expectedSplitK[0] : 1;
                EXPECT_EQ(dispatch.splitK, expected)
                    << "N=" << shape.n << " K=" << shape.k << " M=" << m << " dtype=" << static_cast<int32_t>(dtype);
            }
        }
    }
}

TEST(Nvfp4A16BlackwellDispatchPolicyTest, GenericAlignedShapesUseGemvOnlyAtM1)
{
    for (auto const dtype : kDtypes)
    {
        for (auto const shape :
            {std::array<int32_t, 2>{128, 64}, std::array<int32_t, 2>{256, 128}, std::array<int32_t, 2>{384, 192}})
        {
            auto const decode = getNvfp4A16BlackwellDispatch(
                Nvfp4A16BlackwellDispatchBackend::kDefault, 1, shape[0], shape[1], dtype);
            EXPECT_EQ(decode.kernel, Nvfp4A16BlackwellKernel::kGemv);
            EXPECT_EQ(decode.splitK, 1);

            auto const prefill = getNvfp4A16BlackwellDispatch(
                Nvfp4A16BlackwellDispatchBackend::kDefault, 2, shape[0], shape[1], dtype);
            EXPECT_EQ(prefill.kernel, Nvfp4A16BlackwellKernel::kTcgen05);
            EXPECT_EQ(prefill.splitK, 1);

            for (int32_t const m : kGemvM)
            {
                auto const forced = getNvfp4A16BlackwellDispatch(
                    Nvfp4A16BlackwellDispatchBackend::kGemv, m, shape[0], shape[1], dtype);
                EXPECT_EQ(forced.kernel, Nvfp4A16BlackwellKernel::kGemv);
                EXPECT_EQ(forced.splitK, 1);
            }
        }
    }
}

TEST(Nvfp4A16BlackwellDispatchPolicyTest, ForcedBackendsCoverEveryRequiredCase)
{
    for (auto const& shape : kShapePolicies)
    {
        for (auto const dtype : kDtypes)
        {
            for (int32_t const m : kRequiredM)
            {
                auto const tcgen = getNvfp4A16BlackwellDispatch(
                    Nvfp4A16BlackwellDispatchBackend::kTcgen05, m, shape.n, shape.k, dtype);
                EXPECT_EQ(tcgen.kernel, Nvfp4A16BlackwellKernel::kTcgen05);
                EXPECT_EQ(tcgen.splitK, 1);

                auto const gemv
                    = getNvfp4A16BlackwellDispatch(Nvfp4A16BlackwellDispatchBackend::kGemv, m, shape.n, shape.k, dtype);
                EXPECT_EQ(gemv.kernel, m <= 16 ? Nvfp4A16BlackwellKernel::kGemv : Nvfp4A16BlackwellKernel::kUnsupported)
                    << "N=" << shape.n << " K=" << shape.k << " M=" << m << " dtype=" << static_cast<int32_t>(dtype);
            }
        }
    }
}

TEST(Nvfp4A16BlackwellDispatchPolicyTest, RejectsInvalidConfiguration)
{
    auto const dtype = Nvfp4A16BlackwellDispatchDtype::kFp16;
    EXPECT_EQ(getNvfp4A16BlackwellDispatch(Nvfp4A16BlackwellDispatchBackend::kDefault, 0, 3712, 2688, dtype).kernel,
        Nvfp4A16BlackwellKernel::kUnsupported);
    EXPECT_EQ(getNvfp4A16BlackwellDispatch(Nvfp4A16BlackwellDispatchBackend::kDefault, 1, 129, 64, dtype).kernel,
        Nvfp4A16BlackwellKernel::kUnsupported);
    EXPECT_EQ(getNvfp4A16BlackwellDispatch(Nvfp4A16BlackwellDispatchBackend::kDefault, 1, 128, 65, dtype).kernel,
        Nvfp4A16BlackwellKernel::kUnsupported);
    // M*N*2 reaches the 40-bit TMA byte-stride limit exactly.
    EXPECT_EQ(
        getNvfp4A16BlackwellDispatch(Nvfp4A16BlackwellDispatchBackend::kDefault, 512, 1073741824, 64, dtype).kernel,
        Nvfp4A16BlackwellKernel::kUnsupported);
    EXPECT_EQ(getNvfp4A16BlackwellDispatch(Nvfp4A16BlackwellDispatchBackend::kDefault, 1, 3712, 2688,
                  static_cast<Nvfp4A16BlackwellDispatchDtype>(-1))
                  .kernel,
        Nvfp4A16BlackwellKernel::kUnsupported);
    EXPECT_EQ(
        getNvfp4A16BlackwellDispatch(static_cast<Nvfp4A16BlackwellDispatchBackend>(-1), 1, 3712, 2688, dtype).kernel,
        Nvfp4A16BlackwellKernel::kUnsupported);
}

} // namespace
} // namespace trt_edgellm::plugins
