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

#include "kernels/decodeAttentionKernels/xqaLaunchConfig.h"
#include "common/cudaMacros.h"
#include "kernels/decodeAttentionKernels/decoderXQARunner.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace
{

using trt_edgellm::makeXQALaunchAttributes;
using trt_edgellm::XQALaunchAttributes;
using trt_edgellm::XQALaunchParams;

constexpr uint32_t kCLUSTER_DIM_X{2U};
constexpr int32_t kSM_89{89};
constexpr int32_t kSM_90{90};
constexpr int32_t kSM_110{110};

TEST(XQALaunchConfigTest, EnablesPdlByDefault)
{
    XQALaunchParams const params{};
    EXPECT_TRUE(params.enablePdl);
}

#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
TEST(XQALaunchConfigTest, AddsPdlAttributeForRegularLaunchOnSm90OrNewer)
{
    XQALaunchAttributes const attributes = makeXQALaunchAttributes(false, kCLUSTER_DIM_X, true, kSM_90);

    ASSERT_EQ(attributes.count, 1U);
    EXPECT_EQ(attributes.attrs[0].id, CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION);
    EXPECT_EQ(attributes.attrs[0].value.programmaticStreamSerializationAllowed, 1);
}

TEST(XQALaunchConfigTest, OmitsPdlAttributeForPreSm90)
{
    XQALaunchAttributes const attributes = makeXQALaunchAttributes(false, kCLUSTER_DIM_X, true, kSM_89);

    EXPECT_EQ(attributes.count, 0U);
}

TEST(XQALaunchConfigTest, OmitsPdlAttributeWhenDisabled)
{
    XQALaunchAttributes const attributes = makeXQALaunchAttributes(false, kCLUSTER_DIM_X, false, kSM_110);

    EXPECT_EQ(attributes.count, 0U);
}
#else
TEST(XQALaunchConfigTest, OmitsPdlAttributeForUnsupportedToolchain)
{
    XQALaunchAttributes const attributes = makeXQALaunchAttributes(false, kCLUSTER_DIM_X, true, kSM_110);

    EXPECT_EQ(attributes.count, 0U);
}
#endif // SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH

#if SUPPORTS_CLUSTER_LAUNCH
TEST(XQALaunchConfigTest, PreservesClusterAttributeAndAddsPdl)
{
    XQALaunchAttributes const attributes = makeXQALaunchAttributes(true, kCLUSTER_DIM_X, true, kSM_110);

#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
    ASSERT_EQ(attributes.count, 2U);
#else
    ASSERT_EQ(attributes.count, 1U);
#endif // SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
    EXPECT_EQ(attributes.attrs[0].id, CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION);
    EXPECT_EQ(attributes.attrs[0].value.clusterDim.x, kCLUSTER_DIM_X);
    EXPECT_EQ(attributes.attrs[0].value.clusterDim.y, 1U);
    EXPECT_EQ(attributes.attrs[0].value.clusterDim.z, 1U);

#if SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
    EXPECT_EQ(attributes.attrs[1].id, CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION);
    EXPECT_EQ(attributes.attrs[1].value.programmaticStreamSerializationAllowed, 1);
#endif // SUPPORTS_PROGRAMMATIC_DEPENDENT_LAUNCH
}

TEST(XQALaunchConfigTest, PreservesClusterAttributeWhenPdlIsDisabled)
{
    XQALaunchAttributes const attributes = makeXQALaunchAttributes(true, kCLUSTER_DIM_X, false, kSM_110);

    ASSERT_EQ(attributes.count, 1U);
    EXPECT_EQ(attributes.attrs[0].id, CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION);
}
#endif // SUPPORTS_CLUSTER_LAUNCH

} // namespace
